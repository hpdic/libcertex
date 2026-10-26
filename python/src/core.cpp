/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Python binding of the CUDA C++ library (cuvs::neighbors::certex): certex._core. Arrays arrive by
// DLPack (torch, CuPy, numpy, ...) and outputs are written into arrays that the caller allocates,
// so that the binding owns no memory besides the index. Every call runs on the CUDA stream that the
// caller passes (torch.cuda.current_stream() in certex.Index) and releases the GIL, so that
// searches from several Python threads run concurrently with updates.

#include <cuvs/neighbors/certex.hpp>
#include <cuvs/neighbors/certex_shards.hpp>
#include <cuvs/neighbors/certex_tune.hpp>

#include "../../cpp/src/neighbors/certex/detail/kernels.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace nb = nanobind;

namespace {

namespace certex = cuvs::neighbors::certex;
using cuvs::distance::DistanceType;

using any_array = nb::ndarray<nb::c_contig>;

/**
 * Stream-ordered allocations (cudaMallocAsync) on a device whose RMM resource is still the initial
 * cuda_memory_resource: the temporaries of a search then never synchronize the device (BIGANN-10M
 * on an H100: 17% to 20% more queries per second). A resource that the caller configured is kept,
 * and so is the initial one with CERTEX_KEEP_RMM set.
 */
/** The memory pool of the stream-ordered allocator of every device that uses it. */
auto async_pools() -> std::map<int, cudaMemPool_t>&
{
  static std::map<int, cudaMemPool_t> pools;
  return pools;
}

void use_async_allocation(int device)
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  if (std::getenv("CERTEX_KEEP_RMM") != nullptr) { return; }
  int current = 0;
  if (cudaGetDevice(&current) != cudaSuccess) { return; }
  if (current != device && cudaSetDevice(device) != cudaSuccess) { return; }
  if (rmm::mr::get_current_device_resource_ref() ==
      rmm::device_async_resource_ref{rmm::mr::detail::initial_resource()}) {
    rmm::mr::cuda_async_memory_resource mr{};
    async_pools()[device] = mr.pool_handle();
    rmm::mr::set_current_device_resource(std::move(mr));
  }
  if (current != device) { cudaSetDevice(current); }
}

/** raft::resources of the calling thread and the current device, running on the given stream. */
auto resources(uintptr_t stream) -> raft::resources&
{
  thread_local std::map<int, std::unique_ptr<raft::resources>> per_device;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) { throw std::runtime_error("certex: no CUDA device"); }
  auto& r = per_device[device];
  if (!r) {
    use_async_allocation(device);
    r = std::make_unique<raft::resources>();
  }
  raft::resource::set_cuda_stream(*r, cuda::stream_ref(reinterpret_cast<cudaStream_t>(stream)));
  return *r;
}

enum class dtype { f32, f16, i8, u8 };

auto dtype_of(const any_array& a) -> dtype
{
  if (a.dtype() == nb::dtype<float>()) { return dtype::f32; }
  if (a.dtype() == nb::dtype<int8_t>()) { return dtype::i8; }
  if (a.dtype() == nb::dtype<uint8_t>()) { return dtype::u8; }
  const auto t = a.dtype();
  if (t.code == uint8_t(nb::dlpack::dtype_code::Float) && t.bits == 16 && t.lanes == 1) {
    return dtype::f16;
  }
  throw nb::type_error("certex: vectors must be float32, float16, int8, or uint8");
}

auto on_device(const any_array& a) -> bool
{
  if (a.device_type() == nb::device::cuda::value) { return true; }
  if (a.device_type() == nb::device::cpu::value) { return false; }
  throw nb::type_error("certex: arrays must live in host or CUDA device memory");
}

template <typename T>
auto as_matrix(const any_array& a)
{
  if (a.ndim() != 2) { throw nb::value_error("certex: expected a matrix [n, dim]"); }
  return std::make_pair(static_cast<const T*>(a.data()),
                        std::array<int64_t, 2>{int64_t(a.shape(0)), int64_t(a.shape(1))});
}

template <typename T>
auto device_matrix(const any_array& a)
{
  if (!on_device(a)) { throw nb::value_error("certex: expected an array in device memory"); }
  auto [p, s] = as_matrix<T>(a);
  return raft::make_device_matrix_view<const T, int64_t>(p, s[0], s[1]);
}

template <typename T>
auto host_matrix(const any_array& a)
{
  auto [p, s] = as_matrix<T>(a);
  return raft::make_host_matrix_view<const T, int64_t>(p, s[0], s[1]);
}

template <typename T>
auto device_vector(const any_array& a, const char* what)
{
  if (!on_device(a) || a.ndim() != 1 || a.dtype() != nb::dtype<T>()) {
    throw nb::value_error((std::string("certex: ") + what).c_str());
  }
  return raft::make_device_vector_view<const T, int64_t>(static_cast<const T*>(a.data()),
                                                         int64_t(a.shape(0)));
}

/** Data type T of a certex::index<T>. */
template <typename I>
struct value_of;
template <typename T>
struct value_of<certex::index<T>> {
  using type = T;
};
template <typename I>
using value_t = typename value_of<std::decay_t<I>>::type;

/** The CUDA index of one of the four data types, behind one Python class. */
class Index {
 public:
  using variant = std::variant<certex::index<float>,
                               certex::index<half>,
                               certex::index<int8_t>,
                               certex::index<uint8_t>>;

  explicit Index(variant v) : idx_(std::move(v)) {}

  template <typename F>
  decltype(auto) visit(F&& f)
  {
    return std::visit(std::forward<F>(f), idx_);
  }
  template <typename F>
  decltype(auto) visit(F&& f) const
  {
    return std::visit(std::forward<F>(f), idx_);
  }

  auto data_type() const -> std::string
  {
    return std::visit(
      [](auto const& i) -> std::string {
        using T = value_t<decltype(i)>;
        if constexpr (std::is_same_v<T, float>) { return "float32"; }
        if constexpr (std::is_same_v<T, half>) { return "float16"; }
        if constexpr (std::is_same_v<T, int8_t>) { return "int8"; }
        return "uint8";
      },
      idx_);
  }

 private:
  variant idx_;
};

auto metric_of(const std::string& m) -> DistanceType
{
  if (m == "l2") { return DistanceType::L2Expanded; }
  if (m == "ip") { return DistanceType::InnerProduct; }
  if (m == "cosine") { return DistanceType::CosineExpanded; }
  throw nb::value_error("certex: metric must be l2, ip, or cosine");
}

auto metric_name(DistanceType m) -> std::string
{
  switch (m) {
    case DistanceType::InnerProduct: return "ip";
    case DistanceType::CosineExpanded: return "cosine";
    default: return "l2";
  }
}

template <typename T>
auto build_typed(raft::resources& res, certex::index_params const& p, const any_array& x)
  -> certex::index<T>
{
  if (on_device(x)) { return certex::build(res, p, device_matrix<T>(x)); }
  return certex::build(res, p, host_matrix<T>(x));
}

auto do_build(const any_array& x,
              const std::string& metric,
              int width,
              double variance,
              int anchor_stride,
              int64_t pca_sample,
              bool host_vectors,
              int n_lists,
              bool local_projections,
              bool int8_rows,
              uintptr_t stream) -> Index*
{
  certex::index_params p;
  p.metric            = metric_of(metric);
  p.width             = width;
  p.variance          = variance;
  p.anchor_stride     = anchor_stride;
  p.pca_sample        = pca_sample;
  p.host_vectors      = host_vectors;
  p.n_lists           = n_lists;
  p.local_projections = local_projections;
  p.int8_rows         = int8_rows;
  const auto t        = dtype_of(x);
  nb::gil_scoped_release nogil;
  auto& res = resources(stream);
  Index* out;
  switch (t) {
    case dtype::f32: out = new Index(build_typed<float>(res, p, x)); break;
    case dtype::f16: out = new Index(build_typed<half>(res, p, x)); break;
    case dtype::i8: out = new Index(build_typed<int8_t>(res, p, x)); break;
    default: out = new Index(build_typed<uint8_t>(res, p, x)); break;
  }
  raft::resource::sync_stream(res);
  return out;
}

/** The tuner (certex_tune.hpp) on a host dataset and host queries of one data type: the choice and
 * every candidate. */
auto do_tune(const any_array& x,
             const any_array& q,
             const std::string& metric,
             int k,
             bool host_vectors,
             int n_lists,
             std::vector<int> widths,
             std::vector<int> strides,
             int64_t small_prefix,
             int64_t large_prefix,
             int64_t n_queries,
             int batch_size,
             double memory_budget,
             uintptr_t stream) -> nb::dict
{
  if (on_device(x) || on_device(q)) { throw nb::value_error("certex: tune takes host arrays"); }
  if (dtype_of(x) != dtype_of(q)) {
    throw nb::type_error("certex: dataset and queries of one type");
  }
  certex::index_params p;
  p.metric       = metric_of(metric);
  p.host_vectors = host_vectors;
  p.n_lists      = n_lists;
  certex::tune_params tp;
  tp.widths        = std::move(widths);
  tp.strides       = std::move(strides);
  tp.small_prefix  = small_prefix;
  tp.large_prefix  = large_prefix;
  tp.n_queries     = n_queries;
  tp.batch_size    = batch_size;
  tp.memory_budget = memory_budget;
  const auto t     = dtype_of(x);
  certex::tune_result r;
  {
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    auto run  = [&](auto tag) {
      using T = decltype(tag);
      return certex::tune(res, p, host_matrix<T>(x), host_matrix<T>(q), k, tp);
    };
    switch (t) {
      case dtype::f32: r = run(float{}); break;
      case dtype::f16: r = run(half{}); break;
      case dtype::i8: r = run(int8_t{}); break;
      default: r = run(uint8_t{}); break;
    }
  }
  nb::dict out, model;
  out["width"]          = r.width;
  out["anchor_stride"]  = r.anchor_stride;
  out["predicted_qps"]  = r.predicted_qps;
  model["a_mac"]        = r.model.a_mac;
  model["a_pair"]       = r.model.a_pair;
  model["a_surv"]       = r.model.a_surv;
  model["a_amac"]       = r.model.a_amac;
  model["a_apair"]      = r.model.a_apair;
  model["a_sort"]       = r.model.a_sort;
  model["a_fetch"]      = r.model.a_fetch;
  model["a_mac_large"]  = r.model.a_mac_large;
  model["a_pair_large"] = r.model.a_pair_large;
  model["a_batch"]      = r.model.a_batch;
  out["model"]          = model;
  nb::list cands;
  for (auto const& c : r.candidates) {
    nb::dict d;
    d["width"]         = c.width;
    d["anchor_stride"] = c.anchor_stride;
    d["survivors"]     = c.survivors;
    d["verified"]      = c.verified;
    d["predicted_qps"] = c.predicted_qps;
    d["device_bytes"]  = c.device_bytes;
    d["fits"]          = c.fits;
    cands.append(d);
  }
  out["candidates"] = cands;
  return out;
}

template <typename T>
void check_type(const any_array& a)
{
  bool ok;
  if constexpr (std::is_same_v<T, half>) {
    ok = dtype_of(a) == dtype::f16;
  } else {
    ok = a.dtype() == nb::dtype<T>();
  }
  if (!ok) { throw nb::type_error("certex: the array must have the data type of the index"); }
}

void do_extend(Index& self, const any_array& x, uintptr_t stream)
{
  self.visit([&](auto& idx) {
    using T = value_t<decltype(idx)>;
    check_type<T>(x);
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    if (on_device(x)) {
      certex::extend(res, device_matrix<T>(x), &idx);
    } else {
      certex::extend(res, host_matrix<T>(x), &idx);
    }
    raft::resource::sync_stream(res);
  });
}

void do_remove(Index& self, const any_array& ids, uintptr_t stream)
{
  auto v = device_vector<int64_t>(ids, "ids must be an int64 device vector");
  self.visit([&](auto& idx) {
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    certex::remove(res, v, &idx);
    raft::resource::sync_stream(res);
  });
}

void do_set_labels(Index& self, const any_array& labels, int64_t first, uintptr_t stream)
{
  auto v = device_vector<int32_t>(labels, "labels must be an int32 device vector");
  self.visit([&](auto& idx) {
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    certex::set_labels(res, v, &idx, first);
    raft::resource::sync_stream(res);
  });
}

/**
 * Search into neighbors [nq, k] (int64) and distances [nq, k] (float32). filter: none, a bitset of
 * uint32 words over the vector ids (filter_rows = 0), or a bitmap of filter_rows rows of
 * filter_cols bits (one row per query); query_labels: one int32 label per query.
 */
void do_search(const Index& self,
               const any_array& queries,
               const any_array& neighbors,
               const any_array& distances,
               int anchors,
               int batch_size,
               int probe_lists,
               std::optional<any_array> filter_words,
               int64_t filter_rows,
               int64_t filter_cols,
               std::optional<any_array> query_labels,
               uintptr_t stream)
{
  if (!on_device(neighbors) || neighbors.ndim() != 2 || neighbors.dtype() != nb::dtype<int64_t>() ||
      !on_device(distances) || distances.ndim() != 2 || distances.dtype() != nb::dtype<float>()) {
    throw nb::value_error("certex: outputs must be int64 and float32 device matrices [nq, k]");
  }
  certex::search_params sp;
  sp.anchors     = anchors;
  sp.batch_size  = batch_size;
  sp.probe_lists = probe_lists;
  auto nb_view =
    raft::make_device_matrix_view<int64_t, int64_t>(static_cast<int64_t*>(neighbors.data()),
                                                    int64_t(neighbors.shape(0)),
                                                    int64_t(neighbors.shape(1)));
  auto ds_view =
    raft::make_device_matrix_view<float, int64_t>(static_cast<float*>(distances.data()),
                                                  int64_t(distances.shape(0)),
                                                  int64_t(distances.shape(1)));
  self.visit([&](auto const& idx) {
    using T = value_t<decltype(idx)>;
    check_type<T>(queries);
    auto q = device_matrix<T>(queries);
    std::optional<raft::device_vector_view<const int32_t, int64_t>> ql;
    if (query_labels) { ql = device_vector<int32_t>(*query_labels, "query labels must be int32"); }
    uint32_t* words = nullptr;
    if (filter_words) {
      if (!on_device(*filter_words) || filter_words->dtype() != nb::dtype<int32_t>()) {
        throw nb::value_error("certex: the filter must be a device array of 32-bit words");
      }
      words = static_cast<uint32_t*>(filter_words->data());
    }
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    if (ql) {
      certex::search(res, sp, idx, q, *ql, nb_view, ds_view);
    } else if (words && filter_rows == 0) {
      cuvs::core::bitset_view<uint32_t, int64_t> view(words, filter_cols);
      certex::search(
        res, sp, idx, q, nb_view, ds_view, cuvs::neighbors::filtering::bitset_filter(view));
    } else if (words) {
      cuvs::core::bitmap_view<uint32_t, int64_t> view(words, filter_rows, filter_cols);
      certex::search(
        res, sp, idx, q, nb_view, ds_view, cuvs::neighbors::filtering::bitmap_filter(view));
    } else {
      certex::search(res, sp, idx, q, nb_view, ds_view);
    }
    raft::resource::sync_stream(res);
  });
}

void do_save(const Index& self, const std::string& path, uintptr_t stream)
{
  self.visit([&](auto const& idx) {
    nb::gil_scoped_release nogil;
    auto& res = resources(stream);
    certex::serialize(res, path, idx);
  });
}

auto do_load(const std::string& path, const std::string& data_type, uintptr_t stream) -> Index*
{
  nb::gil_scoped_release nogil;
  auto& res = resources(stream);
  auto one  = [&](auto tag) {
    using T = decltype(tag);
    certex::index<T> idx(res);
    certex::deserialize(res, path, &idx);
    return new Index(std::move(idx));
  };
  if (data_type == "float32") { return one(float{}); }
  if (data_type == "float16") { return one(half{}); }
  if (data_type == "int8") { return one(int8_t{}); }
  if (data_type == "uint8") { return one(uint8_t{}); }
  throw std::invalid_argument("certex: unknown data type " + data_type);
}

}  // namespace

namespace {

/** A sharded index (certex_shards.hpp) of one of the four data types: one index per GPU. */
class Sharded {
 public:
  using variant = std::variant<certex::sharded_index<float>,
                               certex::sharded_index<half>,
                               certex::sharded_index<int8_t>,
                               certex::sharded_index<uint8_t>>;
  explicit Sharded(variant v) : idx_(std::move(v)) {}
  Sharded(const Sharded&)            = delete;  // shards own their GPUs
  Sharded& operator=(const Sharded&) = delete;
  template <typename F>
  decltype(auto) visit(F&& f)
  {
    return std::visit(std::forward<F>(f), idx_);
  }

 private:
  variant idx_;
};

auto do_build_sharded(const any_array& x,
                      std::vector<int> devices,
                      const std::string& metric,
                      int width,
                      int anchor_stride,
                      bool host_vectors,
                      int n_lists,
                      bool local_projections) -> Sharded*
{
  if (on_device(x)) { throw nb::value_error("certex: a sharded index is built from host memory"); }
  for (int dev : devices) {
    use_async_allocation(dev);
  }
  certex::index_params p;
  p.metric            = metric_of(metric);
  p.width             = width;
  p.anchor_stride     = anchor_stride;
  p.host_vectors      = host_vectors;
  p.n_lists           = n_lists;
  p.local_projections = local_projections;
  const auto t        = dtype_of(x);
  nb::gil_scoped_release nogil;
  switch (t) {
    case dtype::f32: return new Sharded(certex::build_sharded(devices, p, host_matrix<float>(x)));
    case dtype::f16: return new Sharded(certex::build_sharded(devices, p, host_matrix<half>(x)));
    case dtype::i8: return new Sharded(certex::build_sharded(devices, p, host_matrix<int8_t>(x)));
    default: return new Sharded(certex::build_sharded(devices, p, host_matrix<uint8_t>(x)));
  }
}

void do_search_sharded(Sharded& self,
                       const any_array& q,
                       any_array neighbors,
                       any_array distances,
                       int anchors,
                       int batch_size,
                       int probe_lists)
{
  if (on_device(q) || on_device(neighbors) || on_device(distances)) {
    throw nb::value_error("certex: a sharded search takes host arrays");
  }
  certex::search_params sp;
  sp.anchors       = anchors;
  sp.batch_size    = batch_size;
  sp.probe_lists   = probe_lists;
  const int64_t nq = int64_t(neighbors.shape(0)), k = int64_t(neighbors.shape(1));
  auto nb_view =
    raft::make_host_matrix_view<int64_t, int64_t>(static_cast<int64_t*>(neighbors.data()), nq, k);
  auto d_view =
    raft::make_host_matrix_view<float, int64_t>(static_cast<float*>(distances.data()), nq, k);
  self.visit([&](auto& idx) {
    using T = value_t<decltype(*idx.shards[0])>;
    check_type<T>(q);
    nb::gil_scoped_release nogil;
    certex::search(sp, idx, host_matrix<T>(q), nb_view, d_view);
  });
}

}  // namespace

NB_MODULE(_core, m)
{
  m.doc() = "CUDA C++ core of certex (cuvs::neighbors::certex)";

  nb::class_<Index>(m, "Index")
    .def_prop_ro("size",
                 [](const Index& s) { return s.visit([](auto const& i) { return i.size(); }); })
    .def_prop_ro(
      "n_removed",
      [](const Index& s) { return s.visit([](auto const& i) { return i.n_removed(); }); })
    .def_prop_ro("dim",
                 [](const Index& s) { return s.visit([](auto const& i) { return i.dim(); }); })
    .def_prop_ro("width",
                 [](const Index& s) { return s.visit([](auto const& i) { return i.width(); }); })
    .def_prop_ro(
      "row_length",
      [](const Index& s) { return s.visit([](auto const& i) { return i.row_length(); }); })
    .def_prop_ro(
      "anchor_stride",
      [](const Index& s) { return s.visit([](auto const& i) { return i.anchor_stride(); }); })
    .def_prop_ro(
      "host_vectors",
      [](const Index& s) { return s.visit([](auto const& i) { return i.host_vectors(); }); })
    .def_prop_ro(
      "has_labels",
      [](const Index& s) { return s.visit([](auto const& i) { return i.has_labels(); }); })
    .def_prop_ro(
      "local_projections",
      [](const Index& s) { return s.visit([](auto const& i) { return i.local_projections(); }); })
    .def_prop_ro(
      "int8_rows",
      [](const Index& s) { return s.visit([](auto const& i) { return i.int8_rows(); }); })
    .def_prop_ro("n_lists",
                 [](const Index& s) { return s.visit([](auto const& i) { return i.n_lists(); }); })
    .def_prop_ro(
      "n_partitioned",
      [](const Index& s) { return s.visit([](auto const& i) { return i.n_partitioned(); }); })
    .def(
      "partition_by_labels",
      [](Index& s, uintptr_t stream) {
        s.visit([&](auto& i) {
          nb::gil_scoped_release nogil;
          auto& res = resources(stream);
          certex::partition_by_labels(res, &i);
          raft::resource::sync_stream(res);
        });
      },
      nb::arg("stream"))
    .def(
      "list_order",
      [](const Index& s, any_array out, uintptr_t stream) {
        s.visit([&](auto const& i) {
          if (!i.mapped()) { throw nb::value_error("certex: the index has no lists"); }
          auto ids = i.row_ids();
          if (!on_device(out) || out.ndim() != 1 || out.dtype() != nb::dtype<int32_t>() ||
              int64_t(out.shape(0)) != ids.extent(0)) {
            throw nb::value_error(
              "certex: list_order needs an int32 device vector of one entry per row");
          }
          nb::gil_scoped_release nogil;
          auto st = reinterpret_cast<cudaStream_t>(stream);
          cudaMemcpyAsync(out.data(),
                          ids.data_handle(),
                          ids.extent(0) * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice,
                          st);
          cudaStreamSynchronize(st);
        });
      },
      nb::arg("out"),
      nb::arg("stream"))
    .def_prop_ro("capacity",
                 [](const Index& s) { return s.visit([](auto const& i) { return i.capacity(); }); })
    .def_prop_ro("screening_bytes_per_vector",
                 [](const Index& s) {
                   return s.visit([](auto const& i) { return i.screening_bytes_per_vector(); });
                 })
    .def_prop_ro(
      "metric",
      [](const Index& s) { return s.visit([](auto const& i) { return metric_name(i.metric()); }); })
    .def_prop_ro("dtype", &Index::data_type)
    .def(
      "reserve",
      [](Index& s, int64_t n, uintptr_t stream) {
        s.visit([&](auto& i) {
          nb::gil_scoped_release nogil;
          auto& res = resources(stream);
          std::unique_lock grow(i.sync().rw);
          i.reserve(res, n);
          raft::resource::sync_stream(res);
        });
      },
      nb::arg("n"),
      nb::arg("stream"))
    .def("extend", &do_extend, nb::arg("x"), nb::arg("stream"))
    .def("remove", &do_remove, nb::arg("ids"), nb::arg("stream"))
    .def("set_labels", &do_set_labels, nb::arg("labels"), nb::arg("first"), nb::arg("stream"))
    .def("search",
         &do_search,
         nb::arg("queries"),
         nb::arg("neighbors"),
         nb::arg("distances"),
         nb::arg("anchors"),
         nb::arg("batch_size"),
         nb::arg("probe_lists"),
         nb::arg("filter_words").none(),
         nb::arg("filter_rows"),
         nb::arg("filter_cols"),
         nb::arg("query_labels").none(),
         nb::arg("stream"))
    .def("save", &do_save, nb::arg("path"), nb::arg("stream"));

  // counters of the searches of the calling thread: certified survivors and exact scores of ordered
  // verification
  m.def("stats_begin", [] {
    static thread_local certex::detail::search_stats st;
    st = {};
    certex::detail::collect_stats(&st);
  });
  m.def("stats_end", [] {
    auto* st = certex::detail::stats_sink();
    certex::detail::collect_stats(nullptr);
    if (st == nullptr) { return std::array<int64_t, 3>{0, 0, 0}; }
    return std::array<int64_t, 3>{st->queries, st->survivors, st->verified};
  });

  m.def("build",
        &do_build,
        nb::arg("x"),
        nb::arg("metric"),
        nb::arg("width"),
        nb::arg("variance"),
        nb::arg("anchor_stride"),
        nb::arg("pca_sample"),
        nb::arg("host_vectors"),
        nb::arg("n_lists"),
        nb::arg("local_projections"),
        nb::arg("int8_rows"),
        nb::arg("stream"),
        nb::rv_policy::take_ownership);
  nb::class_<Sharded>(m, "Sharded")
    .def_prop_ro("size", [](Sharded& s) { return s.visit([](auto& i) { return i.size(); }); })
    .def_prop_ro("n_shards",
                 [](Sharded& s) { return s.visit([](auto& i) { return int(i.shards.size()); }); })
    .def("search",
         &do_search_sharded,
         nb::arg("q"),
         nb::arg("neighbors"),
         nb::arg("distances"),
         nb::arg("anchors"),
         nb::arg("batch_size"),
         nb::arg("probe_lists"));
  m.def("build_sharded",
        &do_build_sharded,
        nb::arg("x"),
        nb::arg("devices"),
        nb::arg("metric"),
        nb::arg("width"),
        nb::arg("anchor_stride"),
        nb::arg("host_vectors"),
        nb::arg("n_lists"),
        nb::arg("local_projections"),
        nb::rv_policy::take_ownership);
  // return the unused memory of the stream-ordered pool of the current device to the driver, so
  // that the device memory in use is the memory that indexes and live buffers hold
  m.def("trim_pool", [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return; }
    auto it = async_pools().find(device);
    if (it == async_pools().end()) { return; }
    nb::gil_scoped_release nogil;
    cudaDeviceSynchronize();
    cudaMemPoolTrimTo(it->second, 0);
  });
  m.def("tune",
        &do_tune,
        nb::arg("x"),
        nb::arg("q"),
        nb::arg("metric"),
        nb::arg("k"),
        nb::arg("host_vectors"),
        nb::arg("n_lists"),
        nb::arg("widths"),
        nb::arg("strides"),
        nb::arg("small_prefix"),
        nb::arg("large_prefix"),
        nb::arg("n_queries"),
        nb::arg("batch_size"),
        nb::arg("memory_budget"),
        nb::arg("stream"));
  m.def("load",
        &do_load,
        nb::arg("path"),
        nb::arg("dtype"),
        nb::arg("stream"),
        nb::rv_policy::take_ownership);
}
