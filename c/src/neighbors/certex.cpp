/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// C API of cuvs::neighbors::certex, in the layout of the cuVS C API (c/src/neighbors).

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cstdint>
#include <dlpack/dlpack.h>
#include <fstream>
#include <string>

#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/certex.h>
#include <cuvs/neighbors/certex.hpp>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/common.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

namespace certex = cuvs::neighbors::certex;

template <typename T>
using index_t = certex::index<T>;

auto to_cpp(cuvsCertexIndexParams const& p) -> certex::index_params
{
  certex::index_params out;
  out.metric            = static_cast<cuvs::distance::DistanceType>(int(p.metric));
  out.width             = p.width;
  out.variance          = p.variance;
  out.anchor_stride     = p.anchor_stride;
  out.pca_sample        = p.pca_sample;
  out.host_vectors      = p.host_vectors;
  out.n_lists           = p.n_lists;
  out.local_projections = p.local_projections;
  out.int8_rows         = p.int8_rows;
  return out;
}

auto to_cpp(cuvsCertexSearchParams const& p) -> certex::search_params
{
  certex::search_params out;
  out.anchors     = p.anchors;
  out.batch_size  = p.batch_size;
  out.probe_lists = p.probe_lists;
  return out;
}

/** Call f with a value of the vector type of a DLPack data type (float, half, int8_t, uint8_t). */
template <typename F>
void dispatch(DLDataType dtype, F&& f)
{
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    f(float{});
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    f(half{});
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    f(int8_t{});
  } else if (dtype.code == kDLUInt && dtype.bits == 8) {
    f(uint8_t{});
  } else {
    RAFT_FAIL("certex: vectors must be float32, float16, int8, or uint8");
  }
}

template <typename T>
auto as_index(cuvsCertexIndex const& index) -> index_t<T>*
{
  RAFT_EXPECTS(index.addr != 0, "certex: the index is empty (build or deserialize it first)");
  return reinterpret_cast<index_t<T>*>(index.addr);
}

template <typename T>
void* build_typed(raft::resources const& res, certex::index_params const& p, DLManagedTensor* x)
{
  index_t<T>* out = nullptr;
  if (cuvs::core::is_dlpack_device_compatible(x->dl_tensor) &&
      x->dl_tensor.device.device_type != kDLCUDAHost) {
    using view = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    out        = new index_t<T>(certex::build(res, p, cuvs::core::from_dlpack<view>(x)));
  } else {
    using view = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    out        = new index_t<T>(certex::build(res, p, cuvs::core::from_dlpack<view>(x)));
  }
  return out;
}

template <typename T>
void extend_typed(raft::resources const& res, DLManagedTensor* x, index_t<T>* idx)
{
  if (cuvs::core::is_dlpack_device_compatible(x->dl_tensor) &&
      x->dl_tensor.device.device_type != kDLCUDAHost) {
    using view = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    certex::extend(res, cuvs::core::from_dlpack<view>(x), idx);
  } else {
    using view = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    certex::extend(res, cuvs::core::from_dlpack<view>(x), idx);
  }
}

template <typename T>
void search_typed(raft::resources const& res,
                  certex::search_params const& p,
                  index_t<T> const& idx,
                  DLManagedTensor* queries,
                  DLManagedTensor* neighbors,
                  DLManagedTensor* distances,
                  cuvsFilter filter)
{
  using q_view = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using n_view = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using d_view = raft::device_matrix_view<float, int64_t, raft::row_major>;
  using w_view = raft::device_vector_view<uint32_t, int64_t>;
  auto q       = cuvs::core::from_dlpack<q_view>(queries);
  auto nb      = cuvs::core::from_dlpack<n_view>(neighbors);
  auto ds      = cuvs::core::from_dlpack<d_view>(distances);
  if (filter.type == NO_FILTER) {
    certex::search(res, p, idx, q, nb, ds);
  } else if (filter.type == BITSET) {
    auto words = cuvs::core::from_dlpack<w_view>(reinterpret_cast<DLManagedTensor*>(filter.addr));
    auto bits  = cuvs::core::bitset_view<uint32_t, int64_t>(words.data_handle(), idx.size());
    certex::search(res, p, idx, q, nb, ds, cuvs::neighbors::filtering::bitset_filter(bits));
  } else if (filter.type == BITMAP) {
    auto words = cuvs::core::from_dlpack<w_view>(reinterpret_cast<DLManagedTensor*>(filter.addr));
    auto bits =
      cuvs::core::bitmap_view<uint32_t, int64_t>(words.data_handle(), q.extent(0), idx.size());
    certex::search(res, p, idx, q, nb, ds, cuvs::neighbors::filtering::bitmap_filter(bits));
  } else {
    RAFT_FAIL("certex: unsupported filter type");
  }
}

/** The vector type of a serialized index from its leading NumPy descriptor ("<f4", "<f2", ...). */
auto dtype_of_file(const char* filename) -> DLDataType
{
  std::ifstream is(filename, std::ios::binary);
  RAFT_EXPECTS(is.good(), "certex: cannot open the index file");
  char prefix[4] = {};
  is.read(prefix, 4);
  const std::string d(prefix + 1, 2);
  if (d == "f4") { return {kDLFloat, 32, 1}; }
  if (d == "f2") { return {kDLFloat, 16, 1}; }
  if (d == "i1") { return {kDLInt, 8, 1}; }
  if (d == "u1") { return {kDLUInt, 8, 1}; }
  RAFT_FAIL("certex: not a CertEx index file");
}

}  // namespace

extern "C" cuvsError_t cuvsCertexIndexParamsCreate(cuvsCertexIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    certex::index_params d;
    *params = new cuvsCertexIndexParams{static_cast<cuvsDistanceType>(int(d.metric)),
                                        d.width,
                                        d.variance,
                                        d.anchor_stride,
                                        d.pca_sample,
                                        d.host_vectors,
                                        d.n_lists,
                                        d.local_projections,
                                        d.int8_rows};
  });
}

extern "C" cuvsError_t cuvsCertexIndexParamsDestroy(cuvsCertexIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCertexSearchParamsCreate(cuvsCertexSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    certex::search_params d;
    *params = new cuvsCertexSearchParams{d.anchors, d.batch_size, d.probe_lists};
  });
}

extern "C" cuvsError_t cuvsCertexSearchParamsDestroy(cuvsCertexSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCertexIndexCreate(cuvsCertexIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsCertexIndex{}; });
}

extern "C" cuvsError_t cuvsCertexIndexDestroy(cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->addr != 0) {
      dispatch(index->dtype,
               [&](auto tag) { delete reinterpret_cast<index_t<decltype(tag)>*>(index->addr); });
    }
    delete index;
  });
}

extern "C" cuvsError_t cuvsCertexIndexGetSize(cuvsCertexIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    dispatch(index->dtype, [&](auto tag) { *size = as_index<decltype(tag)>(*index)->size(); });
  });
}

extern "C" cuvsError_t cuvsCertexIndexGetDim(cuvsCertexIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    dispatch(index->dtype, [&](auto tag) { *dim = as_index<decltype(tag)>(*index)->dim(); });
  });
}

extern "C" cuvsError_t cuvsCertexBuild(cuvsResources_t res,
                                       cuvsCertexIndexParams_t params,
                                       DLManagedTensor* dataset,
                                       cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index->addr == 0, "certex: the index handle already holds an index");
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    const auto p  = to_cpp(*params);
    index->dtype  = dataset->dl_tensor.dtype;
    dispatch(index->dtype, [&](auto tag) {
      index->addr = reinterpret_cast<uintptr_t>(build_typed<decltype(tag)>(*res_ptr, p, dataset));
    });
  });
}

extern "C" cuvsError_t cuvsCertexExtend(cuvsResources_t res,
                                        DLManagedTensor* new_vectors,
                                        cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    dispatch(index->dtype, [&](auto tag) {
      extend_typed<decltype(tag)>(*res_ptr, new_vectors, as_index<decltype(tag)>(*index));
    });
  });
}

extern "C" cuvsError_t cuvsCertexRemove(cuvsResources_t res,
                                        DLManagedTensor* ids,
                                        cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    using view    = raft::device_vector_view<const int64_t, int64_t>;
    auto v        = cuvs::core::from_dlpack<view>(ids);
    dispatch(index->dtype,
             [&](auto tag) { certex::remove(*res_ptr, v, as_index<decltype(tag)>(*index)); });
  });
}

extern "C" cuvsError_t cuvsCertexSetLabels(cuvsResources_t res,
                                           DLManagedTensor* labels,
                                           int64_t first,
                                           cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    using view    = raft::device_vector_view<const int32_t, int64_t>;
    auto v        = cuvs::core::from_dlpack<view>(labels);
    dispatch(index->dtype, [&](auto tag) {
      certex::set_labels(*res_ptr, v, as_index<decltype(tag)>(*index), first);
    });
  });
}

extern "C" cuvsError_t cuvsCertexPartitionByLabels(cuvsResources_t res, cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    dispatch(index->dtype, [&](auto tag) {
      certex::partition_by_labels(*res_ptr, as_index<decltype(tag)>(*index));
    });
  });
}

extern "C" cuvsError_t cuvsCertexSearch(cuvsResources_t res,
                                        cuvsCertexSearchParams_t params,
                                        cuvsCertexIndex_t index,
                                        DLManagedTensor* queries,
                                        DLManagedTensor* neighbors,
                                        DLManagedTensor* distances,
                                        cuvsFilter filter)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    const auto p  = to_cpp(*params);
    RAFT_EXPECTS(queries->dl_tensor.dtype.code == index->dtype.code &&
                   queries->dl_tensor.dtype.bits == index->dtype.bits,
                 "certex: the queries must have the vector type of the index");
    dispatch(index->dtype, [&](auto tag) {
      search_typed<decltype(tag)>(
        *res_ptr, p, *as_index<decltype(tag)>(*index), queries, neighbors, distances, filter);
    });
  });
}

extern "C" cuvsError_t cuvsCertexSearchWithLabels(cuvsResources_t res,
                                                  cuvsCertexSearchParams_t params,
                                                  cuvsCertexIndex_t index,
                                                  DLManagedTensor* queries,
                                                  DLManagedTensor* query_labels,
                                                  DLManagedTensor* neighbors,
                                                  DLManagedTensor* distances)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    const auto p  = to_cpp(*params);
    using l_view  = raft::device_vector_view<const int32_t, int64_t>;
    using n_view  = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
    using d_view  = raft::device_matrix_view<float, int64_t, raft::row_major>;
    auto ql       = cuvs::core::from_dlpack<l_view>(query_labels);
    auto nb       = cuvs::core::from_dlpack<n_view>(neighbors);
    auto ds       = cuvs::core::from_dlpack<d_view>(distances);
    dispatch(index->dtype, [&](auto tag) {
      using T      = decltype(tag);
      using q_view = raft::device_matrix_view<const T, int64_t, raft::row_major>;
      certex::search(
        *res_ptr, p, *as_index<T>(*index), cuvs::core::from_dlpack<q_view>(queries), ql, nb, ds);
    });
  });
}

extern "C" cuvsError_t cuvsCertexSerialize(cuvsResources_t res,
                                           const char* filename,
                                           cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    dispatch(index->dtype, [&](auto tag) {
      certex::serialize(*res_ptr, std::string(filename), *as_index<decltype(tag)>(*index));
    });
  });
}

extern "C" cuvsError_t cuvsCertexDeserialize(cuvsResources_t res,
                                             const char* filename,
                                             cuvsCertexIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index->addr == 0, "certex: the index handle already holds an index");
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);
    index->dtype  = dtype_of_file(filename);
    dispatch(index->dtype, [&](auto tag) {
      using T = decltype(tag);
      auto* p = new index_t<T>(*res_ptr);
      try {
        certex::deserialize(*res_ptr, std::string(filename), p);
      } catch (...) {
        delete p;
        throw;
      }
      index->addr = reinterpret_cast<uintptr_t>(p);
    });
  });
}
