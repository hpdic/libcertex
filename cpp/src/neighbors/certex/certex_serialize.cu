/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuvs/neighbors/certex.hpp>
#include <cuvs/util/numpy_dtype.hpp>

#include "detail/kernels.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/serialize.hpp>

#include <algorithm>
#include <fstream>
#include <shared_mutex>
#include <string>
#include <vector>

namespace cuvs::neighbors::certex {

namespace detail {

constexpr int kSerializationVersion = 1;
constexpr int kMagic                = 0x43455258;  // "CERX"

/** NumPy descriptor of T, padded to 4 bytes: the first bytes of a file name its data type. */
template <typename T>
auto dtype_prefix() -> std::string
{
  auto s = cuvs::util::detail::numpy_dtype_string<T>();
  s.resize(4, '\0');
  return s;
}

// Every quantity that a search reads is stored as saved, so that a loaded index returns the same
// exact neighbors without lifting the vectors again.
template <typename T>
void serialize(raft::resources const& res, std::ostream& os, index<T> const& idx)
{
  std::shared_lock view(idx.sync().rw);
  os << dtype_prefix<T>();
  const int64_t n = idx.size(), d = idx.dim(), na = idx.n_anchors();
  const int MP = idx.row_length();
  raft::serialize_scalar(res, os, kMagic);
  raft::serialize_scalar(res, os, kSerializationVersion);
  raft::serialize_scalar(res, os, int(idx.metric()));
  raft::serialize_scalar(res, os, d);
  raft::serialize_scalar(res, os, idx.width());
  raft::serialize_scalar(res, os, idx.anchor_stride());
  raft::serialize_scalar(res, os, int(idx.host_vectors()));
  raft::serialize_scalar(res, os, n);
  raft::serialize_scalar(res, os, idx.n_removed());
  raft::serialize_scalar(res, os, int(idx.has_labels()));
  raft::serialize_scalar(res, os, int(idx.int8_rows()));
  raft::serialize_scalar(res, os, idx.alpha);
  raft::serialize_scalar(res, os, idx.maxima.lifted);
  raft::serialize_scalar(res, os, idx.maxima.norm);
  raft::serialize_scalar(res, os, idx.maxima.error);
  raft::serialize_scalar(res, os, idx.maxima.high);
  raft::serialize_scalar(res, os, idx.maxima.low);
  raft::serialize_mdspan(res, os, idx.center());
  raft::serialize_mdspan(res, os, idx.basis());
  if (idx.int8_rows()) {  // int8 codes, their scalars, and the largest lifted squared norm
    raft::serialize_scalar(res, os, idx.max_a);
    raft::serialize_mdspan(res,
                           os,
                           raft::make_device_matrix_view<const int8_t, int64_t>(
                             idx.codes().data_handle(), n, idx.code_length()));
    raft::serialize_mdspan(
      res,
      os,
      raft::make_device_matrix_view<const float, int64_t>(idx.row_scalars().data_handle(), n, 5));
  } else {
    raft::serialize_mdspan(
      res, os, raft::make_device_matrix_view<const half, int64_t>(idx.rows().data_handle(), n, MP));
  }
  raft::serialize_mdspan(
    res,
    os,
    raft::make_device_matrix_view<const half, int64_t>(idx.anchor_rows().data_handle(), na, MP));
  raft::serialize_mdspan(
    res,
    os,
    raft::make_device_matrix_view<const T, int64_t>(idx.anchor_vectors().data_handle(), na, d));
  if (idx.host_vectors()) {
    raft::serialize_mdspan(
      res, os, raft::make_host_matrix_view<const T, int64_t>(idx.vectors_ptr(), n, d));
  } else {
    raft::serialize_mdspan(
      res, os, raft::make_device_matrix_view<const T, int64_t>(idx.vectors_ptr(), n, d));
  }
  raft::serialize_mdspan(
    res, os, raft::make_device_vector_view<const uint8_t, int64_t>(idx.removed().data_handle(), n));
  if (idx.has_labels()) {
    raft::serialize_mdspan(
      res,
      os,
      raft::make_device_vector_view<const int32_t, int64_t>(idx.labels().data_handle(), n));
  }
  // k-means lists: offsets, centers, and the map of rows to ids (the map of ids follows from it)
  const int P = idx.n_lists();
  raft::serialize_scalar(res, os, P);
  if (P > 0) {
    raft::serialize_mdspan(
      res,
      os,
      raft::make_host_vector_view<const int64_t, int64_t>(idx.list_offsets().data(), P + 1));
    raft::serialize_mdspan(res, os, idx.list_centers());
    raft::serialize_mdspan(
      res,
      os,
      raft::make_device_vector_view<const int32_t, int64_t>(idx.row_ids().data_handle(), n));
    // local projections: centers, directions, scales, and the certificate maxima of every list
    raft::serialize_scalar(res, os, int(idx.local_projections()));
    if (idx.local_projections()) {
      raft::serialize_mdspan(res, os, idx.local_centers());
      raft::serialize_mdspan(res, os, idx.local_bases());
      std::vector<double> c(6 * P);
      for (int j = 0; j < P; ++j) {
        const auto& M     = idx.local_maxima()[j];
        const double v[6] = {idx.local_alpha()[j], M.lifted, M.norm, M.error, M.high, M.low};
        std::copy(v, v + 6, c.data() + 6 * j);
      }
      raft::serialize_mdspan(
        res, os, raft::make_host_vector_view<const double, int64_t>(c.data(), 6 * P));
    }
  }
}

template <typename T>
void deserialize(raft::resources const& res, std::istream& is, index<T>* out)
{
  char dtype[4];
  RAFT_EXPECTS(is.read(dtype, 4), "certex::deserialize: failed to read the dtype prefix");
  const auto expected = dtype_prefix<T>();
  RAFT_EXPECTS(std::equal(dtype, dtype + 4, expected.begin()),
               "certex::deserialize: the file holds vectors of another data type");
  RAFT_EXPECTS(raft::deserialize_scalar<int>(res, is) == kMagic, "not a certex index");
  RAFT_EXPECTS(raft::deserialize_scalar<int>(res, is) == kSerializationVersion,
               "unsupported certex serialization version");
  const auto metric = cuvs::distance::DistanceType(raft::deserialize_scalar<int>(res, is));
  const auto d      = raft::deserialize_scalar<int64_t>(res, is);
  const auto m      = raft::deserialize_scalar<int>(res, is);
  const auto stride = raft::deserialize_scalar<int>(res, is);
  const bool host   = raft::deserialize_scalar<int>(res, is) != 0;
  const auto n      = raft::deserialize_scalar<int64_t>(res, is);
  const auto n_rm   = raft::deserialize_scalar<int64_t>(res, is);
  const bool labels = raft::deserialize_scalar<int>(res, is) != 0;
  const bool int8   = raft::deserialize_scalar<int>(res, is) != 0;
  index<T> idx(res, metric, d, m, stride, host);
  if (int8) { idx.use_int8_rows(); }
  idx.alpha         = raft::deserialize_scalar<double>(res, is);
  idx.maxima.lifted = raft::deserialize_scalar<double>(res, is);
  idx.maxima.norm   = raft::deserialize_scalar<double>(res, is);
  idx.maxima.error  = raft::deserialize_scalar<double>(res, is);
  idx.maxima.high   = raft::deserialize_scalar<double>(res, is);
  idx.maxima.low    = raft::deserialize_scalar<double>(res, is);
  idx.reserve(res, n);
  idx.set_size(n);
  const int64_t na = idx.n_anchors();
  const int MP     = idx.row_length();
  auto center      = idx.center();
  auto basis       = idx.basis();
  auto rows =
    raft::make_device_matrix_view<half, int64_t>(idx.rows().data_handle(), int8 ? 0 : n, MP);
  auto anchor_rows =
    raft::make_device_matrix_view<half, int64_t>(idx.anchor_rows().data_handle(), na, MP);
  auto anchor_vectors =
    raft::make_device_matrix_view<T, int64_t>(idx.anchor_vectors().data_handle(), na, d);
  raft::deserialize_mdspan(res, is, center);
  raft::deserialize_mdspan(res, is, basis);
  if (int8) {
    idx.max_a  = raft::deserialize_scalar<double>(res, is);
    auto codes = raft::make_device_matrix_view<int8_t, int64_t>(
      idx.codes().data_handle(), n, idx.code_length());
    auto scal =
      raft::make_device_matrix_view<float, int64_t>(idx.row_scalars().data_handle(), n, 5);
    raft::deserialize_mdspan(res, is, codes);
    raft::deserialize_mdspan(res, is, scal);
  } else {
    raft::deserialize_mdspan(res, is, rows);
  }
  raft::deserialize_mdspan(res, is, anchor_rows);
  raft::deserialize_mdspan(res, is, anchor_vectors);
  if (host) {
    auto v = raft::make_host_matrix_view<T, int64_t>(idx.vectors_ptr(), n, d);
    raft::deserialize_mdspan(res, is, v);
  } else {
    auto v = raft::make_device_matrix_view<T, int64_t>(idx.vectors_ptr(), n, d);
    raft::deserialize_mdspan(res, is, v);
  }
  auto removed = raft::make_device_vector_view<uint8_t, int64_t>(idx.removed().data_handle(), n);
  raft::deserialize_mdspan(res, is, removed);
  idx.add_removed(n_rm);
  if (labels) {
    auto lab = raft::make_device_vector_view<int32_t, int64_t>(idx.labels().data_handle(), n);
    raft::deserialize_mdspan(res, is, lab);
    idx.mark_labeled();
  }
  const int P = raft::deserialize_scalar<int>(res, is);
  if (P > 0) {
    std::vector<int64_t> offsets(P + 1);
    auto off_view = raft::make_host_vector_view<int64_t, int64_t>(offsets.data(), P + 1);
    raft::deserialize_mdspan(res, is, off_view);
    auto centers = raft::make_device_matrix<float, int64_t>(res, P, d);
    auto cview   = centers.view();
    raft::deserialize_mdspan(res, is, cview);
    // the maps cover the capacity; rows beyond the stored ones are their own ids
    const int64_t c = idx.capacity();
    auto row_ids    = raft::make_device_vector<int32_t, int64_t>(res, c);
    auto id_rows    = raft::make_device_vector<int32_t, int64_t>(res, c);
    detail::iota_fill(res, row_ids.data_handle(), c);
    auto rview = raft::make_device_vector_view<int32_t, int64_t>(row_ids.data_handle(), n);
    raft::deserialize_mdspan(res, is, rview);
    detail::invert_into(res, row_ids.data_handle(), c, id_rows.data_handle());
    idx.set_lists(std::move(offsets), std::move(centers), std::move(row_ids), std::move(id_rows));
    if (raft::deserialize_scalar<int>(res, is) != 0) {
      auto lc  = raft::make_device_matrix<float, int64_t>(res, P, d);
      auto lb  = raft::make_device_matrix<float, int64_t>(res, int64_t(P) * m, d);
      auto lcv = lc.view();
      auto lbv = lb.view();
      raft::deserialize_mdspan(res, is, lcv);
      raft::deserialize_mdspan(res, is, lbv);
      std::vector<double> c(6 * P);
      auto cv = raft::make_host_vector_view<double, int64_t>(c.data(), 6 * P);
      raft::deserialize_mdspan(res, is, cv);
      std::vector<double> alpha(P);
      std::vector<typename index<T>::certificate_maxima> maxima(P);
      for (int j = 0; j < P; ++j) {
        alpha[j]  = c[6 * j];
        maxima[j] = {c[6 * j + 1], c[6 * j + 2], c[6 * j + 3], c[6 * j + 4], c[6 * j + 5]};
      }
      idx.set_local(std::move(lc), std::move(lb), std::move(alpha), std::move(maxima));
    }
  }
  raft::resource::sync_stream(res);
  *out = std::move(idx);
}

}  // namespace detail

#define CUVS_INST_CERTEX_SERIALIZE(T)                                                          \
  void serialize(raft::resources const& res, std::ostream& os, index<T> const& idx)            \
  {                                                                                            \
    detail::serialize<T>(res, os, idx);                                                        \
  }                                                                                            \
  void serialize(raft::resources const& res, const std::string& filename, index<T> const& idx) \
  {                                                                                            \
    std::ofstream of(filename, std::ios::out | std::ios::binary);                              \
    RAFT_EXPECTS(of, "cannot open %s for writing", filename.c_str());                          \
    detail::serialize<T>(res, of, idx);                                                        \
    RAFT_EXPECTS(of.good(), "error while writing %s", filename.c_str());                       \
  }                                                                                            \
  void deserialize(raft::resources const& res, std::istream& is, index<T>* idx)                \
  {                                                                                            \
    detail::deserialize<T>(res, is, idx);                                                      \
  }                                                                                            \
  void deserialize(raft::resources const& res, const std::string& filename, index<T>* idx)     \
  {                                                                                            \
    std::ifstream in(filename, std::ios::in | std::ios::binary);                               \
    RAFT_EXPECTS(in, "cannot open %s for reading", filename.c_str());                          \
    detail::deserialize<T>(res, in, idx);                                                      \
  }

CUVS_INST_CERTEX_SERIALIZE(float)
CUVS_INST_CERTEX_SERIALIZE(half)
CUVS_INST_CERTEX_SERIALIZE(int8_t)
CUVS_INST_CERTEX_SERIALIZE(uint8_t)

#undef CUVS_INST_CERTEX_SERIALIZE

}  // namespace cuvs::neighbors::certex
