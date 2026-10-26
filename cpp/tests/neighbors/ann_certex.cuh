/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Exactness tests of cuvs::neighbors::certex: the sorted scores of the returned neighbors must
// equal the k smallest scores of exhaustive search in double precision, for every query.

#pragma once

#include <cuvs/neighbors/certex.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include "certex_device.cuh"

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <random>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::certex {

using cuvs::distance::DistanceType;

struct AnnCertexInputs {
  int64_t n_rows;
  int64_t dim;
  int64_t n_queries;
  int k;
  DistanceType metric;
  int width;            // 0: automatic
  bool host_build;      // build and extend from host views
  bool host_vectors;    // full-precision vectors in pinned host memory
  double build_share;   // share of the rows given to build; extend adds the rest
  bool unequal_norms;   // vector norms spread over [0.2, 3.2] (matters for inner products)
  int64_t duplicates;   // copies of the first vector
  int n_lists = 0;      // k-means lists (0: none)
  bool local  = false;  // local projections of the lists
  bool int8   = false;  // int8 screening rows
};

inline auto operator<<(std::ostream& os, const AnnCertexInputs& p) -> std::ostream&
{
  const char* m = p.metric == DistanceType::InnerProduct     ? "ip"
                  : p.metric == DistanceType::CosineExpanded ? "cosine"
                                                             : "l2";
  return os << "{n=" << p.n_rows << " dim=" << p.dim << " nq=" << p.n_queries << " k=" << p.k
            << " metric=" << m << " width=" << p.width << " host_build=" << p.host_build
            << " host_vectors=" << p.host_vectors << " build_share=" << p.build_share
            << " unequal_norms=" << p.unequal_norms << " duplicates=" << p.duplicates
            << " n_lists=" << p.n_lists << " local=" << p.local << " int8=" << p.int8 << "}";
}

template <typename T>
inline auto to_double(T v) -> double
{
  if constexpr (std::is_same_v<T, half>) {
    return double(__half2float(v));
  } else {
    return double(v);
  }
}

/** Exact score of query i and vector j: squared distance, or |q|^2 - 2 <q, x> for inner products.
 */
template <typename T>
auto exact_score(const std::vector<T>& x,
                 const std::vector<T>& q,
                 int64_t i,
                 int64_t j,
                 int64_t d,
                 DistanceType metric) -> double
{
  double s = 0, qq = 0, xx = 0, ip = 0;
  for (int64_t t = 0; t < d; ++t) {
    const double a = to_double(q[i * d + t]), b = to_double(x[j * d + t]);
    s += (a - b) * (a - b);
    ip += a * b;
    qq += a * a;
    xx += b * b;
  }
  if (metric == DistanceType::InnerProduct) { return qq - 2 * ip; }
  if (metric == DistanceType::CosineExpanded) {
    return 2 - 2 * ip / std::sqrt(std::max(qq * xx, 1e-300));  // |q/|q| - x/|x||^2
  }
  return s;
}

/**
 * Number of queries whose result is not exact: the scores of the returned ids (sorted) must equal
 * the smallest scores among the vectors that `keep(i, j)` admits, with -1 exactly beyond their
 * number, and every returned id must be admitted.
 */
template <typename T, typename Keep>
auto count_inexact(const std::vector<T>& x,
                   const std::vector<T>& q,
                   int64_t n,
                   int64_t nq,
                   int64_t d,
                   int k,
                   DistanceType metric,
                   const std::vector<int64_t>& ids,
                   Keep keep) -> int64_t
{
  int64_t bad = 0;
  for (int64_t i = 0; i < nq; ++i) {
    std::vector<double> all;
    for (int64_t j = 0; j < n; ++j) {
      if (keep(i, j)) { all.push_back(exact_score(x, q, i, j, d, metric)); }
    }
    const size_t want = std::min<size_t>(k, all.size());
    std::partial_sort(all.begin(), all.begin() + want, all.end());
    std::vector<double> got;
    bool ok = true;
    for (int t = 0; t < k; ++t) {
      const int64_t j = ids[i * k + t];
      if (j < 0) { continue; }
      ok &= j < n && keep(i, j);
      if (j < n) { got.push_back(exact_score(x, q, i, j, d, metric)); }
    }
    std::sort(got.begin(), got.end());
    ok &= got.size() == want;
    for (size_t t = 0; t < got.size() && ok; ++t) {
      ok = std::fabs(got[t] - all[t]) <= 1e-9 * (1 + std::fabs(all[t]));
    }
    bad += !ok;
  }
  return bad;
}

/**
 * Correlated Gaussian rows (a random mixing matrix gives a decaying spectrum, as in real
 * embeddings), as float, half, or integers of T (scaled and clamped to its range).
 */
template <typename T>
auto make_rows(int64_t rows, int64_t d, bool unequal_norms, std::mt19937& rng) -> std::vector<T>
{
  std::normal_distribution<float> nd(0, 1);
  std::uniform_real_distribution<float> ud(0, 1);
  std::mt19937 mix_rng(1234);  // the same mixing for vectors and queries
  std::vector<float> mix(d * d), g(d);
  for (auto& v : mix) {
    v = nd(mix_rng) / std::sqrt(float(d));
  }
  std::vector<T> out(rows * d);
  for (int64_t i = 0; i < rows; ++i) {
    for (auto& v : g) {
      v = nd(rng);
    }
    const float s = unequal_norms ? 0.2f + 3.0f * ud(rng) : 1.0f;
    for (int64_t a = 0; a < d; ++a) {
      float acc = 0;
      for (int64_t b = 0; b < d; ++b) {
        acc += g[b] * mix[b * d + a];
      }
      acc *= s;
      if constexpr (std::is_same_v<T, float>) {
        out[i * d + a] = acc;
      } else if constexpr (std::is_same_v<T, half>) {
        out[i * d + a] = __float2half(acc);
      } else if constexpr (std::is_same_v<T, int8_t>) {
        out[i * d + a] = int8_t(std::clamp(std::round(acc * 30), -127.0f, 127.0f));
      } else {
        out[i * d + a] = uint8_t(std::clamp(std::round(acc * 30 + 100), 0.0f, 255.0f));
      }
    }
  }
  return out;
}

template <typename T>
class AnnCertexTest : public ::testing::TestWithParam<AnnCertexInputs> {
 public:
  AnnCertexTest() : ps(GetParam()), stream_(raft::resource::get_cuda_stream(handle_).get()) {}

 protected:
  void testCertex()
  {
    constexpr bool is_float = std::is_same_v<T, float> || std::is_same_v<T, half>;
    if (ps.metric == DistanceType::CosineExpanded && !is_float) {
      GTEST_SKIP() << "cosine needs float or half vectors";
    }
    std::mt19937 rng(42);
    auto x = make_rows<T>(ps.n_rows, ps.dim, ps.unequal_norms, rng);
    auto q = make_rows<T>(ps.n_queries, ps.dim, false, rng);
    for (int64_t c = 1; c <= ps.duplicates && c < ps.n_rows; ++c) {
      std::copy_n(x.begin(), ps.dim, x.begin() + c * ps.dim);
    }
    const int64_t n = ps.n_rows, d = ps.dim, nq = ps.n_queries;
    const int64_t n0 = std::max<int64_t>(1, int64_t(ps.build_share * n));

    index_params ip;
    ip.metric            = ps.metric;
    ip.width             = ps.width;
    ip.host_vectors      = ps.host_vectors;
    ip.n_lists           = ps.n_lists;
    ip.local_projections = ps.local;
    ip.int8_rows         = ps.int8;
    auto dx              = raft::make_device_matrix<T, int64_t>(handle_, ps.host_build ? 0 : n, d);
    auto dq              = raft::make_device_matrix<T, int64_t>(handle_, nq, d);
    if (!ps.host_build) { raft::copy(dx.data_handle(), x.data(), n * d, stream_); }
    raft::copy(dq.data_handle(), q.data(), nq * d, stream_);
    auto idx =
      ps.host_build
        ? build(handle_, ip, raft::make_host_matrix_view<const T, int64_t>(x.data(), n0, d))
        : build(
            handle_, ip, raft::make_device_matrix_view<const T, int64_t>(dx.data_handle(), n0, d));
    if (n0 < n) {
      if (ps.host_build) {
        extend(handle_,
               raft::make_host_matrix_view<const T, int64_t>(x.data() + n0 * d, n - n0, d),
               &idx);
      } else {
        extend(
          handle_,
          raft::make_device_matrix_view<const T, int64_t>(dx.data_handle() + n0 * d, n - n0, d),
          &idx);
      }
    }
    ASSERT_EQ(idx.size(), n);
    ASSERT_EQ(idx.host_vectors(), ps.host_vectors);

    auto nb = raft::make_device_matrix<int64_t, int64_t>(handle_, nq, ps.k);
    auto ds = raft::make_device_matrix<float, int64_t>(handle_, nq, ps.k);
    search(handle_, search_params{}, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
    std::vector<int64_t> ids(nq * ps.k);
    std::vector<float> dist(nq * ps.k);
    raft::copy(ids.data(), nb.data_handle(), nq * ps.k, stream_);
    raft::copy(dist.data(), ds.data_handle(), nq * ps.k, stream_);
    raft::resource::sync_stream(handle_);

    EXPECT_EQ(
      count_inexact(x, q, n, nq, d, ps.k, ps.metric, ids, [](int64_t, int64_t) { return true; }),
      0);
    // the distances follow the conventions of brute_force, best first
    for (int64_t i = 0; i < nq; ++i) {
      for (int t = 0; t < ps.k; ++t) {
        const int64_t j = ids[i * ps.k + t];
        const double s  = exact_score(x, q, i, j, d, ps.metric);
        double expected = s;  // squared distance
        if (ps.metric == DistanceType::InnerProduct) {
          expected = 0;
          for (int64_t a = 0; a < d; ++a) {
            expected += to_double(q[i * d + a]) * to_double(x[j * d + a]);
          }
        }
        if (ps.metric == DistanceType::CosineExpanded) { expected = s / 2; }  // 1 - cos
        ASSERT_NEAR(dist[i * ps.k + t], expected, 1e-4 * (1 + std::fabs(expected)));
        if (t > 0) {
          const bool ordered = ps.metric == DistanceType::InnerProduct
                                 ? dist[i * ps.k + t] <= dist[i * ps.k + t - 1]
                                 : dist[i * ps.k + t] >= dist[i * ps.k + t - 1];
          ASSERT_TRUE(ordered);
        }
      }
    }
  }

  raft::device_resources handle_;
  AnnCertexInputs ps;
  cudaStream_t stream_;
};

// clang-format off
inline const std::vector<AnnCertexInputs> inputs = {
  // n,    dim, nq,  k,    metric,                        width, host_build, host_vectors, build_share, unequal_norms, duplicates
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      11,    false,      false,        1.0,         false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      91,    false,      false,        1.0,         false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      false,        0.33,        false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     true,       false,        0.5,         false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     true,       true,         0.5,         false,         0},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      true,         1.0,         false,         0},
  {30000,  96,  200, 10,   DistanceType::InnerProduct,    0,     false,      false,        1.0,         true,          0},
  {30000,  96,  200, 10,   DistanceType::InnerProduct,    0,     false,      false,        0.5,         true,          0},
  {30000,  96,  200, 10,   DistanceType::CosineExpanded,  0,     false,      false,        1.0,         true,          0},
  {30000,  96,  100, 100,  DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0},
  {30000,  96,  50,  1024, DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0},
  {20000,  128, 200, 10,   DistanceType::L2Expanded,      123,   false,      false,        1.0,         false,         500},
  {20000,  128, 200, 10,   DistanceType::L2Expanded,      251,   false,      false,        1.0,         false,         0},
  {25,     32,  50,  10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0},
  {5000,   32,  3001, 10,  DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0},
  // k-means lists: the vectors in list order, the nearest lists probed first
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   64},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      11,    true,       true,         0.5,         false,         0,   100},
  {30000,  96,  200, 10,   DistanceType::InnerProduct,    0,     false,      false,        1.0,         true,          0,   64},
  {30000,  96,  200, 10,   DistanceType::CosineExpanded,  0,     false,      false,        0.5,         true,          0,   32},
  {30000,  96,  100, 100,  DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   256},
  {20000,  128, 200, 10,   DistanceType::L2Expanded,      123,   false,      false,        1.0,         false,         500, 50},
  {300,    32,  50,  10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   300},
  // local projections: every list with its own center and directions
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   64,  true},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      11,    true,       true,         0.5,         false,         0,   100, true},
  {30000,  96,  200, 10,   DistanceType::InnerProduct,    0,     false,      false,        1.0,         true,          0,   64,  true},
  {30000,  96,  200, 10,   DistanceType::CosineExpanded,  0,     false,      false,        0.5,         true,          0,   32,  true},
  {30000,  96,  100, 100,  DistanceType::L2Expanded,      27,    false,      false,        1.0,         false,         0,   256, true},
  {20000,  128, 200, 10,   DistanceType::L2Expanded,      59,    false,      false,        1.0,         false,         500, 50,  true},
  {300,    32,  50,  10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   300, true},
  // int8 screening rows
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   0,   false, true},
  {30000,  96,  200, 10,   DistanceType::L2Expanded,      11,    true,       true,         0.5,         false,         0,   0,   false, true},
  {30000,  96,  200, 10,   DistanceType::InnerProduct,    0,     false,      false,        1.0,         true,          0,   0,   false, true},
  {30000,  96,  200, 10,   DistanceType::CosineExpanded,  0,     false,      false,        0.5,         true,          0,   0,   false, true},
  {30000,  96,  50,  1024, DistanceType::L2Expanded,      27,    false,      false,        1.0,         false,         0,   0,   false, true},
  {20000,  128, 200, 10,   DistanceType::L2Expanded,      123,   false,      false,        1.0,         false,         500, 0,   false, true},
  {25,     32,  50,  10,   DistanceType::L2Expanded,      0,     false,      false,        1.0,         false,         0,   0,   false, true},
};
// clang-format on

}  // namespace cuvs::neighbors::certex
