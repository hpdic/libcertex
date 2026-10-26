/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Sharded search: shards on the devices of the machine (several shards per device when it has
// fewer devices), a shared threshold, and the merge must return the exact neighbors.

#include "../ann_certex.cuh"

#include <cuvs/neighbors/certex_shards.hpp>

namespace cuvs::neighbors::certex {

namespace {

void check_sharded(DistanceType metric, int n_shards, int n_lists)
{
  int n_dev = 0;
  RAFT_CUDA_TRY(cudaGetDeviceCount(&n_dev));
  std::vector<int> devices(n_shards);
  for (int s = 0; s < n_shards; ++s) {
    devices[s] = s % n_dev;
  }
  const int64_t n = 40000, d = 64, nq = 300;
  const int k = 10;
  std::mt19937 rng(31);
  auto x = make_rows<float>(n, d, metric == DistanceType::InnerProduct, rng);
  auto q = make_rows<float>(nq, d, false, rng);
  index_params ip;
  ip.metric  = metric;
  ip.n_lists = n_lists;
  auto idx =
    build_sharded(devices, ip, raft::make_host_matrix_view<const float, int64_t>(x.data(), n, d));
  ASSERT_EQ(idx.size(), n);
  std::vector<int64_t> ids(nq * k);
  std::vector<float> dist(nq * k);
  search(search_params{},
         idx,
         raft::make_host_matrix_view<const float, int64_t>(q.data(), nq, d),
         raft::make_host_matrix_view<int64_t, int64_t>(ids.data(), nq, k),
         raft::make_host_matrix_view<float, int64_t>(dist.data(), nq, k));
  EXPECT_EQ(count_inexact(x, q, n, nq, d, k, metric, ids, [](int64_t, int64_t) { return true; }), 0)
    << n_shards << " shards";
  for (int64_t i = 0; i < nq; ++i) {
    for (int t = 1; t < k; ++t) {
      const bool ordered = metric == DistanceType::InnerProduct
                             ? dist[i * k + t] <= dist[i * k + t - 1]
                             : dist[i * k + t] >= dist[i * k + t - 1];
      ASSERT_TRUE(ordered);
    }
  }
}

}  // namespace

TEST(AnnCertexShards, L2) { check_sharded(DistanceType::L2Expanded, 3, 0); }
TEST(AnnCertexShards, L2Lists) { check_sharded(DistanceType::L2Expanded, 4, 32); }
TEST(AnnCertexShards, InnerProduct) { check_sharded(DistanceType::InnerProduct, 2, 0); }
TEST(AnnCertexShards, Cosine) { check_sharded(DistanceType::CosineExpanded, 5, 16); }

}  // namespace cuvs::neighbors::certex
