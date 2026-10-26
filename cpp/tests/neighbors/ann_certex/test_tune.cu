/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The cost model and the tuner: positive constants, a choice among the candidates that fits, and
// an index built with the choice stays exact.

#include "../ann_certex.cuh"

#include <cuvs/neighbors/certex_tune.hpp>

namespace cuvs::neighbors::certex {

TEST(AnnCertexTune, ChoiceIsExact)
{
  raft::device_resources res;
  auto stream     = raft::resource::get_cuda_stream(res).get();
  const int64_t n = 60000, d = 128, nq = 200;
  const int k = 10;
  std::mt19937 rng(41);
  auto x = make_rows<float>(n, d, false, rng);
  auto q = make_rows<float>(nq, d, false, rng);
  tune_params tp;
  tp.small_prefix = 10000;
  tp.large_prefix = 30000;
  tp.n_queries    = nq;
  auto tr         = tune(res,
                 index_params{},
                 raft::make_host_matrix_view<const float, int64_t>(x.data(), n, d),
                 raft::make_host_matrix_view<const float, int64_t>(q.data(), nq, d),
                 k,
                 tp);
  EXPECT_GT(tr.model.a_mac + tr.model.a_pair, 0.0);
  ASSERT_EQ(tr.candidates.size(), 4u);  // widths 59 and 123, strides 16 and 32
  ASSERT_GT(tr.predicted_qps, 0.0);
  bool chosen = false;
  for (auto const& c : tr.candidates) {
    EXPECT_GT(c.predicted_qps, 0.0);
    chosen |= c.width == tr.width && c.anchor_stride == tr.anchor_stride && c.fits;
  }
  ASSERT_TRUE(chosen);
  index_params ip;
  ip.width         = tr.width;
  ip.anchor_stride = tr.anchor_stride;
  auto dx          = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto dq          = raft::make_device_matrix<float, int64_t>(res, nq, d);
  raft::copy(dx.data_handle(), x.data(), n * d, stream);
  raft::copy(dq.data_handle(), q.data(), nq * d, stream);
  auto idx = build(res, ip, raft::make_const_mdspan(dx.view()));
  auto nb  = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto ds  = raft::make_device_matrix<float, int64_t>(res, nq, k);
  search(res, search_params{}, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
  std::vector<int64_t> ids(nq * k);
  raft::copy(ids.data(), nb.data_handle(), nq * k, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(
    count_inexact(
      x, q, n, nq, d, k, DistanceType::L2Expanded, ids, [](int64_t, int64_t) { return true; }),
    0);
}

}  // namespace cuvs::neighbors::certex
