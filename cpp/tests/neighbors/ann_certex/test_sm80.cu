/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The whole search with the sm80 kernels (cp.async and mma.sync), which also run on Hopper: the
// inputs of the other suites, filtered search by label over a label partition, and a bitset.

#include "../../../src/neighbors/certex/detail/kernels.hpp"
#include "../ann_certex.cuh"

#include <cuvs/core/bitset.hpp>

namespace cuvs::neighbors::certex {

namespace {

/** Runs the search with the sm80 kernels for the lifetime of the object. */
struct UseSm80 {
  UseSm80() { detail::set_kernel_override(detail::screen_kernel::sm80); }
  ~UseSm80() { detail::set_kernel_override(detail::screen_kernel::automatic); }
};

}  // namespace

template <typename T>
class AnnCertexSm80Test : public AnnCertexTest<T> {
 protected:
  void run()
  {
    UseSm80 sm80;
    this->testCertex();
  }
};

typedef AnnCertexSm80Test<float> AnnCertexSm80TestF;
TEST_P(AnnCertexSm80TestF, AnnCertex) { this->run(); }
INSTANTIATE_TEST_CASE_P(AnnCertexSm80Test, AnnCertexSm80TestF, ::testing::ValuesIn(inputs));

typedef AnnCertexSm80Test<uint8_t> AnnCertexSm80TestU8;
TEST_P(AnnCertexSm80TestU8, AnnCertex) { this->run(); }
INSTANTIATE_TEST_CASE_P(AnnCertexSm80Test, AnnCertexSm80TestU8, ::testing::ValuesIn(inputs));

TEST(AnnCertexSm80, PartitionedLabelsAndBitset)
{
  UseSm80 sm80;
  raft::device_resources res;
  auto stream     = raft::resource::get_cuda_stream(res).get();
  const int64_t n = 30000, d = 64, nq = 200, n0 = 20000;
  const int k = 10;
  std::mt19937 rng(21);
  auto x  = make_rows<float>(n, d, false, rng);
  auto q  = make_rows<float>(nq, d, false, rng);
  auto dx = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto dq = raft::make_device_matrix<float, int64_t>(res, nq, d);
  raft::copy(dx.data_handle(), x.data(), n * d, stream);
  raft::copy(dq.data_handle(), q.data(), nq * d, stream);
  auto idx = build(res,
                   index_params{},
                   raft::make_device_matrix_view<const float, int64_t>(dx.data_handle(), n0, d));
  std::vector<int32_t> lab(n), qlab(nq);
  for (auto& v : lab) {
    v = int32_t(rng() % 37);
  }
  for (auto& v : qlab) {
    v = int32_t(rng() % 38);
  }
  auto dlab  = raft::make_device_vector<int32_t, int64_t>(res, n);
  auto dqlab = raft::make_device_vector<int32_t, int64_t>(res, nq);
  raft::copy(dlab.data_handle(), lab.data(), n, stream);
  raft::copy(dqlab.data_handle(), qlab.data(), nq, stream);
  set_labels(
    res, raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle(), n0), &idx);
  partition_by_labels(res, &idx);
  extend(res,
         raft::make_device_matrix_view<const float, int64_t>(dx.data_handle() + n0 * d, n - n0, d),
         &idx);
  set_labels(res,
             raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle() + n0, n - n0),
             &idx,
             n0);
  auto nb = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto ds = raft::make_device_matrix<float, int64_t>(res, nq, k);
  std::vector<int64_t> ids(nq * k);
  search(res,
         search_params{},
         idx,
         raft::make_const_mdspan(dq.view()),
         raft::make_const_mdspan(dqlab.view()),
         nb.view(),
         ds.view());
  raft::copy(ids.data(), nb.data_handle(), nq * k, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(count_inexact(x,
                          q,
                          n,
                          nq,
                          d,
                          k,
                          DistanceType::L2Expanded,
                          ids,
                          [&](int64_t i, int64_t j) { return lab[j] == qlab[i]; }),
            0);
  std::vector<uint32_t> bits((n + 31) / 32, 0);
  for (int64_t j = 0; j < n; ++j) {
    if (lab[j] % 3 == 0) { bits[j / 32] |= 1u << (j % 32); }
  }
  auto dbits = raft::make_device_vector<uint32_t, int64_t>(res, int64_t(bits.size()));
  raft::copy(dbits.data_handle(), bits.data(), bits.size(), stream);
  cuvs::core::bitset_view<uint32_t, int64_t> view(dbits.data_handle(), n);
  search(res,
         search_params{},
         idx,
         raft::make_const_mdspan(dq.view()),
         nb.view(),
         ds.view(),
         cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(view));
  raft::copy(ids.data(), nb.data_handle(), nq * k, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(count_inexact(x,
                          q,
                          n,
                          nq,
                          d,
                          k,
                          DistanceType::L2Expanded,
                          ids,
                          [&](int64_t, int64_t j) { return lab[j] % 3 == 0; }),
            0);
}

}  // namespace cuvs::neighbors::certex
