/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The tensor-core kernels of certex against a direct fp64 evaluation of the same half-precision
// products: every pair whose exact product clears its threshold by more than a tolerance must be
// kept, no pair below it by more than the tolerance may be kept, and the anchor groups must report
// the maximum of every group. Every kernel of the certified pass (cooperative, warp-specialized,
// and sm80, which also runs on Hopper) runs on every shape that it serves.

#include "../../src/neighbors/certex/detail/kernels.hpp"
#include "certex_device.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <random>
#include <set>
#include <vector>

namespace cuvs::neighbors::certex {

using detail::screen_kernel;

struct CertexKernelInputs {
  int B, N, K;
  unsigned seed;
  screen_kernel kernel;
};

inline auto operator<<(std::ostream& os, const CertexKernelInputs& p) -> std::ostream&
{
  const char* k = p.kernel == screen_kernel::pingpong      ? "pingpong"
                  : p.kernel == screen_kernel::cooperative ? "cooperative"
                  : p.kernel == screen_kernel::sm80        ? "sm80"
                                                           : "automatic";
  return os << "{B=" << p.B << " N=" << p.N << " K=" << p.K << " kernel=" << k << "}";
}

class CertexKernelTest : public ::testing::TestWithParam<CertexKernelInputs> {
 protected:
  void run()
  {
    const auto p = GetParam();
    if (p.kernel == screen_kernel::pingpong && !detail::pingpong_supported(p.K)) {
      GTEST_SKIP() << "no warp-specialized pass for rows of " << p.K;
    }
    const int B = p.B, N = p.N, K = p.K;
    std::mt19937 rng(p.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> q(size_t(B) * K), x(size_t(N) * K);
    for (auto& v : q) {
      v = __float2half(nd(rng) * 0.5f);
    }
    for (auto& v : x) {
      v = __float2half(nd(rng) * 0.5f);
    }
    std::vector<double> dot(size_t(B) * N);
    for (int i = 0; i < B; ++i) {
      for (int j = 0; j < N; ++j) {
        double s = 0;
        for (int k = 0; k < K; ++k) {
          s += double(__half2float(q[size_t(i) * K + k])) * __half2float(x[size_t(j) * K + k]);
        }
        dot[size_t(i) * N + j] = s;
      }
    }
    // threshold of each query: its 0.2% quantile from above, so that a few hundred pairs survive
    std::vector<float> theta(B);
    for (int i = 0; i < B; ++i) {
      std::vector<double> row(dot.begin() + size_t(i) * N, dot.begin() + size_t(i + 1) * N);
      std::nth_element(row.begin(), row.begin() + N / 500, row.end(), std::greater<double>());
      theta[i] = float(row[N / 500]);
    }

    const int nseg     = detail::screen_segments(B, N, K, p.kernel);
    const uint32_t cap = uint32_t(std::max<long>(1024, long(B) * N / 50 / nseg * 4));
    const int G        = (N + detail::kTileN - 1) / detail::kTileN;
    auto dq            = raft::make_device_vector<half, int64_t>(res_, int64_t(q.size()));
    auto dx            = raft::make_device_vector<half, int64_t>(res_, int64_t(x.size()));
    auto dth           = raft::make_device_vector<float, int64_t>(res_, B);
    auto oq            = raft::make_device_vector<int32_t, int64_t>(res_, int64_t(cap) * nseg);
    auto ox            = raft::make_device_vector<int32_t, int64_t>(res_, int64_t(cap) * nseg);
    auto ov            = raft::make_device_vector<float, int64_t>(res_, int64_t(cap) * nseg);
    auto cnt           = raft::make_device_vector<uint32_t, int64_t>(res_, nseg);
    auto bv            = raft::make_device_vector<float, int64_t>(res_, int64_t(B) * G);
    auto bi            = raft::make_device_vector<int32_t, int64_t>(res_, int64_t(B) * G);
    raft::copy(dq.data_handle(), q.data(), q.size(), stream_);
    raft::copy(dx.data_handle(), x.data(), x.size(), stream_);
    raft::copy(dth.data_handle(), theta.data(), B, stream_);
    ASSERT_EQ(detail::screen(dq.data_handle(),
                             B,
                             dx.data_handle(),
                             N,
                             K,
                             dth.data_handle(),
                             oq.data_handle(),
                             ox.data_handle(),
                             ov.data_handle(),
                             cnt.data_handle(),
                             cap,
                             detail::Filter{},
                             stream_,
                             p.kernel),
              nseg);
    detail::anchor_groups(
      dq.data_handle(),
      B,
      dx.data_handle(),
      N,
      K,
      bv.data_handle(),
      bi.data_handle(),
      detail::Filter{},
      1,
      stream_,
      p.kernel == screen_kernel::sm80 ? screen_kernel::sm80 : screen_kernel::automatic);
    std::vector<uint32_t> segc(nseg);
    std::vector<int32_t> aq(size_t(cap) * nseg), ax(size_t(cap) * nseg), hbi(size_t(B) * G);
    std::vector<float> av(size_t(cap) * nseg), hbv(size_t(B) * G);
    raft::copy(segc.data(), cnt.data_handle(), nseg, stream_);
    raft::copy(aq.data(), oq.data_handle(), aq.size(), stream_);
    raft::copy(ax.data(), ox.data_handle(), ax.size(), stream_);
    raft::copy(av.data(), ov.data_handle(), av.size(), stream_);
    raft::copy(hbv.data(), bv.data_handle(), hbv.size(), stream_);
    raft::copy(hbi.data(), bi.data_handle(), hbi.size(), stream_);
    raft::resource::sync_stream(res_);

    std::set<std::pair<int, int>> kept;
    int bad = 0;
    for (int sg = 0; sg < nseg; ++sg) {
      ASSERT_LE(segc[sg], cap) << "overflow of segment " << sg;
      for (uint32_t t = 0; t < segc[sg]; ++t) {
        const size_t s = size_t(sg) * cap + t;
        if (!kept.insert({aq[s], ax[s]}).second) {  // duplicate
          ++bad;
          continue;
        }
        const double e = dot[size_t(aq[s]) * N + ax[s]];
        if (std::fabs(av[s] - e) > 1e-3 * (1 + std::fabs(e))) { ++bad; }  // wrong product
        if (e < theta[aq[s]] - 1e-3 * (1 + std::fabs(e))) { ++bad; }      // clearly below
      }
    }
    long missing = 0;
    for (int i = 0; i < B; ++i) {
      for (int j = 0; j < N; ++j) {
        const double e = dot[size_t(i) * N + j];
        if (e >= theta[i] + 1e-3 * (1 + std::fabs(e)) && !kept.count({i, j})) { ++missing; }
      }
    }
    int bad_anchor = 0;
    for (int i = 0; i < B; ++i) {
      for (int g = 0; g < G; ++g) {
        const int lo = g * detail::kTileN, hi = std::min(N, (g + 1) * detail::kTileN);
        double best = -1e300;
        for (int j = lo; j < hi; ++j) {
          best = std::max(best, dot[size_t(i) * N + j]);
        }
        const int j = hbi[size_t(i) * G + g];
        if (j < lo || j >= hi ||
            std::fabs(hbv[size_t(i) * G + g] - best) > 1e-3 * (1 + std::fabs(best)) ||
            std::fabs(dot[size_t(i) * N + j] - best) > 1e-3 * (1 + std::fabs(best))) {
          ++bad_anchor;
        }
      }
    }
    EXPECT_EQ(missing, 0);
    EXPECT_EQ(bad, 0);
    EXPECT_EQ(bad_anchor, 0);
  }

  raft::device_resources res_;
  cudaStream_t stream_ = raft::resource::get_cuda_stream(res_).get();
};

TEST_P(CertexKernelTest, Screen) { this->run(); }

inline auto kernel_inputs() -> std::vector<CertexKernelInputs>
{
  std::vector<CertexKernelInputs> out;
  // B, N, K: rows of 64, 128 (partial tiles), a multiple of 64 beyond 128, a long row
  const std::vector<std::array<int, 4>> shapes = {{300, 10000, 64, 1},
                                                  {129, 5000, 128, 2},
                                                  {1000, 20011, 192, 3},
                                                  {17, 333, 640, 4},
                                                  {64, 7777, 256, 6}};
  for (auto s : shapes) {
    for (auto k : {screen_kernel::cooperative, screen_kernel::pingpong, screen_kernel::sm80}) {
      out.push_back({s[0], s[1], s[2], unsigned(s[3]), k});
    }
  }
  return out;
}

INSTANTIATE_TEST_CASE_P(CertexKernelTest, CertexKernelTest, ::testing::ValuesIn(kernel_inputs()));

}  // namespace cuvs::neighbors::certex
