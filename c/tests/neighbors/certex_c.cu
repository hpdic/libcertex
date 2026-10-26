/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The CertEx C API against exhaustive search in double precision: plain, with lists, with local
// projections, with a bitset filter, and after a save and load.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

extern "C" int run_certex(int64_t n_rows,
                          int64_t n_queries,
                          int64_t dim,
                          int64_t k,
                          float* dataset,
                          float* queries,
                          uint32_t* bitset,
                          int32_t n_lists,
                          int32_t local_projections,
                          const char* save_to,
                          int64_t* neighbors,
                          float* distances);

namespace {

struct Case {
  int32_t n_lists;
  bool local, filter, save;
};

void check(Case c)
{
  const int64_t n = 20000, nq = 100, d = 32, k = 10;
  std::mt19937 gen(7);
  std::normal_distribution<float> normal;
  std::vector<float> x(n * d), q(nq * d);
  for (auto& v : x) {
    v = normal(gen);
  }
  for (auto& v : q) {
    v = normal(gen);
  }
  std::vector<uint32_t> bits((n + 31) / 32, 0);
  auto keep = [&](int64_t i) { return !c.filter || i % 3 == 0; };
  for (int64_t i = 0; i < n; ++i) {
    if (keep(i)) { bits[i / 32] |= 1u << (i % 32); }
  }
  float *dx, *dq, *dd;
  int64_t* dn;
  uint32_t* db;
  cudaMalloc(&dx, x.size() * 4);
  cudaMalloc(&dq, q.size() * 4);
  cudaMalloc(&dd, nq * k * 4);
  cudaMalloc(&dn, nq * k * 8);
  cudaMalloc(&db, bits.size() * 4);
  cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(db, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice);
  const std::string path = std::string(testing::TempDir()) + "certex_c_index.bin";
  ASSERT_EQ(run_certex(n,
                       nq,
                       d,
                       k,
                       dx,
                       dq,
                       c.filter ? db : nullptr,
                       c.n_lists,
                       c.local,
                       c.save ? path.c_str() : nullptr,
                       dn,
                       dd),
            0);
  std::vector<int64_t> ids(nq * k);
  cudaMemcpy(ids.data(), dn, ids.size() * 8, cudaMemcpyDeviceToHost);
  auto dist = [&](int64_t qi, int64_t xi) {
    double s = 0;
    for (int64_t j = 0; j < d; ++j) {
      const double t = double(q[qi * d + j]) - double(x[xi * d + j]);
      s += t * t;
    }
    return s;
  };
  for (int64_t qi = 0; qi < nq; ++qi) {
    std::vector<double> all;
    for (int64_t i = 0; i < n; ++i) {
      if (keep(i)) { all.push_back(dist(qi, i)); }
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::vector<double> got;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t id = ids[qi * k + j];
      ASSERT_GE(id, 0);
      ASSERT_TRUE(keep(id)) << "a filtered-out vector was returned";
      got.push_back(dist(qi, id));
    }
    std::sort(got.begin(), got.end());
    for (int64_t j = 0; j < k; ++j) {
      ASSERT_NEAR(got[j], all[j], 1e-12 * std::max(1.0, all[j])) << "query " << qi;
    }
  }
  cudaFree(dx);
  cudaFree(dq);
  cudaFree(dd);
  cudaFree(dn);
  cudaFree(db);
}

}  // namespace

// Skip every test on a GPU older than compute capability 8.0, which CertEx does not support.
class CertexDeviceEnvironment : public ::testing::Environment {
 public:
  void SetUp() override
  {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 8) { GTEST_SKIP() << "CertEx needs a GPU of compute capability 8.0 or later"; }
  }
};

const auto* const certex_device_environment =
  ::testing::AddGlobalTestEnvironment(new CertexDeviceEnvironment);

TEST(CertexC, Plain) { check({0, false, false, false}); }
TEST(CertexC, Lists) { check({32, false, false, false}); }
TEST(CertexC, LocalProjections) { check({32, true, false, false}); }
TEST(CertexC, Bitset) { check({0, false, true, false}); }
TEST(CertexC, LocalBitset) { check({32, true, true, false}); }
TEST(CertexC, SaveLoad) { check({32, false, false, true}); }
