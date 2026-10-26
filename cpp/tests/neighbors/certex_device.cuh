/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace cuvs::neighbors::certex {

/**
 * Skips every test of the binary on a GPU older than compute capability 8.0, which CertEx does not
 * support (build and search throw there).
 */
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

inline auto* const certex_device_environment =
  ::testing::AddGlobalTestEnvironment(new CertexDeviceEnvironment);

}  // namespace cuvs::neighbors::certex
