/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_certex.cuh"

namespace cuvs::neighbors::certex {

typedef AnnCertexTest<uint8_t> AnnCertexTestU8;
TEST_P(AnnCertexTestU8, AnnCertex) { this->testCertex(); }

INSTANTIATE_TEST_CASE_P(AnnCertexTest, AnnCertexTestU8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::certex
