/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_certex.cuh"

namespace cuvs::neighbors::certex {

typedef AnnCertexTest<float> AnnCertexTestF;
TEST_P(AnnCertexTestF, AnnCertex) { this->testCertex(); }

INSTANTIATE_TEST_CASE_P(AnnCertexTest, AnnCertexTestF, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::certex
