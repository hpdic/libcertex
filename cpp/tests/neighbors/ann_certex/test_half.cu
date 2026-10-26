/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_certex.cuh"

namespace cuvs::neighbors::certex {

typedef AnnCertexTest<half> AnnCertexTestH;
TEST_P(AnnCertexTestH, AnnCertex) { this->testCertex(); }

INSTANTIATE_TEST_CASE_P(AnnCertexTest, AnnCertexTestH, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::certex
