# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Replacing faiss.IndexFlatL2 by an exact GPU index with the same interface."""

import numpy as np

from certex.faiss import IndexCertExL2

d = 64
xb = np.random.default_rng(0).standard_normal((200_000, d), dtype=np.float32)
xq = np.random.default_rng(1).standard_normal((1_000, d), dtype=np.float32)

index = IndexCertExL2(d)  # was: index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xq, 5)
print(index.ntotal, "vectors; first query:", I[0], D[0])
