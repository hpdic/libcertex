# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Filtered search: every vector has a category, and every query asks for its nearest vectors of one category."""

import numpy as np

import certex

rng = np.random.default_rng(0)
xb = rng.standard_normal((500_000, 96), dtype=np.float32)
category = rng.integers(0, 1_000, size=xb.shape[0])  # 1,000 categories: 0.1% selectivity
xq = rng.standard_normal((1_000, 96), dtype=np.float32)
wanted = rng.integers(0, 1_000, size=xq.shape[0])

index = certex.Index(dim=96)
index.add(xb, labels=category)
D, I = index.search(xq, k=10, labels=wanted)
assert (category[I] == wanted[:, None]).all()  # every result has the requested category
print("query 0 asks for category", wanted[0], "and gets", I[0])
