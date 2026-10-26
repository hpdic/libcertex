# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exact 10-NN search over one million random vectors, checked against exhaustive search for a few queries."""

import numpy as np
import torch

import certex

rng = np.random.default_rng(0)
xb = rng.standard_normal((1_000_000, 128), dtype=np.float32)
xq = rng.standard_normal((2_048, 128), dtype=np.float32)

D, I = certex.search(xb, xq, k=10)  # one call
print("nearest neighbors of query 0:", I[0])

# the same answer as exhaustive search in double precision
d = ((torch.from_numpy(xq[:8]).double()[:, None, :] - torch.from_numpy(xb).double()[None]) ** 2).sum(-1)
assert np.array_equal(np.sort(I[:8], 1), np.sort(torch.topk(d, 10, largest=False).indices.numpy(), 1))
print("exact:", True)

# for repeated searches and updates, keep an index
index = certex.Index(dim=128)
index.add(xb)
D2, I2 = index.search(xq, k=10)
assert np.array_equal(I2, I)
