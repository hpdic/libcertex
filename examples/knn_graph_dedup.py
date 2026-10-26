# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exact k-nearest-neighbor graph of a collection and near-duplicate removal: every vector's exact nearest
neighbors in one call, then groups of vectors closer than a threshold."""

import numpy as np

import certex

rng = np.random.default_rng(0)
xb = rng.standard_normal((1_000_000, 128), dtype=np.float32)
copies = rng.choice(xb.shape[0], 20_000, replace=False)
xb = np.concatenate([xb, xb[copies] + 0.01 * rng.standard_normal((len(copies), 128), dtype=np.float32)])

D, I = certex.search(xb, k=10, metric="cosine")  # the exact 10-NN graph (each vector is its own first neighbor)

# near-duplicates: pairs with cosine similarity above a threshold, merged into groups by union-find
parent = np.arange(xb.shape[0])


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


rows, cols = np.nonzero(D > 0.99)
for a, b in zip(rows, I[rows, cols], strict=True):
    if a != b:
        parent[find(a)] = find(b)
roots = np.array([find(i) for i in range(xb.shape[0])])
keep = np.unique(roots, return_index=True)[1]
print(f"{xb.shape[0]} vectors, {xb.shape[0] - len(keep)} near-duplicates removed, {len(keep)} kept")
