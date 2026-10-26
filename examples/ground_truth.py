# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exact ground truth for evaluating an approximate index: the true k nearest neighbors of every query, the recall
of any approximate result against them, and a spot check of the ground truth itself."""

import numpy as np

import certex

rng = np.random.default_rng(0)
xb = rng.standard_normal((2_000_000, 96), dtype=np.float32)
xq = rng.standard_normal((10_000, 96), dtype=np.float32)

_, gt = certex.search(xb, xq, k=100)  # exact ground truth in one call
np.save("groundtruth_k100.npy", gt)
print("ground truth exact on 100 sampled queries:", certex.verify(xb, xq, gt))


def recall(approx_ids, gt_ids, k=10):
    """Fraction of the true k nearest neighbors that an approximate result returns."""
    hits = [len(set(a[:k]) & set(g[:k])) for a, g in zip(approx_ids, gt_ids, strict=True)]
    return sum(hits) / (k * len(gt_ids))


# any approximate result: here a search over a random half of the vectors
half = rng.permutation(xb.shape[0])[: xb.shape[0] // 2]
_, I_half = certex.search(xb[half], xq, k=10)
print("recall@10 of the half search:", recall(half[I_half], gt))
