# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""cuVS-style adapter: the call conventions of cuvs.neighbors.brute_force, with exact results by certified exclusion.

    from certex import cuvs as certex_cuvs
    index = certex_cuvs.build(dataset, metric="sqeuclidean")          # was: brute_force.build(dataset, metric=...)
    distances, neighbors = certex_cuvs.search(index, queries, k)       # was: brute_force.search(index, queries, k)

Inputs are any CUDA array (torch, CuPy, or another array with __dlpack__ or __cuda_array_interface__); outputs are
torch tensors on the GPU, which convert to CuPy without a copy (cupy.asarray). Metrics: "sqeuclidean" (squared
distances), "inner_product" (inner products, largest first), "cosine" (cosine distances 1 - cos, smallest first).
"""

import torch

from .index import Index

_METRIC = {"sqeuclidean": "l2", "l2": "l2", "euclidean": "l2", "inner_product": "ip", "cosine": "cosine"}


def _tensor(a):
    if isinstance(a, torch.Tensor):
        return a
    if hasattr(a, "__dlpack__"):
        return torch.from_dlpack(a)
    return torch.as_tensor(a, device="cuda")


def build(dataset, metric="sqeuclidean", width="auto", storage="device"):
    X = _tensor(dataset)
    index = Index(X.shape[1], metric=_METRIC[metric], width=width, storage=storage)
    index.add(X)
    index.cuvs_metric = metric
    return index


def extend(index, new_vectors):
    return index.add(_tensor(new_vectors))


def search(index, queries, k):
    D, I = index.search(_tensor(queries), k=k)
    if index.cuvs_metric == "euclidean":
        D = D.clamp_min(0).sqrt()
    elif index.cuvs_metric == "cosine":
        D = 1 - D
    return D, I
