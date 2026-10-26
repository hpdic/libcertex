# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Replacing cuvs.neighbors.brute_force by exact search with the same calls, on CUDA arrays."""

import torch

from certex import cuvs as certex_cuvs

dataset = torch.randn(1_000_000, 128, device="cuda")
queries = torch.randn(4_096, 128, device="cuda")

index = certex_cuvs.build(dataset, metric="sqeuclidean")  # was: brute_force.build(dataset, metric=...)
distances, neighbors = certex_cuvs.search(index, queries, 10)  # was: brute_force.search(index, queries, 10)
print(neighbors.shape, neighbors.device)
