# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exact top-k selection for FlashInfer sparse attention over a paged KV cache: certex.flashinfer returns the int32
physical positions of top_k_page_table_transform, computed from the queries and the keys instead of the scores. With
FlashInfer installed, the output is compared with FlashInfer on the same queries and keys."""

import torch

from certex import flashinfer as certex_fi

torch.manual_seed(0)
head_dim, page_size, k = 128, 16, 64
lengths = [5000, 1800, 40]  # three sequences; the last is shorter than k
keys = [torch.randn(n, 1, head_dim, device="cuda", dtype=torch.float16) for n in lengths]  # NHD, one KV head

# the paged KV cache: every sequence occupies its own pages of a shared pool, in a shuffled order
pages = [(n + page_size - 1) // page_size for n in lengths]
pool = torch.randperm(sum(pages), device="cuda").int()
page_table = torch.zeros(len(lengths), max(pages), dtype=torch.int32, device="cuda")
start = 0
for b, p in enumerate(pages):
    page_table[b, :p] = pool[start : start + p]
    start += p

selector = certex_fi.ExactTopK(num_kv_heads=1, head_dim=head_dim, k_max=k)
for slot, K in enumerate(keys):
    selector.append(slot, K)  # in a server: append the keys of every new token of the sequence

q = torch.randn(len(lengths), head_dim, device="cuda", dtype=torch.float16)  # one decoding row per sequence
positions = selector.page_table_transform(q, page_table, k, page_size=page_size)
print(
    "physical positions:",
    tuple(positions.shape),
    positions.dtype,
    "-1 entries per row:",
    (positions < 0).sum(1).tolist(),
)

try:
    import flashinfer
except ImportError:
    flashinfer = None
if flashinfer is not None:  # the same selection through FlashInfer, from float64-rounded scores
    scores = torch.full((len(lengths), max(lengths)), float("-inf"), device="cuda")
    for b, K in enumerate(keys):
        scores[b, : lengths[b]] = (q[b].double() @ K[:, 0].double().T).float()
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    ref = flashinfer.top_k_page_table_transform(scores, page_table, lens, k, page_size=page_size)
    same = all(set(ref[b].tolist()) == set(positions[b].tolist()) for b in range(len(lengths)))
    print("same positions as FlashInfer:", same)

selector.release(0)  # a finished sequence frees its keys
