# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exact top-k attention keys in PyTorch: one call in place of torch.topk(q @ keys.mT, k), then a decoding loop whose
key cache grows by one token per step. Every selection is checked against float64 scores."""

import torch

import certex

torch.manual_seed(0)
n_kv_heads, n_heads, head_dim, k = 8, 32, 128, 64  # grouped-query attention: four query heads per KV head
keys = torch.randn(n_kv_heads, 200_000, head_dim, device="cuda", dtype=torch.bfloat16)
q = torch.randn(n_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # one decoding token


def exact(q, keys, ids, k):
    """Whether the selected keys of every head score the k largest float64 inner products."""
    kv = keys.double()[torch.arange(q.shape[0], device="cuda") // (q.shape[0] // keys.shape[0])]  # (heads, n, d)
    ref = torch.topk(torch.einsum("hd,hnd->hn", q.double(), kv), k, dim=1).values
    got = torch.einsum("hd,hkd->hk", q.double(), kv[torch.arange(q.shape[0], device="cuda")[:, None], ids])
    return bool(torch.equal(torch.sort(got, 1, descending=True).values, ref))


# one call: the exact top 64 keys of every query head
scores, ids = certex.topk(q, keys, k=k)  # (n_heads, k) each, scores largest first
print("one call:", tuple(ids.shape), "exact:", exact(q, keys, ids, k))

# decoding: one index per KV head, extended in place as tokens arrive
cache = certex.KeyCache(n_kv_heads, head_dim, k=k, capacity=keys.shape[1] + 16)
cache.append(keys)
for step in range(16):
    q = torch.randn(n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    scores, ids = cache.topk(q, k=k)
    assert exact(q, keys, ids, k), step
    new = torch.randn(n_kv_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)  # the keys of the new token
    cache.append(new)
    keys = torch.cat([keys, new], dim=1)
print("decoding:", cache.size, "keys after 16 steps, every selection exact")
