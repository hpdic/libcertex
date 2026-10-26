# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""FlashInfer-style adapter: the top-k selection of sparse attention, from queries and keys, exact.

FlashInfer selects the keys of sparse attention in two steps: an indexer writes the scores of every query row over
its sequence, and top_k_page_table_transform (or top_k_ragged_transform) keeps the k largest and returns their int32
positions. This adapter takes the queries and the keys instead of the scores, selects the exact top k by inner product
(float64), and returns the same int32 positions, so that the sparse attention kernel that reads them is unchanged.

    from certex import flashinfer as certex_fi
    sel = certex_fi.ExactTopK(num_kv_heads=8, head_dim=128)
    sel.append(0, k_new)       # the new keys of sequence slot 0, NHD layout: (n, num_kv_heads, head_dim)
    out = sel.page_table_transform(q, src_page_table, k=2048, page_size=64)
                               # was: flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, 2048, ...)
    out = sel.ragged_transform(q, offsets, k=2048)
                               # was: flashinfer.top_k_ragged_transform(scores, offsets, lengths, 2048)
    values, indices = certex_fi.top_k(q, keys, k=64)
                               # was: flashinfer.top_k(q @ keys.T, 64)

Query rows use the NHD layout of FlashInfer. A row of shape (head_dim,) selects for one head: q is (num_rows,
head_dim) and the output (num_rows, k), as in FlashInfer. With (num_rows, num_qo_heads, head_dim), every query head
selects its own k keys from its KV head (grouped-query attention) and the output is (num_rows, num_qo_heads, k).
Row i belongs to the sequence in batch position row_to_batch[i] (default i), which holds the keys of slot
slots[row_to_batch[i]] (default the batch position itself). Every row selects among all keys appended to its slot,
as in decoding. Positions beyond the length of a sequence shorter than k are -1, as in FlashInfer.
"""

import torch

from .attention import KeyCache
from .index import _require


def _tensor(a):
    if isinstance(a, torch.Tensor):
        return a
    if hasattr(a, "__dlpack__"):
        return torch.from_dlpack(a)
    return torch.as_tensor(a, device="cuda")


class ExactTopK:
    """Exact top-k selection over the keys of a batch of sequences: one KeyCache per sequence slot.

    num_kv_heads, head_dim: the shape of the keys. k_max: the largest k that selections will ask for. capacity: the
    expected number of keys per sequence, which reserves memory once per slot.
    """

    def __init__(self, num_kv_heads, head_dim, k_max=2048, capacity=None):
        _require(1 <= k_max <= 2048, f"k_max must be between 1 and 2048, not {k_max}")
        self.num_kv_heads, self.head_dim, self.k_max = int(num_kv_heads), int(head_dim), int(k_max)
        self.capacity = capacity
        self.slots = {}

    def append(self, slot, keys):
        """Append the keys of shape (n, num_kv_heads, head_dim) to a sequence slot; positions continue."""
        K = _tensor(keys)
        _require(
            K.ndim == 3 and K.shape[1] == self.num_kv_heads and K.shape[2] == self.head_dim,
            f"keys must have shape (n, {self.num_kv_heads}, {self.head_dim}), not {tuple(K.shape)}",
        )
        if slot not in self.slots:
            self.slots[slot] = KeyCache(self.num_kv_heads, self.head_dim, k=self.k_max, capacity=self.capacity)
        self.slots[slot].append(K.transpose(0, 1))
        return self

    def release(self, slot):
        """Free the keys of a finished sequence."""
        self.slots.pop(slot, None)

    def length(self, slot):
        return self.slots[slot].size if slot in self.slots else 0

    def select(self, q, k, row_to_batch=None, slots=None):
        """The exact top-k positions of every query row within its sequence: int32, -1 beyond a short sequence."""
        _require(1 <= k <= self.k_max, f"k must be between 1 and k_max={self.k_max}, not {k}")
        Q = _tensor(q)
        single = Q.ndim == 2
        if single:
            Q = Q[:, None, :]
        _require(
            Q.ndim == 3 and Q.shape[2] == self.head_dim and Q.shape[1] % self.num_kv_heads == 0,
            f"q must have shape (num_rows, num_qo_heads, {self.head_dim}) with num_qo_heads a multiple of "
            f"{self.num_kv_heads}, or (num_rows, {self.head_dim})",
        )
        R, H, _ = Q.shape
        b = torch.arange(R, device=Q.device) if row_to_batch is None else _tensor(row_to_batch).to(Q.device).long()
        out = torch.full((R, H, k), -1, dtype=torch.int32, device=Q.device)
        for batch in torch.unique(b).tolist():
            slot = batch if slots is None else int(slots[batch])
            _require(slot in self.slots, f"slot {slot} has no keys: append them first")
            rows = torch.nonzero(b == batch).flatten()
            cache = self.slots[slot]
            if cache.size <= k:  # every position is selected, as in FlashInfer
                out[rows, :, : cache.size] = torch.arange(cache.size, dtype=torch.int32, device=Q.device)
                continue
            _, ids = cache.topk(Q[rows].transpose(0, 1), k)  # (H, m, k)
            out[rows] = ids.transpose(0, 1).to(torch.int32)
        return out[:, 0] if single else out

    def ragged_transform(self, q, offsets, k, row_to_batch=None, slots=None):
        """flashinfer.top_k_ragged_transform from queries: the exact top-k positions plus offsets[i] of row i."""
        local = self.select(q, k, row_to_batch, slots)
        off = _tensor(offsets).to(local.device).to(torch.int32).reshape(-1, *([1] * (local.ndim - 1)))
        return torch.where(local >= 0, local + off, local)

    def page_table_transform(self, q, src_page_table, k, row_to_batch=None, slots=None, page_size=1):
        """flashinfer.top_k_page_table_transform from queries: the exact top-k positions translated to physical
        positions, src_page_table[row_to_batch[i], pos // page_size] * page_size + pos % page_size."""
        local = self.select(q, k, row_to_batch, slots)
        R = local.shape[0]
        if row_to_batch is None:
            b = torch.arange(R, device=local.device)
        else:
            b = _tensor(row_to_batch).to(local.device).long()
        b = b.reshape(-1, *([1] * (local.ndim - 1)))
        pos = local.clamp_min(0).long()
        page = _tensor(src_page_table).to(local.device)[b, pos // page_size].to(torch.int32)
        return torch.where(local >= 0, page * page_size + (pos % page_size).to(torch.int32), local)


def top_k(q, keys, k, sorted=True):
    """flashinfer.top_k(q @ keys.T, k) from queries and keys, exact: q (num_rows, head_dim), keys (n, head_dim).
    Returns (values, indices): the float32 scores, largest first, and their int64 positions, each (num_rows, k)."""
    K, Q = _tensor(keys), _tensor(q)
    _require(
        K.ndim == 2 and Q.ndim == 2 and Q.shape[1] == K.shape[1],
        "q must have shape (num_rows, head_dim) and keys (n, head_dim)",
    )
    values, indices = KeyCache(1, K.shape[1], k=k).append(K[None]).topk(Q[None], k)
    return values[0], indices[0]
