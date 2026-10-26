# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
"""Exact top-k over attention keys: the top-k selection of sparse attention, exact and on the GPU.

    cache = certex.KeyCache(n_kv_heads=8, head_dim=128)   # one exact index per KV head
    cache.append(keys)                                    # (n_kv_heads, n, head_dim), e.g. the post-RoPE keys
    scores, ids = cache.topk(q, k=2048)                   # q: (n_heads, m, head_dim) -> (n_heads, m, k) each

    scores, ids = certex.topk(q, keys, k=64)              # one call, the same as torch.topk(q @ keys.mT, k)

Both return what torch.topk(q @ keys.mT, k, dim=-1) returns, with the scores in descending order, but the k keys of
every query are the exact top k by inner product (float64), whereas a bf16 or fp16 matrix product reorders near
ties among the largest scores. Grouped-query attention is supported: query head h reads KV head h // (n_heads /
n_kv_heads). Keys accumulate with append() as a sequence grows; every search covers all keys appended so far.
"""

import torch

from .index import Index, _require

_TILE = 128  # anchors per group of the threshold stage


def _width(head_dim):
    """The widest screening row that fits the head dimension: m + 5 halves per row, a multiple of 64."""
    return max(59, (head_dim + 5) // 64 * 64 - 5) if head_dim >= 59 else head_dim


def _stride(n, k):
    """An anchor stride that leaves about 16k groups of anchors for n keys (between 1 and 16): the threshold stage
    needs at least 2k groups, and more groups give a tighter threshold for the few extra anchor distances."""
    s = n // (16 * k * _TILE)
    return int(max(1, min(16, s)))


def _keys(x):
    """Keys or queries as float16 on the GPU, or float32 when float16 would overflow (bf16 inputs can)."""
    x = torch.as_tensor(x, device="cuda")
    h = x.to(torch.float16)
    return h if bool(torch.isfinite(h).all()) else x.to(torch.float32)


class KeyCache:
    """Exact top-k over the keys of a growing KV cache: one index per KV head, extended by append().

    n_kv_heads, head_dim: the shape of the keys. k: the largest k that searches will ask for (it sets the anchor
    stride at the first append). capacity: the expected number of keys per head, which reserves memory once.
    """

    def __init__(self, n_kv_heads, head_dim, k=2048, capacity=None):
        _require(n_kv_heads >= 1 and head_dim >= 1, "n_kv_heads and head_dim must be positive")
        _require(1 <= k <= 2048, f"k must be between 1 and 2048, not {k}")
        self.n_kv_heads, self.head_dim, self.k, self.capacity = int(n_kv_heads), int(head_dim), int(k), capacity
        self.heads, self.dtype = None, None
        self.size = 0

    def append(self, keys):
        """Append keys of shape (n_kv_heads, n, head_dim); their ids continue from the keys appended before."""
        K = _keys(keys)
        _require(
            K.ndim == 3 and K.shape[0] == self.n_kv_heads and K.shape[2] == self.head_dim,
            f"keys must have shape ({self.n_kv_heads}, n, {self.head_dim}), not {tuple(K.shape)}",
        )
        if self.heads is None:
            self.dtype = K.dtype
            n = max(K.shape[1], int(self.capacity or 0))
            self.heads = [
                Index(
                    self.head_dim,
                    metric="ip",
                    width=_width(self.head_dim),
                    storage="device",
                    capacity=self.capacity,
                    anchor_stride=_stride(n, self.k),
                )
                for _ in range(self.n_kv_heads)
            ]
        for g, idx in enumerate(self.heads):
            idx.add(K[g].to(self.dtype).contiguous())
        self.size += K.shape[1]
        return self

    def topk(self, q, k=None, batch=2048):
        """The exact top k keys of every query: q of shape (n_heads, m, head_dim) or (n_heads, head_dim), where
        n_heads is a multiple of n_kv_heads. Returns (scores, ids), each of shape (n_heads, m, k) or (n_heads, k)."""
        _require(self.heads is not None, "the cache is empty: append keys first")
        k = self.k if k is None else int(k)
        _require(1 <= k <= min(2048, self.size), f"k must be between 1 and {min(2048, self.size)}, not {k}")
        Q = _keys(q)
        single = Q.ndim == 2
        if single:
            Q = Q[:, None, :]
        _require(
            Q.ndim == 3 and Q.shape[2] == self.head_dim and Q.shape[0] % self.n_kv_heads == 0,
            f"queries must have shape (n_heads, m, {self.head_dim}) with n_heads a multiple of {self.n_kv_heads}",
        )
        H, m, d = Q.shape
        r = H // self.n_kv_heads
        scores = torch.empty((H, m, k), dtype=torch.float32, device="cuda")
        ids = torch.empty((H, m, k), dtype=torch.int64, device="cuda")
        for g, idx in enumerate(self.heads):
            D, I = idx.search(Q[g * r : (g + 1) * r].reshape(r * m, d).to(self.dtype), k=k, batch=batch)
            scores[g * r : (g + 1) * r] = D.reshape(r, m, k)
            ids[g * r : (g + 1) * r] = I.reshape(r, m, k)
        return (scores[:, 0], ids[:, 0]) if single else (scores, ids)


def topk(q, keys, k):
    """The exact top k keys of every query in one call: keys (n_kv_heads, n, head_dim), q (n_heads, m, head_dim)
    or (n_heads, head_dim). The same as torch.topk(q @ keys.mT, k, dim=-1), with exact scores and ids."""
    K = torch.as_tensor(keys)
    _require(K.ndim == 3, "keys must have shape (n_kv_heads, n, head_dim)")
    return KeyCache(K.shape[0], K.shape[2], k=k).append(K).topk(q, k)
