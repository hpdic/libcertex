# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
"""certex.flashinfer: the transforms of flashinfer.top_k_ragged_transform and top_k_page_table_transform, computed
from queries and keys, select the exact top k keys of every query row."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from certex import flashinfer as certex_fi  # noqa: E402

pytestmark = pytest.mark.gpu

KV, H, D = 2, 4, 128


def rand(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).half()


def assert_exact(q, keys, pos, k):
    """q (m, H, d), keys (n, KV, d), pos (m, H, k) local positions: the selected scores equal the exact top k."""
    r = q.shape[1] // keys.shape[1]
    for h in range(q.shape[1]):
        Kh = keys[:, h // r].double()
        ref = torch.topk(q[:, h].double() @ Kh.T, k, dim=1).values
        got = torch.sort(torch.einsum("md,mkd->mk", q[:, h].double(), Kh[pos[:, h].long()]), 1, descending=True).values
        assert bool(((got - ref).abs() <= 1e-12 * ref.abs().clamp_min(1e-30)).all()), f"head {h} is not exact"


def batch():
    """Three sequences in slots 7, 3, 5 (batch positions 0, 1, 2); the last is shorter than k."""
    lens = [40_000, 25_000, 100]
    keys = [rand(n, KV, D, seed=s) for s, n in enumerate(lens)]
    sel = certex_fi.ExactTopK(KV, D, k_max=256)
    for slot, K in zip((7, 3, 5), keys, strict=True):
        for s in range(0, K.shape[0], 10_000):  # keys arrive in steps
            sel.append(slot, K[s : s + 10_000])
    rows = torch.tensor([0, 0, 1, 2, 1, 0], device="cuda")  # row -> batch position
    q = rand(rows.numel(), H, D, seed=9)
    return sel, keys, rows, q, torch.tensor([7, 3, 5])


def test_ragged_transform():
    sel, keys, rows, q, slots = batch()
    offsets = torch.tensor([0, 1000, 2000, 3000, 4000, 5000], device="cuda", dtype=torch.int32)
    out = sel.ragged_transform(q, offsets, 128, row_to_batch=rows, slots=slots)
    assert out.shape == (6, H, 128) and out.dtype == torch.int32
    for i in range(6):
        b = int(rows[i])
        local = out[i] - offsets[i]
        if b == 2:  # 100 keys < k: all of them, then -1
            assert bool((out[i, :, 100:] == -1).all())
            assert bool((torch.sort(local[:, :100], 1).values == torch.arange(100, device="cuda")).all())
        else:
            assert_exact(q[i : i + 1], keys[b], local[None], 128)


def test_page_table_transform():
    sel, keys, rows, q, slots = batch()
    ps = 16
    pages = [(K.shape[0] + ps - 1) // ps for K in keys]
    perm = torch.randperm(sum(pages), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    table = torch.full((3, max(pages)), -1, dtype=torch.int32, device="cuda")
    phys = torch.zeros(sum(pages) * ps, KV, D, dtype=torch.float16, device="cuda")  # a paged key pool
    start = 0
    for b, K in enumerate(keys):
        table[b, : pages[b]] = perm[start : start + pages[b]].int()
        start += pages[b]
        for p in range(pages[b]):
            chunk = K[p * ps : (p + 1) * ps]
            phys[int(table[b, p]) * ps : int(table[b, p]) * ps + chunk.shape[0]] = chunk
    out = sel.page_table_transform(q, table, 64, row_to_batch=rows, slots=slots, page_size=ps)
    for i in range(6):
        b = int(rows[i])
        if b == 2:
            assert bool((out[i, :, 100:] == -1).all())
            continue
        r = H // KV  # the physical keys must be the exact top k
        for h in range(H):
            sc = torch.sort(phys[out[i, h].long(), h // r].double() @ q[i, h].double(), descending=True).values
            ref = torch.topk(keys[b][:, h // r].double() @ q[i, h].double(), 64).values
            assert bool(((sc - ref).abs() <= 1e-12 * ref.abs().clamp_min(1e-30)).all())


def test_single_head_rows():
    keys = rand(30_000, 1, D, seed=4)
    q = rand(8, D, seed=5)
    sel = certex_fi.ExactTopK(1, D, k_max=64).append(0, keys)
    out = sel.ragged_transform(q, torch.zeros(8, dtype=torch.int32), 64, row_to_batch=torch.zeros(8, dtype=torch.long))
    assert out.shape == (8, 64)
    assert_exact(q[:, None], keys, out[:, None], 64)


def test_top_k():
    keys = rand(50_000, D, seed=6)
    q = rand(16, D, seed=7).bfloat16()
    values, indices = certex_fi.top_k(q, keys, 32)
    assert values.shape == indices.shape == (16, 32) and indices.dtype == torch.int64
    assert_exact(q.float()[:, None], keys[:, None], indices[:, None], 32)
