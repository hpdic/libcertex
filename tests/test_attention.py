# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
"""certex.KeyCache and certex.topk: the exact top k attention keys, as torch.topk(q @ keys.mT, k) with exact scores."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certex  # noqa: E402

pytestmark = pytest.mark.gpu


def exact_scores(q, keys, ids):
    """The float64 inner products of every query with its selected keys, sorted in descending order."""
    H, m, d = q.shape
    r = H // keys.shape[0]
    K = keys.double()[torch.arange(H, device=q.device) // r]  # (H, n, d): the KV head of every query head
    s = torch.einsum("hmd,hmkd->hmk", q.double(), K[torch.arange(H, device=q.device)[:, None, None], ids])
    return torch.sort(s, -1, descending=True).values


def assert_exact(q, keys, scores, ids, k):
    H = q.shape[0]
    r = H // keys.shape[0]
    full = q.double() @ keys.double()[torch.arange(H, device=q.device) // r].mT  # (H, m, n)
    ref = torch.topk(full, k, dim=-1).values
    got = exact_scores(q, keys, ids)
    exact = ((got - ref).abs() <= 1e-12 * ref.abs().clamp_min(1e-30)).all()
    assert bool(exact), "selected keys are not the exact top k"
    assert bool((scores[..., :-1] >= scores[..., 1:]).all()), "scores must be in descending order"


def data(n_kv, H, n, m, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randn((n_kv, n, d), generator=g, device="cuda", dtype=torch.float32).half()
    q = torch.randn((H, m, d), generator=g, device="cuda", dtype=torch.float32).half()
    return q, keys


@pytest.mark.parametrize("k", [10, 64, 2048])
def test_gqa_exact(k):
    q, keys = data(n_kv=2, H=4, n=50_000, m=16)
    scores, ids = certex.KeyCache(2, 128, k=k).append(keys).topk(q, k)
    assert scores.shape == ids.shape == (4, 16, k)
    assert_exact(q, keys, scores, ids, k)


def test_append_in_steps():
    q, keys = data(n_kv=2, H=2, n=40_000, m=8)
    cache = certex.KeyCache(2, 128, k=64, capacity=40_000)
    for s in range(0, 40_000, 10_000):
        cache.append(keys[:, s : s + 10_000])
    scores, ids = cache.topk(q, 64)
    assert_exact(q, keys, scores, ids, 64)


def test_single_query_and_function():
    q, keys = data(n_kv=1, H=2, n=30_000, m=1)
    s1, i1 = certex.topk(q[:, 0], keys, 32)  # (n_heads, head_dim) queries
    assert s1.shape == (2, 32)
    assert_exact(q, keys, s1[:, None], i1[:, None], 32)


def test_bf16_inputs():
    q, keys = data(n_kv=2, H=4, n=30_000, m=8)
    scores, ids = certex.topk(q.bfloat16(), keys.bfloat16(), 64)
    assert_exact(q.bfloat16().float(), keys.bfloat16().float(), scores, ids, 64)
