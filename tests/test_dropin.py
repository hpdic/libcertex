# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""The adapters are drop-in replacements: the same calls on cuVS, Faiss, and FlashInfer and on certex give the same
neighbors. Each test runs only where the original library is installed.

    python -m pytest tests/test_dropin.py
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from certex import cuvs as certex_cuvs  # noqa: E402
from certex import flashinfer as certex_fi  # noqa: E402
from certex.faiss import IndexCertExIP, IndexCertExL2  # noqa: E402

pytestmark = pytest.mark.gpu


def data(n=20_000, m=500, d=64, seed=0):
    g = np.random.default_rng(seed)
    return g.standard_normal((n, d), dtype=np.float32), g.standard_normal((m, d), dtype=np.float32)


def same_neighbors(I1, I2, D1, D2, largest=False):
    """The id lists agree, up to the order of near ties, and the distances agree to single precision."""
    D1, D2 = np.sort(np.asarray(D1), 1), np.sort(np.asarray(D2), 1)
    assert np.allclose(D1, D2, rtol=1e-4, atol=1e-4), "distances differ"
    agree = np.mean([len(set(a) & set(b)) == len(a) for a, b in zip(np.asarray(I1), np.asarray(I2), strict=True)])
    assert agree >= 0.99, f"only {agree:.3f} of the queries have the same neighbor set"


def test_cuvs_brute_force():
    brute_force = pytest.importorskip("cuvs.neighbors.brute_force")
    X, Q = data()
    Xd, Qd = torch.from_numpy(X).cuda(), torch.from_numpy(Q).cuda()
    for metric in ("sqeuclidean", "inner_product"):
        ref = brute_force.build(Xd, metric=metric)  # the same two calls on both libraries
        D_ref, I_ref = brute_force.search(ref, Qd, 10)
        ours = certex_cuvs.build(Xd, metric=metric)
        D, I = certex_cuvs.search(ours, Qd, 10)
        D_ref, I_ref = torch.as_tensor(D_ref, device="cuda"), torch.as_tensor(I_ref, device="cuda")
        assert D.shape == D_ref.shape and I.shape == I_ref.shape
        if metric == "inner_product":  # cuVS reports inner products as similarities, as certex does
            D, D_ref = -D, -D_ref
        same_neighbors(I.cpu(), I_ref.cpu(), D.cpu(), D_ref.cpu())


def test_faiss_flat():
    faiss = pytest.importorskip("faiss")
    X, Q = data()
    for ref_cls, ours_cls in ((faiss.IndexFlatL2, IndexCertExL2), (faiss.IndexFlatIP, IndexCertExIP)):
        ref, ours = ref_cls(X.shape[1]), ours_cls(X.shape[1])
        for index in (ref, ours):  # the same calls on both libraries
            index.add(X)
        D_ref, I_ref = ref.search(Q, 10)
        D, I = ours.search(Q, 10)
        assert D.dtype == D_ref.dtype and I.dtype == I_ref.dtype and D.shape == D_ref.shape
        assert ours.ntotal == ref.ntotal
        same_neighbors(I, I_ref, D, D_ref)


def test_flashinfer_page_table_transform():
    flashinfer = pytest.importorskip("flashinfer")
    g = torch.Generator(device="cuda").manual_seed(0)
    d, ps, k = 64, 16, 128
    lens = [3000, 1700, 100]  # the last sequence is shorter than k
    keys = [torch.randn(n, d, generator=g, device="cuda").half() for n in lens]
    q = torch.randn(len(lens), d, generator=g, device="cuda").half()  # one decoding row per sequence
    pages = [(n + ps - 1) // ps for n in lens]
    perm = torch.randperm(sum(pages), generator=g, device="cuda").int()
    table = torch.full((len(lens), max(pages)), 0, dtype=torch.int32, device="cuda")
    start = 0
    for b, p in enumerate(pages):
        table[b, :p] = perm[start : start + p]
        start += p
    scores = torch.full((len(lens), max(lens)), float("-inf"), device="cuda")
    for b, K in enumerate(keys):  # the indexer scores that FlashInfer selects from, exact in fp32 for fp16 inputs
        scores[b, : lens[b]] = (q[b : b + 1].double() @ K.double().T).float()[0]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ref = flashinfer.top_k_page_table_transform(scores, table, lengths, k, page_size=ps)
    sel = certex_fi.ExactTopK(1, d, k_max=k)
    for b, K in enumerate(keys):
        sel.append(b, K[:, None, :])
    ours = sel.page_table_transform(q, table, k, page_size=ps)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype == torch.int32
    assert bool(((ours < 0) == (ref < 0)).all()), "the -1 padding of short sequences differs"
    for b in range(len(keys)):  # map physical positions back to scores: the selected scores must agree
        phys_to_score = {}
        for p in range(pages[b]):
            for o in range(ps):
                pos = p * ps + o
                if pos < lens[b]:
                    phys_to_score[int(table[b, p]) * ps + o] = float(scores[b, pos])
        a = sorted(phys_to_score[int(x)] for x in ours[b] if x >= 0)
        r = sorted(phys_to_score[int(x)] for x in ref[b] if x >= 0)
        assert a == r, f"sequence {b}: the selected scores differ"
