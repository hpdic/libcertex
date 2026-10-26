# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Exactness tests of certex: every returned neighbor list must have the exact k smallest scores, computed here in
double precision by direct differences (or products for inner products) against every stored vector. A test with an
argument Index runs against certex.Index.

    python -m pytest tests          # or: python tests/test_exact.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certex  # noqa: E402
from certex import cuvs as certex_cuvs  # noqa: E402
from certex.faiss import IndexCertExIP, IndexCertExL2  # noqa: E402

try:
    import pytest

    pytestmark = pytest.mark.gpu
except ImportError:  # the file also runs as a plain script
    pass

DEV = "cuda"
IMPLEMENTATIONS = {"cuda": certex.Index}


def data(n, d, nq, seed=0, scale_norms=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = torch.randn(d, d, device=DEV, generator=g) / d**0.5
    X = torch.randn(n, d, device=DEV, generator=g) @ A
    Q = torch.randn(nq, d, device=DEV, generator=g) @ A
    if scale_norms:  # unequal norms matter for inner products
        X = X * (0.2 + torch.rand(n, 1, device=DEV, generator=g) * 3)
    return X, Q


def scores(X, Q, ids, metric):
    """fp64 scores of the given ids: squared distance (l2, cosine on normalized vectors) or -inner product (ip)."""
    Xd, Qd = X.double(), Q.double()
    if metric == "cosine":
        Xd = Xd / Xd.norm(dim=1, keepdim=True)
        Qd = Qd / Qd.norm(dim=1, keepdim=True)
    rows = Xd[ids.clamp_min(0)]
    s = ((Qd[:, None, :] - rows) ** 2).sum(-1) if metric != "ip" else -(Qd[:, None, :] * rows).sum(-1)
    return torch.where(ids >= 0, s, torch.full_like(s, float("inf")))


def exact_k(X, Q, k, metric, live=None, chunk=4096):
    """The exact k smallest fp64 scores of every query over the live vectors."""
    out = []
    Xd = X.double()
    if metric == "cosine":
        Xd = Xd / Xd.norm(dim=1, keepdim=True)
    for i in range(0, Q.shape[0], chunk):
        Qd = Q[i : i + chunk].double()
        if metric == "cosine":
            Qd = Qd / Qd.norm(dim=1, keepdim=True)
        s = ((Qd[:, None, :] - Xd[None]) ** 2).sum(-1) if metric != "ip" else -(Qd @ Xd.T)
        if live is not None:
            s[:, ~live] = float("inf")
        out.append(torch.topk(s, k, dim=1, largest=False).values)
    return torch.cat(out)


def assert_exact(X, Q, ids, k, metric, live=None):
    got = torch.sort(scores(X, Q, torch.as_tensor(ids, device=DEV), metric), 1).values
    ref = exact_k(X, Q, k, metric, live)
    tol = 1e-12 * ref.abs().clamp_min(1e-12)
    bad = ~((got - ref).abs() <= tol).all(1)
    assert int(bad.sum()) == 0, f"{int(bad.sum())} of {Q.shape[0]} queries not exact"
    if live is not None:
        assert bool(live[torch.as_tensor(ids, device=DEV).clamp_min(0)].all()), "a removed vector was returned"


def test_metrics(Index):
    X, Q = data(20_000, 64, 500)
    for metric in ("l2", "cosine", "ip"):
        idx = Index(64, metric=metric)
        idx.add(X)
        D, I = idx.search(Q, k=10)
        assert_exact(X, Q, I, 10, metric)


def test_inner_product_unequal_norms(Index):
    X, Q = data(20_000, 64, 500, scale_norms=True)
    idx = Index(64, metric="ip")
    idx.add(X)
    D, I = idx.search(Q, k=10)
    assert_exact(X, Q, I, 10, "ip")
    ip = (Q.double()[:, None, :] * X.double()[I]).sum(-1)  # returned values are the inner products
    assert torch.allclose(D.double(), ip, rtol=1e-5, atol=1e-5)


def test_widths_and_storage(Index):
    X, Q = data(20_000, 96, 300)
    for width in (11, 59, 91):
        for storage in ("device", "host"):
            idx = Index(96, width=width, storage=storage)
            idx.add(X)
            _, I = idx.search(Q, k=7)
            assert_exact(X, Q, I, 7, "l2")


def test_numpy_uint8(Index):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, size=(20_000, 32), dtype=np.uint8)
    Q = rng.integers(0, 255, size=(200, 32)).astype(np.float32)
    idx = Index(32)
    ids = idx.add(X)
    assert isinstance(ids, np.ndarray) and ids.shape == (20_000,)
    D, I = idx.search(Q, k=5)
    assert isinstance(D, np.ndarray) and I.dtype == np.int64
    assert_exact(torch.from_numpy(X).to(DEV).float(), torch.from_numpy(Q).to(DEV), torch.from_numpy(I), 5, "l2")


def test_insert_and_remove(Index):
    X, Q = data(30_000, 48, 400)
    idx = Index(48, capacity=12_000)
    idx.add(X[:10_000])
    idx.add(X[10_000:20_000])  # grows past the capacity
    idx.add(X[20_000:])
    live = torch.ones(X.shape[0], dtype=torch.bool, device=DEV)
    gone = torch.randperm(X.shape[0], device=DEV)[:3_000]
    idx.remove(gone)
    live[gone] = False
    assert idx.ntotal == 27_000
    _, I = idx.search(Q, k=10)
    assert_exact(X, Q, I, 10, "l2", live)


def test_batches_and_small(Index):
    X, Q = data(5_000, 32, 3_001)
    idx = Index(32)
    idx.add(X)
    for B in (1, 17, 3_001):
        _, I = idx.search(Q[:B], k=10, batch=2048)
        assert_exact(X, Q[:B], I, 10, "l2")
    tiny = Index(32)
    tiny.add(X[:25])  # fewer anchors than k: every row is screened
    _, I = tiny.search(Q[:50], k=10)
    assert_exact(X[:25], Q[:50], I, 10, "l2")


def test_duplicates(Index):
    X, Q = data(10_000, 32, 200)
    X = torch.cat([X, X[:1].repeat(500, 1), X[1:2] + 1e-7])  # identical copies and near ties
    idx = Index(32)
    idx.add(X)
    _, I = idx.search(torch.cat([Q, X[:3]]), k=10)
    assert_exact(X, torch.cat([Q, X[:3]]), I, 10, "l2")


def exact_filtered(X, Q, lab, ql, k, live=None):
    """Exact k smallest fp64 squared distances over the vectors of each query's label (inf where fewer than k)."""
    Xd = X.double()
    out = []
    for i in range(Q.shape[0]):
        msk = lab == ql[i]
        if live is not None:
            msk &= live
        d = ((Q[i].double() - Xd[msk]) ** 2).sum(1)
        v = torch.sort(d).values[:k]
        out.append(torch.cat([v, torch.full((k - v.numel(),), float("inf"), dtype=torch.float64, device=DEV)]))
    return torch.stack(out)


def assert_exact_filtered(X, Q, lab, ql, I, k, live=None):
    I = torch.as_tensor(I, device=DEV)
    got = torch.sort(scores(X, Q, I, "l2"), 1).values
    ref = exact_filtered(X, Q, lab, ql, k, live)
    fin = torch.isfinite(ref)
    assert bool((torch.isfinite(got) == fin).all()), "wrong number of results for a small label"
    bad = ~(((got - ref).abs() <= 1e-12 * ref.abs().clamp_min(1e-12)) | ~fin).all(1)
    assert int(bad.sum()) == 0, f"{int(bad.sum())} of {Q.shape[0]} filtered queries not exact"
    ok = I >= 0
    assert bool((lab[I.clamp_min(0)][ok] == ql[:, None].expand_as(I)[ok]).all()), "a result has another label"


def test_filtered(Index):
    X, Q = data(30_000, 48, 300)
    g = torch.Generator(device=DEV).manual_seed(1)
    for L in (5, 50, 500, 5_000):  # 5,000 labels: most have fewer than 10 vectors
        lab = torch.randint(0, L, (X.shape[0],), device=DEV, generator=g)
        ql = torch.randint(0, L, (Q.shape[0],), device=DEV, generator=g)
        idx = Index(48)
        idx.add(X, labels=lab)
        _, I = idx.search(Q, k=10, labels=ql)
        assert_exact_filtered(X, Q, lab, ql, I, 10)


def test_filtered_updates(Index):
    X, Q = data(20_000, 32, 200)
    g = torch.Generator(device=DEV).manual_seed(2)
    lab = torch.randint(0, 20, (X.shape[0],), device=DEV, generator=g)
    ql = torch.randint(0, 20, (Q.shape[0],), device=DEV, generator=g)
    idx = Index(32, capacity=8_000)
    idx.add(X[:5_000], labels=lab[:5_000])
    idx.add(X[5_000:], labels=lab[5_000:])  # grows past the capacity
    live = torch.ones(X.shape[0], dtype=torch.bool, device=DEV)
    gone = torch.randperm(X.shape[0], device=DEV, generator=g)[:4_000]
    idx.remove(gone)
    live[gone] = False
    _, I = idx.search(Q, k=10, labels=ql)
    assert_exact_filtered(X, Q, lab, ql, I, 10, live)


def test_save_load(Index, tmp_path=None):
    import tempfile

    X, Q = data(20_000, 40, 300)
    g = torch.Generator(device=DEV).manual_seed(3)
    lab = torch.randint(0, 30, (X.shape[0],), device=DEV, generator=g)
    ql = torch.randint(0, 30, (Q.shape[0],), device=DEV, generator=g)
    idx = Index(40)
    idx.add(X, labels=lab)
    gone = torch.randperm(X.shape[0], device=DEV, generator=g)[:2_000]
    idx.remove(gone)
    D0, I0 = idx.search(Q, k=10, labels=ql)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        path = os.path.join(d, "index.certex")
        idx.save(path)
        back = Index.load(path)
    D1, I1 = back.search(Q, k=10, labels=ql)
    assert torch.equal(I0, I1) and back.ntotal == idx.ntotal
    live = torch.ones(X.shape[0], dtype=torch.bool, device=DEV)
    live[gone] = False
    assert_exact_filtered(X, Q, lab, ql, I1, 10, live)


def test_adapters():
    X, Q = data(20_000, 64, 300)
    index = certex_cuvs.build(X, metric="sqeuclidean")
    D, I = certex_cuvs.search(index, Q, 10)
    assert_exact(X, Q, I, 10, "l2")
    for cls, metric in ((IndexCertExL2, "l2"), (IndexCertExIP, "ip")):
        f = cls(64)
        f.add(X.cpu().numpy())
        D, I = f.search(Q.cpu().numpy(), 10)
        assert f.ntotal == 20_000 and D.dtype == np.float32 and I.dtype == np.int64
        assert_exact(X, Q, torch.from_numpy(I), 10, metric)


def test_label_partition():
    """CUDA only: filtered search with and without the label partition, and with a tail of later additions."""
    X, Q = data(30_000, 48, 300)
    g = torch.Generator(device=DEV).manual_seed(6)
    lab = torch.randint(0, 40, (X.shape[0],), device=DEV, generator=g)
    ql = torch.randint(0, 42, (Q.shape[0],), device=DEV, generator=g)  # labels 40 and 41 hold no vector
    for partition in (True, False):
        idx = certex.Index(48, partition_labels=partition)
        idx.add(X[:20_000], labels=lab[:20_000])
        idx.add(X[20_000:22_000], labels=lab[20_000:22_000])  # a tail below an eighth: no refresh
        assert idx.core.n_partitioned == (20_000 if partition else 0)
        idx.add(X[22_000:], labels=lab[22_000:])
        assert idx.core.n_partitioned == (30_000 if partition else 0)
        _, I = idx.search(Q, k=10, labels=ql, batch=100)
        assert_exact_filtered(X, Q, lab, ql, I, 10)


def test_lists():
    """CUDA only: k-means lists with the probe threshold, updates, masks, and removals."""
    X, Q = data(30_000, 48, 300)
    g = torch.Generator(device=DEV).manual_seed(8)
    idx = certex.Index(48, n_lists=64)
    ids = idx.add(X[:20_000])
    assert idx.core.n_lists == 64 and int(ids[-1]) == 19_999
    idx.add(X[20_000:])
    _, I = idx.search(Q, k=10)
    assert_exact(X, Q, I, 10, "l2")
    _, I = idx.search(Q, k=10, probe_lists=0)
    assert_exact(X, Q, I, 10, "l2")
    gone = torch.randperm(X.shape[0], device=DEV, generator=g)[:5_000]
    idx.remove(gone)
    live = torch.ones(X.shape[0], dtype=torch.bool, device=DEV)
    live[gone] = False
    keep = live & (torch.rand(X.shape[0], device=DEV, generator=g) < 0.5)
    _, I = idx.search(Q, k=10, filter=keep)
    assert_exact(X, Q, I, 10, "l2", keep)
    for metric in ("l2", "ip", "cosine"):  # int8 screening rows
        i8 = certex.Index(48, metric=metric, int8_rows=True)
        i8.add(X[:25_000])
        i8.add(X[25_000:])
        _, I = i8.search(Q, k=10)
        assert_exact(X, Q, I, 10, metric)
    for metric in ("l2", "ip", "cosine"):  # local projections of the lists
        loc = certex.Index(48, metric=metric, n_lists=32, local_projections=True)
        loc.add(X[:25_000])
        loc.add(X[25_000:])
        _, I = loc.search(Q, k=10)
        assert_exact(X, Q, I, 10, metric)


def test_masks():
    """CUDA only: a boolean mask over the ids (cuVS bitset) and one row of the mask per query (cuVS bitmap)."""
    X, Q = data(20_000, 48, 300)
    g = torch.Generator(device=DEV).manual_seed(4)
    idx = certex.Index(48)
    idx.add(X)
    keep = torch.rand(X.shape[0], device=DEV, generator=g) < 0.3
    _, I = idx.search(Q, k=10, filter=keep)
    assert_exact(X, Q, I, 10, "l2", keep)
    rows = torch.rand(Q.shape[0], X.shape[0], device=DEV, generator=g) < 0.05
    _, I = idx.search(Q, k=10, filter=rows)
    got = torch.sort(scores(X, Q, I, "l2"), 1).values
    s = ((Q.double()[:, None, :] - X.double()[None]) ** 2).sum(-1).masked_fill(~rows, float("inf"))
    ref = torch.topk(s, 10, dim=1, largest=False).values
    assert bool(((got - ref).abs() <= 1e-12 * ref.abs().clamp_min(1e-12)).all()), "bitmap search not exact"
    assert bool(rows.gather(1, I).all()), "a result outside the mask of its query"


def test_data_types():
    """CUDA only: float16, int8, and uint8 vectors, on the device and streamed from host memory."""
    X, Q = data(20_000, 64, 300)
    for dtype in (torch.float16, torch.int8, torch.uint8):
        if dtype.is_floating_point:
            Xt, Qt = X.to(dtype), Q.to(dtype)
        else:
            lo = -100 if dtype == torch.int8 else 0
            Xt = (X * 30 + lo / 2 + 50 * (lo == 0)).round().clamp(lo, lo + 200).to(dtype)
            Qt = (Q * 30 + lo / 2 + 50 * (lo == 0)).round().clamp(lo, lo + 200).to(dtype)
        for source in ("device", "host"):
            for storage in ("device", "host"):
                idx = certex.Index(64, storage=storage)
                idx.add(Xt if source == "device" else Xt.cpu().numpy() if dtype != torch.float16 else Xt.cpu())
                assert idx.dtype == dtype
                _, I = idx.search(Qt, k=10)
                assert_exact(Xt.float(), Qt.float(), I, 10, "l2")


def test_integer_queries_must_fit():
    X = torch.randint(0, 255, (5_000, 16), device=DEV, dtype=torch.uint8)
    idx = certex.Index(16)
    idx.add(X)
    try:
        idx.search(torch.full((4, 16), 0.5, device=DEV), k=5)
    except ValueError:
        return
    raise AssertionError("fractional queries of a uint8 index must be rejected")


def test_large_k():
    """k up to 2048 (the top-k of sparse attention indexers); k above 1024 merges with 64 slots per lane."""
    X, Q = data(30_000, 32, 64)
    idx = certex.Index(32)
    idx.add(X)
    for k in (100, 1024, 1500, 2048):
        _, I = idx.search(Q, k=k)
        assert_exact(X, Q, I, k, "l2")
    ip = certex.Index(32, metric="ip")
    ip.add(X)
    _, I = ip.search(Q, k=2048)
    assert_exact(X, Q, I, 2048, "ip")


def test_concurrent_search_and_add():
    """CUDA only: searches from several threads while another thread adds and removes vectors."""
    import threading

    X, Q = data(40_000, 32, 200)
    idx = certex.Index(32)
    idx.add(X[:20_000])
    errors = []

    def writer():
        with torch.cuda.stream(torch.cuda.Stream()):
            for i in range(20_000, 40_000, 2_000):
                idx.add(X[i : i + 2_000])
            torch.cuda.current_stream().synchronize()

    sizes = range(20_000, 40_001, 2_000)  # every search sees one of these snapshots
    refs = torch.stack([exact_k(X[:n], Q, 10, "l2") for n in sizes])

    def reader():
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                for _ in range(10):
                    _, I = idx.search(Q, k=10)
                    torch.cuda.current_stream().synchronize()
                    got = torch.sort(scores(X, Q, I, "l2"), 1).values
                    hit = ((got[None] - refs).abs() <= 1e-12 * refs.abs().clamp_min(1e-12)).all(2).any(0)
                    assert bool(hit.all()), f"{int((~hit).sum())} queries match no snapshot"
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert idx.ntotal == 40_000
    _, I = idx.search(Q, k=10)
    assert_exact(X, Q, I, 10, "l2")


def test_one_call_search():
    """certex.search: one call, results in the array type of the queries, all k nearest neighbors without queries."""
    X, Q = data(30_000, 64, 200, seed=6)
    for metric in ("l2", "cosine", "ip"):
        D, I = certex.search(X, Q, k=10, metric=metric)
        assert isinstance(I, torch.Tensor) and I.shape == (200, 10)
        assert_exact(X, Q, I, 10, metric)
    D, I = certex.search(X.cpu().numpy(), Q.cpu().numpy(), k=10)
    assert isinstance(D, np.ndarray) and isinstance(I, np.ndarray)
    assert_exact(X, Q, I, 10, "l2")
    D, I = certex.search(X[:5000], k=5)  # all k nearest neighbors: every vector is its own first neighbor
    assert_exact(X[:5000], X[:5000], I, 5, "l2")
    assert bool((torch.as_tensor(I)[:, 0] == torch.arange(5000, device=DEV)).float().mean() > 0.99)
    D, I = certex.search(X, Q[0], k=10)  # one query vector: one row of results
    assert I.shape == (10,)
    assert_exact(X, Q[:1], I[None], 10, "l2")
    try:
        certex.search(X[:5], Q, k=10)
        raise AssertionError("k beyond the collection must be rejected")
    except ValueError as e:
        assert "exceeds the 5 vectors" in str(e)
    D, I = certex.search(X.cpu().numpy(), Q.cpu().numpy(), k=10, storage="host")
    assert_exact(X, Q, I, 10, "l2")
    try:
        import cupy as cp
    except ImportError:
        return
    D, I = certex.search(cp.asarray(X.cpu().numpy()), cp.asarray(Q.cpu().numpy()), k=10)
    assert isinstance(I, cp.ndarray)
    assert_exact(X, Q, torch.from_dlpack(I), 10, "l2")


def test_verify():
    """certex.verify accepts exact results (ties and near-duplicates included) and catches wrong ones."""
    X, Q = data(20_000, 32, 300, seed=7)
    for metric in ("l2", "cosine", "ip"):
        _, I = certex.search(X, Q, k=10, metric=metric)
        assert certex.verify(X, Q, I, metric=metric, sample=None) == 1.0
    _, I = certex.search(X, Q, k=10)
    wrong = I.clone()
    wrong[:100, 9] = (wrong[:100, 9] + 9_973) % X.shape[0]  # replace the 10th neighbor of 100 queries
    assert abs(certex.verify(X, Q, wrong, sample=None) - 2 / 3) < 1e-6
    dup = torch.cat([X[:5000], X[:5000] + 1e-6, X[:5000] * 1000.0])  # near-duplicates and large norms
    _, I = certex.search(dup, dup[:200], k=4)
    assert certex.verify(dup.cpu().numpy(), dup[:200].cpu().numpy(), I.cpu().numpy(), sample=None) == 1.0


if __name__ == "__main__":
    import inspect

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            if "Index" in inspect.signature(fn).parameters:
                for impl, Index in IMPLEMENTATIONS.items():
                    fn(Index)
                    print("passed", name, impl, flush=True)
            else:
                fn()
                print("passed", name, flush=True)


def test_search_stats():
    X = torch.randn(20000, 32, device=DEV)
    Q = X[:100] + 0.01 * torch.randn(100, 32, device=DEV)
    index = certex.Index(32, width=16)
    index.add(X)
    with certex.search_stats() as st:
        index.search(Q, k=10)
    assert st.queries == 100 and st.survivors >= 10 * st.queries and st.verified >= 10 * st.queries


def test_list_order():
    X = torch.randn(20000, 32, device=DEV)
    index = certex.Index(32, width=16, n_lists=16)
    index.add(X)
    order = index.list_order()
    assert torch.equal(torch.sort(order).values, torch.arange(20000, device=DEV))
    D, I = index.search(X[order[:100]], k=1)
    assert torch.equal(I[:, 0], order[:100])


def test_tune():
    X = np.random.default_rng(0).standard_normal((300000, 32)).astype(np.float32)
    r = certex.tune(X, X[:256], widths=(8, 16), strides=(16,), prefixes=(50000, 200000), n_queries=256)
    assert r["width"] in (8, 16) and r["predicted_qps"] > 0 and len(r["candidates"]) == 2
    assert all(c["survivors"] >= 10 for c in r["candidates"]) and r["model"]["a_mac"] >= 0


def test_sharded_index():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30000, 32)).astype(np.float32)
    Q = X[:200] + 0.01 * rng.standard_normal((200, 32)).astype(np.float32)
    sharded = certex.ShardedIndex(32, devices=[0, 0, 0], width=16)
    sharded.add(X)
    D, I = sharded.search(Q, k=10)
    one = certex.Index(32, width=16)
    one.add(X)
    D1, I1 = one.search(Q, k=10)
    assert sharded.ntotal == 30000 and sharded.core.n_shards == 3
    assert np.array_equal(np.sort(I, 1), np.sort(I1, 1)) and np.allclose(D, D1)


def test_filtered_lists_and_local():
    """The probe of the nearest lists and local projections under labels and bitset filters."""
    X, Q = data(40_000, 48, 300, seed=4)
    g = torch.Generator(device=DEV).manual_seed(5)
    lab = torch.randint(0, 20, (X.shape[0],), device=DEV, generator=g)
    ql = torch.randint(0, 20, (Q.shape[0],), device=DEV, generator=g)
    keep = torch.rand(X.shape[0], device=DEV, generator=g) < 0.3
    for local in (False, True):
        idx = certex.Index(48, width=24, n_lists=32, local_projections=local, partition_labels=False)
        idx.add(X, labels=lab)
        _, I = idx.search(Q, k=10, labels=ql, probe_lists=2)
        assert_exact_filtered(X, Q, lab, ql, I, 10)
        _, I = idx.search(Q, k=10, filter=keep, probe_lists=2)
        I = torch.as_tensor(I, device=DEV)
        assert bool(keep[I.clamp_min(0)].all()), "a filtered-out vector was returned"
        assert_exact(X, Q, I, 10, "l2", live=keep)
