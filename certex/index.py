# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""certex.Index: exact k-nearest-neighbor search on NVIDIA GPUs by certified exclusion.

The index is the CUDA C++ implementation cuvs::neighbors::certex, reached through the extension certex._core; this
module converts arrays, keeps the ids, and runs every call on the current torch CUDA stream.
"""

import threading

import numpy as np
import torch

METRICS = ("l2", "cosine", "ip")
_DTYPES = {torch.float32: "float32", torch.float16: "float16", torch.int8: "int8", torch.uint8: "uint8"}
_TORCH = {v: k for k, v in _DTYPES.items()}
_PREFIX = {b"f4": "float32", b"f2": "float16", b"i1": "int8", b"u1": "uint8"}
_core = None


def _lib():
    """The extension module, loaded on first use (after the cuVS libraries of the RAPIDS wheels)."""
    global _core
    if _core is None:
        try:
            import libcuvs

            libcuvs.load_library()
        except ImportError:  # cuVS from a system or conda installation
            pass
        from . import _core as core

        _core = core
    return _core


def _stream():
    return torch.cuda.current_stream().cuda_stream


def row_len(m):
    """Halves per screening row of width m: m + 1 lifted and 4 certificate coordinates, padded to 64 or 128n."""
    L = m + 5
    return 64 if L <= 64 else 128 * ((L + 127) // 128)


def _to_torch(x):
    """A torch view of numpy, torch, or any DLPack / CUDA-array-interface array (cupy, JAX, ...), and the kind of
    the input, so that results come back in the caller's array type (see _back)."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)), "numpy"
    if isinstance(x, torch.Tensor):
        return x, "torch"
    if hasattr(x, "__dlpack__"):
        return torch.from_dlpack(x), type(x).__module__.split(".")[0]
    if hasattr(x, "__cuda_array_interface__"):
        return torch.as_tensor(x, device="cuda"), type(x).__module__.split(".")[0]
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x))), "numpy"


def _require(ok, message):
    """Reject a call with a ValueError (asserts vanish under python -O)."""
    if not ok:
        raise ValueError(message)


def _back(t, kind):
    """t (a torch tensor) in the array type of an input of this kind: numpy on the host, torch as is, and the arrays of
    other DLPack libraries (cupy, ...) on the GPU without a copy."""
    if kind == "numpy":
        return t.cpu().numpy()
    if kind == "torch":
        return t
    try:
        import importlib

        return importlib.import_module(kind).from_dlpack(t)
    except (ImportError, AttributeError, TypeError):
        return t


def _cast(t, dtype):
    """t in the data type of the index; values that an integer type cannot hold exactly are an error."""
    if t.dtype == dtype:
        return t
    if not dtype.is_floating_point:
        lim = torch.iinfo(dtype)
        f = t.double()
        if not bool(((f == f.round()) & (f >= lim.min) & (f <= lim.max)).all()):
            raise ValueError(f"the index holds {dtype} vectors: values must be integers in [{lim.min}, {lim.max}]")
    return t.to(dtype)


def _pack(mask):
    """A boolean mask, flattened row-major, as cuVS bitset words (bit j of word i is entry 32 i + j)."""
    b = mask.reshape(-1).to(device="cuda", dtype=torch.int64)
    b = torch.nn.functional.pad(b, (0, (-b.numel()) % 32)).view(-1, 32)
    w = (b << torch.arange(32, device=b.device)).sum(1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32).contiguous()


def auto_width(X, metric, target=0.95, sample=200_000):
    """Smallest tile-aligned width (a row of 64, 128, or a multiple of 128 half-precision coordinates) whose principal
    directions capture the target fraction of the variance around the center of the metric. The index applies the
    same rule on the GPU when width="auto"; this function only reports it."""
    d = X.shape[1]
    S = X[torch.randperm(X.shape[0], device=X.device)[:sample]].float()
    if metric != "ip":
        S = S - S.mean(0)
    ev = torch.linalg.eigvalsh(S.double().T @ S.double()).flip(0).clamp_min(0)
    frac = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-300)
    for L in [64, 128] + list(range(256, 4097, 128)):
        m = L - 5
        if m >= d or float(frac[m - 1]) >= target:
            return min(m, d)
    return d


def tune(
    dataset,
    queries,
    metric="l2",
    k=10,
    storage="device",
    n_lists=0,
    widths=(),
    strides=(16, 32),
    prefixes=(1_000_000, 4_000_000),
    n_queries=1024,
    batch=2048,
    memory_budget=0,
):
    """Choose the width and the anchor stride by the cost model on the current GPU: survivors and verified rows per
    query are measured on two prefixes of the dataset and extrapolated by a power law to its full size; returns the
    choice, the calibrated constants, and every candidate with its prediction. dataset and queries: host arrays of
    one data type (a memory map works; only the prefixes are read)."""
    X, _ = _to_torch(dataset)
    Q, _ = _to_torch(queries)
    Q = _cast(Q.cpu(), X.dtype).contiguous()
    return _lib().tune(
        X,
        Q,
        metric,
        k,
        storage == "host",
        n_lists,
        list(widths),
        list(strides),
        prefixes[0],
        prefixes[1],
        n_queries,
        batch,
        float(memory_budget),
        _stream(),
    )


def trim_pool():
    """Return the unused memory of the stream-ordered allocator of the current device to the driver (for memory
    measurements; searches refill the pool as they need)."""
    _lib().trim_pool()


class search_stats:
    """Counters of the searches of the calling thread inside the block: queries, certified survivors (pairs that the
    screening pass keeps), and verified (exact distances computed by ordered verification).

        with certex.search_stats() as st:
            index.search(q)
        st.survivors / st.queries
    """

    def __enter__(self):
        _lib().stats_begin()
        self.queries = self.survivors = self.verified = 0
        return self

    def __exit__(self, *exc):
        torch.cuda.current_stream().synchronize()
        self.queries, self.survivors, self.verified = _lib().stats_end()
        return False


class Index:
    """Exact k-NN index. Every search returns the exact k nearest stored vectors of every query.

    metric: "l2" (squared Euclidean distances, ascending), "ip" (inner products, descending), or "cosine" (cosine
    similarities, descending; computed from the vectors as given, which need not be normalized). width: the number
    m of principal directions kept in the half-precision screening rows ("auto" or an int); any width is exact, it
    only changes the speed. storage:
    "device" keeps the full-precision vectors in GPU memory, "host" keeps them in pinned host memory (more vectors per
    GPU, a few reads over PCIe per query). capacity: preallocated rows for later additions (grown on demand).

    The vectors keep the data type of the first addition (float32, float16, int8, or uint8; other types become
    float32, and cosine needs a float type); later additions and queries are converted to it. Arrays in host memory
    (numpy) are streamed to the GPU in chunks, so that a dataset may exceed GPU memory with storage="host".

    partition_labels: for a labeled index, keep a copy of the screening rows sorted by label, so that a search with
    labels screens only the rows of each query's label (2 to 5 times faster at 10% to 0.1% selectivity, for another
    2 (m + 5) + 4 bytes of GPU memory per vector). The copy is refreshed when additions since the last refresh exceed
    an eighth of the index; the vectors in between are searched with the label filter. Either way the results are
    exact.

    n_lists: k-means lists of the vectors (0: none). The vectors are then stored in list order, and every search first
    screens the lists nearest to each query (probe_lists in search), whose exact distances tighten its threshold; the
    ids stay the order of the additions. local_projections: every list gets its own center and principal directions,
    which tighten the bounds of its rows (fewer exact distances per query); filtered search with local projections
    applies the filter when it verifies the survivors, and a label partition (partition_labels) needs the global one.
    int8_rows: screening rows of int8 codes with one scale per row (about half the device
    memory of half precision at the same width, somewhat more survivors); not combined with lists.
    """

    def __init__(
        self,
        dim,
        metric="l2",
        width="auto",
        storage="device",
        capacity=None,
        anchor_stride=16,
        partition_labels=True,
        n_lists=0,
        local_projections=False,
        int8_rows=False,
    ):
        _require(metric in METRICS, f"metric must be one of {METRICS}, not {metric!r}")
        _require(storage in ("device", "host"), f'storage must be "device" or "host", not {storage!r}')
        self.d, self.metric, self.width, self.storage = dim, metric, width, storage
        self.capacity, self.anchor_stride = capacity, anchor_stride
        self.partition_labels = partition_labels
        self.n_lists, self.local_projections, self.int8_rows = n_lists, local_projections, int8_rows
        self.core = None
        self._writer = threading.Lock()  # additions get consecutive ids

    @property
    def ntotal(self):
        return 0 if self.core is None else int(self.core.size - self.core.n_removed)

    @property
    def dtype(self):
        return None if self.core is None else _TORCH[self.core.dtype]

    def _vectors(self, x, device):
        t, kind = _to_torch(x)
        if t.ndim == 1:  # a single vector
            t = t.reshape(1, -1)
        _require(
            t.ndim == 2 and t.shape[1] == self.d,
            f"expected vectors of dimension {self.d} (a matrix (n, {self.d}) or one vector), "
            f"got shape {tuple(t.shape)}",
        )
        if device or t.is_cuda:
            t = t.cuda()
        if self.core is not None:
            t = _cast(t, self.dtype)
        elif t.dtype not in _DTYPES or (self.metric == "cosine" and not t.dtype.is_floating_point):
            t = t.float()
        return t.contiguous(), kind

    def add(self, x, labels=None):
        """Add vectors, optionally with one non-negative integer label each (for filtered search); returns their ids
        (consecutive integers, never reused after removal). Once an index has labels, every addition needs them."""
        X, kind = self._vectors(x, device=False)
        if labels is not None:
            labels = _to_torch(labels)[0].reshape(-1).to(device="cuda", dtype=torch.int32).contiguous()
            _require(labels.shape[0] == X.shape[0] and int(labels.min()) >= 0, "one non-negative label per vector")
        lib = _lib()
        with self._writer:
            if self.core is not None and self.core.has_labels:
                _require(labels is not None, "this index is labeled: add labels with every addition")
            if self.core is not None and labels is not None and not self.core.has_labels:
                raise ValueError("labels must be given from the first addition on")
            if self.core is None:
                self.core = lib.build(
                    X,
                    self.metric,
                    0 if self.width == "auto" else int(self.width),
                    0.95,
                    self.anchor_stride,
                    1_000_000,
                    self.storage == "host",
                    self.n_lists,
                    self.local_projections,
                    self.int8_rows,
                    _stream(),
                )
                n0 = 0
                if self.capacity is not None and self.capacity > X.shape[0]:
                    self.core.reserve(int(self.capacity), _stream())
            else:
                n0 = self.core.size
                self.core.extend(X, _stream())
            if labels is not None:
                self.core.set_labels(labels, n0, _stream())
                self._refresh_partition()
        ids = torch.arange(n0, n0 + X.shape[0])
        return _back(ids, kind)

    def _refresh_partition(self):
        c = self.core
        # a label partition needs the global projection: with local projections, the filter applies in verification
        stale = 8 * (c.size - c.n_partitioned) > c.size
        if self.partition_labels and not c.local_projections and c.has_labels and stale:
            c.partition_by_labels(_stream())

    def remove(self, ids):
        """Remove vectors by id; they never appear in later results."""
        _require(self.core is not None, "add vectors first")
        t = _to_torch(ids)[0].reshape(-1).to(device="cuda", dtype=torch.int64).contiguous()
        with self._writer:
            self.core.remove(t, _stream())

    def search(self, q, k=10, batch=2048, labels=None, filter=None, anchors=128, probe_lists=2):
        """Exact k nearest neighbors of every query: (distances or similarities, ids), best first.

        labels (one per query): the neighbors of a query are the exact k nearest vectors of its label. filter: a
        boolean mask over the ids (True keeps a vector for every query) or one row of such a mask per query; the
        neighbors are the exact k nearest kept vectors. With either, a query with fewer than k candidates gets id -1
        in the missing entries."""
        _require(self.core is not None, "the index is empty: add vectors first")
        single = _to_torch(q)[0].ndim == 1
        Q, kind = self._vectors(q, device=True)
        nq = Q.shape[0]
        _require(1 <= k <= 2048, f"k must be between 1 and 2048, not {k}")
        _require(k <= self.core.size, f"k = {k} exceeds the {int(self.core.size)} vectors added to the index")
        I = torch.empty((nq, k), dtype=torch.int64, device="cuda")
        D = torch.empty((nq, k), dtype=torch.float32, device="cuda")
        ql, words, rows, cols = None, None, 0, 0
        if labels is not None:
            _require(self.core.has_labels, "the index has no labels")
            ql = _to_torch(labels)[0].reshape(-1).to(device="cuda", dtype=torch.int32).contiguous()
            _require(ql.shape[0] == nq, "one label per query")
        if filter is not None:
            m = _to_torch(filter)[0].to(torch.bool)
            _require(m.shape[-1] == self.core.size, "the filter needs one entry per id")
            _require(m.ndim == 1 or (m.ndim == 2 and m.shape[0] == nq), "a mask over the ids, or one row per query")
            words, rows, cols = _pack(m), (0 if m.ndim == 1 else nq), m.shape[-1]
        if nq > 0:
            self.core.search(Q, I, D, anchors, batch, probe_lists, words, rows, cols, ql, _stream())
        if self.metric == "cosine":
            D = 1 - D  # cosine similarity from the cosine distance; -inf where an entry is missing
        if single:  # one query vector in, one row of results out
            D, I = D[0], I[0]
        return _back(D, kind), _back(I, kind)

    def list_order(self):
        """The ids in the order of the stored rows (list by list) of an index with n_lists; queries taken in this
        order share their nearest lists within a batch."""
        _require(self.core is not None and self.n_lists, "an index with lists")
        out = torch.empty(self.core.size, dtype=torch.int32, device="cuda")
        self.core.list_order(out, _stream())
        return out.long()

    def save(self, path):
        """Write the index to a file; load() restores it without rebuilding."""
        _require(self.core is not None, "add vectors first")
        self.core.save(str(path), _stream())

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            prefix = f.read(4)
        _require(prefix[1:3] in _PREFIX, "not a certex index file")
        core = _lib().load(str(path), _PREFIX[prefix[1:3]], _stream())
        idx = cls(
            core.dim,
            n_lists=core.n_lists,
            local_projections=core.local_projections,
            int8_rows=core.int8_rows,
            metric=core.metric,
            width=core.width,
            storage="host" if core.host_vectors else "device",
            anchor_stride=core.anchor_stride,
        )
        idx.core = core
        with idx._writer:
            idx._refresh_partition()  # the partition is not saved
        return idx

    def info(self):
        c = self.core
        return {
            "metric": self.metric,
            "width": None if c is None else c.width,
            "row_bytes": None if c is None else 2 * c.row_length,
            "screening_bytes_per_vector": None if c is None else c.screening_bytes_per_vector,
            "dtype": None if c is None else c.dtype,
            "storage": self.storage,
            "ntotal": self.ntotal,
        }


class ShardedIndex:
    """Exact k-NN over several GPUs: the vectors are split into contiguous shards, one index per device, built in
    parallel. A search runs in two rounds (the smallest anchor threshold of all shards, then every shard under it)
    and merges the shard answers by exact distance; the results equal those of one index over all vectors.

    devices: the CUDA device of every shard (a device may repeat). The vectors and the queries are host arrays;
    the ids are the row numbers of the one addition."""

    def __init__(
        self,
        dim,
        devices,
        metric="l2",
        width="auto",
        storage="device",
        anchor_stride=16,
        n_lists=0,
        local_projections=False,
    ):
        _require(metric in METRICS and storage in ("device", "host"), "unknown metric or storage")
        self.d, self.devices, self.metric, self.width, self.storage = dim, list(devices), metric, width, storage
        self.anchor_stride, self.n_lists, self.local_projections = anchor_stride, n_lists, local_projections
        self.core = None

    def add(self, x):
        _require(self.core is None, "a sharded index is built by one addition")
        t, _ = _to_torch(x)
        _require(t.ndim == 2 and t.shape[1] == self.d and not t.is_cuda, "a host matrix (n, dim)")
        if t.dtype not in _DTYPES or (self.metric == "cosine" and not t.dtype.is_floating_point):
            t = t.float()
        self.core = _lib().build_sharded(
            t.contiguous(),
            self.devices,
            self.metric,
            0 if self.width == "auto" else int(self.width),
            self.anchor_stride,
            self.storage == "host",
            self.n_lists,
            self.local_projections,
        )
        self._dtype = t.dtype
        return np.arange(t.shape[0])

    @property
    def ntotal(self):
        return 0 if self.core is None else int(self.core.size)

    def search(self, q, k=10, batch=2048, anchors=128, probe_lists=2):
        """Exact k nearest neighbors of every query: (distances or similarities, ids) as numpy arrays."""
        _require(self.core is not None, "add vectors first")
        Q = _cast(_to_torch(q)[0].cpu(), self._dtype).contiguous()
        I = np.empty((Q.shape[0], k), dtype=np.int64)
        D = np.empty((Q.shape[0], k), dtype=np.float32)
        if Q.shape[0] > 0:
            self.core.search(Q, I, D, anchors, batch, probe_lists)
        if self.metric == "cosine":
            D = 1 - D
        return D, I


def search(xb, xq=None, k=10, metric="l2", **index_options):
    """The exact k nearest neighbors in one call: (distances or similarities, ids), best first.

        D, I = certex.search(xb, xq, k=10)

    xb holds the stored vectors and xq the queries (numpy, torch, cupy, or any DLPack array; results come back in the
    type of xq). Without xq, every vector of xb is a query (all k nearest neighbors; a vector's first neighbor is
    itself). metric: "l2", "cosine", or "ip". A host array larger than half of the free GPU memory keeps its
    full-precision vectors in host memory (storage="host"). index_options go to certex.Index; for repeated searches
    over the same vectors, keep a certex.Index instead.
    """
    X, _ = _to_torch(xb)
    _require(X.ndim == 2, "xb must be a matrix (n, dim)")
    if not X.is_cuda:
        free, _ = torch.cuda.mem_get_info()
        size = X.element_size() * X.numel()
        if "storage" not in index_options:
            index_options["storage"] = "host" if size > free // 2 else "device"
        if index_options["storage"] == "device" and 3 * size <= free:
            X = X.cuda()  # one copy to the GPU (while it and the index's copy fit) beats chunked reads of host memory
    index = Index(dim=X.shape[1], metric=metric, **index_options)
    index.add(X)
    D, I = index.search(xb if xq is None else xq, k=k)
    del index
    return D, I


def verify(xb, xq, I, metric="l2", sample=100, seed=0):
    """Check neighbor ids against exhaustive search in double precision; returns the fraction of checked queries whose
    neighbors are exact.

        D, I = certex.search(xb, xq, k=10)
        certex.verify(xb, xq, I)   # 1.0: every checked query has exactly the k best scores

    A random sample of queries (sample of them; None checks all) is searched exhaustively in float64 over all of xb;
    a query is exact when the float64 scores of its returned neighbors, sorted, equal the k best scores (so that ties
    count as exact, whichever tied vector was returned). Works for the results of any library: ids of -1 count as
    missing. xb may be in host or GPU memory and larger than the GPU memory.
    """
    _require(metric in METRICS, f"metric must be one of {METRICS}, not {metric!r}")
    X, _ = _to_torch(xb)
    Q, _ = _to_torch(xq)
    ids, _ = _to_torch(I)
    if Q.ndim == 1:
        Q, ids = Q.reshape(1, -1), ids.reshape(1, -1)
    _require(X.ndim == 2 and Q.ndim == 2 and Q.shape[1] == X.shape[1], "xb and xq need the same dimension")
    _require(ids.shape[0] == Q.shape[0], "one row of ids per query")
    nq, k = ids.shape
    pick = (
        torch.arange(nq)
        if sample is None or sample >= nq
        else torch.randperm(nq, generator=torch.Generator().manual_seed(seed))[:sample]
    )
    Qs = Q[pick.to(Q.device)].cuda().double()
    got_ids = ids[pick.to(ids.device)].cuda().long()
    if metric == "cosine":
        Qs = Qs / Qs.norm(dim=1, keepdim=True)

    def coarse(q, rows):  # smaller is better; only ranks candidates
        if metric == "l2":
            return (q * q).sum(1, keepdim=True) + (rows * rows).sum(1)[None] - 2.0 * q @ rows.T
        if metric == "cosine":
            rows = rows / rows.norm(dim=1, keepdim=True)
        return -(q @ rows.T)

    def exact(cand):  # float64 scores of candidate ids (one row per query) by one formula, smaller is better
        rows = X[cand.clamp_min(0).reshape(-1).to(X.device)].cuda().double().reshape(*cand.shape, -1)
        if metric == "l2":
            s = ((Qs[:, None, :] - rows) ** 2).sum(-1)
        else:
            if metric == "cosine":
                rows = rows / rows.norm(dim=-1, keepdim=True)
            s = -(rows * Qs[:, None, :]).sum(-1)
        return torch.where(cand >= 0, s, torch.full_like(s, float("inf")))

    # candidates by the expanded form, with a margin for its rounding; then the exact formula ranks them
    kc = min(k + 32, X.shape[0])
    cv = torch.full((Qs.shape[0], kc), float("inf"), dtype=torch.float64, device="cuda")
    ci = torch.full((Qs.shape[0], kc), -1, dtype=torch.int64, device="cuda")
    chunk = max(1, (1 << 28) // (8 * X.shape[1]))
    for i in range(0, X.shape[0], chunk):
        s = coarse(Qs, X[i : i + chunk].cuda().double())
        idx = torch.arange(i, i + s.shape[1], device="cuda").expand(s.shape[0], -1)
        top = torch.topk(torch.cat([cv, s], 1), kc, dim=1, largest=False)
        cv, ci = top.values, torch.gather(torch.cat([ci, idx], 1), 1, top.indices)
    best = exact(ci).sort(1).values[:, :k]
    got = exact(got_ids).sort(1).values
    tol = 1e-9 * best.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return float(((got - best).abs() <= tol).all(1).float().mean())
