# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Throughput of certex on BIGANN prefixes against the official ground truth (exactness by fp64 distances).

python bench/bench_bigann.py <millions> [width] [storage] [labels]

With labels = L > 1, every vector and query gets a uniform random label in [0, L), each query searches the vectors of
its label (selectivity 1/L), and exactness is checked against exhaustive search over that label.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certex  # noqa: E402

BIG = os.environ.get("BIGANN", "data/bigann")  # base.1B.u8bin, bigann_query.bvecs, and gnd/ of big-ann-benchmarks


def filtered_truth(X, Q, lab, ql, k=10, rows=1 << 18):
    """Exact k nearest ids of every query among the vectors of its label (integer data: fp64 distances are exact)."""
    out = np.zeros((Q.shape[0], k), dtype=np.int64)
    Qd = Q.double()
    for c in torch.unique(ql).tolist():
        qs = torch.nonzero(ql == c).reshape(-1)
        ids = torch.nonzero(lab == c).reshape(-1)
        best_d = torch.full((qs.numel(), 0), 0.0, dtype=torch.float64, device="cuda")
        best_i = torch.zeros((qs.numel(), 0), dtype=torch.int64, device="cuda")
        for i in range(0, ids.numel(), rows):
            part = ids[i : i + rows]
            Xc = X[part].cuda().double()
            d = (Qd[qs.cuda()] ** 2).sum(1, keepdim=True) + (Xc**2).sum(1)[None] - 2 * Qd[qs.cuda()] @ Xc.T
            best_d = torch.cat([best_d, d], 1)
            best_i = torch.cat([best_i, part.cuda()[None].expand(qs.numel(), -1)], 1)
            v, j = torch.topk(best_d, min(k, best_d.shape[1]), dim=1, largest=False)
            best_d, best_i = v, best_i.gather(1, j)
        out[qs.numpy()] = best_i.cpu().numpy()
    return out


def main():
    n = int(sys.argv[1]) * 1_000_000
    width = sys.argv[2] if len(sys.argv) > 2 else "auto"
    width = int(width) if width != "auto" else width
    storage = sys.argv[3] if len(sys.argv) > 3 else "device"
    Index = certex.Index
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    raw = np.memmap(os.path.join(BIG, "base.1B.u8bin"), dtype=np.uint8, mode="r")
    X = torch.from_numpy(np.ascontiguousarray(raw[8 : 8 + n * 128].reshape(n, 128)))
    q = np.fromfile(os.path.join(BIG, "bigann_query.bvecs"), dtype=np.uint8).reshape(-1, 132)[:, 4:]
    Q = torch.from_numpy(np.ascontiguousarray(q)).float().cuda()
    gt = np.fromfile(os.path.join(BIG, "gnd", f"idx_{n // 1_000_000}M.ivecs"), dtype=np.int32).reshape(-1, 1001)[
        :, 1:11
    ]
    g = torch.Generator().manual_seed(0)
    lab = torch.randint(0, L, (n,), generator=g, dtype=torch.int32) if L > 1 else None
    ql = torch.randint(0, L, (Q.shape[0],), generator=g, dtype=torch.int32) if L > 1 else None
    idx = Index(128, width=width, storage=storage)
    t = time.time()
    idx.add(X, labels=lab)
    torch.cuda.synchronize()
    build = time.time() - t
    idx.search(Q, k=10, labels=ql)
    torch.cuda.synchronize()
    t = time.time()
    D, I = idx.search(Q, k=10, labels=ql)
    torch.cuda.synchronize()
    el = time.time() - t
    if L > 1:
        gt = filtered_truth(X, Q, lab, ql)

    def d2(ids):
        rows = torch.from_numpy(X.numpy()[ids.reshape(-1)]).cuda().double().reshape(ids.shape[0], ids.shape[1], -1)
        return torch.sort(((Q.double()[:, None, :] - rows) ** 2).sum(-1), 1).values

    a = d2(I.cpu().numpy())
    b = d2(gt.astype(np.int64))
    exact = float(((a - b).abs() <= 1e-12 * b.clamp_min(1e-30)).all(1).double().mean())
    print(
        {
            "N": n,
            "width": idx.info()["width"],
            "storage": storage,
            "labels": L,
            "build_s": round(build, 2),
            "qps": round(10000 / el),
            "exact": exact,
        }
    )


if __name__ == "__main__":
    main()
