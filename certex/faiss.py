# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Faiss-style adapter: an index with the interface of faiss.IndexFlatL2 / faiss.IndexFlatIP, exact on the GPU.

    from certex.faiss import IndexCertExL2, IndexCertExIP
    index = IndexCertExL2(d)          # was: faiss.IndexFlatL2(d)
    index.add(xb)                     # float32 numpy arrays, as in Faiss
    D, I = index.search(xq, k)        # squared distances (L2) or inner products (IP), best first

It does not import faiss, so that it also works without Faiss installed; code written against faiss.Index (d,
ntotal, is_trained, metric_type, add, search, remove_ids, reset) runs unchanged. One difference: ids stay stable after
remove_ids, as with faiss.IndexIDMap, whereas faiss.IndexFlat renumbers the remaining vectors.
"""

import numpy as np

from .index import Index

METRIC_INNER_PRODUCT, METRIC_L2 = 0, 1  # the values of faiss.METRIC_INNER_PRODUCT / METRIC_L2


class _IndexCertEx:
    metric = None

    def __init__(self, d, width="auto", storage="device"):
        self.d, self.is_trained = d, True
        self.metric_type = METRIC_L2 if self.metric == "l2" else METRIC_INNER_PRODUCT
        self._args = dict(width=width, storage=storage)
        self.reset()

    @property
    def ntotal(self):
        return self._index.ntotal

    def reset(self):
        self._index = Index(self.d, metric=self.metric, **self._args)

    def train(self, x):
        pass

    def add(self, x):
        self._index.add(np.ascontiguousarray(x, dtype=np.float32))

    def search(self, x, k):
        D, I = self._index.search(np.ascontiguousarray(x, dtype=np.float32), k=k)
        return D.astype(np.float32), I.astype(np.int64)

    def remove_ids(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self._index.remove(ids)
        return int(ids.size)


class IndexCertExL2(_IndexCertEx):
    metric = "l2"


class IndexCertExIP(_IndexCertEx):
    metric = "ip"
