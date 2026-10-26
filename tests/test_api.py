# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Interface checks that need no GPU: argument validation, metric names, and the screening width rule."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certex  # noqa: E402
from certex import cuvs as certex_cuvs  # noqa: E402
from certex.faiss import METRIC_INNER_PRODUCT, METRIC_L2, IndexCertExIP, IndexCertExL2  # noqa: E402
from certex.index import row_len  # noqa: E402


def test_version():
    assert certex.__version__.count(".") == 2


def test_rejects_unknown_metric_and_storage():
    with pytest.raises(ValueError):
        certex.Index(8, metric="hamming")
    with pytest.raises(ValueError):
        certex.Index(8, storage="disk")


def test_empty_index():
    idx = certex.Index(8)  # no GPU needed until the first addition
    assert idx.ntotal == 0 and idx.info()["width"] is None and idx.dtype is None
    with pytest.raises(ValueError):
        idx.search(torch.zeros(1, 8))


def test_mask_packing():
    if not torch.cuda.is_available():
        pytest.skip("packing runs on the GPU")
    from certex.index import _pack

    m = torch.zeros(70, dtype=torch.bool)
    m[[0, 31, 32, 69]] = True
    w = _pack(m).cpu().numpy().view("uint32")
    assert w.tolist() == [1 | 1 << 31, 1, 1 << 5]


def test_row_length():
    assert [row_len(m) for m in (11, 59, 60, 123, 124, 251)] == [64, 64, 128, 128, 256, 256]


def test_auto_width_follows_the_spectrum():
    g = torch.Generator().manual_seed(0)
    low = torch.randn(4000, 8, generator=g) @ torch.randn(8, 256, generator=g)  # rank 8
    assert certex.auto_width(low, "l2") == 59
    full = torch.randn(4000, 256, generator=g)  # flat spectrum
    assert certex.auto_width(full, "l2") > 200


def test_adapter_conventions():
    assert IndexCertExL2.metric == "l2" and IndexCertExIP.metric == "ip"
    assert (METRIC_INNER_PRODUCT, METRIC_L2) == (0, 1)
    assert certex_cuvs._METRIC["sqeuclidean"] == "l2" and certex_cuvs._METRIC["inner_product"] == "ip"
