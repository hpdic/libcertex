# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""certex: exact k-nearest-neighbor search on NVIDIA GPUs by certified exclusion.

    import certex
    D, I = certex.search(xb, xq, k=10)   # one call: the exact 10 nearest neighbors of every query
    certex.verify(xb, xq, I)             # 1.0: sampled queries match exhaustive search in float64

    index = certex.Index(dim=768, metric="cosine")   # an index for repeated searches and updates
    index.add(xb)                    # numpy, torch, or cupy, any number of times
    D, I = index.search(xq, k=10)    # the exact 10 nearest neighbors of every query

Adapters with the call conventions of other libraries: certex.cuvs (cuVS brute_force style), certex.faiss (Faiss
Index style), and certex.flashinfer (the top-k transforms of FlashInfer sparse attention).
"""


def _preload_nvjitlink():
    """Load the nvJitLink of the nvidia-nvjitlink wheel by path, before torch can load an older copy that
    LD_LIBRARY_PATH names (libcuvs needs the newer one, and a loaded library is shared by its soname)."""
    import ctypes
    import importlib.util
    import os

    try:
        spec = importlib.util.find_spec("nvidia.nvjitlink")
    except ImportError:  # no nvidia wheels (a CPU-only environment, or cuVS from conda)
        return
    for d in [] if spec is None else list(spec.submodule_search_locations or []):
        lib = os.path.join(d, "lib", "libnvJitLink.so.12")
        if os.path.isfile(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return


try:
    _preload_nvjitlink()
except OSError:  # cuVS from a system or conda installation
    pass

from .attention import KeyCache, topk  # noqa: E402
from .index import Index, ShardedIndex, auto_width, search, search_stats, trim_pool, tune, verify  # noqa: E402

__version__ = "0.1.0"
__all__ = [
    "KeyCache",
    "topk",
    "Index",
    "ShardedIndex",
    "auto_width",
    "search",
    "search_stats",
    "trim_pool",
    "tune",
    "verify",
    "__version__",
]
