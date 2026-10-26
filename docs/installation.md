# Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install certex
```

certex needs Linux (x86-64), Python 3.11 or newer, an NVIDIA driver for CUDA 12, and an NVIDIA GPU of compute
capability 8.0 or newer. The wheel holds the CUDA C++ library and its Python binding; pip installs the cuVS libraries
(`libcuvs-cu12`) that it builds on. These need the CUDA 12.9 libraries, so PyTorch comes first from the CUDA 12.9
index: the default PyTorch wheels on PyPI are built for CUDA 13, and with them pip falls back to an old PyTorch release.

| GPU generation | Compute capability | Examples | Screening kernels |
| --- | --- | --- | --- |
| Ampere | 8.0, 8.6 | A100, A30, A10, A40, RTX 30 series | cp.async and mma.sync |
| Ada Lovelace | 8.9 | L4, L40, L40S, RTX 40 series | cp.async and mma.sync |
| Hopper | 9.0 | H100, H200, GH200 | TMA and WGMMA |
| Blackwell | 10.0, 12.0 | B200, GB200, RTX PRO 6000, RTX 50 series | cp.async and mma.sync |
| Volta, Turing | 7.0, 7.5 | V100, T4 | not supported: build and search raise `ValueError` |

The kernels are chosen at run time. Check an installation in one line, which searches random vectors and compares a
sample of the results with exhaustive search in double precision:

```bash
python -c "import torch, certex; x = torch.randn(100_000, 64); print(certex.verify(x, x[:100], certex.search(x, x[:100], k=10)[1]))"
```

It prints `1.0`. The exactness tests run from a clone with `python -m pytest -m gpu -q`.

## Building from source

A build needs the CUDA toolkit 12.9 (the toolkit of the release wheels and of the cuVS CUDA 12 builds; 12.6 also
works, and the module compiles with CUDA 13.0), CMake 4.0 or newer (the cuVS CMake package requires it), and Ninja.
pip compiles the CUDA library and the binding in one step; the CUTLASS headers are fetched at configure time, or taken
from a local tree:

```bash
git clone https://github.com/hpdic/libcertex.git && cd libcertex
CERTEX_CUTLASS_DIR=/path/to/cutlass pip install -e ".[dev]"     # CERTEX_CUTLASS_DIR is optional
```

The C++ library alone, with its tests and benchmarks, builds with CMake against an installed cuVS, as described in
`cpp/CMakeLists.txt`.
