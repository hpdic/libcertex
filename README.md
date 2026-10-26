# certex

[![CI](https://github.com/hpdic/libcertex/actions/workflows/ci.yml/badge.svg)](https://github.com/hpdic/libcertex/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Zarc paper](https://img.shields.io/badge/Zarc%20paper-10.5281%2Fzenodo.23674583-6f42c1.svg)](https://doi.org/10.5281/zenodo.23674583)
[![CertEx paper](https://img.shields.io/badge/CertEx%20paper-arXiv%3A2610.21325-b31b1b.svg)](https://arxiv.org/abs/2610.21325)

Exact k-nearest-neighbor search on NVIDIA GPUs by certified exclusion. certex returns the exact k nearest
neighbors of every query, at the speed of a half-precision screen: a lower bound evaluated on the tensor cores, with
its rounding error folded into the same product, proves that almost every stored vector cannot be a neighbor, and
only the few survivors are verified with exact double-precision distances.

Every result equals exhaustive search in double precision. No setting trades accuracy for speed: there is no graph
and no recall setting, options such as k-means lists change only the speed, and additions and removals take effect
at once, also while queries run. On one
H100, certex answers 176,700 exact 10-NN queries per second over BIGANN-10M and searches 500 million BIGANN vectors
exactly within 71 GB of device memory. It supports Euclidean, cosine, and
inner-product metrics, filtered search by label, and host-memory storage for collections larger than GPU memory, and
it is a drop-in for cuVS (C++, C, and Python), Faiss, `torch.topk`, and the sparse-attention top-k of FlashInfer.

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129   # PyTorch for CUDA 12.9, as cuVS needs
pip install certex            # Linux, Python >= 3.11, CUDA 12, an NVIDIA GPU of compute capability >= 8.0
```

## Quickstart

One call returns the exact nearest neighbors:

```python
import certex

D, I = certex.search(xb, xq, k=10)  # numpy, torch, or cupy; exact neighbors, best first
D, I = certex.search(xb, k=10)  # all k nearest neighbors of every vector of xb
certex.verify(xb, xq, I)  # 1.0: sampled queries match exhaustive search in float64
```

For repeated searches and updates, keep an index:

```python
index = certex.Index(dim=768, metric="cosine")  # "l2", "cosine", or "ip"
index.add(xb)  # numpy, torch, or cupy; call again to insert
D, I = index.search(xq, k=10)  # exact neighbors, best first

tagged = certex.Index(dim=768)  # filtered search: one integer label per vector and query
tagged.add(xb, labels=lb)
D, I = tagged.search(xq, k=10, labels=lq)  # exact k nearest vectors of each query's label

index.remove(ids)  # removed vectors never come back
index.save("vectors.certex")  # certex.Index.load restores it without a rebuild
```

## Interfaces

certex keeps the calls of the libraries that it replaces, so that moving code to exact search changes little more
than an import. `tests/test_dropin.py` checks the Python interfaces against the original libraries.

| Interface | Drop-in for | Example |
|---|---|---|
| `certex.cuvs` (Python) | `cuvs.neighbors.brute_force` | [from_cuvs.py](examples/from_cuvs.py) |
| `certex.faiss` (Python) | `faiss.IndexFlatL2`, `faiss.IndexFlatIP` | [from_faiss.py](examples/from_faiss.py) |
| `certex.topk`, `certex.KeyCache` | `torch.topk(q @ keys.mT, k)` | [sparse_attention.py](examples/sparse_attention.py) |
| `certex.flashinfer` | FlashInfer `top_k_page_table_transform`, `top_k_ragged_transform`, `top_k` | [flashinfer_sparse_attention.py](examples/flashinfer_sparse_attention.py) |
| `cuvs::neighbors::certex` (C++) | the cuVS index modules (`build`, `extend`, `search`) | [cpp/exact_search.cu](examples/cpp/exact_search.cu) |
| `cuvsCertex*` (C) | the cuVS C API | [c/exact_search.c](examples/c/exact_search.c) |

cuVS and Faiss from Python:

```python
from certex import cuvs as certex_cuvs  # instead of cuvs.neighbors.brute_force

index = certex_cuvs.build(dataset, metric="sqeuclidean")
distances, neighbors = certex_cuvs.search(index, queries, k)

from certex.faiss import IndexCertExL2  # instead of faiss.IndexFlatL2

index = IndexCertExL2(d)
index.add(xb)
D, I = index.search(xq, k)
```

Sparse attention in PyTorch and FlashInfer, with the exact top-k keys of every query:

```python
scores, ids = certex.topk(q, keys, k=64)  # instead of torch.topk(q @ keys.mT, 64)

cache = certex.KeyCache(n_kv_heads=8, head_dim=128, k=2048)  # keys that grow during decoding
cache.append(keys)  # (n_kv_heads, n, head_dim)
scores, ids = cache.topk(q, k=2048)

from certex import flashinfer as certex_fi  # instead of the top-k transforms of FlashInfer

sel = certex_fi.ExactTopK(num_kv_heads=8, head_dim=128)
sel.append(0, k_new)  # keys of sequence slot 0, (n, num_kv_heads, head_dim)
out = sel.page_table_transform(q, page_table, k=2048, page_size=64)  # int32 physical positions
```

C++ and C, with the conventions of cuVS:

```cpp
#include <cuvs/neighbors/certex.hpp>

auto index = cuvs::neighbors::certex::build(res, cuvs::neighbors::certex::index_params{}, dataset);
cuvs::neighbors::certex::search(res, cuvs::neighbors::certex::search_params{}, index, queries, neighbors, distances);
```

```c
#include <cuvs/neighbors/certex.h>

cuvsCertexBuild(res, index_params, &dataset, index);
cuvsCertexSearch(res, search_params, index, &queries, &neighbors, &distances, filter);
```

Both examples build with the C++ library (`cmake -S cpp -B build`, see [docs/cpp.md](docs/cpp.md)) as
`build/certex_example_cpp` and `build/certex_example_c`; [docs/cpp.md](docs/cpp.md) and [docs/c.md](docs/c.md) describe the
two APIs.

## Documentation

The [documentation](docs/index.md) covers installation, the user guide (metrics and storage, insertions and
removals, filtered search, persistence, adapters), how certified exclusion works, benchmarks, and the API reference.
Build it locally with `pip install -e ".[docs]" && mkdocs serve`.

Runnable examples are in [examples/](examples/): one-call search, exact ground truth for evaluating approximate
indexes, an exact k-NN graph with near-duplicate removal, filtered search, and every interface of the table above.
The [Zarc paper](https://doi.org/10.5281/zenodo.23674583) describes the design of the library (search pipeline, kernels, memory tiers, updates, metrics, and
filters), proves that each mechanism keeps every answer exact, and reports its evaluation.

## Benchmarks

| Dataset | Width | Vectors in | Device memory | Queries per second | Exact queries |
|---|---|---|---|---|---|
| BIGANN-10M | 123 | host | 5.1 GB | 176,700 | 100% |
| BIGANN-100M | 123 | host | 29.0 GB | 18,700 | 100% |
| BIGANN-500M | 59 | host | 71 GB | 5,233 | 100% |
| BIGANN-10M, filtered at 10%, 1%, 0.1% | 123 | device | | 324,400, 392,000, 402,700 | 100% |

One H100 NVL, batches of 2,048 queries, k = 10, anchor stride 16 (32 at 500M). These are the measurements of the
[Zarc paper](https://doi.org/10.5281/zenodo.23674583); [docs/benchmarks.md](docs/benchmarks.md) lists more configurations.

certex is a CUDA C++ library laid out as a cuVS module (`cuvs::neighbors::certex`, in [cpp/](cpp/)), with a Python
binding.

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) (commits are signed off under the DCO) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues as described in [SECURITY.md](SECURITY.md).

## Citation

The library is described in the Zarc paper, and its certified screening comes from the CertEx paper:

```bibtex
@misc{zhao2026zarc,
  author       = {Dongfang Zhao},
  title        = {{Zarc: Exact Nearest Neighbors over a Billion Vectors in Seconds}},
  howpublished = {Zenodo preprint},
  year         = {2026},
  doi          = {10.5281/zenodo.23674583},
  url          = {https://doi.org/10.5281/zenodo.23674583}
}

@article{zhao2026certex,
  author  = {Dongfang Zhao},
  title   = {{CertEx: Exact Visual Retrieval at Scale by Certified Exclusion}},
  journal = {arXiv preprint arXiv:2610.21325},
  year    = {2026}
}
```

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
