# Changelog

All notable changes to certex are recorded here. The format follows Keep a Changelog, and versions follow semantic
versioning.

## [0.1.0] - 2026-10-26

The first public release.

### Added
- `certex.search(xb, xq, k)` and `certex.Index`: exact k-nearest-neighbor search on NVIDIA GPUs by certified
  exclusion, with squared Euclidean, cosine, and inner-product metrics; insertions and removals while searches run;
  filtered search by labels, cuVS bitset filters, or per-query masks; full-precision vectors in GPU or pinned host
  memory; saving and loading.
- Partition thresholds and per-partition local projections, multi-GPU sharding with a shared threshold
  (`certex.ShardedIndex`), a calibrated cost model that chooses the screening width (`certex.tune`), and compact
  int8 screening rows.
- `k` up to 2048.
- Sparse attention: `certex.topk` (instead of `torch.topk(q @ keys.mT, k)`), `certex.KeyCache` for keys that grow
  during decoding, and `certex.flashinfer` with the top-k transforms of FlashInfer.
- Adapters with the calls of cuVS brute force (`certex.cuvs`) and Faiss flat indexes (`certex.faiss`).
- `certex.verify(xb, xq, I)`: checks the neighbors of any library against exhaustive search in double precision.
- The CUDA C++ library as a cuVS module (`cuvs::neighbors::certex`) and its C API (`cuvsCertex*`), with Hopper and
  Ampere kernels.
- Examples for every interface: Python search and index, cuVS and Faiss adapters, PyTorch and FlashInfer sparse
  attention, and complete C++ and C programs (`examples/cpp`, `examples/c`, built with the C++ library); documentation.
- Documentation that cites the Zarc paper ([doi:10.5281/zenodo.23674583](https://doi.org/10.5281/zenodo.23674583)),
  which describes the library, beside the CertEx paper of its certified screening.
- `tests/test_dropin.py`: the same calls on cuVS brute force, Faiss flat indexes, and FlashInfer's
  `top_k_page_table_transform` and on the certex adapters give the same neighbors.
