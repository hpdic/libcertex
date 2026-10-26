# certex

certex returns the exact k nearest neighbors of every query on an NVIDIA GPU, at the speed of a half-precision
screen. Instead of computing every distance in full precision, it proves for almost every stored vector that the
vector cannot be a neighbor, and computes exact distances only for the few vectors that it cannot exclude.

The screening bound carries a certificate for its own rounding error, so that no true neighbor is ever excluded, and
every result equals exhaustive search in double precision. No setting trades accuracy for speed: there is no graph and
no recall setting, options such as k-means lists change only the speed, and insertions and removals take effect
immediately, also while queries run. On one H100, certex answers 176,700 exact 10-NN queries per second over BIGANN-10M
and searches 500 million BIGANN vectors exactly within 71 GB of device memory. It is a drop-in for cuVS (C++, C, and
Python), Faiss flat indexes, `torch.topk`, and the sparse-attention top-k of FlashInfer, so that existing code changes
in one line.

```python
import certex

D, I = certex.search(xb, xq, k=10)  # the exact 10 nearest neighbors of every query
```

certex is the library of *Zarc: Exact Nearest Neighbors over a Billion Vectors in Seconds*
([doi:10.5281/zenodo.23674583](https://doi.org/10.5281/zenodo.23674583)), which builds on the certified screening of *CertEx: Exact Visual Retrieval at
Scale by Certified Exclusion* ([arXiv:2610.21325](https://arxiv.org/abs/2610.21325)).
