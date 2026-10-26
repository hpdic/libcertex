# Quickstart

```python
import numpy as np
import certex

xb = np.random.rand(1_000_000, 128).astype(np.float32)  # stored vectors
xq = np.random.rand(10_000, 128).astype(np.float32)  # queries

D, I = certex.search(xb, xq, k=10)  # exact neighbors, nearest first
```

`certex.search` builds a temporary index, searches it, and frees it. Without queries, `certex.search(xb, k=10)`
returns the k nearest neighbors of every vector of `xb` (each vector is its own first neighbor). A NumPy array larger
than about half of the free GPU memory keeps its full-precision vectors in host memory.

For repeated searches over the same vectors, and for insertions and removals, keep an index:

```python
index = certex.Index(dim=128)  # squared Euclidean distances
index.add(xb)
D, I = index.search(xq, k=10)  # exact neighbors, nearest first
print(I[0], D[0])
```

Inputs may be NumPy arrays, PyTorch tensors on any device, CuPy arrays, or any other DLPack array, in float32,
float16, int8, or uint8. The results have the type of the queries: NumPy in, NumPy out; a CUDA tensor in, CUDA
tensors out; CuPy in, CuPy out.

Searches are fastest in batches: every batch screens the whole collection once, so that a batch of 2,048 queries
costs little more than a single query. See [Benchmarks](benchmarks.md) for the effect of the batch size.
