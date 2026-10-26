# Moving from Faiss or cuVS

## Faiss

```python
from certex.faiss import IndexCertExL2  # instead of faiss.IndexFlatL2

index = IndexCertExL2(d)
index.add(xb)
D, I = index.search(xq, k)
```

`IndexCertExL2` and `IndexCertExIP` have the attributes and methods of Faiss flat indexes (`d`, `ntotal`,
`is_trained`, `metric_type`, `add`, `search`, `remove_ids`, `reset`). One difference: ids stay stable after
`remove_ids`, as with `faiss.IndexIDMap`, whereas `faiss.IndexFlat` renumbers the remaining vectors.

## cuVS

```python
from certex import cuvs as certex_cuvs  # instead of cuvs.neighbors.brute_force

index = certex_cuvs.build(dataset, metric="sqeuclidean")
distances, neighbors = certex_cuvs.search(index, queries, k)
```

Inputs are any CUDA array (PyTorch, CuPy, or another `__dlpack__` array); outputs are CUDA tensors that convert to
CuPy without a copy.
