# Choosing the width

Every width and anchor stride returns the same exact answer, so that the choice only changes the speed and the device
memory. `certex.tune` chooses them with a cost model calibrated on the current GPU:

```python
r = certex.tune(xb, xq, k=10, storage="host", widths=(59, 123), strides=(16, 32))
r["width"], r["anchor_stride"], r["predicted_qps"]
index = certex.Index(dim=xb.shape[1], width=r["width"], anchor_stride=r["anchor_stride"], storage="host")
```

The tuner times the screening kernels on random rows, searches two prefixes of the dataset (1M and 4M vectors by
default) for every candidate, extrapolates the survivors and the verified rows per query to the full size by a power
law, and prices a batch with the calibrated constants. It returns the candidate of highest predicted throughput that
fits the device memory, together with every candidate (`r["candidates"]`) and the constants (`r["model"]`). `xb` may
be a memory map: only the prefixes are read. On BIGANN with host vectors, the tuner picks the measured fastest
configuration from 100M to 500M vectors (width 123 at 100M, width 59 at 200M and 500M) in about 5 seconds.

`certex.search_stats()` counts the survivors and the verified rows of the searches inside a `with` block, which shows
how tight the threshold of a configuration is:

```python
with certex.search_stats() as st:
    index.search(xq, k=10)
print(st.survivors / st.queries, st.verified / st.queries)
```
