# Lists and local projections

Every search screens all stored vectors, so that its threshold decides how many survivors need an exact distance. On
clustered data, k-means lists tighten that threshold:

```python
index = certex.Index(dim=768, n_lists=1024)  # k-means lists, vectors stored list by list
D, I = index.search(xq, k=10, probe_lists=2)  # the 2 nearest lists of each query are screened first
```

The exact distances to the survivors of the nearest lists give k distinct vectors, whose k-th distance lowers the
threshold of the full pass; the answer stays exact, because any k distinct vectors bound the k-th distance from above.
On BIGANN-10M with 1,024 lists, the survivors per query fall from 2,635 to 527. `Index.list_order()` returns the ids
in stored order, so that a self-join can take its queries list by list and share the nearest lists within a batch.

Local projections give every list its own center and principal directions:

```python
index = certex.Index(dim=768, width=255, n_lists=256, local_projections=True)
```

They help most where one global projection leaves many survivors, that is at narrow widths of high-dimensional
embeddings: on DINOv2 embeddings (768 dimensions) at width 255, the global projection with the anchor threshold leaves
842,000 survivors per query, local projections with the probe leave 4,100, and the throughput rises from 2,600 to
62,000 queries per second. At wide rows, where the global projection already captures most of the variance, the query
rows of every list cost more than they save.
