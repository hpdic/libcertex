# Several GPUs

`certex.ShardedIndex` splits the vectors into contiguous shards, one index per GPU, built in parallel:

```python
index = certex.ShardedIndex(dim=128, devices=[0, 1, 2, 3], width=59, storage="host")
index.add(xb)  # a host array; one addition builds every shard
D, I = index.search(xq, k=10)  # numpy results, ids are the row numbers of xb
```

A search runs in two rounds. Every shard first computes the threshold of each query from its own anchors; the smallest
of these thresholds bounds the k-th distance over the whole collection, because any k distinct vectors of one shard
bound it. Every shard then searches under the shared threshold and returns its exact k nearest vectors within it, and
the host merges the shard answers by exact distance. Only one threshold per query and k pairs of id and distance per
shard cross the interconnect. The results equal those of one index over all vectors. A device may appear more than
once in `devices`, which splits one GPU into several shards (useful for tests). With `storage="host"`, each shard keeps
its full-precision vectors in pinned host memory and its GPU holds only the screening rows and the anchors.
