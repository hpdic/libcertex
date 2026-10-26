# Filtered search

Give every vector one non-negative integer label, and every query the label it must match:

```python
index = certex.Index(dim=128)
index.add(xb, labels=lb)  # one label per vector
D, I = index.search(xq, k=10, labels=lq)  # exact k nearest vectors of each query's label
```

The results are exact at every selectivity: certex screens with a threshold from vectors of the query's own label and
never keeps a vector of another label, so that it needs no graph connectivity among the matching vectors. A labeled
index keeps a copy of its screening rows sorted by label, so that the queries of a label screen only the rows of that
label and filtered search gets faster as the selectivity drops (on BIGANN-10M, 324,400 queries per second at 10% and
402,700 at 0.1%). The copy costs another 2 (m + 5) + 4 bytes of GPU memory per vector;
`certex.Index(..., partition_labels=False)` screens every row with the label filter instead. When a
label has fewer than k vectors, the missing entries have id `-1`. Once an index has labels, every addition needs them.

A boolean mask selects the vectors directly, either one mask for every query (a cuVS bitset) or one row per query (a
cuVS bitmap):

```python
D, I = index.search(xq, k=10, filter=keep)  # keep: [n] booleans over the ids, True keeps a vector
D, I = index.search(xq, k=10, filter=rows)  # rows: [n_queries, n], one mask per query
```

The mask has one entry for every id, removed ones included (`n` is the largest id plus one).

Filters combine with the other options of an index. With k-means lists (`n_lists`), the probe of the nearest lists
verifies only vectors that the filter keeps, so that the threshold it sets bounds the k-th admitted distance. With
local projections, the filter applies when the survivors are verified; a label partition needs the global projection,
so that a labeled index with local projections filters instead of partitioning.
