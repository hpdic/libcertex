# Insertions and removals

```python
index = certex.Index(dim=128, capacity=50_000_000)  # optional: preallocate for later additions
index.add(xb_first)  # the first addition fixes the projection
ids = index.add(xb_more)  # later additions are inserted, no rebuild
index.remove(ids[:1000])  # removed vectors never appear again
```

Ids are consecutive integers in the order of addition and are never reused. Insertions project the new vectors with
the existing projection and raise the bounds of the rounding certificate to cover them, so that every search stays
exact; buffers grow automatically beyond `capacity`.

Searches may run concurrently with additions and removals from another thread. Each search batch reads one consistent
view of the index (the rows, the anchors, and the certificate bounds published by the last completed update), so that
its results are exact for the vectors of that view.
