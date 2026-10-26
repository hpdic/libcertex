# Saving and loading

```python
index.save("vectors.certex")
index = certex.Index.load("vectors.certex")
```

The file holds the vectors, the screening rows, the projection, and the certificate bounds, so that loading restores
the index without projecting the vectors again and returns the same results.
