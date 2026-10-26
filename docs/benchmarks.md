# Benchmarks

Reproduce with `python bench/bench_bigann.py <millions> <width> <storage> [labels]` on BIGANN (SIFT1B)
prefixes, which checks every result against the official ground truth by exact distances (against exhaustive search
over the label of each query for filtered search).

| Dataset | Width | Anchor stride | Vectors in | Device memory | Queries per second | Exact queries |
|---|---|---|---|---|---|---|
| BIGANN-10M | 123 | 16 | host | 5.1 GB | 176,700 | 100% |
| BIGANN-10M | 123 | 32 | host | 5.1 GB | 183,300 | 100% |
| BIGANN-100M | 123 | 16 | device | 43.6 GB | 18,700 | 100% |
| BIGANN-100M | 123 | 16 | host | 29.0 GB | 18,700 | 100% |
| BIGANN-100M | 123 | 32 | host | 29.7 GB | 20,300 | 100% |
| BIGANN-500M | 59 | 16 | host | 75 GB | 4,717 | 100% |
| BIGANN-500M | 59 | 32 | host | 71 GB | 5,233 | 100% |
| BIGANN-10M, filtered at 10% | 123 | 16 | device | | 324,400 | 100% |
| BIGANN-10M, filtered at 1% | 123 | 16 | device | | 392,000 | 100% |
| BIGANN-10M, filtered at 0.1% | 123 | 16 | device | | 402,700 | 100% |

One NVIDIA H100 NVL (94 GB), batches of 2,048 queries, k = 10, 10,000 queries; these are the measurements of the
[Zarc paper](https://doi.org/10.5281/zenodo.23674583). Filtered search uses the label partition (the default); with the label filter over every row
(`partition_labels=False`), it runs at 116,900, 61,500, and 50,500 queries per second.
