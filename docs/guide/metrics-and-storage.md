# Metrics and storage

## Metrics

| `metric` | returned value | order |
|---|---|---|
| `"l2"` (default) | squared Euclidean distance | smallest first |
| `"cosine"` | cosine similarity (vectors are normalized on entry) | largest first |
| `"ip"` | inner product | largest first |

Inner-product search is exact for vectors of any norm: certex bounds the inner-product score directly instead of
appending a coordinate that would turn it into Euclidean search, because rounding that coordinate can reorder near
ties.

## Width

`width` is the number of principal directions kept in the half-precision screening rows. Every width returns the same
exact results; a wider row excludes more vectors per comparison but costs more memory and compute. The default,
`"auto"`, takes the smallest row of 64, 128, or a multiple of 128 coordinates whose directions capture 95% of the
variance of the first vectors added.

## Storage

The screening rows always live in GPU memory, at 2 bytes per coordinate of the row. The full-precision vectors live in
GPU memory (`storage="device"`, fastest) or in pinned host memory (`storage="host"`), where the GPU reads only the few
rows it must verify. Host storage lets one GPU hold several times more vectors at the same speed: on BIGANN-100M at
width 123, 18,700 queries per second with the vectors in host memory and 29.0 GB of device memory, against 18,700 with
the vectors in device memory and 43.6 GB.
