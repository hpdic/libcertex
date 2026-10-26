# Sparse attention

Sparse attention attends, for every query, only to the top k keys of the preceding context. LLM code selects these
keys with a bf16 or fp16 matrix product followed by `torch.topk`, whose rounding reorders near ties among the
largest scores. certex selects the exact top k by inner product, verified in double precision, and returns them in
the layout of the code that it replaces.

## PyTorch

```python
import certex

scores, ids = certex.topk(q, keys, k=64)  # instead of torch.topk(q @ keys.mT, 64)
```

`keys` has shape `(n_kv_heads, n, head_dim)` and `q` has shape `(n_heads, m, head_dim)` or `(n_heads, head_dim)`.
Grouped-query attention is supported: query head `h` reads KV head `h // (n_heads // n_kv_heads)`. Both outputs have
shape `(n_heads, m, k)` (or `(n_heads, k)`), with the scores in float32 and largest first, as `torch.topk` returns them.
Inputs may be float32, float16, or bfloat16 on the GPU.

`certex.topk` builds an index for every call. During decoding the keys grow by one token per step, so a
`certex.KeyCache` keeps one index per KV head and extends it in place:

```python
cache = certex.KeyCache(n_kv_heads=8, head_dim=128, k=2048)
cache.append(prompt_keys)  # (n_kv_heads, n, head_dim), e.g. the keys after rotary position embedding
for step in range(max_new_tokens):
    scores, ids = cache.topk(q, k=2048)
    ...
    cache.append(new_keys)  # (n_kv_heads, 1, head_dim)
```

`k` is the largest k that the searches will ask for, at most 2048; `capacity` reserves memory for the expected
number of keys per head once.

## FlashInfer

FlashInfer selects the keys of sparse attention in two steps: an indexer writes the scores of every query row, and
`top_k_page_table_transform` or `top_k_ragged_transform` keeps the k largest and returns their int32 positions.
`certex.flashinfer` takes the queries and the keys instead of the scores and returns the same int32 positions, so
that the attention kernel that reads them is unchanged:

```python
from certex import flashinfer as certex_fi

sel = certex_fi.ExactTopK(num_kv_heads=8, head_dim=128, k_max=2048)
sel.append(slot, k_new)  # keys of one sequence slot, NHD layout: (n, num_kv_heads, head_dim)
out = sel.page_table_transform(q, src_page_table, k=2048, row_to_batch=row_to_batch, page_size=64)
out = sel.ragged_transform(q, offsets, k=2048, row_to_batch=row_to_batch)
values, indices = certex_fi.top_k(q, keys, k=64)  # instead of flashinfer.top_k(q @ keys.T, 64)
sel.release(slot)  # when the sequence finishes
```

Query rows use the NHD layout of FlashInfer. A row of shape `(head_dim,)` selects for one head and gives an output of
shape `(num_rows, k)`; with `(num_rows, num_qo_heads, head_dim)`, every query head selects its own keys from its KV
head and the output has shape `(num_rows, num_qo_heads, k)`. Row `i` belongs to batch position `row_to_batch[i]`,
which reads the keys of slot `slots[row_to_batch[i]]` (by default the batch position itself). Positions beyond the
length of a sequence shorter than k are -1, as in FlashInfer. `tests/test_dropin.py` checks the output against
FlashInfer on the same queries and keys.

Every row selects among all keys appended to its slot, as in decoding.
