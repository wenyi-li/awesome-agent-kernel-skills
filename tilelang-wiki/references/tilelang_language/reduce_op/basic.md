# Reduction Operations: Basic

Tile reductions reduce a shared or fragment tile along one dimension into a
smaller shared or fragment output tile. Prefer the typed wrappers in normal
kernels:

```python
T.reduce_sum(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_max(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_min(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_absmax(buffer, out, dim=-1, clear=True, batch=1)
```

`dim` supports negative Python-style indexing. If the input shape is
`[X, D, Y]` and `dim` points to `D`, the output shape must be either `[X, Y]` or
`[X, 1, Y]`.

## Common Patterns

Per-row absolute max, as used by FP8 cast kernels:

```python
y_local = T.alloc_fragment((blk_m, group_size), T.float32)
y_amax_local = T.alloc_fragment((blk_m,), T.float32)

T.copy(X[row_start:row_stop, col_start:col_stop], y_local)
T.reduce_absmax(y_local, y_amax_local, dim=1)
```

Softmax max and sum, as used by attention kernels:

```python
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(scores, scores_max, dim=1, clear=False)
T.reduce_sum(scores, scores_sum, dim=1)
```

Use `clear=True` when the reduction should initialize the output for this
operation. Use `clear=False` when you have already initialized the output or
want the reduction to combine into existing values.

## Cumulative Sum

```python
T.cumsum(src, dst=None, dim=0, reverse=False)
```

`T.cumsum` computes an inclusive cumulative sum along `dim`. If `dst` is
omitted, the result is written back into `src`.

```python
T.cumsum(prefix_tile, dim=1)
T.cumsum(src=shared_tile, dst=out_tile, dim=0, reverse=True)
```

`dst`, when provided, must have the same rank and extents as `src`.
