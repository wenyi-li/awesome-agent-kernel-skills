# Reduction Operations: Advanced

This page covers the generic reducer API, batching, NaN behavior, scans, and
warp-level reductions.

## Generic Reduction API

```python
T.reduce(buffer, out, reduce_type, dim, clear, batch=1, nan_propagate=False)
```

`reduce_type` may be `"sum"`, `"abssum"`, `"max"`, `"absmax"`, `"min"`,
`"bitand"`, `"bitor"`, or `"bitxor"`. The typed wrappers call this generic
API:

```python
T.reduce_abssum(buffer, out, dim=-1, batch=1)
T.reduce_bitand(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_bitor(buffer, out, dim=-1, clear=True, batch=1)
T.reduce_bitxor(buffer, out, dim=-1, clear=True, batch=1)
```

Tile reductions operate on shared and fragment buffers. If input and output
scopes differ, the macro allocates fragment temporaries and inserts the needed
copies. Other scope combinations are rejected.

## Batching

`batch` must be at least `1`. Values greater than `1` request batched all-reduce
lowering, where each backend call covers multiple output elements. The backend
requires a compile-time constant output element count, and `batch` must evenly
divide the per-thread output element count.

`T.finalize_reducer(reducer, batch=1)` emits the final all-reduce operation for
a reducer buffer. It uses the same batching rule.

## NaN Propagation

`nan_propagate=True` is accepted by `reduce_max`, `reduce_min`, and
`reduce_absmax`. It is only meaningful for float16 and bfloat16 CUDA min/max
paths, where it requests NaN-propagating CUDA half or bfloat16 intrinsics. The
default path may ignore a NaN in favor of the other operand.

## Cumulative Sum Caveats

```python
T.cumsum(src, dst=None, dim=0, reverse=False)
```

`dim` is normalized like Python indexing and an out-of-range dimension raises
`ValueError`. When `dst` is provided, rank and extents must match `src`.

For fragment input, `T.cumsum` copies through shared memory, runs the tile scan
there, and copies back. For non-fragment input, it emits the cumsum tile
operation directly.

## Warp Reductions

```python
T.warp_reduce_sum(value)
T.warp_reduce_max(value)
T.warp_reduce_min(value)
T.warp_reduce_bitand(value)
T.warp_reduce_bitor(value)
```

Warp reductions operate on scalar register expressions, not tile buffers. Each
thread provides one value, and the returned expression has the same dtype as the
input value.
