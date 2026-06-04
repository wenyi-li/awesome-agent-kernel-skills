# Miscellaneous Language APIs: Basic

This page covers general helpers that appear around normal TileLang kernels:
atomics, debug output, dynamic symbols, simple buffer views, and small
expression utilities.

## Atomics

Use atomics when multiple threads or CTAs can update the same destination:

```python
T.atomic_add(dst[i], value)
T.atomic_min(dst[i], value)
T.atomic_max(dst[i], value)
```

TileLang also supports tile-region atomics when extents can be inferred from
buffer or region arguments:

```python
T.atomic_add(dQ[row_start:row_stop, head, :], dq)
```

For scalar/addressed atomics, `memory_order` may be one of:

```text
relaxed, consume, acquire, release, acq_rel, seq_cst
```

`return_prev=True` is supported for the scalar direct paths, but not for
tile-region atomics. `T.atomic_add(..., use_tma=True)` requests the TMA reduce
path for supported SM90+ cases.

Prefer reductions or per-block accumulation when the access pattern is
statically separable.

## Debug Print And Assert

`T.print` can print a message, scalar expression, or buffer:

```python
T.print(msg="entered kernel")
T.print(tx, msg="thread id")
T.print(A_shared, msg="tile")
```

For shared and fragment buffers, TileLang prints from the lane selected by:

```text
warp_group_id * 128 + warp_id * 32
```

Global and local buffers do not use that single-lane guard. Fragment buffers
are copied through shared memory before printing. Unsupported buffer scopes
raise `ValueError`.

`T.device_assert(condition, msg="", no_stack_info=False)` emits CUDA
device-side assert calls when CUDA is available:

```python
T.device_assert(idx < N, "idx out of range")
```

By default, TileLang adds source-stack information to the message. Passing
`no_stack_info=True` skips stack collection.

## Dynamic Symbols

Use `T.dynamic` when a dimension must stay symbolic in the generated kernel:

```python
M, N = T.dynamic("M, N")
K = T.dynamic("K", "int64")
```

Comma-separated or whitespace-separated names return a tuple. `T.symbolic` is a
deprecated alias; prefer `T.dynamic`.

`T.index_to_coordinates(index, shape)` converts a row-major flat index into
coordinates:

```python
i, j = T.index_to_coordinates(flat, (BM, BN))
```

## Boolean Buffer Reductions

`T.any_of(buffer)` and `T.all_of(buffer)` reduce a boolean buffer or buffer
region:

```python
if T.any_of(flags[base:base + 32]):
    ...
```

For buffer regions, only the last dimension may have extent greater than one.
These are not variadic scalar predicate helpers; use normal boolean
expressions for scalar predicates.

## Small Utilities

`T.clamp(value, min_val, max_val)` is expression sugar:

```python
x = T.clamp(x, 0.0, 1.0)
```

`T.reshape(src, shape)` and `T.view(src, shape=None, dtype=None)` create buffer
views over the same storage. The source and target must have the same total bit
count. `reshape` preserves dtype; `view` can change shape, dtype, or both.

