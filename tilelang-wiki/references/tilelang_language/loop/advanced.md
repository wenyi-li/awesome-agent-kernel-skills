# Advanced Loops

This page covers loop controls that are useful after the basic `T.Parallel`,
`T.Pipelined`, and `T.serial` patterns are already clear.

## Parallel Loop Hints

`T.Parallel` accepts optional lowering hints:

```python
for i, j in T.Parallel(
    block_M,
    block_N,
    coalesced_width=4,
    loop_layout=layout,
    prefer_async=True,
):
    ...
```

`coalesced_width` requests a coalescing width. `loop_layout` attaches a fragment
layout as `parallel_loop_layout` on the outermost generated parallel loop.
`prefer_async=True` adds an async-copy preference for the loop subtree.

A `loop_layout` must match the dimensionality of the parallel nest. For a
two-dimensional `T.Parallel`, the layout must describe two input dimensions.
The annotation belongs on the outermost generated parallel loop; inner loops do
not carry their own layout annotation.

## Manual Pipeline Scheduling

`T.Pipelined` also accepts manual scheduling metadata:

```python
for k in T.Pipelined(num_tiles, order=[1, 0], stage=[0, 1]):
    base: T.int32 = k * block_K
    T.copy(A[base], A_shared)
    T.copy(A_shared, B[base])
```

`order` and `stage` describe executable pipeline statements, such as copies,
fills, GEMMs, reductions, stores, waits, and commits. Replayable scalar aliases
do not consume entries. In the example above, `base` is replayed where it is
used, so the two annotation entries correspond to the two copy statements.

Prefer `num_stages` alone for ordinary software pipelining. Reach for manual
`order`, `stage`, `sync`, and `group` only when following a kernel pattern that
needs explicit scheduling.

## `T.unroll` And `T.Unroll`

`T.unroll(...)` marks a loop for unrolling. `T.Unroll` is an alias.

```python
for k in T.unroll(4):
    acc += a[k] * b[k]

for k in T.unroll(0, K, 2, explicit=True):
    ...
```

`T.unroll` supports the same one-argument and start/stop/step forms as
`T.serial`. It adds a `pragma_unroll_explicit` annotation unless one is already
present. `unroll_factor=` also adds `pragma_unroll_factor`, but current
front-end checking around `explicit` and `unroll_factor` is narrow; use it only
when a kernel pattern has already validated that form.

## `T.vectorized` And `T.Vectorized`

`T.vectorized(...)` marks a loop as vectorized. `T.Vectorized` is an alias.

```python
for v in T.vectorized(local_size):
    local[v] = global_buf[offset + v]
```

The helper supports `T.vectorized(n)` and `T.vectorized(start, stop)`. It does
not expose a `step` argument.

## `T.Persistent`

`T.Persistent(domain, wave_size, index, group_size=8)` constructs a persistent
loop with wave and group metadata. This is a lower-level scheduling construct,
not a default loop for ordinary kernels. Use it when the surrounding kernel is
already written as a persistent scheduling pattern.
