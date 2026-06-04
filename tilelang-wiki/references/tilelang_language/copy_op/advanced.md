# Advanced Copy Operations

This page covers explicit asynchronous, TMA, cluster, gather/scatter, and
transpose copy helpers. Start with `T.copy` unless the kernel needs one of
these lower-level paths.

## Copy Annotations

`T.copy` accepts lowering hints:

```python
T.copy(
    src,
    dst,
    coalesced_width=4,
    disable_tma=True,
    eviction_policy="evict_last",
    loop_layout=layout,
)
```

`coalesced_width` requests a coalescing width. `disable_tma=True` records that
TMA should not be used for this copy. `eviction_policy` can be
`"evict_normal"`, `"evict_first"`, or `"evict_last"`. `loop_layout` records a
parallel-loop layout hint for the generated SIMT copy loop.

`annotations={...}` can pass the same lowering keys directly. Values already
present in `annotations` take precedence over the individual keyword arguments.

## `T.async_copy`

`T.async_copy(src, dst, *, coalesced_width=None, annotations=None,
loop_layout=None)` emits the explicit async-copy operation.

```python
T.async_copy(A[by * block_M, k * block_K], A_shared)
# independent work
T.ptx_wait_group(0)
T.sync_threads()
```

It uses the same region normalization and scalar-store shortcut as `T.copy`,
but it does not insert the wait needed before reading the destination. Use it
only when the kernel explicitly manages wait and synchronization.

## `T.tma_copy`

`T.tma_copy(src, dst, *, barrier=None, eviction_policy=None, annotations=None)`
emits an explicit TMA producer operation.

```python
mbars = T.alloc_barrier([128, 128])
T.tma_copy(
    A[row_start : row_stop, k_start : k_stop],
    A_shared[stage, :, :],
    barrier=mbars[stage],
)
```

For global-to-shared TMA loads, pass a barrier and manage the matching wait
before consuming the shared tile. For shared-to-global TMA stores, the helper
emits the producer/store side and leaves store waits to user code so stores can
be batched.

`T.tma_copy` performs the same pairwise extent legalization as `T.copy`, but it
does not use the scalar-store shortcut.

## `T.copy_cluster`

`T.copy_cluster` is the cluster-aware copy API:

```python
T.copy_cluster(src, dst, cluster_mask=0b11)
T.copy_cluster(src_shared, dst_shared, dst_block=1, remote_barrier=done[0])
```

Use `cluster_mask` for TMA multicast. Use `dst_block` for SM-to-SM
shared-memory copy inside a cluster. The Python API can accept both keys, but
they describe different modes; choose one mode per call.

`remote_barrier` is for asynchronous SM-to-SM completion signaling.
`coalesced_width` and `loop_layout` apply to SIMT fallback paths.
`eviction_policy` is relevant to TMA multicast.

## Blackwell Gather/Scatter TMA

Blackwell gather/scatter helpers target four-row TMA tile operations:

```python
T.tma_gather4(src, dst, col, rows, barrier=mbar)
bytes_ = T.tma_gather4_bytes(K_box, "float16")
T.tma_scatter4(src, dst, col, rows)
```

`T.tma_gather4` loads four arbitrary rows from a rank-2 global buffer into a
rank-2 shared tile of shape `(4, K_box)`. `T.tma_scatter4` stores a `(4, K_box)`
shared tile to four arbitrary rows of a rank-2 global buffer. Both helpers
require matching dtypes, exactly four row indices, and unit innermost global
stride when explicit strides are present. `T.tma_gather4` also requires a
barrier.

`T.tma_gather4_bytes(K_box, dtype)` returns `4 * K_box * sizeof(dtype)` for the
matching transaction. Non-default `swizzle` values are deprecated; annotate the
shared layout instead.

## `T.transpose`

`T.transpose(src, dst)` emits a tile transpose operation.

```python
T.transpose(tile_mn, tile_nm)
```

Both source and destination must provide extents, and both must be at least
2D. Use it for shared-memory tile transposes, not as a general Python tensor
transform.

## Deprecated Alias

`T.c2d_im2col(...)` is a deprecated alias for `T.im2col(...)` and is scheduled
for removal in TileLang 0.14.0. Use `T.im2col(...)` in new kernels.
