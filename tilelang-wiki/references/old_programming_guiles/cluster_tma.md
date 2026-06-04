# Cluster TMA

This guide covers TileLang's cluster-aware data movement on NVIDIA SM90+
targets:

- TMA multicast: one descriptor TMA load can deliver the same global-memory tile
  to multiple CTAs in a cluster.
- SM-to-SM shared-memory copy: one CTA can write into another CTA's shared
  memory inside the same cluster.

The public API for both features is `T.copy_cluster(...)`. Internally it is
lowered as a `tl.tileop.copy` with cluster annotations.

For the baseline `T.copy`, `T.tma_copy`, and synchronization terminology used by
this guide, see `instructions.md`.

These features require a kernel launched with thread-block clusters:

```python
with T.Kernel(num_blocks, threads=128, cluster_dims=2) as bx:
    rank = T.block_rank_in_cluster()
```

`cluster_dims` may be an integer, a list, or a tuple. `2` is normalized to
`(2, 1, 1)`. `(1, 1, 1)` is treated as no cluster launch.

Cluster support is CUDA-only in the current implementation path and is guarded
by SM90+ code. The CUDA runtime wrapper launches clustered kernels with
`cudaLaunchKernelEx` and `cudaLaunchAttributeClusterDimension`.

## Cluster Helpers

TileLang exposes the following helper operations:

| Operation | Meaning |
| --- | --- |
| `T.block_rank_in_cluster()` | Returns the 1-D CTA rank inside the cluster (`%cluster_ctarank`). |
| `T.cluster_arrive_relaxed()` | Emits `barrier.cluster.arrive.relaxed.aligned`. |
| `T.cluster_arrive()` | Emits `barrier.cluster.arrive.aligned`. |
| `T.cluster_wait()` | Emits `barrier.cluster.wait.aligned`. |
| `T.cluster_sync()` | Emits cluster arrive followed by cluster wait. |
| `T.alloc_cluster_barrier(counts)` | Allocates an mbarrier buffer in `shared.cluster_barrier` scope. |
| `T.mbarrier_wait_parity(barrier, parity)` | Waits for an mbarrier phase. |

The examples in `examples/gemm_sm100` and
`examples/blockscaled_gemm_sm100` show the common
Blackwell/SM100 pattern: launch `cluster_dims=2`, read
`T.block_rank_in_cluster()`, allocate cluster barriers, and use mbarriers to
coordinate 2-CTA tensor-core work.

## API

```python
T.copy_cluster(
    src,
    dst,
    *,
    dst_block=None,
    cluster_mask=None,
    remote_barrier=None,
    eviction_policy=None,
    coalesced_width=None,
    loop_layout=None,
)
```

Use exactly one cluster-copy mode in normal code:

- `cluster_mask=...` selects TMA multicast for global-to-shared TMA loads.
- `dst_block=...` selects SM-to-SM shared-memory copy.

The implementation does not currently reject passing both. Since CUDA lowering
checks `dst_block` first, such a call will take the SM-to-SM path and the
multicast mask will not do what you intended.

## TMA Multicast

TMA multicast is selected by passing `cluster_mask`:

```python
T.copy_cluster(A[by * BM, k * BK], A_shared, cluster_mask=0b0011)
```

`cluster_mask` is an integer bitmask of CTA ranks in the current cluster. The
lowest set rank issues `tma_load_multicast`; CTAs in the mask receive the tile;
CTAs outside the mask issue the regular TMA load for their own tile.

Important constraints:

- The source must be a global-memory tile and the destination must be shared
  memory.
- The copy must select the descriptor-based TMA load path. The 1-D bulk path
  explicitly rejects multicast.
- `cluster_mask` is effectively compile-time: the Python API type is `int`, and
  the C++ lowering only reads an integer immediate from annotations.
- Multicast is a load feature. It is not used for shared-to-shared SM-to-SM
  copy.

Minimal shape:

```python
@T.prim_func
def kernel(A: T.Tensor((M, K), "float16"), B: T.Tensor((M, K), "float16")):
    with T.Kernel(T.ceildiv(M, BM), threads=128, cluster_dims=2) as bx:
        rank = T.block_rank_in_cluster()
        A_shared = T.alloc_shared((BM, BK), "float16")

        # Rank 0 issues the multicast load for ranks 0 and 1.
        T.copy_cluster(A[bx * BM, 0], A_shared, cluster_mask=0b11)

        # Both ranks can consume their local shared-memory copy after the
        # synchronous TMA-load sequence emitted by the copy lowering.
        T.copy(A_shared, B[bx * BM, 0])
```

In the CUDA lowering, multicast wraps the normal TMA load as:

1. Compute the minimum CTA rank set in `cluster_mask`.
2. If `block_rank_in_cluster() == min_rank`, emit `tma_load_multicast`.
3. Else if the rank is not in the mask, emit the regular TMA load.
4. Else emit no load instruction for that CTA.

## SM-to-SM Cluster Copy

SM-to-SM copy is selected by passing `dst_block`:

```python
T.copy_cluster(src_shared, dst_shared, dst_block=1)
```

Both `src` and `dst` must be `shared` or `shared.dyn` buffers. The destination
CTA rank may be a constant or a runtime `PrimExpr`.

There are two synchronization styles.

### Async Bulk Copy With Remote Barrier

When `remote_barrier` is provided and the source/destination regions have
matching element counts, TileLang tries to use cluster TMA store:

```python
barrier = T.alloc_cluster_barrier([1])

T.cluster_sync()

if rank == 0:
    T.copy_cluster(
        src_shared,
        dst_shared,
        dst_block=1,
        remote_barrier=barrier[0],
    )

if rank == 1:
    T.mbarrier_wait_parity(barrier[0], 0)
```

For a contiguous region, the lowering emits one guarded call to:

```text
tl::tma_store_cluster(dst_ptr, src_ptr, dst_cta, size_bytes, barrier)
```

Only one elected thread issues the bulk copy. The hardware writes to the
destination CTA's shared memory and signals the destination CTA's mbarrier.

For a non-contiguous but same-shaped region, TileLang recursively decomposes the
copy into contiguous rows and emits one `tma_store_cluster` per row. The barrier
arrival count is updated to the number of emitted row copies.

### SIMT Fallback

Without `remote_barrier`, or when the bulk path cannot prove matching element
counts and per-dimension extents, TileLang falls back to elementwise cluster
stores:

```python
T.copy_cluster(src_shared, dst_shared, dst_block=1)
```

The fallback lowers destination stores to `ptx_cluster_store`, and CUDA codegen
uses:

```text
cooperative_groups::this_cluster().map_shared_rank(...)
```

If a remote barrier was supplied but the copy falls back to SIMT stores,
TileLang appends a shared-memory sync and a single-thread cluster-barrier
arrival so the destination CTA can still wait on the requested mbarrier.

## Synchronization Rules

- Use `T.cluster_sync()` after shared/barrier allocation and before a CTA pushes
  data into another CTA's shared memory. This ensures all CTAs reached the
  allocation/barrier initialization point.
- For `remote_barrier` copies, the destination CTA waits with
  `T.mbarrier_wait_parity(remote_barrier, parity)`.
- For fallback copies without a remote barrier, use cluster-level coordination
  such as `T.cluster_sync()` before reading the destination shared memory.
- Allocate enough barrier arrivals for the number of producers. Multi-row TMA
  decomposition updates the recorded count for that barrier allocation, but
  multiple independent producer CTAs still require the logical count to match
  the number of arrivals you expect.

## Practical Pattern

For 2-CTA kernels, a common layout is:

```python
@T.prim_func
def kernel(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(2, threads=128, cluster_dims=2) as bid:
        rank = T.block_rank_in_cluster()
        src = T.alloc_shared((N,), "float32")
        dst = T.alloc_shared((N,), "float32")
        done = T.alloc_cluster_barrier([1])

        for i in T.Parallel(N):
            src[i] = A[i]

        T.cluster_sync()

        if rank == 0:
            T.copy_cluster(src, dst, dst_block=1, remote_barrier=done[0])

        if rank == 1:
            T.mbarrier_wait_parity(done[0], 0)
            for i in T.Parallel(N):
                B[i] = dst[i]
```

For multicast, keep the mental model separate: use `cluster_mask` only on a
global-to-shared TMA load when more than one CTA needs the same global tile.

## Implementation Pointers

- The public API is exposed through TileLang cluster-copy helpers.
- Cluster rank, arrive, wait, and sync helpers are part of the language frontend.
- Cluster barrier allocation follows the same shared-memory barrier model used by other TileLang synchronization APIs.
- CUDA lowering provides the TMA multicast, SM-to-SM copy, and cluster-store codegen paths described above.
