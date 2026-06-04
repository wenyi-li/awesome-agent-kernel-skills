# Kernel, Warpgroup, And Cluster Builtins: Basic

This page covers the launch-frame and target helpers that appear in normal
TileLang kernels. It assumes the conventional imports:

```python
import tilelang
import tilelang.language as T
```

## `T.Kernel`

`T.Kernel` creates a launch region inside `@tilelang.jit` or `@T.prim_func`:

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    ...
```

Positional arguments are the grid extents for `blockIdx.x`, `blockIdx.y`, and
`blockIdx.z`. A one-dimensional launch may be unpacked as either `bx` or
`(bx,)`:

```python
with T.Kernel(T.ceildiv(N, BN), threads=128) as bx:
    ...
```

`threads=` accepts an integer or a short tuple/list. `threads=128` means
`(128, 1, 1)`, and omitted GPU thread extents default to 128 threads. Most
tiled kernels use `T.Parallel`, `T.copy`, and tile operators inside the launch
instead of assigning work directly to lanes.

Use direct thread bindings only when the kernel really needs lane-level control:

```python
with T.Kernel(blocks, threads=(128, 2)) as bx:
    tx = T.get_thread_binding(0)
    ty = T.get_thread_binding(1)
```

The binding helpers read the current launch frame:

```python
T.get_thread_binding(dim=0)
T.get_thread_bindings()
T.get_block_binding(dim=0)
T.get_block_bindings()
T.get_thread_extent(dim=0)
T.get_thread_extents()
T.get_block_extent(dim=0)
T.get_block_extents()
```

They must be called inside a `T.Kernel` region.

## Cluster Launch Basics

`cluster_dims=` requests a clustered thread-block launch:

```python
with T.Kernel(num_tiles, threads=128, cluster_dims=2) as bid:
    rank = T.block_rank_in_cluster()
```

An integer such as `2` is normalized to `(2, 1, 1)`. `(1, 1, 1)` is treated as
no cluster launch. Cluster launches are target-specific; use them for kernels
that intentionally use clustered CTAs, cluster barriers, TMA multicast, or
SM-to-SM shared-memory copy.

The common cluster helpers are:

```python
rank = T.block_rank_in_cluster()
T.cluster_sync()
```

`T.block_rank_in_cluster()` returns the one-dimensional CTA rank in the current
cluster. `T.cluster_sync()` is the full cluster barrier, equivalent to an
arrive followed by a wait. Use it after cluster-shared setup and before one CTA
reads data or barriers initialized by another CTA.

## Barrier And Mbarrier Basics

For normal block-level synchronization:

```python
T.sync_threads()
T.sync_warp()
```

Manual async/TMA pipelines and SM100 examples often use mbarriers:

```python
ready = T.alloc_barrier([32] * num_stages)
done = T.alloc_barrier([1] * num_stages)

T.mbarrier_wait_parity(done[k % num_stages], ((k // num_stages) & 1) ^ 1)
T.tma_copy(src, dst, barrier=ready[k % num_stages])
T.mbarrier_arrive(ready[k % num_stages])
T.mbarrier_wait_parity(ready[k % num_stages], (k // num_stages) & 1)
```

`T.barrier_wait(...)` and `T.barrier_arrive(...)` are aliases for the mbarrier
parity wait and arrive helpers. Prefer compiler-managed `T.Pipelined` and
ordinary `T.copy` unless you are explicitly managing async barriers.

## Warpgroup Usage

Most users should reach for `T.gemm`, `T.wgmma_gemm`, or the SM100 GEMM
operators instead of low-level warpgroup intrinsics.

When a kernel explicitly splits producer and consumer work by raw thread id,
register-allocation hints may appear near those regions:

```python
tx = T.get_thread_binding()

if tx < 128:
    T.set_max_nreg(240, 1)
    ...
else:
    T.set_max_nreg(80, 0)
    ...
```

Treat these as performance-sensitive target hints. Keep them close to the
region they control and validate generated code on the intended architecture.

