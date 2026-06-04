# Miscellaneous Language APIs: Advanced

This page collects low-level pointer, synchronization, lane, memory, random,
PDL, and raw TIR/TVM helpers. Prefer higher-level TileLang tile operations when
they express the same work.

## Access Pointers

`T.access_ptr` creates a handle-typed pointer from a buffer-like location:

```python
ptr = T.access_ptr(A[i], "r")
wide = T.access_ptr(A[i, j], "r", 1, 8)
```

Full form:

```python
T.access_ptr(
    base,
    access_type="r",
    *extents,
    offset=0,
    extent=None,
    ignore_last_ndim=0,
)
```

`base` may be a `Buffer`, `BufferLoad`, `BufferRegion`, or a variable bound to
one of those values. `access_type` may be `"r"`, `"w"`, `"rw"`, or the integer
mask equivalent. If `extent=` is omitted, TileLang infers one element for a
`BufferLoad`, the product of region extents for a `BufferRegion`, or the full
shape product for a `Buffer`.

Nonzero `offset` is supported only for one-dimensional buffers in this wrapper.
For multi-dimensional buffers, express offsets through indexing.

## TMA Store And Proxy Helpers

These are explicit async/TMA hooks:

```python
T.fence_proxy_async()
T.tma_store_arrive()
T.tma_store_wait(count=0)
T.create_tma_descriptor(...)
T.tma_load(...)
T.tma_load_2sm(...)
```

`T.fence_proxy_async()` emits a shared-memory proxy fence.
`T.tma_store_arrive()` commits a TMA store group.
`T.tma_store_wait(count=0)` waits until at most `count` store groups remain
outstanding.

Descriptor and load helpers are mostly lowering hooks. User kernels should
normally use `T.copy`, `T.tma_copy`, or `T.copy_cluster`.

Manual mbarrier and cp.async helpers are also available:

```python
T.mbarrier_expect_tx(mbarrier, tx)
T.mbarrier_arrive_expect_tx(mbarrier, tx)
T.cp_async_barrier_noinc(barrier)
```

Use these only when manually managing async transaction counts. Compiler-managed
pipeline lowering and `T.tma_copy(..., barrier=...)` should be the first choice
for normal tiled pipelines.

## Lane, Warp, Shuffle, And Vote

Lane and warp index helpers:

```python
T.get_lane_idx(warp_size=None)
T.get_warp_idx_sync(warp_size=None)
T.get_warp_idx(warp_size=None)
T.get_warp_group_idx(warp_size=None, warps_per_group=None)
T.shuffle_elect(thread_extent)
```

`get_warp_group_idx` requires `warp_size` when `warps_per_group` is supplied.
`shuffle_elect(0)` elects one lane in the whole thread block; positive values
elect one lane per logical group.

Shuffle and vote helpers:

```python
T.shfl_sync(value, srcLane, width=32, mask=0xFFFFFFFF)
T.shfl_xor(value, delta, width=32, mask=0xFFFFFFFF)
T.shfl_down(value, delta, width=32, mask=0xFFFFFFFF)
T.shfl_up(value, delta, width=32, mask=0xFFFFFFFF)
T.any_sync(predicate, mask=0xFFFFFFFF)
T.all_sync(predicate, mask=0xFFFFFFFF)
T.ballot_sync(predicate, mask=0xFFFFFFFF)
T.ballot(predicate)
T.activemask()
T.match_any_sync(value, mask=0xFFFFFFFF)
T.match_all_sync(value, mask=0xFFFFFFFF)
```

On CUDA these map to sync shuffle/vote/ballot intrinsics. On HIP, several masks
are ignored because wavefront operations are full-wavefront. Ballot helpers
return `uint64`; CUDA ballots are zero-extended from 32 lanes, while HIP
ballots can cover 64 lanes. `match_any_sync` and `match_all_sync` are CUDA-only
in the current codegen path.

Thread-block vote barriers combine synchronization with predicate reduction:

```python
T.syncthreads_count(predicate)
T.syncthreads_and(predicate)
T.syncthreads_or(predicate)
```

## Explicit Global Memory Access

`T.__ldg(x[i])` emits an explicit read-only cache load on CUDA and falls back
to a regular load on non-CUDA backends. If you pass a buffer directly, also
pass a one-dimensional index:

```python
v = T.__ldg(A[i])
v = T.__ldg(A, i)
```

Fixed-width global load/store helpers:

```python
T.ldg32(src, pred=None)
T.ldg64(src, pred=None)
T.ldg128(src, pred=None)
T.ldg256(src, pred=None)
T.stg32(dst, value, pred=None)
T.stg64(dst, value, pred=None)
T.stg128(dst, value, pred=None)
T.stg256(dst, value, pred=None)
```

They take buffer-like source or destination arguments and optional predicates.
The load helpers return packed unsigned vector values: `uint32`, `uint32x2`,
`uint32x4`, or `uint32x8`. Use these only when fixed-width global memory
instructions are intentional; use `T.copy` for ordinary tile movement.

AMD LDS transpose helpers are available for MFMA-oriented shared-memory load
patterns:

```python
T.ds_read_tr16_b64(src)
T.ds_read_tr8_b64(src)
```

They return `uint32x2`.

## Grid And Global Synchronization

```python
T.sync_grid()
T.sync_global()
```

`T.sync_grid()` emits a grid sync intrinsic. `T.sync_global()` computes current
thread and block information from the launch frame and emits a global storage
sync call. It also prints those values during expression construction, so treat
it as a specialized/debug path.

## Random Numbers

CUDA CURAND-style helpers:

```python
T.rng_init(seed, seq=None, off=0, generator="curandStatePhilox4_32_10_t")
T.rng_rand()
T.rng_rand_float(bit=32, dist="uniform")
```

Supported generator names are:

```text
curandStateMRG32k3a_t
curandStatePhilox4_32_10_t
curandStateXORWOW_t
```

If `seq` is omitted, TileLang derives it from
`threadIdx.x + blockIdx.x * blockDim.x`. `rng_rand` returns `uint32`.
`rng_rand_float` supports `bit=32` or `64` and `dist="uniform"` or `"normal"`.
Validate target availability before using these in portable kernels.

## Programmatic Dependent Launch

```python
T.pdl_trigger()
T.pdl_sync()
```

These emit low-level PDL intrinsics. Treat them as target-specific hooks for
programmatic dependent launch workflows.

## Assumptions And Branch Hints

`T.assume(condition)` and `T.likely(condition)` are raw TIR-style helpers
re-exported through the language surface. Use `T.assume` for facts the compiler
may rely on, such as a cluster rank bound already guaranteed by the launch
shape:

```python
rank = T.block_rank_in_cluster()
T.assume(rank < 2)
```

Use these sparingly. A wrong assumption can make generated code incorrect.

## DP4A, Atomics, And Raw TIR Exports

`T.dp4a(A, B, C)` emits a DP4A call over read pointers for `A` and `B` and a
read-write pointer for `C`.

Wider atomic helpers are also available:

```python
T.atomic_addx2(dst, value, return_prev=False)
T.atomic_addx4(dst, value, return_prev=False)
T.atomic_load(src, memory_order="seq_cst")
T.atomic_store(dst, src, memory_order="seq_cst")
```

`T.loop_break()` emits a loop-break intrinsic. In eager/JIT code, ordinary
Python `break` is usually clearer.

`tilelang.language` also re-exports a broad TIR script and math surface,
including `T.prim_func`, `T.macro`, `T.const`, `T.Tensor`, `T.Layout`,
`T.Fragment`, `T.serial`, `T.grid`, `T.thread_binding`, `T.ceildiv`,
`T.if_then_else`, `T.call_extern`, `T.call_intrin`, PTX intrinsics, fast/IEEE
math helpers, packed half2-style helpers, and dtype constructors. Use the
TileLang helper when one exists; drop to raw TIR, PTX, or TVM intrinsics when
implementing a missing primitive or debugging generated code.
