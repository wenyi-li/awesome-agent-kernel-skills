# Instructions

This guide summarizes the TileLang instruction surface that appears in the
current implementation and in the working examples. The most reliable pattern is
to write kernels with high-level tile operations first, then use explicit
low-level instructions only when a kernel needs manual scheduling.

Implementation anchors:
- `import tilelang.language.copy_op`
- `import tilelang.language.allocate`
- `import tilelang.language.gemm_op`
- `import tilelang.language.reduce_op`
- `import tilelang.language.atomic`

Example anchors:
- `examples/gemm/example_gemm.py`
- `examples/gemm/example_gemm_autotune.py`
- `examples/quickstart.py`
- `examples/online_softmax/online_softmax.py`
- `examples/gemm_splitk/example_tilelang_gemm_splitk.py`

## Context

This guide assumes the surrounding kernel structure introduced in
`language_basics.md`: a `@tilelang.jit` or `@T.prim_func` kernel, a launch
region created with `T.Kernel(...)`, explicit memory-scope allocations, and a
small number of tile operations inside the kernel body.

The examples below focus on instruction behavior rather than on the whole kernel
around them.

## Data Movement

### `T.copy`

```python
T.copy(src, dst, *,
       coalesced_width=None,
       disable_tma=False,
       eviction_policy=None,
       annotations=None,
       loop_layout=None)
```

`T.copy` is the default instruction for moving a tile between buffers or buffer
regions. It accepts a `Buffer`, `BufferRegion`, or `BufferLoad` on either side.

Common uses:

```python
T.copy(A[by * block_M, k * block_K], A_shared)
T.copy(B[k * block_K, bx * block_N], B_shared)
T.copy(C_local, C[by * block_M, bx * block_N])
```

Important behavior from the implementation:
- If both arguments are full buffers, their shapes must be structurally equal.
- If both arguments are scalar `BufferLoad`s with no region extent, TileLang
  lowers the operation to a direct scalar store.
- Otherwise, TileLang tries to infer extents from the arguments and legalize
  them pairwise. Missing extents are treated as size-1 dimensions. This is
  limited syntactic sugar, not general NumPy-style broadcasting.
- `coalesced_width`, `disable_tma`, `eviction_policy`, and `loop_layout` are
  passed as annotations to the lowering pipeline. `annotations={...}` takes
  precedence over the individual keyword arguments.

Use `T.copy` for ordinary global/shared/fragment movement unless you are writing
an explicitly asynchronous pipeline.

### `T.async_copy`

```python
T.async_copy(src, dst, *, coalesced_width=None,
             annotations=None, loop_layout=None)
```

`T.async_copy` is the explicit `cp.async` path for asynchronous global-to-shared
copy. The Python frontend emits a `tl.tileop.async_copy` operation; the backend
is expected to lower it through `ptx_cp_async` and commit a copy group.

The key semantic difference from `T.copy` is synchronization: `T.async_copy` does
not wait for the data before the next statement. Insert an explicit wait before
consuming the destination.

```python
T.async_copy(A[by * block_M, k * block_K], A_shared)

# independent work here

T.ptx_wait_group(0)
T.sync_threads()
T.gemm(A_shared, B_shared, C_local)
```

Use this only when you are intentionally managing overlap. For normal software
pipelined loops, prefer `T.copy` inside `T.Pipelined(...)`.

### `T.tma_copy`

```python
T.tma_copy(src, dst, *, barrier=None,
           eviction_policy=None, annotations=None)
```

`T.tma_copy` exposes an explicit TMA producer operation. It is used by advanced
SM90/SM100 and warp-specialized examples such as:

- `examples/gemm_sm100/gemm_tcgen5mma_ws.py`
- `examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`
- `examples/warp_specialize/example_warp_specialize_gemm_softpipe_stage2.py`

For global-to-shared loads, pass a barrier allocated with `T.alloc_barrier(...)`:

```python
mbars = T.alloc_barrier([128, 128])
T.tma_copy(A[by * block_M, ko * block_K], A_shared, barrier=mbars[0])
```

The implementation documents this as user-managed synchronization:
- TMA loads issue the producer side and require a barrier.
- TMA stores omit the final wait so multiple stores can be batched before an
  explicit wait.

If you do not need that control, use `T.copy` and let the lowering pipeline
choose legal copy mechanisms.

### Other Movement Helpers

- `T.copy_cluster(...)`: cluster-aware copy for TMA multicast or SM-to-SM shared
  memory copy. Use only in cluster/TMA kernels.
- `T.transpose(src, dst)`: transposes a 2D buffer tile.
- `T.c2d_im2col(...)`: image-to-column helper used by convolution-style kernels.

## Allocation

The common allocation instructions are:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
C_local = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((block_M,), T.float32)
scalar = T.alloc_var("int32", init=0)
```

Use these by scope:
- `T.alloc_shared(shape, dtype, scope="shared.dyn")`: shared memory tile. `bool`
  uses `"shared"` internally because shared-memory merging does not handle bool.
- `T.alloc_fragment(shape, dtype, scope="local.fragment")`: fragment/register
  tile for tile operations such as GEMM and reductions.
- `T.alloc_local(shape, dtype, scope="local")`: thread-local storage.
- `T.alloc_var(dtype, init=None, scope="local.var")`: one-element scalar buffer.
  It supports `T.alloc_var("int32", 1)`, `T.alloc_var("int32", init=1)`, and the
  older positional scope form.
- `T.alloc_global(shape, dtype, scope="global")`: global workspace allocated by
  backend APIs. The source code warns that user-managed framework workspaces are
  usually preferable.
- `T.empty(shape, dtype=...)`: declares an eager-style output tensor that should
  be returned by the JIT function.

Advanced allocation:
- `T.alloc_barrier(arrive_count)` and `T.alloc_cluster_barrier(arrive_count)`:
  allocate shared-memory barrier buffers. `arrive_count` can be an int or a list
  of ints.
- `T.alloc_tmem(shape, dtype)`: Blackwell tensor-memory buffer. Shape must be
  2D and is intended for TCGEN5 MMA paths.
- `T.alloc_reducer(shape, dtype, op="sum", replication=None)`: reducer buffer
  for reductions inside parallel loops. Valid ops are `"sum"`, `"max"`, and
  `"min"`; valid replication values are `"all"` and `"none"`.
- Descriptor helpers: `T.alloc_wgmma_desc()`,
  `T.alloc_tcgen05_smem_desc()`, and `T.alloc_tcgen05_instr_desc()`.

For the broader memory-scope model and the canonical minimal kernel skeleton,
see `language_basics.md`. For dtype-specific allocation behavior, see
`type_system.md`.

## Compute

### `T.gemm`

```python
T.gemm(A_shared, B_shared, C_local,
       transpose_A=False,
       transpose_B=False,
       policy=T.GemmWarpPolicy.Square,
       clear_accum=False,
       k_pack=1,
       mbar=None)
```

`T.gemm` is the default synchronous tile GEMM instruction. It selects the backend
implementation during lowering:
- CUDA paths use MMA/WGMMA/TCGEN5 depending on target and operand pattern.
- HIP paths use MFMA-style lowering.
- If WGMMA or TCGEN5 is selected, the high-level `T.gemm` path inserts the
  corresponding wait implicitly.

Use `transpose_B=True` when the shared B tile is stored as `(block_N, block_K)`,
as in `examples/gemm/example_gemm_autotune.py`:

```python
T.copy(B[bx * block_N, k * block_K], B_shared)
T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

Use `policy=T.GemmWarpPolicy.FullRow` for attention-style row-oriented GEMM
patterns, as shown in the flash-attention examples.

Manual asynchronous GEMM exists for specialist kernels:
- `T.wgmma_gemm(...)` plus `T.wait_wgmma(...)` on Hopper.
- `T.tcgen05_gemm(...)` plus mbarrier waits on Blackwell.

Start with `T.gemm` unless the example you are following already uses the manual
form.

For the common pipelined GEMM loop structure, see `software_pipeline.md`. For
tuned GEMM kernels that vary block sizes and `num_stages`, see `autotuning.md`.

### Elementwise Work

Use `T.Parallel(...)` to apply elementwise operations across a tile:

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)
```

This is the pattern in `examples/quickstart.py`. Common scalar math functions
come from the TIR/TileLang language namespace, including `T.exp`, `T.log`,
`T.sqrt`, `T.rsqrt`, `T.max`, `T.min`, `T.if_then_else`, `T.infinity`, and
`T.clamp`.

`T.clear(buf)` and `T.fill(buf, value)` are convenience forms for setting all
elements in a tile.

### Reductions and Scans

Reduction functions take an input buffer, an output buffer, a dimension, and a
`clear` flag:

```python
T.reduce_sum(A_pow_local, A_powsum, dim=1)
T.reduce_max(x, max_x, dim=1, clear=True)
T.reduce_absmax(y_local, y_amax_local, dim=1)
```

Supported high-level reductions:
- `T.reduce_sum`
- `T.reduce_max`
- `T.reduce_min`
- `T.reduce_abssum`
- `T.reduce_absmax`
- `T.reduce_bitand`
- `T.reduce_bitor`
- `T.reduce_bitxor`

Implementation details:
- Negative `dim` values are legalized against the input rank.
- Output shape must match the input shape with the reduced dimension removed,
  or with that dimension kept as extent 1.
- Reductions accept fragment and shared buffers. When needed, TileLang creates
  fragment temporaries and copies between scopes.
- `batch > 1` batches multiple output elements per all-reduce call. It must be
  compatible with the output element count.
- `nan_propagate=True` is supported for max/min/absmax on CUDA float16/bfloat16
  paths.

`T.cumsum(src, dst=None, dim=..., reverse=False)` computes cumulative sums. The
GDN cumsum example uses both fragment and shared inputs:

```python
T.cumsum(G_fragment, dim=1, reverse=reverse)
T.cumsum(G_shared, dim=1, reverse=reverse)
```

Warp reducers are scalar intrinsics:
- `T.warp_reduce_sum(value)`
- `T.warp_reduce_max(value)`
- `T.warp_reduce_min(value)`
- `T.warp_reduce_bitand(value)`
- `T.warp_reduce_bitor(value)`

## Views

`T.reshape(buffer, shape)` and `T.view(buffer, shape=None, dtype=None)` create a
new view of the same storage. Examples use this to reinterpret shared buffers
before reducing:

```python
s_reshaped = T.reshape(s, (block_N, block_Q, heads))
acc_dkv = T.view(KV_shared, shape=[BS // split_store, D], dtype=accum_dtype)
```

Use these when the storage layout is already correct and only the logical shape
or dtype view changes.

## Atomics

Common atomics:

```python
T.atomic_add(dst, value, memory_order=None,
             return_prev=False, use_tma=False)
T.atomic_max(dst, value, memory_order=None, return_prev=False)
T.atomic_min(dst, value, memory_order=None, return_prev=False)
```

`T.atomic_add` is used heavily in split-K GEMM and backward kernels:

```python
for i, j in T.Parallel(block_M, block_N):
    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])
```

Implementation-backed constraints:
- Scalar/addressed atomics use extern element intrinsics. `return_prev=True` is
  supported on this scalar path.
- Tile-region atomics infer and legalize extents like `T.copy`.
- `return_prev=True` is not supported for tile-region atomics.
- Memory order strings are `"relaxed"`, `"consume"`, `"acquire"`, `"release"`,
  `"acq_rel"`, and `"seq_cst"`.
- `use_tma=True` on `atomic_add` requests a TMA/cp.reduce path where supported.

Vectorized atomics such as `T.atomic_addx2` and `T.atomic_addx4` appear in some
DeepSeek examples and should be treated as low-level optimization tools.

## Synchronization

Most high-level kernels rely on the compiler pipeline to insert necessary
producer/consumer synchronization around shared-memory tile operations. Add
explicit synchronization when you use manual async copies, custom shared-memory
protocols, or atomic/shared-memory staging.

Common sync instructions:

```python
T.sync_threads()
T.sync_threads(barrier_id, arrive_count)
T.sync_warp()
T.sync_warp(mask)
T.sync_grid()
```

Advanced barrier/TMA instructions include:
- `T.alloc_barrier(...)`
- `T.mbarrier_wait_parity(...)`
- `T.mbarrier_arrive(...)`
- `T.mbarrier_expect_tx(...)`
- `T.tma_store_arrive(...)`
- `T.tma_store_wait(...)`
- `T.fence_proxy_async(...)`

Warp-group instructions include:
- `T.warpgroup_arrive()`
- `T.warpgroup_commit_batch()`
- `T.warpgroup_wait(num_mma)`
- `T.wait_wgmma(id)`

## Warp Intrinsics

TileLang exposes CUDA/HIP-style warp helpers:

- Vote/ballot: `T.any_sync`, `T.all_sync`, `T.ballot_sync`, `T.ballot`,
  `T.activemask`.
- Shuffle: `T.shfl_sync`, `T.shfl_xor`, `T.shfl_down`, `T.shfl_up`.
- Match: `T.match_any_sync`, `T.match_all_sync` on CUDA targets that support
  them.
- Block-wide predicates: `T.syncthreads_count`, `T.syncthreads_and`,
  `T.syncthreads_or`.

These are low-level tools. Prefer tile reductions when reducing full tiles.

## Diagnostics

`T.print(obj, msg="", warp_group_id=0, warp_id=0)` prints scalars or buffers and
dispatches to different helpers for global, shared, local, and fragment buffers.
The buffer-print helpers gate output by selected warp/group IDs where that is
implemented; scalar-expression printing is a direct print intrinsic and should
not be described as "one thread only" behavior.

`T.device_assert(condition, msg="", no_stack_info=False)` emits a device-side
assert. The sparse utility code uses this to validate metadata invariants.

Use both sparingly; they affect generated code and runtime behavior.

## Annotations and Hints

Common hints:

```python
T.use_swizzle(panel_size=10, enable=True)
T.annotate_layout({...})
T.annotate_safe_value(var, ...)
T.annotate_l2_hit_ratio(buf, ratio)
```

`T.use_swizzle` is common in GEMM examples to improve L2 locality:

```python
T.use_swizzle(panel_size=10, enable=enable_rasteration)
```

Pass configs can also affect instruction lowering. For example, many attention
examples enable fast math:

```python
@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
```

## Dtypes

TileLang accepts dtype strings and TileLang dtype constants:

```python
dtype = "float16"
dtype = T.float16
accum_dtype = T.float32
```

Examples generally use `T.float16`, `T.bfloat16`, and `T.float32` in kernel
factories. Dtypes are normalized internally before lowering.

## Practical Guidance

- Start from a working example with the same dataflow. For GEMM, copy the
  `example_gemm.py` structure. For tuned GEMM, copy `example_gemm_autotune.py`.
- Prefer `T.copy`, `T.gemm`, `T.reduce_*`, and `T.Parallel` before reaching for
  manual TMA, WGMMA, or barrier instructions.
- Treat region inference as convenience syntax, not full broadcasting.
- Put explicit waits/barriers next to explicit async instructions so the data
  dependency is obvious in the source.
- Keep accumulator dtypes explicit. Most GEMM and attention examples accumulate
  in `T.float32` even when inputs and outputs are fp16/bf16.
