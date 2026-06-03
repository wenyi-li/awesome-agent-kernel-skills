# Language Basics

This page introduces the core TileLang (tile‑lang) DSL that you’ll use to write
high‑performance kernels. It focuses on how to define a kernel, express
iteration, move data across memory scopes, and run it with JIT.

The examples use the conventional aliases:

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## 1. Defining a Kernel with `@T.prim_func`

TileLang kernels are TIR (TVM IR) functions produced by the `@T.prim_func`
decorator. Arguments are annotated with shapes and dtypes via `T.Tensor` or
`T.Buffer`.

Note on dtypes
- You can pass dtypes as a string (e.g., 'float32'), a TileLang dtype (e.g., `T.float32`),
  or a framework dtype (e.g., `torch.float32`). TileLang normalizes all of these.
  See Type System for details.

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((N,), dtype),    # dtype could be 'float32' | T.float32 | torch.float32
    B: T.Tensor((N,), dtype),
    C: T.Tensor((N,), dtype),
):
    ...  # kernel body
```

- Shapes may be concrete integers or symbolic. For symbolic, you can pass
  Python ints through the outer `@jit` wrapper (shown below), or annotate with
  `T.dyn` when you want a named symbolic dimension.

```python
# Named symbolic dimension (optional)
K = T.dyn['K']
@T.prim_func
def uses_dyn(A: T.Tensor((K,), 'float32')):
    ...
```

### Dynamic symbolic dimensions: two ways

TileLang supports two complementary ways to introduce symbolic (dynamic) dims:

- Type-level annotations via `T.dyn[...]` (recommended for function signatures)
  - Use in `T.Tensor((T.dyn['K'], ...), dtype)` or bind once then reuse (as above).
  - Inside the kernel body, prefer reading from the buffer’s shape, e.g. `M = A.shape[0]`.

- Term-level variables via `T.dynamic(name, dtype)`
  - Creates a TIR `tir.Var` you can use directly in expressions/loops.
  - Handy when you need to reference the dimension symbol in the body.

```python
# 1) Annotation-only symbol; read the bound size via shape
K = T.dyn['K']  # dtype defaults to int32
@T.prim_func
def foo(A: T.Tensor((K,), 'float32')):
    N = A.shape[0]
    for i in T.serial(N):
        ...

# 2) Explicit Var symbol usable in the body
K = T.dynamic('K', 'int32')   # or T.dynamic('K') defaults to int32
@T.prim_func
def bar(A: T.Tensor((K,), 'float32')):
    for i in T.serial(K):
        ...
```

Notes
- `T.symbolic(name, dtype)` is a deprecated alias of `T.dynamic`; prefer `T.dynamic`.
- Under `@jit`, concrete sizes come from the actual tensor arguments at the first call.
- Symbols in annotations do not need to be separate kernel arguments; TileLang binds them from argument shapes.

## 2. Launching Work with `T.Kernel`

`with T.Kernel(...)` declares a launch context and creates block/thread
bindings. For GPU backends, specify a grid and threads per block.

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...  # bx/by are blockIdx.x/y
```

You rarely need raw thread indices; most kernels use structured loops
(`T.serial`, `T.unroll`, `T.Parallel`, `T.Pipelined`) inside a `T.Kernel`.

## 3. Loops and Control Flow

Core loop constructs map to familiar hardware patterns:

- `T.serial(start, stop[, step])`: plain for‑loop
- `T.unroll(start, stop[, step])`: unrolled loop
- `T.Parallel(ext0, ext1, ...)`: nested parallel loops (elementwise‑friendly)
- `T.Pipelined(iters, num_stages=N)`: software pipelining for producer/consumer

```python
for i in T.serial(N):
    ...

for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    # overlap copy/compute across stages
    ...
```

Conditionals use standard Python `if`/`else`. Guard edges with predicates when
tile sizes do not divide problem sizes evenly.

## 4. Memory Scopes and Allocation

TileLang exposes key software‑managed scopes:

- Global: device memory (default for `T.Tensor` arguments)
- Shared: on‑chip, block‑visible (`T.alloc_shared(shape, dtype)`)
- Fragment and scalars: per‑thread fragments and scalar vars but in Shared View
  (`T.alloc_fragment`, `T.alloc_var`)

```python
A_shared = T.alloc_shared((BM, BK), 'float16')
B_shared = T.alloc_shared((BK, BN), 'float16')
C_local  = T.alloc_fragment((BM, BN), 'float32')
T.clear(C_local)  # zero accumulators
```

## 5. Moving Data: `T.copy`

Use `T.copy(src, dst)` to move tiles between scopes. It accepts buffers,
buffer regions, or buffer loads; extents are inferred or can be broadcast.

```python
# Global -> Shared (tile copy), extents inferred from dst
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[ko * BK, bx * BN], B_shared)

# Fragment -> Global (store back)
T.copy(C_local, C[by * BM, bx * BN])
```

`T.copy` performs coalescing and scope‑specific lowering during compilation.

If you need explicitly asynchronous global->shared prefetch for manual pipelining,
use `T.async_copy(src, dst)`. Unlike `T.copy`, it does not auto-insert any wait:
you must explicitly insert `T.ptx_wait_group(...)` before consuming `dst`. A
shared-memory barrier is still required for cross-thread consumption, but in
most TileLang programs you do not need to write it manually because
`ThreadSync("shared")` will insert the necessary `T.tvm_storage_sync("shared")`
before the first read from `dst`.

## 6. A Minimal End‑to‑End Example (Vector Add)

```python
import tilelang
import tilelang.language as T
from tilelang import jit

@jit  # infers target from tensors at first call
def add(N: int, block: int = 256, dtype: str = 'float32'):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                # Optional — LegalizeSafeMemoryAccess inserts a guard when an access may be OOB
                C[gi] = A[gi] + B[gi]

    return add_kernel

# Host side (PyTorch shown; NumPy/DLPack also supported)
import torch
N = 1 << 20
A = torch.randn(N, device='cuda', dtype=torch.float32)
B = torch.randn(N, device='cuda', dtype=torch.float32)
C = torch.empty(N, device='cuda', dtype=torch.float32)

kernel = add(N)
kernel(A, B, C)  # runs on GPU
torch.testing.assert_close(C, A + B)
```

Notes
- The `@jit` wrapper returns a callable kernel after the first compilation.
- You can pass compile‑time tunables (tile sizes, dtypes) through the outer
  Python function and bake them into the generated TIR.

## 7. Tiled GEMM Skeleton

Below is a minimal pattern for a tiled GEMM using shared memory staging and a
fragment accumulator. It mirrors the quickstart style found in the repository.

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), 'float16'),
    B: T.Tensor((K, N), 'float16'),
    C: T.Tensor((M, N), 'float16'),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), 'float16')
        B_s = T.alloc_shared((BK, BN), 'float16')
        C_f = T.alloc_fragment((BM, BN), 'float32')
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)  # lowered to tensor‑core/ISA specific kernels

        T.copy(C_f, C[by * BM, bx * BN])
```

## 8. Debugging and Printing

Use `T.print` inside a kernel for quick introspection. TileLang emits printing
from a single thread for shared/fragment scopes to avoid floods.

```python
T.print(C_f, msg='accumulator:')
T.print(A_s, msg='A tile:')
T.print(C[0], msg='C[0] = ')
```

## 9. Where to Go Next

- Control flow details: see Programming Guides → Control Flow
- Memory topics: see Programming Guides → (removed cache/layout); basics are covered inline
- Autotuning tile sizes and mappings: Programming Guides → Autotuning
- Operator examples (GEMM, GEMV, attention): see Deep Learning Operators
# Instructions

This page summarizes the core TileLang “instructions” available at the DSL
level, how they map to hardware concepts, and how to use them correctly.

## Quick Categories
- Data movement: `T.copy`, `T.async_copy`, `T.c2d_im2col`, staging Global ↔ Shared ↔ Fragment
- Compute primitives: `T.gemm`/`T.gemm_sp`, elementwise math (`T.exp`, `T.max`),
  reductions (`T.reduce_sum`, `T.cumsum`, warp reducers)
- Control helpers: `T.clear`/`T.fill`, `T.reshape`/`T.view`
- Diagnostics: `T.print`, `T.device_assert`
- Advanced: atomics, memory barriers, warp‑group ops

## Data Movement

Use `T.copy(src, dst, *, coalesced_width=None, disable_tma=False, eviction_policy=None, loop_layout=None)`
to move tiles between memory scopes. It accepts `tir.Buffer`, `BufferLoad`, or
`BufferRegion`; extents are inferred or broadcast when possible.

```python
# Global → Shared tiles (extents inferred from dst)
T.copy(A[by * BM, ko * BK], A_s)
T.copy(B[ko * BK, bx * BN], B_s)

# Fragment/Register → Global (store result)
T.copy(C_f, C[by * BM, bx * BN])
```

Semantics
- Extents are deduced from arguments; missing sides broadcast to the other’s rank.
- Access patterns are legalized and coalesced during lowering. Explicit
  vectorization is not required in HL mode.
- Safety: the LegalizeSafeMemoryAccess pass inserts boundary guards when an
  access may be out‑of‑bounds and drops them when proven safe.

### `T.copy` vs `T.async_copy`

TileLang supports both synchronous and explicitly-asynchronous copies.

`T.copy(src, dst, ...)` (synchronous semantics)
- Intended default for most TileLang programs.
- The compiler is free to lower it to different mechanisms (SIMT copy, `ldmatrix`,
  TMA, `cp.async`, etc.) depending on target/hints, but the observable semantics
  are *synchronous*: after the statement, it is safe to use `dst`.
- If `T.copy` lowers to `cp.async`, TileLang will still preserve synchronous
  semantics by emitting the required `commit`/`wait` (and any required
  synchronization) so that consuming `dst` is correct.

`T.async_copy(src, dst, ...)` (explicit async semantics)
- Intended for writing manual pipelines or warp-specialized code where you want
  to overlap global->shared copies with compute.
- Lowers through `cp.async` and emits:
  - `ptx_cp_async(...)`
  - `ptx_commit_group()`
  - No `ptx_wait_group(...)` is auto-inserted.
- You must explicitly insert `T.ptx_wait_group(...)` before consuming `dst`.
- A barrier is still required when `dst` is produced cooperatively and consumed
  across threads. In most TileLang programs you do not need to write it
  manually: `ThreadSync("shared")` will insert the required
  `T.tvm_storage_sync("shared")` before the first read from `dst`. If you want
  explicit control (or if you're writing very low-level code), you can insert
  `T.tvm_storage_sync("shared")` yourself (or `T.tvm_storage_sync("warp")` for
  warp-local consumption).
- This op is intentionally strict: if the copy cannot be lowered to `cp.async`
  (e.g., wrong scopes, unsupported vector width), compilation fails instead of
  silently falling back to a synchronous copy.

Example (manual async prefetch)
```python
# Prefetch into shared asynchronously (emits cp.async + commit).
T.async_copy(A[by * BM, ko * BK], A_s)

# ... independent work here ...

# Before consuming A_s, ensure the async copies are completed.
T.ptx_wait_group(0)
# The required shared-memory barrier will be inserted automatically before the
# first read from A_s by ThreadSync("shared") in the default lowering pipeline.
T.gemm(A_s, B_s, C_f)
```

Other helpers
- `T.c2d_im2col(img, col, ...)`: convenience for conv‑style transforms.

## Compute Primitives

GEMM and sparse GEMM
- `T.gemm(A_shared, B_shared, C_fragment)`: computes a tile GEMM using shared
  inputs and a fragment accumulator; lowered to target‑specific tensor cores.
- `T.gemm_sp(...)`: 2:4 sparse tensor core variant (see examples and README).

Reductions and scans
- `T.reduce_sum`, `T.reduce_max`, `T.reduce_min`, `T.cumsum`, plus warp
  reducers (`T.warp_reduce_sum`, etc.).
- Allocate and initialize accumulators via `T.alloc_fragment` + `T.clear` or
  `T.fill`.

Elementwise math
- Most math ops mirror TVM TIR: `T.exp`, `T.log`, `T.max`, `T.min`, `T.rsqrt`,
  `T.sigmoid`, etc. Compose freely inside loops.

Reshape/view (no copy)
- `T.reshape(buf, new_shape)` and `T.view(buf, shape=None, dtype=None)` create
  new views that share storage, with shape/dtype checks enforced.

## Synchronization (HL usage)

In HL pipelines, you usually don’t need to write explicit barriers. Passes such
as PipelinePlanning/InjectSoftwarePipeline/InjectTmaBarrier orchestrate
producer/consumer ordering and thread synchronization behind the scenes.

If you need debugging or explicit checks:
- `T.device_assert(cond, msg='')` emits device‑side asserts on CUDA targets.
- `T.print(obj, msg='...')` prints scalars or buffers safely from one thread.

## Putting It Together: GEMM Tile

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), 'float16'),
    B: T.Tensor((K, N), 'float16'),
    C: T.Tensor((M, N), 'float16'),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), 'float16')
        B_s = T.alloc_shared((BK, BN), 'float16')
        C_f = T.alloc_fragment((BM, BN), 'float32')
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)  # Global → Shared
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)             # compute into fragment

        T.copy(C_f, C[by * BM, bx * BN])      # store back
```

## Instruction Reference (Concise)

Below is a concise list of TileLang instructions grouped by category. For full
signatures, behaviors, constraints, and examples, refer to API Reference
(`autoapi/tilelang/index`).

Data movement
- `T.copy(src, dst, ...)`: Move tiles between Global/Shared/Fragment.
- `T.async_copy(src, dst, ...)`: Explicit async global→shared copy via `cp.async`.
- `T.transpose(src, dst)`: Transpose a 2D shared buffer: `dst[j, i] = src[i, j]`.
- `T.c2d_im2col(img, col, ...)`: 2D im2col transform for conv.

Memory allocation and descriptors
- `T.alloc_shared(shape, dtype, scope='shared.dyn')`: Allocate shared buffer.
- `T.alloc_fragment(shape, dtype, scope='local.fragment')`: Allocate fragment.
- `T.alloc_var(dtype, [init], scope='local.var')`: Scalar var buffer (1 elem).
- `T.alloc_barrier(arrive_count)`: Allocate and initialize one or more mbarriers.
- `T.alloc_tmem(shape, dtype)`: Tensor memory (TMEM) buffer (Hopper+).
- `T.deallocate_tmem(buffer)`: Explicitly release a TMEM buffer at the current site.
- `T.alloc_reducer(shape, dtype, op='sum', replication=None)`: Reducer buf.
- `T.alloc_descriptor(kind, dtype)`: Generic descriptor allocator.
  - `T.alloc_wgmma_desc(dtype='uint64')`
  - `T.alloc_tcgen05_smem_desc(dtype='uint64')`
  - `T.alloc_tcgen05_instr_desc(dtype='uint32')`
- `T.empty(shape, dtype='float32')`: Declare function output tensors.

Compute primitives
- `T.gemm(A_s, B_s, C_f)`: Tile GEMM into fragment accumulator.
- `T.gemm_sp(...)`: Sparse (2:4) tensor core GEMM.
- Reductions: `T.reduce_sum/max/min/abssum/absmax`, bitwise `and/or/xor`.
- Scans: `T.cumsum`, finalize: `T.finalize_reducer`.
- Warp reducers: `T.warp_reduce_sum/max/min/bitand/bitor`.
- Elementwise math: TIR ops (`T.exp`, `T.log`, `T.max`, `T.min`, `T.rsqrt`, ...).
- Fast math: `T.__log/__log2/__log10/__exp/__exp2/__exp10/__sin/__cos/__tan`.
- IEEE math: `T.ieee_add/sub/mul/fmaf` (configurable rounding).
- Helpers: `T.clear(buf)`, `T.fill(buf, value)`.
- Views: `T.reshape(buf, shape)`, `T.view(buf, shape=None, dtype=None)`.

Diagnostics
- `T.print(obj, msg='')`: Print scalar/buffer from one thread.
- `T.device_assert(cond, msg='')`: Device-side assert (CUDA).

Logical helpers
- `T.any_of(a, b, ...)`, `T.all_of(a, b, ...)`: Multi-term predicates.

Annotation helpers
- `T.use_swizzle(panel_size=..., enable=True)`: Rasterization hint.
- `T.annotate_layout({...})`: Attach explicit layouts to buffers.
- `T.annotate_safe_value(var, ...)`: Safety/const hints.
- `T.annotate_l2_hit_ratio(buf, ratio)`: Cache behavior hint.

Synchronization helpers
- `T.sync_threads([barrier_id, arrive_count])`: Block-wide barrier (`__syncthreads()`).
- `T.sync_warp([mask])`: Warp-wide barrier (`__syncwarp([mask])`).
- `T.sync_grid()`: Cooperative grid barrier (requires cooperative launch).
- `T.pdl_trigger()`: Signal programmatic launch completion for the current kernel.
- `T.pdl_sync()`: Wait until kernel dependencies are satisfied.

Warp-vote / warp-ballot (CUDA ≥ 9 / HIP)
- `T.any_sync(predicate[, mask])` → `int32`: Non-zero if ANY lane in `mask` has non-zero predicate (`__any_sync`). `mask` defaults to `0xFFFFFFFF`.
- `T.all_sync(predicate[, mask])` → `int32`: Non-zero if ALL lanes in `mask` have non-zero predicate (`__all_sync`). `mask` defaults to `0xFFFFFFFF`.
- `T.ballot_sync(predicate[, mask])` → `uint64`: Bitmask of lanes in `mask` with non-zero predicate. CUDA: `__ballot_sync` zero-extended to 64 bits; HIP: `__ballot` returns natively as `uint64`, covering all 64 wavefront lanes. `mask` defaults to `0xFFFFFFFF`.
- `T.ballot(predicate)` → `uint64`: Full-warp/wavefront ballot (mask = `0xFFFFFFFF`). No truncation on HIP.
- `T.activemask()` → `uint64`: Bitmask of currently active lanes. CUDA: `__activemask` zero-extended to 64 bits; HIP: `__ballot(1)` as `uint64`.

Block-wide predicated sync
- `T.syncthreads_count(predicate)` → `int32`: Sync all threads; return count with non-zero predicate (`__syncthreads_count`).
- `T.syncthreads_and(predicate)` → `int32`: Sync; non-zero iff ALL threads have non-zero predicate (`__syncthreads_and`).
- `T.syncthreads_or(predicate)` → `int32`: Sync; non-zero iff ANY thread has non-zero predicate (`__syncthreads_or`).

Warp-shuffle (intra-warp data exchange). All accept a trailing `mask` kwarg that defaults to `0xFFFFFFFF`.
- `T.shfl_sync(value, src_lane[, width, mask])`: Broadcast value from `src_lane` to all lanes (`__shfl_sync`).
- `T.shfl_xor(value, delta[, width, mask])`: XOR-swap across lanes (`__shfl_xor_sync`).
- `T.shfl_down(value, delta[, width, mask])`: Shift down by `delta` lanes (`__shfl_down_sync`).
- `T.shfl_up(value, delta[, width, mask])`: Shift up by `delta` lanes (`__shfl_up_sync`).

Warp-match (CUDA sm_70+, not supported on HIP). `mask` defaults to `0xFFFFFFFF`.
- `T.match_any_sync(value[, mask])` → `uint32`: Bitmask of lanes in `mask` whose `value` matches the calling lane's (`__match_any_sync`).
- `T.match_all_sync(value[, mask])` → `uint32`: Returns `mask` if all lanes in `mask` agree on `value`, else 0 (`__match_all_sync`). The C-level `int*` predicate output is hidden; reconstruct it as `result != 0`.

> **Note on HIP:** `any_sync`/`all_sync` ignore the mask and call `__any`/`__all` directly. `ballot_sync`, `ballot`, and `activemask` call `__ballot` which returns `uint64` natively on 64-thread wavefronts — no truncation occurs. Shuffle intrinsics lower to `__shfl`/`__shfl_xor`/`__shfl_down`/`__shfl_up` (mask ignored). `syncthreads_count/and/or` have identical signatures on both platforms. `match_any_sync` and `match_all_sync` have no HIP equivalent and will fail to codegen on HIP.

Atomics
- `T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False)`.
- `T.atomic_addx2(dst, value, return_prev=False)`; `T.atomic_addx4(...)`.
- `T.atomic_max(dst, value, memory_order=None, return_prev=False)`.
- `T.atomic_min(dst, value, memory_order=None, return_prev=False)`.
- `T.atomic_load(dst)`, `T.atomic_store(dst, value)`.

Custom intrinsics
- `T.dp4a(A, B, C)`: 4‑element dot‑product accumulate.
- `T.clamp(x, lo, hi)`: Clamp to [lo, hi].
- `T.loop_break()`: Break from current loop via intrinsic.

Barriers, TMA, warp‑group
- Barriers: `T.alloc_barrier(arrive_count)`.
- Parity ops: `T.mbarrier_wait_parity(barrier, parity)`, `T.mbarrier_arrive(barrier)`.
- Expect tx: `T.mbarrier_expect_tx(...)`; sugar: `T.barrier_wait(id, parity=None)`.
- TMA: `T.create_tma_descriptor(...)`, `T.tma_load(...)`,
  `T.tma_store_arrive(...)`, `T.tma_store_wait(...)`.
- Proxy/fences: `T.fence_proxy_async(...)`, `T.warpgroup_fence_operand(...)`.
- Warp‑group: `T.warpgroup_arrive()`, `T.warpgroup_commit_batch()`,
  `T.warpgroup_wait(num_mma)`, `T.wait_wgmma(id)`.

Lane/warp index
- `T.get_lane_idx(warp_size=None)`: Lane id in warp.
- `T.get_warp_idx_sync(warp_size=None)`: Canonical warp id (sync).
- `T.get_warp_idx(warp_size=None)`: Canonical warp id (no sync).
- `T.get_warp_group_idx(warp_size=None, warps_per_group=None)`: Group id.

Register control
- `T.set_max_nreg(reg_count, is_inc)`, `T.inc_max_nreg(n)`, `T.dec_max_nreg(n)`.
- `T.annotate_producer_reg_dealloc(n=24)`, `T.annotate_consumer_reg_alloc(n=240)`.
- `T.no_set_max_nreg()`, `T.disable_warp_group_reg_alloc()`.

## Notes on Dtypes

Dtypes accept three equivalent forms:
- String: `'float32'`
- TileLang dtype: `T.float32`
- Framework dtype: `torch.float32`
All are normalized internally. See Type System for details.
# Control Flow

This guide covers the control‑flow primitives in TileLang and how they lower to
efficient GPU code. You will use these to structure loops, handle boundaries,
and express pipelined compute.

## Overview
- Conditionals: `if` / `elif` / `else`, ternary (`x if c else y`)
- Loops: `T.serial`, `T.unroll`, `T.Parallel`, `T.Pipelined`
- While loops: `while` with a TIR condition
- Flow control: Python `break` / `continue`
- Safety: automatic OOB guards via the LegalizeSafeMemoryAccess pass

The examples assume `import tilelang.language as T`.

## Conditionals

Standard Python `if`/`elif`/`else` is supported inside `@T.prim_func` kernels.
Conditions should be TIR expressions (e.g., `i < N`). Python plain booleans are
treated as compile‑time constants and will be folded.

```python
for i in T.serial(N):
    if i < N:            # TIR condition
        C[i] = A[i] + B[i]
    else:
        pass

# Ternary
x = (A[i] if i < N else 0)
```

Short‑circuit boolean ops are supported. For multi‑dimensional bounds, use
`T.any_of` / `T.all_of` for clarity:

```python
if T.all_of(i < M, j < N):
    C[i, j] = A[i, j] + B[i, j]
```

Boundary handling note
- The LegalizeSafeMemoryAccess pass automatically inserts guards when an access
  may be out‑of‑bounds, and elides them when proven safe. You can often omit
  explicit `if` checks for simple edge handling, but keep them when you need
  custom logic or clarity.

## Loops

### Serial

`T.serial` creates a plain for‑loop. Common forms:

```python
for i in T.serial(N):
    ...                     # 0..N-1

for i in T.serial(0, N, 2):
    ...                     # 0, 2, 4, ...
```

### Unroll

`T.unroll` requests loop unrolling for small trip counts.

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

Advanced: TileLang forwards unroll hints to TIR; factor/explicit knobs are
available for expert tuning.

### Parallel (elementwise)

`T.Parallel(ext0, ext1, ...)` builds nested loops that map well to elementwise
operations. The body receives all indices in one `for` header:

```python
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

Optional hints:
- `coalesced_width=` controls memory coalescing width (used for vectorization checks).
- `loop_layout=` accepts a `T.Fragment` to annotate the layout of the entire
  nested parallel loop. The annotation is attached to the outermost loop only
  and must have `InputDim == number of nested parallel extents`.

### Pipelined (software pipelining)

`T.Pipelined(iters, num_stages=...)` overlaps producer/consumer stages (e.g.,
Global→Shared copies with compute). This is the backbone of GEMM/attention
pipelines.

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_s)  # stage: copy A tile
    T.copy(B[ko * BK, bx * BN], B_s)  # stage: copy B tile
    T.gemm(A_s, B_s, C_f)             # stage: compute
```

### Persistent (advanced)

`T.Persistent(domain, wave_size, index, group_size=...)` exposes persistent
thread‑block style looping. It is an advanced construct that TileLang lowers in
later passes and is typically used by specialized templates.

## While Loops

`while` is supported when the condition is a TIR expression. Avoid infinite
loops; TileLang will error if it detects a constant‑true condition.

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

## Break and Continue

Use Python `break`/`continue` to exit or skip within `T.serial`/`T.unroll`/
`T.Parallel`/`while` loops. Keep the body clean after a `break`/`continue` for
readability; the compiler will ignore the dead path.

## Putting It Together: Residual Tile Handling

Below is a typical edge pattern for a 2D kernel. With LegalizeSafeMemoryAccess,
the explicit guard can be omitted when you don’t need a custom edge path.

```python
for i, j in T.Parallel(M, N):
    gi = by * BM + i
    gj = bx * BN + j
    if T.all_of(gi < M, gj < N):     # optional in many cases
        C[gi, gj] = A[gi, gj] + B[gi, gj]
```

## Debugging Conditions

Use `T.print` to inspect values under predicates. For buffers, TileLang prints
from a single thread to avoid duplicate outputs.

```python
if i == 0:
    T.print(C, msg='C tile:')
```
