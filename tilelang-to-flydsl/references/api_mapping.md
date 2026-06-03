# TileLang → FlyDSL API Mapping

This is the symbol-by-symbol translation dictionary. Use it as a lookup, not a
recipe — many TileLang primitives have *no* one-line FlyDSL equivalent because
they are higher-level. Those rows say "**no direct equivalent**" and point to
`idioms.md` for the multi-step pattern.

Imports assumed:

```python
# TileLang
import tilelang
from tilelang import language as T

# FlyDSL
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, buffer_ops, rocdl, range_constexpr, const_expr, arith
from flydsl.expr import math as fmath
from flydsl.expr.typing import T as flyT       # MLIR type factory (collides with TileLang T!)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.vector import ReductionOp, full
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
```

> **Important naming collision.** TileLang's `T` is `tilelang.language` (the
> entire DSL). FlyDSL's `T` is `flydsl.expr.typing.T`, an MLIR type factory.
> In converted code import the latter as `flyT` (or just spell types as
> `fx.Float32`, `fx.Int32`, etc.) to avoid confusion.

---

## 1. Decorators and kernel structure

| TileLang | FlyDSL |
|---|---|
| `@tilelang.jit` factory function | A plain Python function returning the `@flyc.jit` launcher. The factory pattern is identical — it captures compile-time constants in a closure. |
| `@tilelang.jit(pass_configs={...TL_DISABLE_WARP_SPECIALIZED: True...})` | No FlyDSL counterpart. Drop the pass_configs; FlyDSL does not warp-specialize automatically. |
| `@T.prim_func` | `@flyc.kernel` (kernel body) plus `@flyc.jit` (host launcher). Two functions, not one. |
| `with T.Kernel(gx, gy, gz, threads=N) as (pid_x, pid_y, pid_z):` | Inside `@flyc.kernel`, read `bid = fx.block_idx.{x,y,z}` and `tid = fx.thread_idx.x`. The grid/block shape is set by the `.launch(grid=, block=)` call inside the `@flyc.jit` host function. |
| `T.get_thread_binding(axis=0)` | `fx.thread_idx.x` (or `.y`, `.z`). |
| `T.get_block_binding(axis=0)` | `fx.block_idx.x`. |
| `T.dynamic('name')` | Pass the value as a runtime parameter typed `fx.Int32` on the `@flyc.jit` host function. The kernel body sees it as an `fx.Int32` value. |
| `T.symbolic('name')` | Same — runtime `fx.Int32` parameter. |
| Inferred grid `T.ceildiv(N, BLOCK)` | Compute on the host side inside the `@flyc.jit` body using normal Python arithmetic on `fx.Int32` (`(N + BLOCK - 1) // BLOCK`). |

## 2. Types and tensor declarations

| TileLang | FlyDSL |
|---|---|
| `T.Tensor[(M, N), T.float32]` | `fx.Tensor` annotation (no shape/dtype encoded; FlyDSL discovers them via DLPack at host boundary). |
| `T.StridedTensor[(B, M, N), (M*S, S, 1), dtype]` | `fx.Tensor` plus `flyc.from_dlpack(...).mark_layout_dynamic(leading_dim=k, divisibility=d)` at the call site for non-contiguous inputs. |
| `T.float32` / `T.float16` / `T.bfloat16` | `fx.Float32` / `fx.Float16` / `fx.BFloat16`. |
| `T.float8_e4m3` / `T.float8_e5m2` | `fx.Float8E4M3FN` / `fx.Float8E5M2`. (Note: AMD historically uses the FNUZ variants — `fx.Float8E4M3FNUZ` — on gfx94x; check the target arch.) |
| `T.int32` / `T.int64` / `T.int8` | `fx.Int32` / `fx.Int64` / `fx.Int8`. Unsigned: `fx.Uint32` etc. |
| `T.bool` | `fx.Boolean`. |
| `T.dtype(...)` (runtime dtype value) | No direct equivalent; FlyDSL picks dtypes at trace time via `Constexpr` parameters (e.g., `dtype_str: fx.Constexpr[str]`) and a small dispatch helper like `dtype_to_elem_type`. See `kernels/kernels_common.py` for the canonical pattern. |
| `T.const('M, N, K')` (multi-symbol) | A list of `fx.Constexpr[int]` parameters on the kernel/launcher. |
| Constexpr int parameter | `param: fx.Constexpr[int]`. |

## 3. Memory allocation

| TileLang | FlyDSL |
|---|---|
| `T.alloc_shared((M, N), dtype)` | `SmemAllocator(None, arch=arch).allocate_array(flyT.f16, M*N)` outside the `@flyc.kernel`; inside, get a typed view via `lds_a(allocator.get_base())` returning a `SmemPtr`. Must call `allocator.finalize()` inside the `@flyc.jit` body, in `gpu_module_body`. See `references/idioms.md` §3. |
| `T.alloc_local((N,), dtype)` | `fx.memref_alloca(memref_type, layout)` where `memref_type = fx.MemRefType.get(elem_type, fx.LayoutType.get(N, 1), fx.AddressSpace.Register)` and `layout = fx.make_layout(N, 1)`. |
| `T.alloc_fragment((M, N), dtype)` | When the fragment is the destination of a tiled copy/MMA, use `fx.make_fragment_like(partition_src)` or `thr_mma.make_fragment_A/B/C(tensor)`. When it is just a per-thread scratch buffer, treat it as `T.alloc_local`. |
| `T.alloc_var(dtype)` / `T.alloc_var(dtype, init=v)` | A Python variable holding a `fx.Float32`/`fx.Int32`/etc. value. If it must persist across an `scf.for` loop, pass it through `range(..., init=[...])` as a loop-carried value. |
| `T.alloc_reducer((1,), dtype, op='min', replication='all')` | No first-class reducer in FlyDSL. Implement with `Vector.reduce(ReductionOp.MIN/MAX/ADD)` for per-thread, then a manual block reduction (wave_reduce + LDS round-trip) for cross-thread. See `idioms.md` §4. |
| `T.alloc_global` | Rare; FlyDSL kernels typically receive globals via `fx.Tensor` arguments. |
| `T.alloc_barrier`, `T.alloc_descriptor` (TMA) | NVIDIA-specific; unused on AMD targets. Drop or rewrite as a `gpu.barrier()` synchronisation. |

## 4. Loops

| TileLang | FlyDSL |
|---|---|
| `for i in T.unroll(N):` | `for i in range_constexpr(N):` (compile-time unrolled). |
| `for i in T.vectorized(VEC):` | No first-class vector loop. The vectorisation is achieved by *choosing* a vectorising copy atom (`fx.rocdl.BufferCopy128b()` for 128-bit, etc.) and a register memref of shape `(VEC, 1)`. The body then becomes a single `Vector` op. See `idioms.md` §1. |
| `for i in T.serial(N):` / `for i in T.serial(start, stop, step):` | `for i in range(N):` (lowers to `scf.for`). |
| `for i in T.Parallel(N):` | **No direct equivalent.** Distribute by hand: compute `idx = tid + base` and guard with `if idx < N:`. For 2-D `T.Parallel(M, N)`, decode into `tid // N_per_row` and `tid % N_per_row`, or use a tiled copy. See `idioms.md` §2. |
| `for i, j in T.Parallel(M, N, loop_layout=...)` | Pre-build a `tiled_copy` whose thread layout matches the supplied `loop_layout`, then `tiled_copy.get_slice(tid).partition_S/D` partitions the loop. |
| `for ko in T.Pipelined(K_iters, num_stages=k):` | No first-class SW pipeline. Use `range(K_iters, init=[...])` to carry prefetch buffers across iterations and issue the next-iteration loads inside the body. The `prefetch-data-load` skill (separate, in this repo's `.claude/skills/`) describes the canonical 2-stage pattern. |
| `for i in T.Persistent(...):` | Persistent kernels: a `range(...)` outer loop in FlyDSL that re-derives the work-item from a stride-1 work counter. Manual. |

## 5. Synchronisation

| TileLang | FlyDSL |
|---|---|
| `T.sync_threads()` | `gpu.barrier()`. |
| `T.shfl_sync(value, lane)` | `value.shuffle(lane, WARP_SIZE)` on a `Vector`/`ArithValue`, or `rocdl.ds_bpermute(idx, src)` for arbitrary permute. For XOR shuffle: `value.shuffle_xor(off, WARP_SIZE)`. |
| `T.shfl_xor(value, mask)` | `value.shuffle_xor(mask, WARP_SIZE)`. |
| `T.shfl_down(value, off)` | Use `value.shuffle(lane + off, WARP_SIZE)` or `rocdl.ds_bpermute`. |
| `T.barrier_arrive`, `T.barrier_wait` (mbarriers) | NVIDIA-specific; drop on AMD. |

## 6. Data movement

| TileLang | FlyDSL |
|---|---|
| `T.copy(global_tensor, shared_buffer)` | The 4-step layout pattern: `fx.rocdl.make_buffer_tensor(global)` → `fx.zipped_divide(buf, (BLOCK_M, BLOCK_N))` → `fx.slice(divided, (None, bid))` → `fx.make_tiled_copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype), tv_layout, tile_mn)` → `tiled_copy.get_slice(tid).partition_S(...)/partition_D(...)` → `fx.copy(copy_atom, src_part, dst_part)`. The destination must be a memref or LDS view. |
| `T.copy(shared_buffer, global_tensor)` | Same as above, but `partition_S` on shared and `partition_D` on the buffer-backed global. |
| `T.copy(local_buffer, global_tensor[base:])` | Single-tile store with `fx.copy_atom_call(copy_atom, register_memref, fx.slice(global_buf, (None, idx)))`. |
| `arr[i, j] = scalar` (single-element write) | Build a 1-element register memref with `fx.memref_alloca(scalar_reg_ty, fx.make_layout(1, 1))`, store the scalar with `fx.memref_store_vec`, then `fx.copy_atom_call(BufferCopy16b/32b, reg, fx.slice(global, (None, idx)))`. |
| `T.async_copy(...)` | `fx.rocdl` async copy intrinsics (gfx950+) or use `BufferCopy128b()` which is async-friendly on CDNA3. |
| `T.tma_copy(...)` | NVIDIA-specific; on AMD use `BufferCopy*` atoms, on gfx1250 use TDM ops via `fx.rocdl.tdm_ops`. |
| `T.transpose(src, dst)` | No direct equivalent. Two `fx.copy` calls with mismatched layouts, or use `fx.rocdl.ds_read_tr16_b64` (gfx950) for an LDS-transpose intrinsic. |

## 7. GEMM

| TileLang | FlyDSL |
|---|---|
| `T.gemm(A_shared, B_shared, C_local)` | A *much* bigger expansion. The minimal MFMA pattern: `mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))`, `tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))`, `thr_mma = tiled_mma.thr_slice(tid)`, then `frag_A = thr_mma.make_fragment_A(A_block)`, `frag_B = thr_mma.make_fragment_B(B_block)`, `frag_C = thr_mma.make_fragment_C(C_block)`, `fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)`. Production GEMMs (preshuffle, blockscale) add a SW pipeline and an explicit prefetch — see `kernels/preshuffle_gemm.py`. |
| `T.gemm(..., transpose_A=True)` | Pick a different MFMA shape variant (`mfma_f32_16x16x4f32`, `mfma_f32_16x16x16f16`, ...) whose source layouts match. There is no boolean transpose flag. |
| `T.gemm_sp(...)` (2:4 sparse) | Not supported in FlyDSL today. Surface to user. |
| `T.wgmma_gemm(...)` (Hopper) | NVIDIA-specific. Re-derive on AMD via MFMA atoms. |
| `T.tcgen05_gemm(...)` (Blackwell) | NVIDIA-specific. |

## 8. Fill / clear

| TileLang | FlyDSL |
|---|---|
| `T.clear(buf)` | `frag.fill(0)` if `frag` is the result of `make_fragment_*`; otherwise iterate with `range_constexpr` and write zeros via `memref_store`. |
| `T.fill(buf, value)` | Same as `clear` but with `value` instead of zero. For LDS, write through `SmemPtr.store` per-element or per-vector. |

## 9. Reductions

| TileLang | FlyDSL |
|---|---|
| `T.reduce_max(buf, out_buf, dim=-1)` | Manual: load to register, run `Vector.reduce(ReductionOp.MAX)` for the per-thread part, then `wave_reduce` (butterfly via `shuffle_xor`) and a 2-stage block reduce through LDS. Steal the `wave_reduce` and `block_reduce` helpers from `kernels/softmax_kernel.py`. |
| `T.reduce_sum(...)` / `T.reduce_min(...)` / `T.reduce_abssum(...)` / `T.reduce_absmax(...)` | Same pattern as `reduce_max`, with the corresponding `ReductionOp` and arithmetic. |
| `T.reduce_bitand` / `T.reduce_bitor` / `T.reduce_bitxor` | `Vector.reduce(ReductionOp.AND/OR/XOR)`, identical block-reduction structure. |
| `T.cumsum(...)` | Manual scan; no first-class FlyDSL primitive. |
| `T.warp_reduce_sum(value)` / `T.warp_reduce_max(value)` | Single-shot wave butterfly: write a `wave_reduce` helper using `value.shuffle_xor(off, WARP_SIZE)` for `off ∈ {WARP/2, WARP/4, ..., 1}`. |
| `T.finalize_reducer(reducer)` | After the manual block-reduction, the result already lives at LDS slot 0; just load it back. |

## 10. Atomics

| TileLang | FlyDSL |
|---|---|
| `T.atomic_add(dst, value)` | `fx.rocdl` raw buffer atomic intrinsics (`raw_ptr_buffer_atomic_fadd`, etc.). For shared memory atomics use the corresponding `ds_atomic_*` intrinsics. |
| `T.atomic_max` / `T.atomic_min` | Same — pick the matching ROCDL intrinsic. |

## 11. Math intrinsics

| TileLang | FlyDSL |
|---|---|
| `T.exp(x)` / `T.exp2(x)` | `fmath.exp(x)` / `fmath.exp2(x)`. For maximum performance on a single VALU cycle: `rocdl.exp2(flyT.f32, x)` (lower precision than `fmath.exp2`). |
| `T.log(x)` / `T.log2(x)` | `fmath.log(x)` / `fmath.log2(x)`. |
| `T.sqrt(x)` / `T.rsqrt(x)` | `fmath.sqrt(x)` / `fmath.rsqrt(x)`. |
| `T.max(a, b)` / `T.min(a, b)` | `a.maximumf(b)` / `a.minimumf(b)` for floats; `a.maximumi(b)` / `a.minimumi(b)` for ints. (These come from `Numeric` operator overloads.) |
| `T.abs(x)` | `fmath.absf(x)` for floats, `fmath.absi(x)` for ints. |
| `T.infinity(T.float32)` | `fx.Float32(float("inf"))`. |
| `T.max_value(T.int32)` | `fx.Int32(0x7fffffff)`. |
| `T.Select(cond, t, f)` | `cond.select(t, f)` when `cond` is a `Numeric`/`ArithValue`, otherwise `arith.select(cond, t, f)`. |
| `T.if_then_else(cond, t, f)` | Same as `T.Select`. |
| `T.ceildiv(a, b)` | `(a + b - 1) // b` (Python integer division on `fx.Int32`/`fx.Index`) or `fx.ceil_div(a, b)` for compile-time IntTuple math. |
| `T.assume(cond)` | No direct equivalent; `arith.assume` in MLIR is rarely needed because FlyDSL's layout system carries divisibility through the `mark_layout_dynamic(divisibility=d)` host hook. |

## 12. Annotations / layout hints

| TileLang | FlyDSL |
|---|---|
| `T.annotate_layout(buf, layout)` | Build the appropriate `make_layout`/`make_ordered_layout` and pass it to the allocator/memref. Layout in FlyDSL is *always* explicit, so there is no separate annotation pass. |
| `T.use_swizzle(panel_size=...)` | Construct a `fx.Swizzle` descriptor and `fx.apply_swizzle(ptr, swizzle)`, or — for L2 swizzle — re-derive `bid` from a swizzled mapping in Python. |
| `T.annotate_safe_value(...)` | No counterpart. |
| `T.Fragment(shape, forward_fn=...)` | A `tv_layout` constructed via `make_layout` / `make_layout_tv` that *matches* what `forward_fn` produces. Reverse-engineer the function: `forward_fn(i, j) -> (thread_id, local_id)` defines the thread/value layout that `make_tiled_copy` consumes. See `idioms.md` §3 on transpose for an example. |

## 13. Printing / debugging

| TileLang | FlyDSL |
|---|---|
| `T.print(x)` | `fx.printf("x={}", x)`. |
| `T.device_assert(cond, msg)` | Not supported; use `if not cond: fx.printf(...)` patterns or skip. |

## 14. Common helpers in TileKernels

| TileKernels helper | FlyDSL drop-in |
|---|---|
| `tile_kernels.utils.ceil_div(x, y)` | Plain Python `(x + y - 1) // y`. |
| `tile_kernels.utils.align(x, y)` | `((x + y - 1) // y) * y`. |
| `tile_kernels.utils.is_power_of_two(x)` | `x > 0 and (x & (x - 1)) == 0`. |

These are pure Python; carry them over verbatim.
