# FlyDSL Kernel Authoring — full `@flyc.kernel` / `@flyc.jit` reference

## The two decorators

- `@flyc.kernel` wraps a Python function into an MLIR GPU kernel.
- `@flyc.jit` wraps a launcher that calls `kernel(...).launch(grid=..., block=..., stream=...)`.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir

@flyc.kernel
def my_kernel(
    A: fx.Tensor,              # GPU tensor (memref via DLPack)
    B: fx.Tensor,
    N: fx.Constexpr[int],      # compile-time constant (bakes into cache key)
):
    tid = gpu.thread_idx.x     # returns Int32
    bid = gpu.block_idx.x
    # ... kernel body ...

@flyc.jit
def launch(
    A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(A, B, N).launch(
        grid=(N // 256,), block=(256,), stream=stream
    )
```

## Parameter types

| Type | Meaning at host boundary |
|------|--------------------------|
| `fx.Tensor` | GPU tensor, auto-converted from `torch.Tensor` via DLPack |
| `fx.Constexpr[int]` | Compile-time constant; different values produce different compiled kernels |
| `fx.Int32`, `fx.Int64` | Dynamic integer scalars, auto-converted from Python `int` |
| `fx.Stream` | CUDA/HIP stream; pass `torch.cuda.current_stream()` at the call site |

## Thread/block hierarchy

```python
from flydsl.expr import gpu

tid_x = gpu.thread_idx.x   # Int32
bid_x = gpu.block_idx.x
bdim_x = gpu.block_dim.x
gdim_x = gpu.grid_dim.x
gpu.barrier()              # workgroup sync
```

## Control flow

```python
from flydsl.expr import range_constexpr

for i in range_constexpr(N):   # unrolled at compile time
    ...

for i in range(n):             # scf.for at runtime
    ...
```

### scf.for with loop-carried values (software pipelining)

Use `init=` on `range()` to create an `scf.for` with explicit SSA phi nodes
for loop-carried state. This is required for prefetch / software pipelining.

Three critical pitfalls (verified by real debugging):

1. **Loop bounds must be `arith.index()`, NOT Python ints.** If you write
   `range(0, 15, 1, init=...)`, the AST rewriter treats constant bounds as a
   Python range and unrolls — silently ignoring `init=`. Use
   `arith.index(0)`, `arith.index(15)`, `arith.index(1)`.
2. **All `init` values must be raw MLIR `ir.Value`s.** Unwrap FlyDSL numeric
   wrappers via `ir_value()`:
   ```python
   def _unwrap(v): return v.ir_value() if hasattr(v, "ir_value") else v
   init_state = [_unwrap(v) for v in raw_list]
   ```
3. **Clear `SmemPtr._view_cache` before epilogue.** If `SmemPtr.get()` is
   called inside the loop body, the cached `memref.view` is scoped to the
   loop; using it outside causes an SSA dominance error. Fix:
   `my_smem_ptr._view_cache = None` after the loop, before epilogue.

## Frontend semantic restrictions (will trip you up)

1. **Do NOT mutate captured outer variables inside helper functions.** Reads
   are fine; writes must be through explicit parameters and return values.
2. **Do NOT define a value inside `if/else` and use it outside the branch.**
   Keep a single explicit definition path.
3. **Avoid early `return`; no `return`/`yield` inside `if/else`.** Single
   explicit exit keeps the frontend happy.

```python
# GOOD
if cond:
    out = v0
else:
    out = v1
return out
```

## Arithmetic

```python
from flydsl.expr import arith

c42   = fx.Index(42)                           # index constant
c3_14 = fx.Float32(3.14)                       # f32 constant
mask  = fx.Int32(0xFF)
# Legacy path (also works): arith.constant(42, index=True),
# arith.constant(3.14, type=T.f32())

result = arith.addf(a, b)      # float add
result = arith.mulf(a, b)      # float multiply
result = arith.maximumf(a, b)  # float max (works on scalar AND vector)
result = arith.select(cond, t, f)

is_less = arith.cmpf(a, b, predicate="olt")
```

### Internal `Vector` + `Numeric` types (preferred)

```python
from flydsl.expr.typing import Vector as Vec, Float32, BFloat16

acc = Vec(frag_C.load())         # wrap vector<Nxf32> -> Vector
val = acc[idx]                    # element extract -> Float32
v_f32 = Vec(raw).bitcast(Float32) # vector<Nxi32> -> vector<Nxf32>
bf16_val = f32_val.to(BFloat16)   # type convert
result = (val * scale_a) * scale_b
zeros = Vec.filled(N, 0.0, Float32)
idx = fx.Int32(gpu.block_id("x") * tile_m)
```

### Arith op availability

| Op | Function | Works on vectors | Notes |
|---|---|---|---|
| Add | `a + b` or `arith.addf(a, b)` | Yes | |
| Mul | `a * b` or `arith.mulf(a, b)` | Yes | |
| Neg | `arith.negf(a)` | Yes | |
| Max | `arith.maximumf(a, b)` | Yes | |
| Cmp | `arith.cmpf(a, b, pred)` | Yes | returns i1 / vec<i1> |
| Select | `arith.select(c, t, f)` | Yes | |
| Abs | NOT in `arith`. Use `negf+cmpf+select` | NO | |
| FMA | Not exposed; use `mulf+addf` | | |
| Splat const | `arith.constant_vector(val, vty)` | | |

## Shared memory (LDS) with `SmemAllocator`

```python
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext

allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")
lds_a = allocator.allocate_array(T.f16, 8192)
lds_b = allocator.allocate_array(T.f16, 8192)

@flyc.kernel
def my_kernel(A: fx.Tensor, ...):
    lds_base = allocator.get_base()
    lds_a_ptr = lds_a(lds_base)
    val = lds_a_ptr.load([idx])
    lds_a_ptr.store(val, [idx])

    # finalize INSIDE the GPU module body (before kernel launches)
    comp_ctx = CompilationContext.get_current()
    with ir.InsertionPoint(comp_ctx.gpu_module_body):
        allocator.finalize()
```

## Printf debugging

```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `FLYDSL_DUMP_IR` | false | Dump IR at each pipeline stage |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` | false | Emit DWARF debug info (source-to-asm) |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | true | On-disk kernel cache (auto-invalidates on source change) |
| `FLYDSL_RUNTIME_CACHE_DIR` | `~/.flydsl/cache` | Cache dir |
| `FLYDSL_COMPILE_OPT_LEVEL` | 2 | 0-3 |

## Common compile errors

- `arith.constant` takes `type=...` as a KEYWORD (not positional).
- `buffer_ops.buffer_load` offset is in ELEMENTS (not bytes) for the wrapped
  `buffer_ops` API; the raw ROCDL intrinsic uses bytes.
- Dynamic shapes: use `fx.Int32` not `fx.Constexpr[int]`. Different
  `Constexpr` values produce different compiled kernels.
- Tensor alignment: `flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`.
- `SmemAllocator.finalize()` must be called INSIDE the GPU module body (use
  `CompilationContext.get_current().gpu_module_body`).
- `DLTensorAdaptor` caching bug: do NOT use `flyc.from_dlpack()` with
  pre-wrapped tensors when calling a `@jit` function with varying `Constexpr`
  values — pass raw `torch.Tensor` objects instead.
