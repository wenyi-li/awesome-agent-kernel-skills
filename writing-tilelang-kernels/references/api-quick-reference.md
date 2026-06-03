# TileLang API Quick Reference (v0.1.9)

---

## Structure

| API | Description |
|-----|-------------|
| `@tilelang.jit(out_idx=[], target="cuda")` | JIT compilation decorator. `out_idx` lists output buffer indices. |
| `@T.prim_func` | Declares a TileLang kernel function. |
| `T.Tensor((shape), dtype)` | Declares a kernel buffer parameter with given shape and dtype. |
| `T.Kernel(grid_x, grid_y, threads=N)` | Opens a launch context defining grid dimensions and thread count. |
| `tilelang.compile(func, target, execution_backend)` | Explicit ahead-of-time compilation (alternative to `@jit`). |

---

## Memory Allocation

| API | Description |
|-----|-------------|
| `T.alloc_shared(shape, dtype)` | Allocate on-chip shared memory (visible to all threads in a block). |
| `T.alloc_fragment(shape, dtype)` | Allocate register-file memory using shared-view notation (warp-private). |
| `T.alloc_local(shape, dtype)` | Allocate thread-private local memory. |
| `T.alloc_var(dtype)` | Allocate a single scalar variable. |
| `T.alloc_global(shape, dtype)` | Dynamically allocate a global memory buffer. |

---

## Data Movement

| API | Description |
|-----|-------------|
| `T.copy(src, dst)` | Copy data between memory scopes. Extents are auto-inferred from shapes. |
| `T.async_copy(src, dst)` | Explicit async copy. Must pair with `T.ptx_wait_group` for synchronization. |

---

## Compute -- GEMM

| API | Description |
|-----|-------------|
| `T.gemm(A_shared, B_shared, C_fragment, transpose_A=False, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)` | Tile-level GEMM via tensor cores. Operands in shared memory, accumulator in fragment. |

---

## Compute -- Reductions

| API | Description |
|-----|-------------|
| `T.reduce_sum(src, dst, dim=N)` | Sum reduction along dimension `dim`. |
| `T.reduce_max(src, dst, dim=N)` | Max reduction along dimension `dim`. |
| `T.reduce_min(src, dst, dim=N)` | Min reduction along dimension `dim`. |
| `T.cumsum(src, dst, dim=N)` | Prefix sum (inclusive scan) along dimension `dim`. |

---

## Compute -- Elementwise Math

| API | Description |
|-----|-------------|
| `T.exp`, `T.exp2`, `T.log`, `T.log2` | Exponential and logarithm functions. |
| `T.rsqrt`, `T.sigmoid` | Reciprocal square root and sigmoid activation. |
| `T.max(a, b)`, `T.min(a, b)` | Element-wise binary max/min. |
| `T.abs` | Absolute value. |
| `T.cast(value, dtype)` | Explicit type conversion. |
| `T.if_then_else(cond, true_val, false_val)` | Branchless conditional select. |

---

## Buffer Operations

| API | Description |
|-----|-------------|
| `T.clear(buf)` | Zero-fill a buffer. |
| `T.fill(buf, value)` | Fill a buffer with a scalar value. |
| `T.atomic_add(dst, value)` | Atomic addition to global memory. |

---

## Loop Constructs

| API | Description |
|-----|-------------|
| `T.Pipelined(iters, num_stages=N)` | Software-pipelined loop. Overlaps data movement with compute across `N` stages. |
| `T.Parallel(dim0, dim1, ...)` | Parallel elementwise loop over the given dimensions. |
| `T.serial(start, stop)` | Sequential loop from `start` to `stop`. |
| `T.unroll(start, stop)` | Compile-time unrolled loop from `start` to `stop`. |
| `T.Persistent(domain, wave_size, index)` | Persistent thread block loop for tile-streaming kernels. |

---

## Annotations and Hints

| API | Description |
|-----|-------------|
| `T.use_swizzle(panel_size=10, enable=True)` | Enable L2 cache swizzle for improved spatial locality. |
| `T.annotate_layout({buf: layout})` | Set explicit memory layout on a buffer. |
| `T.annotate_l2_hit_ratio(buf, ratio)` | Hint expected L2 cache hit ratio for a buffer. |

---

## Constants and Utilities

| API | Description |
|-----|-------------|
| `T.infinity(dtype)` | Typed positive infinity constant. |
| `T.ceildiv(a, b)` | Ceiling division: `ceil(a / b)`. |

---

## Dynamic Shapes

| API | Description |
|-----|-------------|
| `T.dynamic("name")` | Declare a symbolic dimension by name (resolved at launch time). |
| `T.dyn["name"]` | Shorthand alias for `T.dynamic("name")`. |

---

## Data Types

**Floating point:** `T.float16`, `T.bfloat16`, `T.float32`, `T.float64`

**Integer:** `T.int8`, `T.int16`, `T.int32`, `T.int64`

**Unsigned integer:** `T.uint8`, `T.uint16`, `T.uint32`

**FP8 (sub-byte):** `T.float8_e4m3`, `T.float8_e5m2`

---

## Debugging

| API | Description |
|-----|-------------|
| `T.print(obj, msg='')` | Print a value from thread 0 (single-thread debug output). |
| `T.device_assert(cond, msg='')` | Device-side assertion. Fails the kernel if `cond` is false. |

---

## Profiling and Validation

| API | Description |
|-----|-------------|
| `kernel.get_kernel_source()` | Return the generated CUDA source code as a string. |
| `kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)` | Create a profiler with the specified tensor supply strategy. |
| `profiler.do_bench(warmup=25, rep=100, backend="event")` | Benchmark kernel latency (returns time in ms). |
| `profiler.assert_allclose(ref_prog, rtol, atol)` | Check correctness against a reference program within tolerances. |
| `profiler.assert_consistent(repeat=N)` | Run `N` times and check outputs match (detects race conditions). |

---

## TensorSupplyType

| Enum Value | Description |
|------------|-------------|
| `tilelang.TensorSupplyType.Normal` | Random values from a normal distribution. |
| `tilelang.TensorSupplyType.Integer` | Random integers. Best for debugging (exact arithmetic). |
| `tilelang.TensorSupplyType.Randn` | Standard normal distribution (mean=0, std=1). |
| `tilelang.TensorSupplyType.Auto` | Automatic selection based on dtype and context. |
