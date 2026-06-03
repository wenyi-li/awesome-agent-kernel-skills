# Autotuning Guide

Complete walkthrough for using TileLang's AutoTuner to automatically find optimal kernel configurations.

## Table of Contents

1. [When to Autotune](#when-to-autotune)
2. [Programmatic API (Recommended)](#programmatic-api-recommended)
3. [Generating Config Search Spaces](#generating-config-search-spaces)
4. [Interpreting Results](#interpreting-results)
5. [Complete Example](#complete-example)
6. [Known Issues](#known-issues)

---

## When to Autotune

Autotuning is most valuable when:
- You have a kernel that works but you don't know the optimal tile sizes
- The kernel will be used at multiple problem sizes (different configs may be optimal)
- Manual tuning has plateaued and you want to explore more configurations
- You're deploying to a new GPU architecture

Autotuning is NOT needed when:
- The kernel is memory-bound (tile sizes matter less)
- You already know the optimal config from prior experiments
- The kernel is a one-off experiment (manual tuning is faster for < 5 configs)

## Programmatic API (Recommended)

The programmatic `AutoTuner.from_kernel()` API is the reliable way to autotune. The `@tilelang.autotune` decorator has a cache serialization bug when `ref_prog` is a function -- use the programmatic API instead.

### Basic Usage

```python
from tilelang.autotuner import AutoTuner
import tilelang as tl

# 1. Define your kernel function (must return a @T.prim_func)
@tl.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K, num_stages=2, threads=128,
              dtype=tl.language.float16, accum_dtype=tl.language.float32):
    @tl.language.prim_func
    def kernel(
        A: tl.language.Tensor((M, K), dtype),
        B: tl.language.Tensor((K, N), dtype),
        C: tl.language.Tensor((M, N), dtype),
    ):
        with tl.language.Kernel(
            tl.language.ceildiv(N, block_N),
            tl.language.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = tl.language.alloc_shared((block_M, block_K), dtype)
            B_shared = tl.language.alloc_shared((block_K, block_N), dtype)
            C_local = tl.language.alloc_fragment((block_M, block_N), accum_dtype)
            tl.language.clear(C_local)
            for ko in tl.language.Pipelined(
                tl.language.ceildiv(K, block_K), num_stages=num_stages
            ):
                tl.language.copy(A[by * block_M, ko * block_K], A_shared)
                tl.language.copy(B[ko * block_K, bx * block_N], B_shared)
                tl.language.gemm(A_shared, B_shared, C_local)
            tl.language.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

# 2. Define configs to search
configs = [
    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 256},
]

# 3. Define reference for correctness check
def ref_program(A, B):
    return A @ B

# 4. Create and run autotuner
autotuner = (
    AutoTuner.from_kernel(my_kernel, configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        warmup=3,
        rep=20,
    )
)

result = autotuner.run()
```

### AutoTuner API Reference

**`AutoTuner.from_kernel(kernel_fn, configs)`**
- `kernel_fn`: The JIT-decorated kernel function
- `configs`: List of dicts, each mapping parameter names to values. Only include parameters you want to vary -- fixed parameters (M, N, K, dtype) are passed separately.

**`.set_compile_args(**kwargs)`**
- `out_idx`: Same as `@tilelang.jit(out_idx=...)` 
- `target`: `"auto"` (inferred), `"cuda"`, or specific target string

**`.set_profile_args(**kwargs)`**
- `supply_type`: `tl.TensorSupplyType.Integer` (best for correctness) or `tl.TensorSupplyType.Normal`
- `ref_prog`: Reference function for correctness validation
- `warmup`: Warmup time in ms (default 25)
- `rep`: Repetition time in ms (default 100)

**`.run(**kwargs)`**
- Returns a result object with `.config`, `.latency`, `.kernel`
- Configs that fail to compile or produce incorrect results are automatically skipped

### Result Object

```python
result = autotuner.run()
result.config    # dict: the winning configuration
result.latency   # float: latency in ms
result.kernel    # compiled kernel object, ready to use
```

## Generating Config Search Spaces

### Manual Grid

For a focused search, enumerate specific configurations:

```python
configs = []
for bm in [64, 128]:
    for bn in [64, 128]:
        for bk in [32, 64]:
            for stages in [2, 3]:
                for threads in [128, 256]:
                    configs.append({
                        "block_M": bm, "block_N": bn, "block_K": bk,
                        "num_stages": stages, "threads": threads,
                    })
# 2 * 2 * 2 * 2 * 2 = 32 configs
```

### Practical Config Sets

**Quick search** (3-5 configs, ~5 seconds):
```python
configs = [
    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
]
```

**Thorough search** (16-32 configs, ~30-60 seconds):
```python
configs = [
    {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns, "threads": thr}
    for bm in [64, 128]
    for bn in [64, 128]
    for bk in [32, 64]
    for ns in [2, 3]
    for thr in [128, 256]
]
```

### Filtering Invalid Configs

Some configs exceed shared memory or violate constraints. The autotuner skips configs that fail to compile, but you can pre-filter to save time:

```python
def is_valid_config(cfg, dtype_bytes=2):
    shared = (cfg["block_M"] * cfg["block_K"] + cfg["block_K"] * cfg["block_N"]) * dtype_bytes * cfg["num_stages"]
    return shared <= 228 * 1024  # 228 KB max on Blackwell

configs = [c for c in configs if is_valid_config(c)]
```

## Interpreting Results

### Good Signs

- Best config is 10-50% faster than default (expected for compute-bound kernels)
- Multiple configs are within 5% of each other (stable optimum)
- Best config beats cuBLAS reference

### Warning Signs

- All configs have similar latency → kernel may be memory-bound (tile sizes don't matter much)
- Best config has very large tiles → may not generalize to other problem sizes
- Best config at 4096x4096 is different from best at 1024x1024 → need per-size tuning

### Per-Size Tuning

If your kernel runs at multiple problem sizes, autotune at each size:

```python
for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
    autotuner = (
        AutoTuner.from_kernel(lambda bM, bN, bK, **kw: my_kernel(M, N, K, bM, bN, bK, **kw), configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(supply_type=tl.TensorSupplyType.Integer, ref_prog=ref_program)
    )
    result = autotuner.run()
    print(f"M={M}, N={N}, K={K}: best={result.config}, latency={result.latency:.4f}ms")
```

## Complete Example

```python
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner

M, N, K = 4096, 4096, 4096

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_stages=2, threads=128,
           dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

def ref_program(A, B):
    return A @ B

configs = [
    {"block_M": 64,  "block_N": 64,  "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 256},
]

autotuner = (
    AutoTuner.from_kernel(matmul, configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tilelang.TensorSupplyType.Integer,
        ref_prog=ref_program,
        warmup=3,
        rep=20,
    )
)

result = autotuner.run()
print(f"Best config: {result.config}")
print(f"Best latency: {result.latency:.4f} ms")
print(f"Best TFLOPS: {2 * M * N * K / result.latency * 1e-9:.1f}")

# Use the best kernel
best_kernel = result.kernel
import torch
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = best_kernel(a, b)
```

## Known Issues

### `@tilelang.autotune` Decorator Bug

The decorator syntax has a cache serialization bug when `ref_prog` is a function:

```python
# THIS WILL FAIL with: TypeError: Object of type function is not JSON serializable
@tilelang.autotune(configs=configs, ref_prog=ref_program, ...)
def my_kernel(...):
    ...
```

**Workaround**: Always use the programmatic `AutoTuner.from_kernel()` API instead. It does not have this bug and provides the same functionality.

### Config Compilation Failures

Some configs may fail to compile (e.g., shared memory exceeds limits). The autotuner silently skips these and reports the best among successful configs. If ALL configs fail, you'll get an error -- check that at least one config is valid.

### Timing Variance

For very fast kernels (< 0.1 ms), timing variance between runs can be significant. Increase `rep` to 200+ for more stable results. The autotuner uses the same `do_bench` infrastructure as manual profiling.
