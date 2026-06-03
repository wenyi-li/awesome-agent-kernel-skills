---
name: optimizing-tilelang-programs
description: >
  How to optimize TileLang GPU kernels for better performance. Use this skill
  whenever the user has a working TileLang kernel that is too slow, wants to tune
  tile sizes or pipeline stages, needs to improve TFLOPS or bandwidth utilization,
  wants to use the AutoTuner, or asks questions like "how do I make my kernel faster",
  "what tile sizes should I use", "how do I autotune", or "why is my kernel slower
  than cuBLAS". Also trigger when the user mentions block sizes, num_stages, threads
  per block, shared memory pressure, occupancy, swizzle, vectorization, split-K,
  or any performance tuning in TileLang context. Even for vague requests like
  "improve performance" or "speed up this kernel" when TileLang is involved.
---

# Optimizing TileLang Programs

## Prerequisites

Before optimizing, ensure:
1. **Correctness is verified** -- use `profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)`
2. **You have a baseline measurement** -- use the **profiling-tilelang-programs** skill to get latency and TFLOPS
3. **You know whether the kernel is compute-bound or memory-bound** -- this determines which optimizations matter
4. **You know the specific bottleneck** -- use ncu to determine if the kernel is tensor core bound, CUDA core (FMA) bound, memory bound, or latency bound (see **profiling-tilelang-programs** skill §Bottleneck Diagnosis and `references/ncu-bottleneck-guide.md` in the profiling skill)

## The Change-One-Thing Method

Optimization is empirical. GPU performance is full of non-obvious interactions between tile sizes, pipeline depth, thread count, and hardware constraints. The only reliable approach:

1. Measure baseline performance
2. Change exactly one parameter
3. Re-measure
4. Record results in a table
5. Keep the change if it helps, revert if it doesn't
6. Repeat

Never change multiple parameters at once -- you won't know which one helped.

### Results Table Template

```
| Config | Latency (ms) | TFLOPS | Notes |
|--------|-------------|--------|-------|
| baseline: bM=128 bN=128 bK=32 stages=2 thr=128 | ... | ... | |
| change bK=64 | ... | ... | +X% ← keep |
| change thr=256 | ... | ... | +Y% from baseline |
| change stages=3 | ... | ... | -Z% ← revert |
```

## Optimization Checklist

Work through these in order. Each builds on the previous. For detailed explanations and code examples, read `references/optimization-checklist.md`.

### 1. Tile Sizes (Biggest Impact)

The single most important parameter. Larger tiles amortize memory access overhead but require more shared memory and registers.

Tile sizes have the largest impact on GEMM throughput. Going from 64x64 to 128x128 typically gives the biggest single improvement.

**Rules of thumb:**
- Start with 128x128 for GEMM kernels
- For memory-bound kernels (elementwise, reduction), smaller tiles (64x64) are fine
- Tile sizes must be powers of 2 for most hardware
- Total shared memory per tile = `(block_M * block_K + block_K * block_N) * dtype_bytes * num_stages`

### 2. Inner Tile (block_K)

Controls the reduction dimension tile. Larger block_K means fewer iterations in the pipelined loop but more shared memory per stage.

Increasing block_K (e.g. 32→64) can improve throughput on compute-bound kernels, at the cost of more shared memory per stage. Always measure — the gain depends on the kernel and GPU.

**When to increase:** When shared memory allows it and the kernel is compute-bound.

### 3. Pipeline Stages (num_stages)

Controls software pipelining depth in `T.Pipelined`. More stages overlap more memory transfers with compute, but each stage costs additional shared memory.

- `num_stages=0`: No pipelining (useful for debugging)
- `num_stages=2`: Double-buffered — good default
- `num_stages=3`: Triple-buffered — may help or hurt depending on shared memory pressure

The difference between 2 and 3 stages is typically small. Always measure on your target GPU.

**Guidelines:**
- `num_stages=2` is a safe default
- `num_stages=3` helps when memory latency is the bottleneck (large K dimension)
- More stages = more shared memory; check that total doesn't exceed GPU limits
- `num_stages=0` disables pipelining entirely (useful for isolating issues)

### 4. Thread Count

Threads per block affects occupancy and parallelism within each block.

Increasing from 128 to 256 threads can improve throughput when tiles are large enough. The gain is modest and GPU-dependent — always measure.

**Guidelines:**
- 128 is a safe starting point
- 256 can help when there is enough work per block
- Higher thread count increases register pressure, potentially reducing occupancy

### 5. L2 Cache Swizzle

`T.use_swizzle(panel_size=10, enable=True)` reorders block execution to improve L2 cache locality for the B matrix in GEMM.

Swizzle helps most when N is large relative to L2 cache size. For small problems the effect is minimal. Try it for larger problems (N >= 8192).

### 6. Memory Optimizations

For memory-bound kernels (elementwise, reductions, normalization):

- **Vectorized loads**: Ensure inner dimensions are multiples of 8 (fp16) for coalesced access
- **Shared memory staging**: Use `T.alloc_shared` even for elementwise ops to enable coalesced global reads
- **Minimize global memory writes**: Fuse epilogue operations into the same kernel

### 7. Epilogue Fusion

Fuse post-GEMM operations (activation, bias, scaling) into the kernel to avoid extra global memory round-trips:

```python
# Instead of: C = matmul(A, B); C = sigmoid(C)  (2 kernels, 2 global writes)
# Do this: fuse sigmoid into the GEMM kernel (1 kernel, 1 global write)
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# Epilogue: applied in registers before writeback
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.sigmoid(C_local[i, j])

T.copy(C_local, C[by * block_M, bx * block_N])
```

### 8. Layout Annotations

For kernels using `T.atomic_add` (e.g., backward pass dQ accumulation), layout annotations can improve performance by matching the fragment layout:

```python
T.annotate_layout({dQ: make_dq_layout(dQ)})
```

This is primarily relevant for backward kernels -- see the **testing-fwd-bwd-kernels** skill.

## Autotuning

When manual tuning is tedious, use TileLang's `AutoTuner` to search over configurations automatically.

**Use the programmatic API** (the `@tilelang.autotune` decorator has a cache serialization bug when passing `ref_prog` as a function):

```python
from tilelang.autotuner import AutoTuner
import tilelang as tl

# Define configs to search
configs = [
    {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 128},
    {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 256},
]

def ref_program(A, B):
    return A @ B

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
print(f"Best config: {result.config}")
print(f"Best latency: {result.latency:.4f} ms")
best_kernel = result.kernel
```

For the full autotuning walkthrough (config generation, search strategies, interpreting results), read `references/autotuning-guide.md`.

## Before/After Benchmarking Template

Always measure before and after optimization. Use this template:

```python
import tilelang
import tilelang.language as T
import torch

M, N, K = 4096, 4096, 4096

def ref_program(A, B):
    return A @ B

# Baseline kernel
baseline = my_kernel(M, N, K, block_M=128, block_N=128, block_K=32)
profiler = baseline.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
base_latency = profiler.do_bench(warmup=25, rep=100, return_mode="median")
base_tflops = 2 * M * N * K / base_latency * 1e-9

# Optimized kernel
optimized = my_kernel(M, N, K, block_M=128, block_N=128, block_K=64)
profiler_opt = optimized.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
profiler_opt.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
opt_latency = profiler_opt.do_bench(warmup=25, rep=100, return_mode="median")
opt_tflops = 2 * M * N * K / opt_latency * 1e-9

# Reference (cuBLAS via PyTorch)
ref_latency = profiler.do_bench(ref_program, warmup=25, rep=100, return_mode="median")
ref_tflops = 2 * M * N * K / ref_latency * 1e-9

print(f"Baseline:  {base_latency:.4f} ms ({base_tflops:.1f} TFLOPS)")
print(f"Optimized: {opt_latency:.4f} ms ({opt_tflops:.1f} TFLOPS) [{opt_tflops/base_tflops:.2f}x]")
print(f"cuBLAS:    {ref_latency:.4f} ms ({ref_tflops:.1f} TFLOPS)")
```

## Important Caveats

- **Optimal config varies by problem size.** A config that beats cuBLAS at 4096x4096 may lose at smaller sizes. Always benchmark at your target production size.

- **Compilation happens once.** Each unique set of compile-time parameters produces a new compiled kernel. Changing tile sizes means recompilation, not just re-running the same code. This is fast (< 1 second for simple kernels) but matters for autotuning.

- **Shared memory limits.** Total shared memory per block must fit within GPU limits (typically 48KB default, up to 164-228KB with opt-in depending on GPU architecture). Formula: `(block_M * block_K + block_K * block_N) * dtype_bytes * num_stages`. If exceeded, the kernel either fails to launch or silently uses slower memory.

- **TMA alignment on Blackwell/Hopper.** Inner dimensions must be multiples of 8 (fp16) or 4 (fp32) due to TMA hardware requirements. This constrains which tile sizes are valid.

## Common Pitfalls

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Changing multiple params at once | Can't tell which change helped | Change one thing, measure, repeat |
| Optimizing at wrong problem size | Config is suboptimal for production | Always benchmark at target sizes |
| Ignoring correctness after changes | Fast wrong answers are useless | Re-run assert_allclose after every change |
| Too many pipeline stages | Shared memory overflow, perf regression | Start with 2, increase only if latency improves |
| Using `@tilelang.autotune` decorator with `ref_prog` | Cache serialization error | Use programmatic `AutoTuner.from_kernel()` API instead |
| Assuming larger tiles always help | Diminishing returns, register pressure | 128x128 is a good starting point for most GEMM |

## Escalation

- Kernel produces wrong results after optimization → use the **debugging-tilelang-programs** skill
- Need accurate baseline measurements → use the **profiling-tilelang-programs** skill
- Writing a new kernel from scratch → use the **writing-tilelang-kernels** skill
- Optimizing backward pass → use the **testing-fwd-bwd-kernels** skill for architecture guidance, then come back here for tuning

For the full optimization checklist with code examples, read `references/optimization-checklist.md`.
For the complete autotuning walkthrough, read `references/autotuning-guide.md`.
