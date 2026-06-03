---
name: profiling-tilelang-programs
description: >
  How to benchmark and profile TileLang GPU kernels for performance measurement.
  Use this skill whenever the user wants to measure kernel latency, compare TileLang
  performance against PyTorch/cuBLAS/Triton, use do_bench or get_profiler, calculate
  TFLOPS or bandwidth, run Nsight Compute or Nsight Systems on a TileLang kernel,
  understand occupancy or shared memory usage, or figure out why a kernel is slow.
  Also trigger when the user mentions benchmarking, profiling, timing, throughput,
  performance numbers, or latency in the context of TileLang or GPU kernels. Even
  for simple questions like "how fast is my kernel" or "how do I time this".
---

# Profiling TileLang Programs

## Rule #1: Verify Correctness First

Always validate correctness before profiling. A fast wrong answer is useless.

```python
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
```

## TileLang's Built-in Profiling

### Method 1: Profiler from JIT Kernel (Recommended)

```python
import tilelang

profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

# Basic benchmark
latency = profiler.do_bench(warmup=25, rep=100)
print(f"Latency: {latency:.4f} ms")

# With specific return mode
latency = profiler.do_bench(warmup=25, rep=100, return_mode="median")

# With percentile reporting
quantiles = profiler.do_bench(warmup=25, rep=100, quantiles=[0.5, 0.95, 0.99])
# Returns a list: [median, p95, p99]
```

### Method 2: Benchmark Reference for Comparison

```python
def ref_program(A, B):
    return A @ B

profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

# Verify correctness
profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)

# Benchmark both
tl_latency = profiler.do_bench(warmup=25, rep=100, return_mode="median")
ref_latency = profiler.do_bench(ref_program, warmup=25, rep=100, return_mode="median")

print(f"TileLang:  {tl_latency:.4f} ms")
print(f"Reference: {ref_latency:.4f} ms")
print(f"Speedup:   {ref_latency/tl_latency:.2f}x")
```

### Method 3: Standalone do_bench

```python
from tilelang.profiler import do_bench

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

# With out_idx=[-1]: pass only input args (kernel returns the output)
latency = do_bench(lambda: kernel(a, b), warmup=25, rep=100)

# Without out_idx: pass all args including pre-allocated output
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
latency = do_bench(lambda: kernel(a, b, c), warmup=25, rep=100)
```

### do_bench Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 25 | Warmup time in ms (auto-calculated iteration count) |
| `rep` | 100 | Repetition time in ms (auto-calculated iteration count) |
| `n_warmup` | None | Override: exact number of warmup iterations |
| `n_repeat` | None | Override: exact number of repeat iterations |
| `backend` | `"event"` | Timing backend (see below) |
| `return_mode` | `"mean"` | `"mean"`, `"median"`, `"min"`, `"max"` |
| `quantiles` | None | e.g., `[0.5, 0.95]` -- returns list of floats |

### Timing Backends

| Backend | Method | Best for |
|---------|--------|----------|
| `"event"` | CUDA events | General-purpose timing (default) |
| `"cupti"` | torch.profiler / CUPTI | When you need per-kernel breakdowns |
| `"cudagraph"` | CUDA graph replay | Very fast kernels where launch overhead matters |

`do_bench` automatically flushes the L2 cache (256MB buffer) between iterations for consistent measurements.

### Consistency Check

Detect race conditions by checking that repeated runs produce identical output:

```python
profiler.assert_consistent(repeat=10)
```

## Computing Derived Metrics

### TFLOPS (for GEMM-like kernels)

```python
tflops = 2 * M * N * K / latency_ms * 1e-9
print(f"TFLOPS: {tflops:.1f}")
```

The factor of 2 accounts for multiply + add per element. For GEMM with epilogue (e.g., ReLU, sigmoid), add the epilogue ops but they're usually negligible.

### Effective Bandwidth (for memory-bound kernels)

```python
bytes_moved = (input_elements + output_elements) * bytes_per_element
bandwidth_gb_s = bytes_moved / latency_ms * 1e-6
print(f"Bandwidth: {bandwidth_gb_s:.1f} GB/s")
```

Example for elementwise add (C = A + B, all fp16):
```python
bytes_moved = (M * N * 2 + M * N * 2 + M * N * 2)  # A + B + C, 2 bytes each (fp16)
bandwidth_gb_s = bytes_moved / latency_ms * 1e-6
```

### Roofline Analysis

Compare achieved performance against peak:
- **Compute-bound**: achieved TFLOPS vs peak TFLOPS of the GPU
- **Memory-bound**: achieved bandwidth vs peak memory bandwidth

If achieved TFLOPS is far from peak but bandwidth is near peak, the kernel is memory-bound. Optimize memory access, not compute.

## Complete Benchmarking Template

```python
import tilelang
import tilelang.language as T
import torch

# ... kernel definition ...

M, N, K = 4096, 4096, 4096
kernel = my_kernel(M, N, K, block_M=128, block_N=128, block_K=64)

def ref_program(A, B):
    return A @ B

profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

# 1. Correctness
profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
print("Correctness: PASS")

# 2. Benchmark
tl_latency = profiler.do_bench(warmup=25, rep=100, return_mode="median")
ref_latency = profiler.do_bench(ref_program, warmup=25, rep=100, return_mode="median")

tl_tflops = 2 * M * N * K / tl_latency * 1e-9
ref_tflops = 2 * M * N * K / ref_latency * 1e-9
speedup = ref_latency / tl_latency

print(f"TileLang:  {tl_latency:.4f} ms ({tl_tflops:.1f} TFLOPS)")
print(f"Reference: {ref_latency:.4f} ms ({ref_tflops:.1f} TFLOPS)")
print(f"Speedup:   {speedup:.2f}x")

# 3. Source inspection (optional)
print(f"\nGenerated CUDA ({len(kernel.get_kernel_source())} chars)")
```

## Vendor Profilers

### Nsight Compute (Single-Kernel Deep Dive)

Best for understanding why a specific kernel is slow. Shows occupancy, memory throughput, compute throughput, warp stalls.

```bash
# Basic profile
ncu --target-processes all python script.py

# Full metrics, save report
ncu --set full -o profile_report python script.py

# Open report in GUI
ncu-ui profile_report.ncu-rep
```

**Key metrics to examine:**
- SM Occupancy (%) -- how many warps are active vs maximum
- Memory Throughput (%) -- fraction of peak memory bandwidth achieved
- Compute Throughput (%) -- fraction of peak compute achieved
- L2 Hit Rate -- effectiveness of L2 cache (higher is better)
- Shared Memory Bank Conflicts -- source of wasted cycles
- Register Usage per Thread -- affects occupancy

### Nsight Systems (Timeline Analysis)

Best for understanding kernel launch overhead, CPU/GPU overlap, and multi-kernel workflows.

```bash
nsys profile -o timeline_report python script.py
nsys-ui timeline_report.nsys-rep
```

### compute-sanitizer (Memory Errors)

Not a profiler, but essential for debugging memory issues:

```bash
compute-sanitizer --tool memcheck python script.py
```

### Fallback When Vendor Tools Aren't Available

If ncu/nsys aren't installed, use `do_bench(backend="cupti")` which uses torch.profiler internally:

```python
latency = profiler.do_bench(warmup=25, rep=100, backend="cupti")
```

## Bottleneck Diagnosis with ncu

Beyond basic latency measurement, ncu reveals *why* a kernel is slow. The approach is to find the **shortest stave of the barrel** — whichever hardware resource has the highest utilization relative to its own peak is the bottleneck.

### Quick Pipe Check

```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  --kernel-id ::1: --target-processes all python script.py
```

### How to Read the Output

1. **Compare SM Throughput vs DRAM Throughput** — the higher one indicates whether the kernel is compute-dominated or memory-dominated. If both are low, it's latency-dominated (hardware underutilized).

2. **For compute-dominated kernels, compare compute pipes:**
   - **Tensor pipe** (`sm__pipe_tensor_cycles_active`): tensor core MMA instructions → `T.gemm`
   - **FMA pipe** (`sm__pipe_fma_cycles_active`): scalar/vector FP math → `T.Parallel` elementwise work
   - The pipe with the highest utilization is the compute bottleneck

3. **For memory-dominated kernels**: check L2 hit rate and shared memory bank conflicts to determine whether DRAM, L2, or shared memory is the stave.

4. **For latency-dominated kernels**: check warp stall reasons (`smsp__warp_issue_stalled_*`) — the dominant stall reason tells you what warps are waiting for (barriers, memory, compute backpressure).

Each diagnosis maps to specific TileLang actions (tile sizes, pipeline depth, epilogue fusion, etc.). For the complete decision tree, metric names, and worked examples, read `references/ncu-bottleneck-guide.md`.

## Inspecting Generated CUDA Source

```python
cuda_source = kernel.get_kernel_source()
print(cuda_source)
```

Look for:
- `__launch_bounds__(threads, ...)` -- threads per block and min blocks
- `__shared__` declarations -- total shared memory size
- `#pragma unroll` -- loop unrolling
- Vectorized loads (`uint4`, `float4`) -- memory access width
- Number of registers (check with ncu)

## Common Pitfalls

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Benchmarking without warmup | First run includes JIT compilation time | Use `warmup=25` minimum (do_bench handles this) |
| Manual timing without L2 flush | Artificially fast due to cached data | Use do_bench (flushes automatically) or allocate 256MB flush buffer |
| Comparing different dtypes | Apples-to-oranges comparison | Ensure TileLang and reference use same dtype |
| `out_idx=[-1]` + standalone do_bench with all args | `Kernel expected N inputs, but M provided` | Pass only input args (N-1), kernel returns output |
| Using `cudagraph` for launch-heavy workloads | Underestimates real latency (no launch overhead) | Use `"event"` backend for realistic numbers |
| Benchmarking at wrong problem size | Optimal config varies by size | Always benchmark at target production sizes |
| Single-run timing | High variance from noise | Use median over many repeats |

## Escalation

- Kernel is correct but slow → use the **optimizing-tilelang-programs** skill
- Kernel produces wrong results → use the **debugging-tilelang-programs** skill first
- Need to profile fwd+bwd pair together → use Nsight Systems for timeline view

For detailed GPU metrics reference, read `references/metrics-reference.md`.
For the full ncu/nsys bottleneck diagnosis guide (pipe breakdown, warp stalls, TileLang actions), read `references/ncu-bottleneck-guide.md`.
