# ncu/nsys Bottleneck Diagnosis Guide

How to use Nsight Compute (ncu) and Nsight Systems (nsys) to identify what limits a TileLang kernel's performance, and what to do about it.

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [The Shortest-Stave Method](#2-the-shortest-stave-method)
3. [Level 1: Compute vs Memory vs Latency](#3-level-1-compute-vs-memory-vs-latency)
4. [Level 2: Compute Pipe Breakdown](#4-level-2-compute-pipe-breakdown)
5. [Level 2: Memory Subsystem Breakdown](#5-level-2-memory-subsystem-breakdown)
6. [Level 2: Latency — Warp Stall Analysis](#6-level-2-latency--warp-stall-analysis)
7. [ncu Metric Names Reference](#7-ncu-metric-names-reference)
8. [Diagnosis → TileLang Action](#8-diagnosis--tilelang-action)
9. [nsys for Multi-Kernel Analysis](#9-nsys-for-multi-kernel-analysis)
10. [Worked Example](#10-worked-example)

---

## 1. Quick Start

Three commands to profile a TileLang kernel:

```bash
# 1. Full ncu analysis (save to file for later inspection)
ncu --set full -o report --target-processes all python script.py

# 2. Quick pipe-level bottleneck check (prints to terminal)
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  --target-processes all python script.py

# 3. Target a specific kernel (skip warmup/JIT kernels)
ncu --kernel-id ::2: --set full -o report python script.py
```

TileLang kernels go through JIT compilation on first invocation, so the first few CUDA kernel launches may be compilation-related. Use `--kernel-id ::N:` to skip to the Nth kernel launch (0-indexed), or `--kernel-name "regex"` to filter by name.

## 2. The Shortest-Stave Method

GPU performance analysis is about finding the **shortest stave of the barrel** — which hardware resource is most saturated relative to its own peak. The kernel's throughput is limited by whichever pipe or subsystem is closest to its ceiling.

Do NOT use fixed threshold numbers (e.g., ">60% means bound"). Thresholds vary across GPUs, architectures, and kernel types. Instead:

1. Collect utilization metrics for all major pipes/subsystems
2. Compare them against each other — the one with the highest % of its own peak is the bottleneck
3. If nothing is near its peak, the kernel is latency-bound (hardware is underutilized)

This method works across all GPU architectures without needing architecture-specific cutoff values.

## 3. Level 1: Compute vs Memory vs Latency

Start with the **GPU Speed Of Light Throughput** section in ncu. It reports two key numbers:

- **SM Throughput** (% of peak): How busy the compute units are
- **DRAM Throughput** (% of peak): How busy the memory system is

Compare these two numbers:

| SM vs DRAM | Classification | Next step |
|-----------|---------------|-----------|
| SM is the higher one | **Compute-dominated** | Go to §4 — which compute pipe? |
| DRAM is the higher one | **Memory-dominated** | Go to §5 — which memory subsystem? |
| Both are low | **Latency-dominated** | Go to §6 — what are warps waiting for? |
| Both are high | **Well-balanced** | Near-optimal; try minor tuning |

## 4. Level 2: Compute Pipe Breakdown

When SM throughput dominates, determine *which* compute pipe is the stave. Modern GPUs have multiple compute pipes that operate in parallel:

### Tensor Pipe (T.gemm)

Metric: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`

The tensor core pipe handles matrix multiply-accumulate (MMA) instructions. In TileLang, this corresponds to `T.gemm`. If the tensor pipe has the highest utilization among compute pipes, your kernel's performance is limited by the rate of tensor core operations.

**What this means for TileLang**: The GEMM is the bottleneck, which is often the desired state for compute-heavy kernels. The epilogue, data movement, and other work are NOT the limiting factor.

### FMA Pipe (T.Parallel elementwise)

Metric: `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`

The FMA (fused multiply-add) pipe handles scalar and vector floating-point operations. In TileLang, this corresponds to elementwise work in `T.Parallel` loops — activation functions, bias addition, scaling, etc.

**What this means for TileLang**: The epilogue or elementwise work is the bottleneck, not the GEMM. Consider fusing operations or reducing the amount of scalar math.

### ALU Pipe (address computation, transcendentals)

Metric: `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active`

Handles integer arithmetic (address calculations, index computation) and special functions (exp, log, rsqrt, etc. via SFU). High ALU utilization often indicates excessive address arithmetic or heavy use of transcendental functions.

**What this means for TileLang**: If your epilogue uses functions like `T.exp`, `T.log`, `T.rsqrt`, `T.sigmoid`, the special function unit may be the stave. Consider whether the transcendental can be approximated or restructured.

### Interpreting the Pipe Breakdown

Compare the three pipe utilizations side by side. The highest one is your compute bottleneck:

```bash
ncu --metrics \
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active \
  --target-processes all python script.py
```

A well-optimized GEMM kernel should show tensor pipe as the dominant pipe. If FMA or ALU dominates, the kernel is spending more time on non-GEMM work than on the matrix multiply itself.

## 5. Level 2: Memory Subsystem Breakdown

When DRAM throughput dominates, determine which level of the memory hierarchy is the stave.

### DRAM (Global Memory)

Metric: `dram__throughput.avg.pct_of_peak_sustained_elapsed`

High DRAM throughput relative to other memory subsystems means the kernel's working set exceeds what the caches can hold. Data is streaming from/to device memory.

**Check L2 hit rate**: `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum / lts__t_sectors_srcunit_tex_op_read.sum`

Low L2 hit rate confirms data is not being reused — it passes through L2 to DRAM on every access.

### L2 Cache

High L2 throughput with moderate DRAM throughput can indicate L2 thrashing — data nominally fits in L2 but the access pattern causes evictions and re-fetches.

### Shared Memory

Metric: `l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed`

High shared memory throughput combined with bank conflicts means the shared memory access pattern is inefficient:

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --target-processes all python script.py
```

Bank conflicts cause serialization — multiple threads in a warp want the same bank simultaneously, so accesses are replayed.

## 6. Level 2: Latency — Warp Stall Analysis

When both SM and DRAM throughput are far from their peaks, the hardware is underutilized. Warps are spending time waiting instead of executing. The **warp stall reasons** tell you what they're waiting for.

```bash
ncu --metrics \
  smsp__warp_issue_stalled_barrier.pct,\
  smsp__warp_issue_stalled_long_scoreboard.pct,\
  smsp__warp_issue_stalled_short_scoreboard.pct,\
  smsp__warp_issue_stalled_math_pipe_throttle.pct,\
  smsp__warp_issue_stalled_mio_throttle.pct,\
  smsp__warp_issue_stalled_lg_throttle.pct,\
  smsp__warp_issue_stalled_not_selected.pct \
  --target-processes all python script.py
```

The dominant stall reason (highest percentage) is the stave:

| Stall Reason | What warps are waiting for | TileLang implication |
|-------------|--------------------------|---------------------|
| `stall_barrier` | `__syncthreads()` or pipeline barriers | Too many synchronization points relative to work per sync. Reduce `num_stages` or increase tile sizes. |
| `stall_long_scoreboard` | Global memory load/store completion | Memory latency not hidden. Increase `num_stages` for deeper pipelining, or increase occupancy. |
| `stall_short_scoreboard` | Shared memory or L1 cache operation | Shared memory access latency. Check for bank conflicts. |
| `stall_math_pipe_throttle` | Compute pipe is busy (backpressure) | Actually compute-bound — reclassify and go to §4. |
| `stall_mio_throttle` | Memory instruction queue is full | Memory pipe saturated at instruction level but DRAM not fully utilized. Tiles may be too small. |
| `stall_lg_throttle` | Local/global memory pipe full | Similar to mio_throttle — memory instruction backpressure. |
| `stall_not_selected` | Scheduler has eligible warps but can't dispatch | Dependency chains within warps, or instruction mix prevents dual-issue. Low ILP. |

### Occupancy

Low occupancy is a common root cause of latency-bound behavior:

```bash
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__registers_per_thread,\
  launch__shared_mem_per_block_driver \
  --target-processes all python script.py
```

If achieved occupancy is far below theoretical occupancy, check for load imbalance between blocks. If theoretical occupancy itself is low, the limiter is either:
- **Registers per thread** (`launch__registers_per_thread`): reduce tile sizes or pipeline depth
- **Shared memory per block** (`launch__shared_mem_per_block_driver`): reduce `num_stages` or `block_K`
- **Block size**: increase `threads` if below 128

## 7. ncu Metric Names Reference

| Diagnostic Question | ncu Metric | Short Form |
|--------------------|-----------|-----------|
| Overall compute utilization | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM % |
| Overall memory utilization | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM % |
| Tensor core pipe utilization | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | Tensor % |
| FMA pipe utilization | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA % |
| ALU pipe utilization | `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | ALU % |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy % |
| Registers per thread | `launch__registers_per_thread` | Regs |
| Shared memory per block | `launch__shared_mem_per_block_driver` | SMEM bytes |
| L2 hit rate | `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum.pct_of_peak_sustained_elapsed` | L2 hit % |
| Shared memory bank conflicts (load) | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Bank conflicts |
| Warp stall: barrier | `smsp__warp_issue_stalled_barrier.pct` | Stall barrier % |
| Warp stall: global memory | `smsp__warp_issue_stalled_long_scoreboard.pct` | Stall gmem % |
| Warp stall: shared memory | `smsp__warp_issue_stalled_short_scoreboard.pct` | Stall smem % |
| Warp stall: compute throttle | `smsp__warp_issue_stalled_math_pipe_throttle.pct` | Stall math % |

Combine multiple metrics in one command using commas:
```bash
ncu --metrics metric1,metric2,metric3 --target-processes all python script.py
```

## 8. Diagnosis → TileLang Action

Once you identify the bottleneck (the shortest stave), apply the corresponding optimization:

| Bottleneck | Evidence (relative pattern) | TileLang Action |
|-----------|---------------------------|----------------|
| Tensor core bound | Tensor pipe is the most-utilized compute pipe | Kernel GEMM is well-optimized. Increase arithmetic intensity: larger tiles, or reduce epilogue work to let T.gemm dominate more. |
| CUDA core (FMA) bound | FMA pipe is the most-utilized compute pipe | Fuse elementwise ops into GEMM epilogue instead of separate kernels. Move work into `T.gemm` where possible. Reduce scalar math in `T.Parallel` loops. |
| ALU / SFU bound | ALU pipe is the most-utilized compute pipe | Transcendental functions (exp, log, rsqrt) dominate. Consider `T.exp2`/`T.log2` (faster HW path), or restructure math. |
| DRAM bound | DRAM throughput is the most-utilized subsystem | Increase tile sizes for data reuse. Use `T.use_swizzle` for L2 locality. Stage through shared memory. Vectorize loads (inner dim multiple of 8 for fp16). |
| L2 bound | L2 throughput high, DRAM moderate, L2 hit rate low | `T.use_swizzle(panel_size=10)`. Increase `block_K` for temporal reuse of A/B tiles. |
| Shared memory bound | Shared memory pipe is the most-utilized; bank conflicts present | Check T.copy and T.gemm operand access patterns. Unusual shared memory layouts may cause bank conflicts. |
| Low occupancy | Achieved occupancy far below theoretical | Reduce registers: smaller tiles, fewer pipeline stages. Reduce shared memory: fewer `num_stages` or smaller `block_K`. |
| Sync overhead | `stall_barrier` is the dominant stall reason | Reduce `num_stages` (fewer barriers per loop iteration). Increase tile sizes (more compute work between syncs). |
| Memory latency | `stall_long_scoreboard` dominates | Increase `num_stages` to overlap more loads with compute. Increase occupancy to give the scheduler more warps to choose from. |
| Launch overhead | nsys shows large gaps between kernel invocations | Fuse TileLang kernels where possible. Use CUDA graphs for inference (`do_bench(backend="cudagraph")`). |

## 9. nsys for Multi-Kernel Analysis

nsys captures a timeline of all GPU activity, which is essential for multi-kernel workflows like TileLang fwd+bwd pipelines.

### Running nsys

```bash
# Basic timeline
nsys profile -o timeline --target-processes all python script.py

# Text summary (no GUI needed)
nsys stats timeline.nsys-rep

# With CUDA API trace for launch overhead analysis
nsys profile --trace=cuda,nvtx -o timeline python script.py
```

### Reading the Summary

`nsys stats` prints tables showing per-kernel and per-API-call statistics:

```
CUDA Kernel Statistics:
  Time(%)  Avg(us)  Instances  Name
  85.3     23.4     100        my_kernel_0x...
  14.7     4.1      100        another_kernel_0x...

CUDA API Statistics:
  Time(%)  Avg(us)  Instances  Name
  45.2     123.4    1          cudaMalloc
  32.1     0.8      200        cudaLaunchKernel
```

### What to Look For

| Pattern | What it means | Action |
|---------|--------------|--------|
| One kernel dominates Time(%) | That kernel is the optimization target | Profile it with ncu for pipe-level analysis |
| Large gaps between kernel bars in GUI timeline | CPU overhead between launches (Python, JIT, memory allocation) | Pre-allocate buffers; move setup outside hot loop |
| Many tiny kernels (Avg < 10 μs) | Launch overhead dominates total time | Fuse operations into fewer TileLang kernels |
| `cudaMalloc` taking significant time | Memory allocation during computation | Pre-allocate all tensors before the compute loop |
| First kernel invocation much slower | TileLang JIT compilation overhead | Normal — ensure warmup in benchmarks |
| GPU idle while CPU is busy | CPU is the bottleneck (data prep, Python overhead) | Overlap CPU/GPU work; reduce Python in hot path |

### Identifying the Bottleneck Kernel in a Pipeline

For fwd+bwd TileLang workflows with multiple kernels:

1. Run `nsys stats` to see which kernel takes the most wall time
2. Profile that specific kernel with ncu: `ncu --kernel-name "kernel_name" --set full python script.py`
3. Apply the shortest-stave analysis from §2-§6
4. Re-run nsys to verify the bottleneck shifted

### Comparing Event vs CUDAGraph Timing

If `do_bench(backend="event")` gives significantly higher latency than `do_bench(backend="cudagraph")`, the difference is kernel launch overhead. Solutions:
- Fuse TileLang kernels (combine fwd phases into one kernel where possible)
- Use CUDA graphs for inference deployment
- Increase kernel granularity (larger tiles = fewer blocks = fewer launches for multi-kernel scenarios)

## 10. Worked Example

### Scenario: GEMM + Sigmoid Fusion Kernel

A TileLang kernel does `C = sigmoid(A @ B)` and you want to know what's limiting its performance.

### Step 1: Quick pipe check

```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  --kernel-id ::1: --target-processes all python gemm_sigmoid.py
```

Suppose the output shows:
```
SM Throughput:      72%
DRAM Throughput:    35%
Tensor pipe:        45%
FMA pipe:           68%
```

### Step 2: Interpret

- SM throughput (72%) > DRAM throughput (35%) → **compute-dominated**
- FMA pipe (68%) > tensor pipe (45%) → **FMA is the stave**, not tensor cores
- The sigmoid epilogue is consuming more compute cycles than the GEMM itself

### Step 3: Diagnose

The sigmoid function (`1 / (1 + exp(-x))`) involves exp and division, which go through both FMA and ALU pipes. The FMA pipe is more saturated than the tensor pipe, so the epilogue is the bottleneck.

### Step 4: TileLang action

Options to reduce FMA pressure:
- **Increase tile sizes**: larger tiles mean more T.gemm work per sigmoid element, shifting the ratio toward tensor cores
- **Simplify the epilogue**: if approximate sigmoid is acceptable, use a polynomial approximation instead of the full exp-based sigmoid
- **Increase block_K**: more GEMM iterations per epilogue application

### Step 5: Verify

Re-run the same ncu command after the change. If the tensor pipe is now the highest-utilized compute pipe, the epilogue is no longer the bottleneck — the optimization worked.
