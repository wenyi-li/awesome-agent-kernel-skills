---
name: cuda-auto-tune
description: "NCU-driven iterative optimization workflow for CUDA/CUTLASS/Triton/CuTe DSL kernels. MANDATORY: every optimization MUST start with NCU profiling, followed by multi-dimensional analysis, then targeted code modification, then re-profiling to verify. Supports roofline, memory hierarchy, warp stalls, instruction mix, occupancy, divergence analysis. Provides implementation-specific code modifications: Native CUDA (launch config, memory patterns, async copy, Tensor Core), CUTLASS (ThreadblockShape, stages, epilogue, schedule policy, alignment), Triton (autotune params, compiler hints, tl.* API patterns), CuTe DSL (threads_per_cta, elems_per_thread, tiled_copy, copy atom, shared memory, warp/cta reduce). Use when optimizing any CUDA kernel performance."
---

# NCU-driven iterative kernel optimization (CUDA / CUTLASS / Triton / CuTe DSL)

## GATE CHECK (enforce before any optimization)

```
STOP — Do you have NCU profile data for this kernel?
  NO  → Go to Step 1. Do NOT touch any kernel code.
  YES → Go to Step 2.
```

**Hard rules — violation of any rule invalidates the entire optimization:**
- NEVER change kernel code, launch config, or template parameters without NCU data.
- ALL recommendations MUST cite specific NCU metric values as evidence.
- Each iteration MUST cover at minimum: roofline, memory hierarchy, warp stalls, occupancy.
- The optimization playbook MUST match the kernel implementation type.
- After EVERY code change, re-profile and compare with `--diff`.
- Stop iterating when improvements plateau or metrics approach hardware ceiling.

---

## Mandatory optimization loop

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Profile (NCU --set full)                                   │
│      ↓                                                              │
│  Step 2: Multi-dimensional analysis + identify kernel type          │
│      ↓                                                              │
│  Step 3: Apply type-specific playbook (one change per iteration)    │
│      ↓                                                              │
│  Step 4: Re-profile + diff → improved? → loop or stop               │
│      ↑                                           │                  │
│      └───────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Profile with NCU (REQUIRED — no data = no optimization)

### Option A: Profiling script (recommended)

```bash
# Native CUDA / CUTLASS binaries
bash cuda-auto-tune/scripts/ncu_profile.sh ./kernel report_v1

# Triton / Python
bash cuda-auto-tune/scripts/ncu_profile.sh "python your_kernel.py" report_v1

# CuTe DSL / Python
bash cuda-auto-tune/scripts/ncu_profile.sh "python your_cutedsl_kernel.py" report_v1
```

The script collects `--set full` → exports CSV → runs deep analysis → generates reports.

### Option B: Manual profiling

```bash
ncu --set full -o report_v1 --target-processes all ./your_kernel
ncu --import report_v1.ncu-rep --page raw --csv > report_v1.csv
python3 cuda-auto-tune/scripts/ncu_analyse.py report_v1.csv
```

### Kernel-name filters (reduce noise)

```bash
# CUTLASS only
ncu --set full -o report_v1 --target-processes all \
    --kernel-name "cutlass_\|sm90_\|ampere_" ./cutlass_program

# Triton only
ncu --set full -o report_v1 --target-processes all \
    --kernel-name "triton_" "python triton_kernel.py"

# CuTe DSL (kernel name often generic — use --type override in analysis)
python3 cuda-auto-tune/scripts/ncu_analyse.py report_v1.csv --type cutedsl
```

### Expected outputs

```
ncu_reports/
├── report_v1.ncu-rep           # Full binary report
├── report_v1.csv               # Raw metrics CSV
├── report_v1_analysis.md       # Deep analysis report
└── report_v1_summary.txt       # Per-kernel summary
```

---

## Step 2: Multi-dimensional analysis

### 2.1 Identify implementation type

Determine the kernel type from NCU "Function Name" and source context:

| Type | Detection signals |
|------|-------------------|
| **Native CUDA** | No library prefix; hand-written `__global__` functions |
| **CUTLASS** | `cutlass_` prefix, `smXX_xmma_`, contains `tensorop` or `cutlass` |
| **Triton** | `triton_` prefix, contains `triton`, encoded suffixes (e.g. `_0d1d...e`) |
| **CuTe DSL** | Generic names from `@cute.kernel`; confirm via source imports (`cutlass.cute`, `cute.compile`) or `--type cutedsl` |
| **Library** | `cublas*`, `cudnn*` — baseline/reference only, not optimizable |

### 2.2 Common diagnostics (ALL kernel types — always run)

| Dimension | Key NCU metrics | Output |
|-----------|----------------|--------|
| **Roofline** | SM throughput, memory throughput | compute-bound / memory-bound / latency-bound / balanced |
| **Memory hierarchy** | L1/L2 hit rate, coalescing ratio, DRAM throughput | cache efficiency + bandwidth sub-bottleneck (DRAM/L2/L1) |
| **Warp stalls** | PC sampling stall reasons (long_scoreboard, wait, barrier, ...) | top stall reasons with percentages |
| **Instruction mix** | pipe FMA/ALU/LSU/Tensor utilization | pipeline imbalance, Tensor Core usage |
| **Occupancy** | active warps %, limiter breakdown (register/smem/warp/block) | limiting factor + register count + smem size |
| **Memory hazards** | bank conflicts, register spills (local store sectors) | severity and root cause |
| **Divergence** | avg threads executed vs avg threads active (true) | divergence percentage |

### 2.3 Type-specific focus

| Type | Key focus areas |
|------|----------------|
| Native CUDA | launch config (block size, grid), memory access patterns, async copy (cp.async/TMA), Tensor Core opportunity |
| CUTLASS | ThreadblockShape, WarpShape, stages, alignment, schedule policy, epilogue fusion, CTA swizzle |
| Triton | `num_warps`, `num_stages`, `BLOCK_*` sizes, compiler hints (`tl.multiple_of`, `tl.max_contiguous`), `tl.dot` config |
| CuTe DSL | `threads_per_cta`, `elems_per_thread`, CopyAtom (`num_bits_per_copy`), `tiled_copy` layout, smem staging, `cta_reduce` pattern |

### 2.4 Bottleneck classification decision tree

```
SM% > MEM% + 20  →  COMPUTE_BOUND
MEM% > SM% + 20  →  MEMORY_BOUND
  ├─ DRAM throughput > 70%        → DRAM-Bound (near HBM ceiling)
  ├─ L2 hit < 50%, DRAM > 40%    → DRAM-Bound (L2 miss driven)
  ├─ L1 hit < 20%, L2 hit >= 50% → L2-Bound
  └─ L1 hit < 20%                → L1-Bound
SM% < 40 AND MEM% < 40           →  LATENCY_BOUND
SM% > 60 AND MEM% > 60           →  BALANCED (near peak)
```

### 2.5 Conclusion template (REQUIRED after every analysis)

```
=== Conclusion ===
Kernel:    {kernel_name}
Type:      {Native CUDA | CUTLASS | Triton | CuTe DSL}
Arch:      SM_{arch}
Overall:   {COMPUTE_BOUND | MEMORY_BOUND | LATENCY_BOUND | BALANCED}
Duration:  {duration_us} us
Roofline:  SM {sm}%, MEM {mem}%, DRAM {dram}%
Occupancy: {occ}% (theoretical: {theo}%), limited by {limiter}
Regs/Thread: {regs}, Smem/Block: {smem} KB

Findings (sorted by severity):
  [CRITICAL] {finding}: {NCU evidence with numbers} -> {specific action}
  [WARNING]  {finding}: {NCU evidence with numbers} -> {specific action}
  [INFO]     {finding}: {NCU evidence with numbers}

Optimization priorities:
  1. {highest_priority} (expected gain: Nx, evidence: {metric}={value})
  2. {second_priority}  (expected gain: Nx, evidence: {metric}={value})
  3. {third_priority}   (expected gain: Nx, evidence: {metric}={value})
```

---

## Step 3: Apply type-specific playbook

No intuition-only edits. Every change MUST directly address an NCU finding.
Apply ONE change per iteration, then re-profile (Step 4).

---

### 3.1 Playbook: Native CUDA

#### 3.1.1 Launch configuration

| NCU finding | Action | Code pattern |
|------------|--------|-------------|
| Occupancy < 50%, block size < 128 | Increase block size to 128–256 | `kernel<<<grid, 256>>>` |
| Registers are occupancy limiter | Cap registers via `__launch_bounds__` | `__global__ void __launch_bounds__(256, 2) kernel()` |
| Grid too small (< SM count) | Ensure enough blocks for full SM coverage | `grid = (N + block - 1) / block` with sufficient N |
| Occupancy low, blocks limiter | Reduce block size to fit more blocks per SM | Try 128 instead of 256 |

#### 3.1.2 Memory access optimization

| NCU finding | Action | Code pattern |
|------------|--------|-------------|
| Load coalescing ratio > 8 | Ensure warp-contiguous addressing, AoS→SoA | `data[threadIdx.x + blockIdx.x * blockDim.x]` |
| Store coalescing ratio > 8 | Use shared memory staging for scatter writes | Write to smem first, then coalesced writeback |
| L1 hit rate < 20% | Use `__shared__` for frequently reused data | Tile into shared memory with `__syncthreads()` |
| L2 hit rate < 50% | Use L2 persistence hints (Ampere+) | `cudaAccessPolicyWindow` for hot data ranges |
| DRAM throughput > 80% | Reduce data movement: mixed precision, compression | `half` / `__nv_bfloat16` for bandwidth-sensitive ops |
| Bank conflicts > 100K | Pad shared memory or swizzle layout | `__shared__ float smem[32][33];` (pad +1) |
| Register spills > 0 | Reduce per-thread state, use `__launch_bounds__` | Simplify accumulators, split into sub-kernels |

#### 3.1.3 Latency hiding and pipelining

| NCU finding | Action | Code pattern |
|------------|--------|-------------|
| stall_long_scoreboard > 30% (SM>=80) | Use `cp.async` + double buffering | `__pipeline_memcpy_async(&smem, &gmem, size)` |
| stall_long_scoreboard > 30% (SM>=90) | Use TMA for bulk async transfers | `cute::copy(tma_load, ...)` or CuTe TMA atoms |
| stall_barrier > 25% | Reduce sync frequency, use warp primitives | `__shfl_sync()`, `cooperative_groups` |
| stall_wait > 30%, long_scoreboard < 15% | Pipeline over-buffered, reduce depth | Remove one buffer stage |
| stall_math_pipe_throttle > 20% | Compute saturated (positive signal) | Consider Tensor Core or reduce FLOPs |

#### 3.1.4 Tensor Core utilization

| NCU finding | Action |
|------------|--------|
| pipe_tensor < 5%, FP16/BF16 workload with GEMM-like pattern | Use WMMA (`wmma::mma_sync`) or inline PTX (`mma.sync`) |
| pipe_tensor < 5%, but data is FP32 | Use TF32 path via `wmma::mma_sync` with `nvcuda::wmma::precision::tf32` |
| pipe_fma_fp16 > 10%, pipe_tensor < 5% | Switch from scalar FP16 FMA to Tensor Core path |

#### 3.1.5 Vectorized memory access

```
// NCU evidence: coalescing ratio > 4 for 32-bit loads
// Before: scalar loads
float val = input[idx];

// After: vectorized 128-bit load (4x float)
float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
```

---

### 3.2 Playbook: CUTLASS

#### 3.2.1 Kernel config parsing

CUTLASS kernel names encode configuration. Extract:
- Architecture: `sm80_`, `sm90_`, `ampere_`, `hopper_`
- Compute type: `tensorop` vs `simt`
- Tile shape: `128x128x32`, `256x128x64`
- Pipeline stages: trailing `x3`, `x5`
- Alignment: `align8`
- Schedule (3.x): `WarpSpecialized`, `WarpSpecializedCooperative`, `WarpSpecializedPingpong`

#### 3.2.2 Tile shape and occupancy

| NCU finding | Action |
|------------|--------|
| Occupancy < 40%, smem is limiter | Reduce ThreadblockShape (e.g., 256x128→128x128) or reduce stages |
| Occupancy < 40%, registers are limiter | Use smaller WarpShape (e.g., 64x64→32x32) to reduce per-thread regs |
| SM throughput < 30%, grid is small | Increase ThreadblockShape to process more elements per CTA |
| SM throughput > 80%, MEM < 40% | Already compute-bound; increase pipeline stages for more overlap |

#### 3.2.3 Pipeline stages

| NCU finding | Action |
|------------|--------|
| stall_long_scoreboard > 30% | Increase stages (Ampere: 3→5, Hopper: 2→3) |
| stall_wait > 30%, long_scoreboard < 15% | Pipeline over-buffered; reduce stages to save smem |
| Smem limiter + stages > 3 | Reduce stages to free smem for higher occupancy |

#### 3.2.4 Alignment and vectorization

| NCU finding | Action |
|------------|--------|
| Load coalescing > 4, alignment < 8 | Increase CUTLASS alignment to 8 (128 bytes); pad matrix leading dims to multiples of alignment |
| SIMT path used but data supports TensorOp | Switch to `tensorop` CUTLASS configuration (2–8x speedup) |
| TensorOp configured but pipe_tensor < 5% | Check alignment requirements — LD must be multiple of InstructionShape::kK |

#### 3.2.5 Schedule and architecture

| NCU finding | Action |
|------------|--------|
| CUTLASS 2.x on SM>=90 | Upgrade to CUTLASS 3.x with WarpSpecialized + TMA (1.2–1.5x gain) |
| L2 hit rate < 50% on large GEMM | Add ThreadblockSwizzle (2.x: `GemmIdentityThreadblockSwizzle<N>`, 3.x: `StreamK` or tile swizzle) |
| stall_long_scoreboard > 30% on Hopper | Switch to `WarpSpecializedCooperative` schedule with TMA loads |

#### 3.2.6 Epilogue fusion

| NCU finding | Action |
|------------|--------|
| Multiple CUTLASS kernels back-to-back (e.g., GEMM + bias + activation) | Fuse into single kernel via CUTLASS epilogue visitor tree |
| High DRAM traffic (read+write GB > expected) | Move post-GEMM ops into epilogue to eliminate intermediate tensors |

---

### 3.3 Playbook: Triton

#### 3.3.1 Kernel classification

Triton kernel subtypes (from kernel name):
- `triton_poi_`: Inductor pointwise (auto-generated)
- `triton_red_`: Inductor reduction (auto-generated)
- `triton_per_`: Inductor persistent reduction (auto-generated)
- Custom `@triton.jit`: hand-written kernel (fully tunable)

**Inductor-generated kernels**: optimize at PyTorch level (`torch._inductor.config`), or rewrite as custom `@triton.jit` if this is a hot path.

#### 3.3.2 num_warps tuning

| NCU finding | Action |
|------------|--------|
| Registers >= 128, num_warps >= 8 | **CRITICAL**: reduce num_warps (try 4 or 2) |
| Registers >= 64, num_warps >= 8 | Reduce num_warps to 4 |
| Occupancy < 40%, register-limited | Reduce num_warps AND/OR reduce BLOCK_* tile sizes |
| SM throughput < 30%, few warps | Increase num_warps to improve latency hiding |

#### 3.3.3 num_stages tuning

| NCU finding | Action |
|------------|--------|
| stall_long_scoreboard > 30% | Increase num_stages (2→3→4 on Ampere, 2→3 on Hopper) |
| stall_wait > 30%, long_scoreboard < 15% | Decrease num_stages (over-buffered) or increase tile work |
| Smem is occupancy limiter | Decrease num_stages (each stage doubles smem buffer) |
| On Hopper + long_scoreboard high | Also consider `tl.make_block_ptr()` for TMA-based loads |

#### 3.3.4 BLOCK_* tile size tuning

| NCU finding | Action |
|------------|--------|
| Register pressure high | Reduce BLOCK_M, BLOCK_N, or BLOCK_K |
| SM throughput low, compute-bound opportunity | Increase BLOCK_M/BLOCK_N for more compute per tile |
| DRAM bandwidth near ceiling | Increase BLOCK_K for more data reuse before writeback |

#### 3.3.5 Memory access optimization

| NCU finding | Action | Code pattern |
|------------|--------|-------------|
| Load coalescing > 8 | Add stride hints | `tl.multiple_of(stride, 16)` and `tl.max_contiguous(offsets, BLOCK)` |
| Uncoalesced on transposed input | Use structured pointers | `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` |
| L1 hit rate low | Verify access pattern continuity | Ensure innermost dim stride == 1 |

#### 3.3.6 Tensor Core utilization

| NCU finding | Action |
|------------|--------|
| pipe_tensor < 5%, kernel uses `tl.dot` | 1) `allow_tf32=True` for fp32; 2) BLOCK_K multiple of 16; 3) check dtypes are fp16/bf16/tf32/fp8 |
| pipe_tensor < 5%, no `tl.dot` in code | GEMM-like pattern missing — restructure to use `tl.dot` |

#### 3.3.7 Triton autotune integration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(...):
    ...
```

When NCU reveals the bottleneck, narrow autotune configs to the promising region instead of blind search.

---

### 3.4 Playbook: CuTe DSL

#### 3.4.1 Key tuning parameters

| Parameter | Effect | Typical range |
|-----------|--------|--------------|
| `threads_per_cta` | Warps per CTA; affects occupancy, barrier cost, reduce cost | 128–512 |
| `elems_per_thread` | Elements per thread; affects register pressure, data reuse | 4–32 |
| `num_bits_per_copy` | CopyAtom width; affects vectorized load/store bandwidth | 32, 64, 128 |
| Smem staging buffer | Pipeline depth × tile size; affects smem footprint | Minimize for occupancy |

#### 3.4.2 Occupancy optimization

| NCU finding | Action |
|------------|--------|
| Occupancy < 40%, registers are limiter | Reduce `elems_per_thread` or reduce `threads_per_cta`; add `--maxrregcount=128` to `cute.compile()` |
| Occupancy < 40%, smem is limiter | Reduce `threads_per_cta` (fewer warps → smaller reduce buffer) or reduce staging buffer count |
| Registers >= 128, warps >= 8 | **CRITICAL**: reduce `threads_per_cta` to 128 or 256 |

#### 3.4.3 Memory access (TiledCopy)

| NCU finding | Action |
|------------|--------|
| Load coalescing > 8 | 1) Increase `num_bits_per_copy` to 128; 2) verify `t_layout` distributes threads along contiguous addresses; 3) ensure `from_dlpack()` uses `assumed_align=16` |
| stall_long_scoreboard > 30% | 1) Increase `num_bits_per_copy` to 128; 2) increase `elems_per_thread` for more reuse; 3) on SM>=80 use CpAsyncOp copy atom; 4) add double-buffering |
| stall_wait > 30%, long_scoreboard < 15% | Pipeline over-buffered; increase `elems_per_thread` for more compute per stage or reduce pipeline depth |

#### 3.4.4 Synchronization and reduction

| NCU finding | Action |
|------------|--------|
| stall_barrier > 25% | 1) Reduce `threads_per_cta` (fewer warps at barrier); 2) replace second `sync_threads` with shuffle broadcast (if warps <= 32); 3) merge multiple `cta_reduce` calls |
| High barrier + small reduction | Use warp-only reduce without smem for small element counts |
| Multiple sync_threads per iteration | Minimize sync points; use async pipeline commit/wait patterns |

#### 3.4.5 Thread divergence

| NCU finding | Action |
|------------|--------|
| Divergence > 20% | Adjust `threads_per_cta * elems_per_thread` to closely match problem dimension N, reducing predicated-off threads |
| Predicated copies show high divergence | Ensure N is divisible by `threads_per_cta * elems_per_thread` or use tail-handling strategy |

#### 3.4.6 Compute optimization

| NCU finding | Action |
|------------|--------|
| pipe_tensor < 5%, FP16 GEMM-like ops | Use `cute.make_mma_atom()` with MmaOp for Tensor Core path |
| pipe_fma high but pipe_tensor low (non-GEMM ops like RMSNorm/LayerNorm) | Tensor Core not applicable for reductions — focus on memory and barrier optimization instead |

#### 3.4.7 Cache invalidation for re-profiling

CuTe DSL compiles Python to CUDA via JIT. After code changes:
```bash
# Clear compilation cache to ensure re-compilation
rm -rf __pycache__/ .cache/ /tmp/cutlass_cute_cache/
# Then re-profile
bash cuda-auto-tune/scripts/ncu_profile.sh "python your_cutedsl_kernel.py" report_v2
```

---

## Step 4: Re-profile and verify (REQUIRED after every change)

### 4.1 Re-profile

```bash
# Clear JIT caches first
rm -rf ~/.triton/cache            # Triton
rm -rf __pycache__/ .cache/       # CuTe DSL

# Profile updated version
bash cuda-auto-tune/scripts/ncu_profile.sh ./kernel_v2 report_v2
# or
bash cuda-auto-tune/scripts/ncu_profile.sh "python kernel_v2.py" report_v2
```

### 4.2 Compare against baseline

```bash
python3 cuda-auto-tune/scripts/ncu_analyse.py ncu_reports/report_v2.csv --diff ncu_reports/report_v1.csv
```

### 4.3 Verification checklist

| Check | Criteria |
|-------|---------|
| Duration improved? | `gpu__time_duration.sum` decreased |
| Target bottleneck improved? | The specific metric that triggered the change improved |
| No new bottlenecks? | No new CRITICAL findings in the diff report |
| At hardware ceiling? | SM throughput > 80% or DRAM throughput > 85% means near peak |

### 4.4 Iteration log template

Track each iteration for accountability:

```
=== Iteration {N} ===
Change:  {what was changed and why}
NCU evidence: {metric}={before_value} -> {finding}
Report: report_v{N}.csv

Result:
  Duration: {before} us -> {after} us ({delta}%)
  Target metric: {metric}={before} -> {after}
  New findings: {any new issues introduced}

Decision: {CONTINUE to next bottleneck | STOP — at ceiling | ROLLBACK — regression}
```

---

## Quick reference: high-signal NCU metrics

| Metric | NCU key |
|--------|---------|
| Duration | `gpu__time_duration.sum [us]` |
| SM throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed [%]` |
| Memory throughput | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed [%]` |
| DRAM throughput | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed [%]` |
| L1 hit rate | `l1tex__t_sector_hit_rate.pct [%]` |
| L2 hit rate | `lts__t_sector_hit_rate.pct [%]` |
| Load coalescing | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` |
| Register spills | `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum [sector]` |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active [%]` |
| Warp eligibility | `smsp__warps_eligible.avg.per_cycle_active [warp]` |
| Registers/thread | `launch__registers_per_thread [register/thread]` |
| Smem/block | `launch__shared_mem_per_block [Kbyte/block]` |

---

## Summary

This skill enforces a strict **profile → analyze → change → verify** loop.
No NCU data = no optimization. No metric evidence = no code change.
Each kernel type (Native CUDA / CUTLASS / Triton / CuTe DSL) has a dedicated playbook
with NCU-metric-to-action mappings. Every change is tracked and verified by re-profiling.
