---
id: technique-blackwell-optimization
title: "Blackwell SM100 Kernel Optimization"
type: technique
architectures: [sm100, sm100a]
tags: [blackwell-optimization, tcgen05, tmem, clc, 2sm-cooperative, warp-specialization, ping-pong-scheduling, software-exp, nvfp4, swizzling]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tcgen05-mma, hw-tmem, hw-clc, hw-2sm-cooperative]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, technique-warp-specialization, technique-persistent-kernels, technique-ping-pong-scheduling, technique-software-exp, technique-pipeline-stages, hw-nvfp4, hw-2sm-cooperative]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-flash-attention-4, pr-cutlass-2139, blog-colfax-cutlass, doc-ptx-isa-sm100]
aliases: [Blackwell, B200, B100, GB200, "SM100", "tcgen05", "UMMA", "TMEM"]
---

## Scope

Blackwell covers compute capability 10.x GPUs: B200/B100/GB200/GB300 (`sm100`, `sm100a`). Blackwell is a fundamental rearchitecture of NVIDIA's tensor core pipeline: single-thread MMA issuance, dedicated Tensor Memory (TMEM), Cluster Launch Control (CLC) hardware scheduling, 2-SM cooperative MMA, and native FP4/NVFP4 support.

The dominant optimization challenge on Blackwell is **asymmetric hardware scaling**: tensor core throughput doubled compared to Hopper, but shared memory bandwidth and SFU throughput stayed flat. This shifts the roofline bottleneck from MMA compute to SMEM traffic and non-matmul operations (especially softmax exponentials in attention).

## Architecture Quick Reference

| Resource | Per SM | vs Hopper |
|---|---|---|
| Max warps | 64 | Same |
| Max threads | 2048 | Same |
| Shared memory | 228 KB | Same |
| TMEM | 256 KB | **New** (dedicated accumulator storage) |
| 32-bit registers | 65,536 | Same |
| Tensor cores | 4 (5th gen) | 2× BF16 throughput |
| BF16 tensor core ops/clock | 8192 | 2× (Hopper: 4096) |
| SFU units | 16 | **Same** as Hopper |
| SMEM bandwidth | 128 B/clock | **Same** as Hopper |
| CLC | Yes | **New** (hardware work scheduler) |

| Precision | Peak TFLOPS (B200) |
|---|---|
| NVFP4 | 4500 |
| FP8 | 2250 |
| BF16/FP16 | 2250 |
| TF32 | 1125 |
| FP64 (tensor) | 1125 |

## Critical Asymmetric Hardware Scaling

This is the single most important insight for Blackwell optimization:

| Resource | Hopper H100 | Blackwell B200 | Scaling |
|---|---|---|---|
| BF16 Tensor Cores | 4096 ops/clock/SM | **8192 ops/clock/SM** | **2×** |
| Shared Memory BW | 128 B/clock/SM | 128 B/clock/SM | **1× (unchanged)** |
| Exponential (SFU) | 16 ops/clock/SM | 16 ops/clock/SM | **1× (unchanged)** |
| FMA units | 128/SM | 128/SM | **1× (unchanged)** |

This means SMEM traffic and non-matmul ops (softmax exp, activation functions, reductions) now dominate the forward-pass roofline for many workloads. A kernel that was compute-bound on Hopper may become memory-bound on Blackwell. Roofline analysis must account for these shifted bottlenecks explicitly.

## Decision Table

| Symptom | Blackwell action | Check |
|---|---|---|
| SMEM becoming compute bottleneck | Use 2-SM cooperative MMA (`cta_group::2`), halving per-SM SMEM bandwidth demand | MMA throughput rises; SMEM pressure drops |
| SFU bottleneck on softmax/attention | Replace hardware `ex2.approx` with software-emulated 2^x via FMA polynomial (FA4 technique) | SFU utilization drops; FMA utilization rises |
| Launch overhead dominates small problems | Use CLC persistent kernel with dynamic tile scheduling | Launch overhead → ~20 cycles per tile fetch |
| Tail effect on irregular grids | CLC persistent scheduling or `try_cancel` pattern | Last wave idle time near zero |
| MMA stalled waiting for TMA | Deepen pipeline stages (3-5); verify 128B swizzled SMEM to eliminate bank conflicts | Pipeline bubbles shrink |
| MMA results not visible to epilogue | Insert `tcgen05.mma.fence::before_thread_sync` + `__syncthreads()` before reading TMEM | Epilogue reads correct values |
| Output store bandwidth limited | Use TMA async_store + `stmatrix` + output double buffering | Store bandwidth up to 3× improvement |
| Tile size too small for peak throughput | Increase to m128×n256 for 1-SM, or m256×n256 for 2-SM cooperative | Utilizes full tcgen05 MMA tile |
| Kernel works on SM90 but fails silently on SM100 | Verify 128B swizzle (wgmma works with 64B; tcgen05 does not) | Correct results on SM100 |
| FP4/MXFP4 quantization not reaching peak | Use `kind::f8f6f4` or `kind::mxf4nvf4` with UE8M0 block scales | Throughput matches expected 2× over FP8 |

## tcgen05.mma Programming Model

`tcgen05.mma` is the defining instruction of Blackwell. It replaces Hopper's `wgmma.mma_async` with a fundamentally different model:

| Property | Hopper wgmma | Blackwell tcgen05 |
|---|---|---|
| Issuing scope | Warpgroup (128 threads) | **Single thread** |
| Accumulator storage | Registers (128+ regs) | **TMEM** (256 KB/SM) |
| Operand A source | Registers (ldmatrix) | **SMEM directly** (no ldmatrix) |
| Operand B source | Shared memory (descriptor) | SMEM (descriptor) |
| SMEM swizzle | 64B or 128B | **128B mandatory** |
| Synchronization | commit_group / wait_group | **Async fence-based** |

### Single-Thread Issuance

```cuda
// Only one thread issues the MMA — typically thread 0
__device__ void issue_tcgen05_mma(
    uint32_t tmem_addr,     // TMEM accumulator address
    uint64_t desc_a,        // A operand: SMEM descriptor
    uint64_t desc_b,        // B operand: SMEM descriptor
    int accumulate           // 0 = init, 1 = accumulate
) {
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, %4;"
            :
            : "r"(tmem_addr), "l"(desc_a), "l"(desc_b),
              "r"(0),  // scale descriptor (unused for f16)
              "n"(accumulate)
        );
    }
}
```

### Critical Fences

```cuda
// Before reading MMA results from TMEM, this fence is MANDATORY:
asm volatile("tcgen05.mma.fence::before_thread_sync;" ::);
__syncthreads();  // All threads synchronize after fence
// Now safe to read from TMEM
```

Forgetting the fence produces undefined (stale) values silently.

### SMEM Descriptor Construction

```cuda
// tcgen05 requires 128-byte swizzled shared memory descriptors
__device__ uint64_t make_smem_desc_128b(void* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // Base address (bits 0-13)
    desc |= (uint64_t)(addr >> 4);
    // Leading dimension stride (bits 16-29)
    desc |= (uint64_t)((stride_bytes >> 4) & 0x3FFF) << 16;
    // 128-byte swizzle mode (bits 62-63) — MANDATORY
    desc |= (uint64_t)(3) << 62;
    return desc;
}
```

### MMA Variants

| Variant | A/B Type | Acc | K | Notes |
|---|---|---|---|---|
| `kind::f16` | FP16/BF16 | FP32 | 16 | Standard half-precision |
| `kind::tf32` | TF32 | FP32 | 8 | Training precision |
| `kind::f8f6f4` | FP8/FP6/FP4 | FP32 | 32 | Block-scaled narrow precision |
| `kind::i8` | INT8 | INT32 | 32 | Integer quantized inference |
| `kind::mxf8` | MXFP8 | FP32 | 32 | Microscaling FP8 |
| `kind::mxf4` | MXFP4 | FP32 | 64 | Microscaling FP4 |
| `kind::mxf4nvf4` | NVFP4/MXFP4 | FP32 | 64 | Mixed NVFP4/MXFP4 |

## TMEM (Tensor Memory)

TMEM is a 256 KB/SM dedicated accumulator storage organized as 128 rows × 512 columns (32-bit elements). Eliminates the register pressure that limited Hopper tile sizes.

### TMEM Lifecycle

```cuda
// Allocate at kernel start
uint32_t tmem_acc = tmem_alloc(256);  // 256 columns for 128×256 tile

// Mainloop: MMA writes to TMEM
for (int k = 0; k < num_k_tiles; k++) {
    issue_tcgen05_mma(tmem_acc, desc_a, desc_b, k > 0 ? 1 : 0);
}

// Fence before reading
asm volatile("tcgen05.mma.fence::before_thread_sync;" ::);
__syncthreads();

// Read from TMEM for epilogue
float4 vals = tmem_load_f32x4(tmem_acc + col_offset);

// Deallocate before CTA exit (critical for persistent kernels)
tmem_dealloc(tmem_acc, 256);
```

### TMEM Double-Buffering

```cuda
uint32_t tmem_acc[2];
tmem_acc[0] = tmem_alloc(256);
tmem_acc[1] = tmem_alloc(256);

int buf = 0;
for (int tile = 0; tile < num_tiles; tile++) {
    // MMA on current buffer
    issue_tcgen05_mma(tmem_acc[buf], desc_a, desc_b, 0);
    asm volatile("tcgen05.mma.fence::before_thread_sync;" ::);
    __syncthreads();

    // Epilogue on current buffer (can overlap with next MMA)
    epilogue(tmem_acc[buf], output[tile]);
    buf ^= 1;
}
```

### TMEM Budget

With 512 total columns:

| Tile Size | Columns | Max Buffers |
|---|---|---|
| 128×128 FP32 | 128 | 4 |
| 128×192 FP32 | 192 | 2 |
| 128×256 FP32 | 256 | 2 |
| 256×256 FP32 (2-SM) | 256 | 2 |

## 2-SM Cooperative MMA

Two adjacent CTAs in a thread block cluster cooperatively execute a single MMA. Each SM contributes 128 rows from its TMEM partition, producing a 256×256 output tile in half the time.

```cuda
// 2-SM cooperative MMA
// CTA_group::2 indicates two CTAs share the MMA
// Each CTA stages half of operand B → halved SMEM bandwidth per SM
if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::f16 "
        "[%0], %1, %2, %3, 1;"
        :
        : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0)
    );
}
```

Practical rules:
- 2-SM mode requires thread block cluster configuration.
- Both CTAs must have identical SMEM layouts.
- Most impactful when SMEM bandwidth is the bottleneck (common on Blackwell due to asymmetric scaling).
- Halves per-SM operand-B bandwidth demand.

## CLC Persistent Kernels

CLC (Cluster Launch Control) replaces the traditional grid scheduler with hardware-assisted persistent scheduling. A fixed number of CTAs (equal to SM count) remain resident and dynamically fetch work tiles.

```cuda
__global__ void __launch_bounds__(512)
persistent_gemm_clc(const __grid_constant__ GemmParams params) {
    while (true) {
        TileCoord tile;
        if (!clc_get_next_tile(&tile)) {
            if (clc_try_cancel()) return;  // All work done
            continue;  // Spurious: another CTA may push work
        }
        compute_tile(params, tile.m, tile.n);
    }
}
```

Scheduling latency: **~20 cycles** (CLC) vs **300+ cycles** (traditional grid launch).

## Software-Emulated Exponential

FlashAttention-4's signature technique. Tensor core throughput doubled on Blackwell while SFU throughput stayed flat (16 ops/clock/SM). The softmax exponential becomes the bottleneck.

Solution: Replace hardware `ex2.approx` (SFU) with a 4-FMA polynomial (Cody-Waite range reduction + Horner evaluation):

```cuda
__device__ float software_exp2(float x) {
    // Range reduction: n = rint(x), r = x - n, r in [-0.5, 0.5]
    float n = rintf(x);
    float r = x - n;

    // Horner polynomial for 2^r on [-0.5, 0.5]
    const float c1 = 0.6931471805599453f;   // ln(2)
    const float c2 = 0.2402265069591007f;
    const float c3 = 0.05550410866482158f;
    const float c4 = 0.009618129107628477f;

    float poly = c4;
    poly = fmaf(poly, r, c3);  // FMA 1
    poly = fmaf(poly, r, c2);  // FMA 2
    poly = fmaf(poly, r, c1);  // FMA 3
    poly = fmaf(poly, r, 1.0f);// FMA 4

    return ldexpf(poly, (int)n);
}
```

Key insight: 4 FMAs × 128 FMA units = 512 potential exp ops/clock vs SFU's 16 exp ops/clock = **8× throughput** for the exponential. Latency is similar (~16 cycles for 4 dependent FMAs vs ~20 cycles for ex2), but throughput is the win.

When to use:
- **Attention kernels on Blackwell**: FA4 measured 1.1-1.3× speedup over cuDNN from this technique on B200.
- **Any kernel SFU-limited**: If profiling shows SFU utilization near 100% while FMA is low.
- **Not on Hopper**: SFU-to-MMA ratio is balanced on SM90.

## Ping-Pong Scheduling (FlashAttention-4)

FA4's warp-specialized pipeline with ping-pong scheduling:

```
Time ──────────────────────────────────────────>
Warp 0 (TMA load):    | Q0 load | Q1 load | Q0' load | Q1' load |
Warp 1 (MMA):         | MMA(Q0) | MMA(Q1) | MMA(Q0') | MMA(Q1') |
Warp 2 (softmax):     |         | softmax0|          | softmax1 |
Warp 3 (correction):  |         |         | corr0    | corr1    |
Warp 4-15 (epilogue): |                  | store0   | store1   |
```

Two query tile groups (Q0, Q1) alternate through the pipeline with explicitly managed TMEM buffers. Achieves 1,605-1,613 TFLOPS (71% utilization) on B200 — 1.3× faster than cuDNN 9.13, 2.7× faster than Triton.

## Performance Progression

From the HGEMM case study (4096×4096×4096 BF16 GEMM, B200, cuBLAS baseline = 1763 TFLOPS, peak = 2250):

| Stage | Technique | TFLOPS | % cuBLAS |
|---|---|---|---|
| 1 | Naive CUDA core GEMM | 5 | 0.3% |
| 2 | + TMA + tcgen05.mma | 155 | 8.7% |
| 3 | + 128B swizzled SMEM | 288 | 16.4% |
| 4 | + TMA async_store + stmatrix | 293 | 16.7% |
| 5 | + 2-SM cooperative MMA | 360 | 20.4% |
| 6 | + Software pipelining (double buffer) | 1429 | 81% |
| 7 | + Epilogue pipeline (output double buffer) | 1493 | 85% |
| 8 | + **CLC persistent kernel** | **1772** | **100.6%** |
| 9 | + Thread block swizzle | Final | Tuning |

The final CLC-persistent kernel **surpasses cuBLAS** by 0.6%.

## NVFP4 / FP4 Native Support

Blackwell adds hardware FP4 (E2M1) with UE8M0 block scaling via `kind::f8f6f4`:

```cuda
// NVFP4 MMA with UE8M0 block scales
// E2M1 data (2-bit mantissa, 1-bit exponent) + FP8 block scale per 16 elements
asm volatile(
    "tcgen05.mma.cta_group::1.kind::f8f6f4 "
    "[%0], %1, %2, %3, %4;"
    :
    : "r"(tmem_addr), "l"(desc_a), "l"(desc_b),
      "r"(0),
      "r"(scale_desc)  // UE8M0 block scale descriptor
);
```

Per-tile (1×128 or 128×128) block scaling preserves accuracy while achieving 2× throughput over FP8.

## Compilation and Tooling

- Target `sm_100a` or `sm_100`: `-gencode arch=compute_100,code=sm_100`
- CUDA 12.8+ required for Blackwell features
- Nsight Compute with Blackwell-specific sections for TMEM, CLC, tcgen05 profiling
- CUTLASS 4.5.0+ provides native SM100 schedules: `KernelScheduleSm100CpAsyncWarpSpecialized`, `KernelTmaWarpSpecializedCooperative`

## Anti-Patterns

- Using `wgmma` patterns on Blackwell. tcgen05 is single-thread, TMEM-backed — the programming model is different.
- 64-byte SMEM swizzle. tcgen05 requires 128-byte swizzle; 64-byte produces silent wrong results.
- Forgetting `tcgen05.mma.fence::before_thread_sync` before reading TMEM.
- Not deallocating TMEM in persistent kernels — causes hangs.
- Assuming Hopper occupancy tradeoffs apply. TMEM eliminates register pressure for accumulators.
- Not accounting for asymmetric hardware scaling. Kernels that were compute-bound on Hopper may be SMEM-bound on Blackwell.
- Using hardware `ex2.approx` in attention softmax without checking SFU utilization. The software exp2 technique should be considered when SFU > 80% utilization.
- Launching non-persistent grids for large problems. CLC persistent kernels provide 57%+ improvement over static grids.
