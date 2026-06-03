---
id: technique-hopper-optimization
title: "Hopper SM90 Kernel Optimization"
type: technique
architectures: [sm90, sm90a]
tags: [hopper-optimization, wgmma, tma, cluster, fp8, warp-specialization, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-warp-specialization, technique-pipeline-stages, technique-register-budgeting, pattern-compute-bound, lang-cuda-cpp, migration-wgmma-to-tcgen05]
sources: [doc-nvidia-tuning-guide, pr-cutlass-2139, blog-flash-attention-4]
blackwell_relevance: "Hopper wgmma patterns form the baseline that tcgen05 replaces; understanding wgmma → tcgen05 migration context is essential for cross-architecture kernel work."
aliases: [Hopper, H100, H200, H800, "SM90", "SM90a", "wgmma", "Thread Block Cluster"]
---

## Scope

Hopper covers compute capability 9.x GPUs: H100/H200/H800 (`sm90`, `sm90a`). Hopper introduced three major innovations over Ampere: warpgroup MMA (`wgmma`), the Tensor Memory Accelerator (TMA), and Thread Block Clusters with distributed shared memory. Hopper also brought FP8 tensor core support, doubling peak throughput over BF16.

Hopper GPUs remain the dominant training platform in 2026. Optimization patterns center on `wgmma`-driven warp-specialized pipelines, TMA-based async data movement, and cluster-level cooperation.

## Architecture Quick Reference

| Resource | Per SM | Notes |
|---|---|---|
| Max warps | 64 | 2048 threads |
| Max thread blocks | 32 | |
| Shared memory | 228 KB | Configurable up to 228 KB per block |
| 32-bit registers | 65,536 | |
| Tensor cores | 4 (4th gen) | FP16/BF16/TF32/FP64/INT8/FP8 |
| L2 cache | 50 MB | Up from 40 MB on A100 |
| Max cluster size (portable) | 8 SMs | Non-portable: up to 16 on H100 |
| NVLink | 900 GB/s | 18 links × 50 GB/s bidirectional |

| Precision | Peak TFLOPS (H100 SXM) |
|---|---|
| FP8 | 1,979 |
| BF16/FP16 | 989 |
| TF32 | 495 |
| FP64 (tensor) | 67 |
| FP32 (CUDA) | 67 |

## Decision Table

| Symptom | Hopper action | Check |
|---|---|---|
| Compute-bound GEMM not reaching peak | Use `wgmma.mma_async` with warp specialization; TMA for operand staging | Tensor pipe utilization > 70% |
| High register pressure from wgmma accumulators | Increase tile granularity or split accumulators across warpgroups | Occupancy stable; no spills |
| TMA throughput below expectation | Align to 128-byte; use multicast within clusters | HBM bandwidth > 80% peak |
| Pipeline stalls between TMA and MMA | Deepen pipeline stages (3-5); use mbarrier for producer-consumer | Scoreboard stalls drop significantly |
| FP8 accuracy loss | Use E4M3 for forward, E5M2 for backward; per-tensor scaling | Accuracy within 0.5% of BF16 baseline |
| Tail effect on irregular grids | Use CLC equivalent or persistent kernel with dynamic work stealing | Last wave idle time < 5% |
| Cluster DSM underutilized | Redesign SMEM access for cross-SM coalescing; avoid non-unit strides | Cross-SM bandwidth utilization rises |
| L2 cache thrashing | Apply cache policy hints (`evict_first`, `no_allocate`) via PTX | L2 hit rate improves |

## WGMMA Programming Model

`wgmma.mma_async` is Hopper's native tensor core instruction. A warpgroup (4 warps = 128 threads) collectively issues the MMA. This is fundamentally different from Ampere's synchronous `mma.sync` (single warp) and Blackwell's `tcgen05` (single thread).

### Key wgmma Properties

| Property | Value |
|---|---|
| Issuing scope | Warpgroup (4 warps, 128 threads) |
| Typical tile | m64 × nN × k16 (N multiples of 8, max m64×n256×k16) |
| Operand A source | Registers (loaded via ldmatrix) |
| Operand B source | Shared memory (via descriptor) |
| Accumulator storage | Registers (high pressure: 128+ regs per warpgroup) |
| Async model | `commit_group` / `wait_group` |
| SMEM swizzle | 64-byte or 128-byte |

```cuda
// WGMMA warpgroup-coordinated MMA
// All 128 threads in the warpgroup participate
__device__ void hopper_wgmma(
    float* acc,           // Register accumulators
    const uint32_t* a,    // A frag in registers (ldmatrix output)
    uint64_t desc_b       // B descriptor in shared memory
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, "
        "%20, "
        "1, 1, 0, 1;\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b)
    );
}
```

### WGMMA Synchronization

```cuda
// After issuing one or more wgmma operations, commit and wait:
asm volatile("wgmma.commit_group.sync.aligned;" ::);
// ... optionally issue more MMAs in another group ...
asm volatile("wgmma.wait_group.sync.aligned 0;" ::);
// After wait_group 0, all outstanding MMAs are complete
// Accumulators in registers are ready for epilogue
```

Practical rules:

- Keep operand and accumulator fragments live for the shortest useful window.
- Use `__launch_bounds__` only after checking spills and occupancy.
- 64-byte swizzle is common for Hopper SMEM layouts; 128-byte works too. Verify with `ncu` bank conflict counters.
- wgmma output goes to registers — high tile sizes cause register pressure that reduces occupancy.

## TMA (Tensor Memory Accelerator)

TMA is a dedicated hardware unit for async bulk data movement between global memory and shared memory. It supports 1D–5D tensor transfers and SM-to-SM communication within clusters.

### TMA Core Pattern

```cuda
// Single thread issues TMA; all threads wait on mbarrier
__device__ void tma_load_tile(
    const CUtensorMap* tma_desc,
    int tile_m, int tile_n,
    half* smem_dst
) {
    if (threadIdx.x == 0) {
        // Issue async TMA copy
        cp.async.bulk.tensor.2d.shared.global.tile(
            smem_dst, tma_desc, tile_m, tile_n
        );
        // mbarrier arrive signals completion to waiting threads
        mbarrier_arrive(mbar);
    }
    // All threads wait for TMA completion
    mbarrier_wait(mbar, phase);
    phase ^= 1;
}
```

### TMA Multicast

Within a thread block cluster, TMA can multicast data to multiple SMs:

```cuda
// TMA multicast: broadcast tile to all CTAs in cluster
// Reduces total HBM traffic proportionally to cluster size
cp.async.bulk.tensor.2d.shared.global.tile.multicast(
    smem_dst, tma_desc, tile_m, tile_n, multicast_mask
);
```

### TMA Checklist

- Create `CUtensorMap` descriptors with correct box dimensions and swizzle mode.
- 128-byte aligned global memory addresses maximize bandwidth (91% vs 72% at 64-byte alignment).
- Sequential reads achieve ~95% theoretical HBM bandwidth; random access drops to ~15%.
- TMA multicast within clusters can halve HBM traffic for commonly used operands.

## Thread Block Clusters

Hopper introduced a new hierarchy level: thread block clusters. SMs within a cluster can cooperatively load data and share results via distributed shared memory (DSM).

```
Cluster (up to 8 SMs portable, 16 non-portable)
├── SM 0: CTA 0 (shared memory → DSM export)
├── SM 1: CTA 1 (can read CTA 0's DSM via TMA)
├── ...
└── SM N: CTA N
```

DSM access rules:
- Coalesced, 32-byte aligned access patterns.
- Avoid non-unit strides — they fragment cross-SM transactions.
- Use `cudaLaunchCluster` or `cudaLaunchKernel` with cluster attribute.

## FP8 Training on Hopper

FP8 is production-ready on H100/H200 as of 2025. It delivers ~2× throughput over BF16 with equivalent accuracy when properly configured.

| Format | Usage | Mantissa | Exponent |
|---|---|---|---|
| E4M3 | Forward pass, inference | 3 bits | 4 bits |
| E5M2 | Backward pass (gradients) | 2 bits | 5 bits |

```cuda
// FP8 wgmma: requires scale factors
// The scale_desc encodes per-matrix or per-block scaling
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
    "{%0, %1, ...}, {%N, ...}, %desc_b, %scale_a, %scale_b, 1, 1, 0, 1;"
    : "+f"(acc[0]), "+f"(acc[1]), ...
    : "r"(a_frag[0]), ..., "l"(desc_b), "r"(scale_a), "r"(scale_b)
);
```

Practical rules:
- E4M3 for forward activations and weights; E5M2 for backward gradient computation.
- Per-tensor or per-128-block scaling maintains accuracy for most models.
- For optimal tensor core utilization, batch dimensions should be multiples of 16 (94% vs 61% utilization for batch=127).

## Hopper Warp Specialization

Unlike Blackwell's 16-warp CTA with single-thread MMA, Hopper warp specialization centers on warpgroup coordination:

```
Typical Hopper warp-specialized GEMM layout:
Warp 0-3:   Producer warpgroup (TMA loads, SMEM staging)
Warp 4-7:   Consumer warpgroup (wgmma, register accumulators)
Warp 8-11:  Epilogue / output (optional: separate warpgroup)
Warp 12-15: Tile scheduler / prefetch
```

### FlashAttention-3 Pattern

FlashAttention-3 (Tri Dao, 2024) achieves ~75% GPU utilization on H100 via:

1. **Warp specialization**: Separate warps for TMA load, MMA compute, softmax, and epilogue
2. **TMA async transfers**: Overlap data movement with computation
3. **Interleaved block matmul/softmax**: Hides softmax latency behind MMA
4. **FP8 forward path**: 1.5–2× over FA2 BF16 forward

FA3 key metrics on H100:
- 1.5–2× speedup over FlashAttention-2
- ~75% GPU utilization for typical LLM attention shapes
- FP8 forward achieves ~2× throughput over BF16

## Occupancy and Shared Memory Tuning

Hopper has 228 KB SMEM per SM (39% more than A100's 164 KB). This enables deeper pipelines and larger tiles.

| Strategy | SMEM per block | Occupancy | Pipeline |
|---|---|---|---|
| Minimal | 48 KB | 4 blocks/SM | 2 stages |
| Moderate | 72 KB | 3 blocks/SM | 3 stages |
| Aggressive | 114 KB | 2 blocks/SM | 4-5 stages |

Tuning rule: measure, don't assume. A 2-stage pipeline with 4 blocks/SM can beat 5 stages at 2 blocks/SM when compute is not the bottleneck.

## Compilation and Tooling

- Target `sm_90a` or `sm_90` for H100/H200/H800: `-gencode arch=compute_90,code=sm_90`
- CUDA 12.4+ (12.6+ for FP8 features, 12.8+ for latest features)
- Use Nsight Compute `--section-folder-restore` with Hopper-specific sections
- Profile `sm__warps_active.avg.pct_of_peak`, `l1tex__throughput.avg.pct_of_peak`, `dram__throughput.avg.pct_of_peak`

## Anti-Patterns

- Using Ampere `mma.sync` on Hopper. Use `wgmma.mma_async` for peak throughput.
- Using Blackwell `tcgen05` patterns on Hopper. Hopper has no TMEM — accumulators stay in registers.
- Ignoring TMA for data movement. TMA is the fastest path for GMEM→SMEM on Hopper; cp.async is strictly slower.
- Not using thread block clusters for cooperative workloads. DSM and TMA multicast are free throughput multipliers.
- Blindly increasing pipeline stages without checking SMEM pressure. Deeper pipelines can reduce occupancy.
- FP8 training without per-tensor scaling. Unscaled FP8 gradients diverge within hundreds of steps.
- Ignoring L2 cache policy. Hopper's 50 MB L2 is a significant resource — use `evict_first` on streaming data.
