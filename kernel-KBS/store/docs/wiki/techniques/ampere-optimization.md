---
id: technique-ampere-optimization
title: "Ampere SM80/SM86 Kernel Optimization"
type: technique
architectures: [sm80, sm86, sm87]
tags: [ampere-optimization, cp-async, ldmatrix, vectorized-loads, shared-memory-optimization, swizzling, register-budgeting]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-vectorized-loads, technique-pipeline-stages, technique-swizzling, pattern-memory-bound, lang-cuda-cpp, technique-register-budgeting]
sources: [doc-cutile-python-dsl, pr-flashinfer-812, pr-flashinfer-844, pr-vllm-38427, pr-flashinfer-2679, pr-flashinfer-2927, pr-flashinfer-2928, pr-cutlass-2340, doc-nvidia-tuning-guide]
aliases: [Ampere, A100, A10G, A40, "RTX 3090", "SM80", "SM86", "SM87", "cp.async", ldmatrix]
---

## Scope

Ampere covers compute capability 8.x GPUs: A100 (`sm80`), GA10x/GA102-family devices (`sm86`, `sm87`). The canonical optimization pipeline is tile-centric: stage global memory into shared memory, use `cp.async` where available, feed tensor cores with `ldmatrix` + `mma.sync`, and tune occupancy against register/shared-memory pressure.

Ampere GPUs remain the most prevalent datacenter architecture as of 2026, and optimization techniques continue to evolve. This page synthesizes corpus evidence plus community benchmarks that push Ampere kernels beyond cuBLAS.

## Architecture Quick Reference

| Resource | A100 / SM80 | SM86 / SM87 class | Notes |
|---|---|---|
| Max warps | 64 | 48-64 by SKU | Use the actual device query for occupancy limits. |
| Max thread blocks | 32 | 16-32 by SKU | Small CTAs can hit block-count limits before warp limits. |
| Shared memory opt-in | up to ~163 KB/block | ~99 KB/block on common SM86 SKUs | Do not reuse A100 tile/stage choices blindly on GA10x. |
| 32-bit registers | 65,536 | 65,536 | Register pressure is often the first decode/attention limiter. |
| Tensor cores | 4 (3rd gen) | 4 (3rd gen) | FP16/BF16/TF32/INT8/INT4 support varies by product and driver path. |
| L1/SMEM pool | up to 192 KB | smaller on common GA10x SKUs | Configurable partition, but SMEM opt-in caps still matter. |
| Max FP16 TFLOPS | A100-class peak | lower by SKU | Tune against measured roofline, not architecture label alone. |

## Decision Table

| Symptom | Ampere action | Check |
|---|---|---|
| Long scoreboard stalls on tiled GEMM/attention | Add or deepen `cp.async` multi-stage GMEM→SMEM pipeline | Global load dependency stalls drop without raising shared-memory pressure too far |
| Tensor core underuse | Use `ldmatrix` from XOR-swizzled SMEM and `mma.sync` fragments; align register bases to multiples of 4 | Tensor pipe utilization rises; no bank-conflict spike |
| Excessive address computation instructions | Use strided swizzling with precomputed per-thread stride offsets | IMAD/SHF/LOP3 instruction count drops by 90%+ |
| Register pressure in attention/MLA decode | Use asymmetric warp roles or reduce per-thread live fragments | Occupancy improves without spilling |
| Bank conflicts despite padding | Apply XOR-based swizzle (Swizzle<3,3,3>) to shared memory layout | ldmatrix and cp.async both bank-conflict-free |
| `mma.sync` producing extra MOV/IMAD instructions | Align register bases: 4-reg align for D/A/C operands, 2-reg align for B operand | Instruction count matches cuBLAS reference |
| Ampere-specific runtime bug | Dispatch all SM 8x variants explicitly (`sm80`, `sm86`, `sm87`) | A100 and RTX/A-series paths both hit the intended implementation |
| Tile candidates fail during autotune/JIT | Filter by per-SM shared-memory budget before launching tactic probes | Unsupported candidates are skipped cleanly; no sticky CUDA errors |
| CuTe DSL emits Hopper-only PDL PTX | Clamp or reject `enable_pdl=True` on SM < 90 | Ampere norm/fused kernels compile without `griddepcontrol` PTX |
| Memory-bound kernels | Coalesce/vectorize loads; choose cache policy by reuse; use CTA-wide cooperative loads | Memory throughput rises; compute-side tweaks do not dominate |
| Low SM occupancy on small tiles | Reduce per-thread register count via register budgeting; use `__launch_bounds__` | Occupancy rises without spills |

## `cp.async` Pipeline

Use `cp.async` on Ampere to overlap global-to-shared copies with compute. The `cp.async.cg` variant bypasses L1 cache via `LDGSTS.E.BYPASS`, ideal for streaming GEMM data.

```cuda
template <int BYTES>
__device__ __forceinline__ void cp_async_cg(void* smem_dst, const void* gmem_src) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(smem_addr), "l"(gmem_src), "n"(BYTES)
    );
}

template <int STAGES>
__device__ void ampere_copy_pipeline(char* smem, const char* gmem, int tile_bytes) {
    #pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        cp_async_cg<16>(smem + s * tile_bytes, gmem + s * tile_bytes);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}
```

Practical rules:

- Prefer 16-byte aligned copies and coalesced per-warp access.
- Commit after issuing a group of copies; wait only when the consumer stage needs the data.
- If adding stages increases shared-memory usage enough to reduce resident CTAs, benchmark 2-stage vs 3-stage explicitly.
- Use `cp.async.cg` (bypass L1) for streaming GEMM data; use `cp.async.ca` (cache all) only when the data will be reused.

## Tensor Core Mainloop

Ampere tensor-core kernels load matrix fragments from shared memory with `ldmatrix`, then execute `mma.sync`. This is the key contrast with Blackwell `tcgen05`, which reads operands directly from shared memory, and Hopper `wgmma`, which uses warpgroup-scoped async MMA.

```cuda
__device__ __forceinline__ void load_a_frag(uint32_t* a, const void* smem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(uint32_t* d, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}
```

### Register Alignment for ldmatrix and mma

At the SASS level, `LDSM` and `HMMA` have strict register alignment requirements:

- **4-register operands** (D, A, C in HMMA; outputs in LDSM): base register **must be a multiple of 4**
- **2-register operands** (B in HMMA): base register **must be a multiple of 2**

Misalignment forces the compiler to insert `IMAD.MOV.U32` / `MOV` copy instructions, which can be 100x more numerous than in cuBLAS reference kernels. Use `__align__` or explicit register variable ordering to ensure alignment.

## Shared Memory Swizzling (XOR-Based)

Naive row-major SMEM layouts cause bank conflicts on `ldmatrix` (loading 8x8 tiles with 32 threads). XOR-based swizzling eliminates these:

```cuda
// XOR swizzle: Swizzle<3,3,3> on Shape<_64,_8> with Stride<_1,_64>
// Produces both cp.async-bank-conflict-free and ldmatrix-bank-conflict-free layout
//
// Swizzle encoding: B = 3 (log2 of bytes per bank), S = 3 (log2 of row stride in elements),
//                   D = 3 (log2 of number of interleaved rows)
```

### Strided Swizzling (Advanced, 2025+)

Encoding swizzling as explicit per-thread strides rather than recomputed offsets achieves dramatic instruction reduction:

| Metric | Before | After |
|---|---|---|
| Address calculation instructions | Baseline | **-90%** |
| SMEM address registers for LDSM | ~40 | **5** |
| Register spilling | Baseline | **-94%** |

```cuda
// Stride-based swizzle: precompute per-thread stride offsets once
// stride[i] magnitude = 2^i, sign determined by i-th bit of thread_id
__device__ int compute_swizzle_stride(int thread_id, int bit) {
    return (1 << bit) * (((thread_id >> bit) & 1) ? 1 : -1);
}
```

## CTA-Level Cooperative Global→Shared Loads

Instead of having each warp independently load contiguous rows, use all warps in the CTA cooperatively:

```cuda
// CTA-wide cooperative load pattern
// Change stride from per-warp to CTA-wide
// Replaces per-warp __syncwarp() with __syncthreads()
// Reduces SMEM address registers for LDGSTS from ~14 to 1 register
__device__ void cta_cooperative_load(
    half* smem, const half* gmem,
    int tile_m, int tile_k, int lda
) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int total_elems = tile_m * tile_k;

    for (int idx = tid; idx < total_elems; idx += total_threads) {
        int row = idx / tile_k;
        int col = idx % tile_k;
        smem[row * tile_k + col] = gmem[row * lda + col];
    }
    __syncthreads();
}
```

## Performance Progression (Community Benchmarks)

| Optimization Step | % of cuBLAS |
|---|---|
| Direct GMEM load (no SMEM) | 7.6% |
| Hierarchical tile load (GMEM→SMEM→RF) | 54.7% |
| + SMEM padding (bank conflict reduction) | 59.7% |
| + `cp.async` asynchronous copy | 64.2% |
| + GMEM→SMEM double buffer | 76.8% |
| + SMEM→RF double buffer | 77.8% |
| + Multi-stage pipeline + XOR swizzle | **101.9%** |

This progression is validated on SM86 (A10); the same methodology applies to A100 with larger tile sizes (A100 has 164 KB SMEM vs A10's 99 KB).

## Production Portability Guardrails

Ampere optimization is not a single A100 recipe. Production code should preserve the architecture split through dispatch, autotuning, JIT cache keys, and test coverage.

### SM80 Versus SM86/SM87

`sm80` has a larger shared-memory budget than common `sm86`/`sm87` devices. `pr-flashinfer-2927` is the best current corpus example: it moved CUTLASS candidate filtering to a shared `tile_fits_smem(sm, m, n, k_elem, config, stages=2)` predicate and reused that predicate in the Ampere mainloop path.

Practical rules:

- Compute tile SMEM as a function of architecture, tile shape, element size, and pipeline stage count.
- Reject candidates before the tactic probe, not after a launch failure.
- Keep skipped-tactic logging concise in normal runs and detailed only at debug level.
- Synchronize after failed tactic probes in an autotuner so sticky async CUDA errors do not contaminate later CUDA graph capture.

### Hopper-Only Features Must Be Guarded

Programmatic Dependent Launch (PDL/GDC) is not an Ampere feature. `pr-flashinfer-2928` fixed a production failure mode where CuTe DSL norm kernels could emit `griddepcontrol.wait` / `griddepcontrol.launch_dependents` PTX when `enable_pdl=True`, even on Ampere. That PTX requires SM90-or-newer targets.

Guardrail:

```python
def normalize_enable_pdl(enable_pdl, major: int) -> bool:
    if major < 9:
        return False
    return bool(enable_pdl)
```

Use this kind of guard at the API or dispatch boundary before JIT compilation. It is cheaper and clearer than allowing an invalid PTX path to reach ptxas.

### CuTe DSL Example Hygiene

`pr-cutlass-2340` fixed the precompile path for `examples/python/CuTeDSL/ampere/elementwise_apply.py`. Treat Ampere examples as compatibility tests, not disposable samples: if the example precompile path breaks, LLM-generated larger kernels will often fail for tooling reasons before the real kernel logic can be evaluated.

## Attention And MLA Decode

The corpus contains multiple Ampere-facing MLA/attention references:

- `pr-flashinfer-812` generalizes MLA templates for A100 by varying `CTA_TILE_KV` and `NUM_STAGES`.
- `pr-flashinfer-844` adds a CuTe SM80 MLA decode path and mentions asymmetric warp configuration to handle register-file pressure.
- `pr-flashinfer-2679` uses a `cp.async` pipeline in a decode-style state kernel.

Useful starting strategy:

1. Tune `CTA_TILE_KV` / tile-K before changing math.
2. Try 2-stage and 3-stage copy pipelines.
3. Split warp roles when one stage dominates register pressure.
4. Keep the SM80 path separate from SM90/SM100 paths when the instruction pipeline differs.

## Triton On Ampere

For Triton, do not assume all Ampere devices behave like A100. `pr-vllm-38427` specifically widened a batch-invariant Triton matmul path to all SM 8x variants (`sm80`, `sm86`, `sm87`).

Checklist:

- Dispatch on compute capability families, not only `sm80`.
- Keep configs for `sm80` and `sm86/sm87` separable when tile sizes or occupancy differ.
- For matmul, search `BLOCK_M/N/K`, `num_warps`, and `num_stages` together; a larger tile that forces fewer stages can regress.
- Preserve correctness across A100 and RTX/A-series test paths before promoting one autotune winner globally.

### Triton Ampere Tile Tuning Ranges

| Parameter | A100 Range (SM80) | A10/A40 Range (SM86) |
|---|---|---|
| BLOCK_M | 64, 128, 256 | 32, 64, 128 |
| BLOCK_N | 64, 128, 256 | 32, 64, 128 |
| BLOCK_K | 32, 64, 128 | 32, 64 |
| num_warps | 4, 8 | 4, 8 |
| num_stages | 2, 3, 4 | 2, 3 |

## Evidence Map

| Source | Evidence | KBS takeaway |
|---|---|---|
| `pr-flashinfer-812` | A100 MLA template exposes `CTA_TILE_KV` and `NUM_STAGES` | Tune tile-K and pipeline stages together. |
| `pr-flashinfer-844` | SM80 CuTe MLA decode uses asymmetric warp roles | Register pressure can require role allocation changes, not only smaller tiles. |
| `pr-flashinfer-2679` | Decode-style BF16 state kernel uses a `cp.async` pipeline | Use cooperative staging when hidden-state loads dominate. |
| `pr-vllm-38427` | Triton matmul dispatch widened from exact SM80 to all SM8x | Use family predicates for SM8x paths meant for A100, RTX 30xx, A-series, and Orin. |
| `pr-flashinfer-2927` | Ampere mainloop candidates use `tile_fits_smem` | Filter tile/stage candidates against per-SM SMEM limits before probing. |
| `pr-flashinfer-2928` | PDL is clamped off on SM < 90 | Guard Hopper-only PTX before CuTe DSL JIT on Ampere. |
| `pr-cutlass-2340` | CuTe DSL Ampere example precompile path fixed | Keep small Ampere examples healthy as LLM/JIT smoke tests. |

## Evidence-Backed Checklist

- Query by `architecture=sm80`, `sm86`, and `sm87` separately before assuming one Ampere answer.
- Use `cp.async` only on SM80+ paths and keep a non-`cp.async` fallback for older compatibility code.
- Use `ldmatrix` + `mma.sync` register alignment checks for tensor-core kernels; watch generated MOV/IMAD bloat.
- Use per-SM shared-memory predicates for every autotuned tile candidate and every `num_stages` value.
- Gate PDL, TMA, WGMMA, and Blackwell `tcgen05` code away from Ampere dispatch.
- Test A100 plus at least one SM86/SM87 target when promoting a Triton or CuTe DSL Ampere path.

## Numerical Precision Guide

| Format | Mantissa | Exponent | Use Case |
|---|---|---|---|
| TF32 | 10 bits | 8 bits | Training: FP32 range, near-FP16 speed |
| BF16 | 7 bits | 8 bits | Training: same exp range as FP32, no gradient scaling |
| FP16 | 10 bits | 5 bits | Inference: widely supported, adequate for most models |
| INT8 | — | — | Quantized inference: 2x throughput on A100 |

Use `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` or PTX-level `mma.sync` for direct tensor core control.

## Anti-Patterns

- Porting Blackwell `tcgen05` advice back to Ampere. Ampere still needs the `ldmatrix`/register-fragment pipeline.
- Porting Hopper `wgmma` patterns to Ampere. Ampere uses synchronous `mma.sync`, not async warpgroup MMA.
- Treating occupancy as the only goal. Register cuts that create spills often lose even if theoretical occupancy improves.
- Increasing `num_stages` without checking shared-memory residency (A100 limit: ~163 KB/block).
- Targeting only A100 when product code must cover `sm86` or `sm87`.
- Assuming PDL is harmless on Ampere just because a higher-level API exposes `enable_pdl`.
- Letting failed autotune candidates leave sticky async CUDA errors before CUDA graph capture.
- Using naive row-major SMEM without swizzle — bank conflicts can erase 2x+ of tensor-core gains.
- Misaligning register bases for `ldmatrix`/`mma` operands — causes silent instruction bloat.
