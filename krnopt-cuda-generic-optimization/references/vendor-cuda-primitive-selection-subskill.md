# Vendor CUDA Primitive Selection

Use this reference when the diagnosed CUDA hotspot looks like a manual implementation of a standard primitive. The default optimization posture is:

```text
1. Precompiled vendor/runtime libraries
   cuBLAS, cuBLASLt, cuDNN, NCCL, framework/runtime kernels

2. CCCL header-only CUDA core primitives
   CUB, Thrust, libcu++ / cuda:: APIs

3. Header/template or generated-kernel libraries
   CUTLASS/CuTe, CUTLASS DSL, TileLang/CuTile-style generators,
   domain wrappers such as DeepGEMM or FlashInfer

4. Handcrafted CUDA kernels
   hand-written SIMT, WGMMA/TMA, fused, persistent, or specialized kernels
```

Prefer the highest tier that supports the required operation and contract. Move down the ladder only when a higher tier lacks a required feature, fails measured performance on the real workload, cannot satisfy layout or scale contracts, cannot be integrated in the build or binding path, or would prevent a deeply customized fused operator that is central to the win.

CCCL is the modern umbrella for CUB, Thrust, and libcu++. These components are header-only and are normally available through the CUDA Toolkit include path when compiling with `nvcc`; recommending them is usually an include/API choice, not a link-time or runtime-library dependency. Keep using concrete API names when they are the actionable choice: CUB for device/block/warp collectives and primitives, Thrust for high-level parallel algorithms, and libcu++/`cuda::` for CUDA-aware standard library facilities such as atomics, barriers, pipelines, spans, and memory utilities.

## Pattern To Primitive Map

| Manual code shape | First candidates | Notes |
|---|---|---|
| Dense GEMM / matmul | cuBLAS, cuBLASLt | Start with precompiled BLAS when shapes, dtype, epilogue, and layout fit. |
| Batched, grouped, or strided GEMM | cuBLASLt grouped/batched APIs when supported, then CUTLASS/CuTe or domain grouped-GEMM wrappers | For MoE, route backend/regime choice to `krnopt-cuda-domain-optimization` if the question is not a local primitive swap. |
| FP8 or block-scaled GEMM | cuBLASLt support when available, then CUTLASS/CuTe, CUTLASS DSL, TransformerEngine, DeepGEMM, or FlashInfer-style paths | Check scale layout, architecture target, accumulator contract, and conversion overhead before accepting. |
| Scan, prefix sum, reduction, histogram, sort, select, compaction | CCCL CUB or framework primitives | Do not hand-roll common routing plumbing until CUB shape, temporary storage, and stream constraints are checked. |
| High-level vector-style transforms, reductions, sorting, and scans outside a fused kernel | CCCL Thrust or framework primitives | Check launch count, iterator/materialization overhead, allocator behavior, and stream policy before accepting. |
| In-kernel block/warp collectives, atomics, barriers, pipelines, spans, and CUDA-aware standard library utilities | CCCL CUB and libcu++/`cuda::` APIs | Prefer maintained CUDA core building blocks over local ad hoc versions when they fit the scope and compiler/toolkit version. |
| Convolution, normalization, frontend-fusible matmul-like blocks | cuDNN frontend when it fits | cuDNN can own both primitive choice and supported fusion patterns, but only inside its operation graph. |
| Collective communication | NCCL or framework distributed primitives | Handcrafted communication should require topology, fusion, or protocol evidence. |
| Tiny-M, irregular, custom epilogue, or deeply fused operator | Try the highest viable tier, then consider CUTLASS/CuTe or handcrafted kernels | Small shape overhead, descriptor cost, and conversion traffic can make standard paths lose. |

## Accept / Reject Gates

Before recommending a library-backed path, state:

- operation and dtype/scale contract
- target architecture and required build flags
- for CCCL/header-only choices, the required headers, minimum CUDA Toolkit or CCCL version, and whether `nvcc` already provides the include path
- expected data layout conversion or packing cost
- workspace ownership and whether the caller can provide it
- stream compatibility and hidden synchronization risk
- correctness and numerical tolerance gate
- timing workload and accept/reject threshold

Reject or defer a higher-tier primitive when:

- it does not support a required dtype, scale layout, epilogue, grouping mode, or dynamic-shape contract
- layout conversion, quantization, dequantization, or pointer setup erases the expected win
- tiny-M, skewed groups, or irregular scheduling make library overhead dominate
- the required build, include, link, or runtime-loading path is unavailable
- the measured path is slower on the real workload
- the optimization depends on fusing producer, consumer, routing, or reduction stages in a way the standard primitive cannot express

## Routing Boundary

Stay in `krnopt-cuda-generic-optimization` when the decision is a diagnosed local primitive mismatch and one primary experiment can test a standard-library-style substitution.

Escalate to `krnopt-cuda-structural-optimization` when the substitution changes hot-path boundaries, stage order, materialization, ownership, scheduling, or decomposition across multiple kernels.

Escalate to `krnopt-cuda-domain-optimization` when the question is a MoE or other workload-domain backend/regime choice, such as decode versus prefill grouped GEMM, dispatch/combine strategy, expert MLP fusion, padding strategy, or persistent scheduling.

Treat library integration mechanics as implementation checks: stream binding, handle lifetime, descriptors, heuristics, workspace, linker flags, runtime loading, and hidden synchronization checks. Route source edits through `krnopt-cuda-coding` once the primitive choice is made.

## Embedded Checks

- Treat library availability as a build and binding fact, not an assumption. Verify headers and version compatibility for CCCL components; verify linked libraries, runtime loading, handle lifetime, stream use, workspace, and architecture flags for runtime libraries such as cuBLASLt, cuDNN, and NCCL; verify include paths and target architecture flags for CUTLASS/CuTe or generated kernels.
- For grouped GEMM and MoE-like work, include routing, gather/scatter, padding, expert imbalance, scale layout, and descriptor setup in the cost model. A faster GEMM primitive can still lose if it increases surrounding movement or launch overhead.
- For tiny, ragged, or heavily fused operators, benchmark the standard primitive on the real workload before accepting it. Descriptor setup, packing, data-layout conversion, and temporary storage can dominate even when the primitive is strong in isolation.
- Local upstream checkouts can be useful as reference material, but do not make them a dependency unless the task explicitly asks to vendor, wrap, or reimplement that code in repo-owned files.
