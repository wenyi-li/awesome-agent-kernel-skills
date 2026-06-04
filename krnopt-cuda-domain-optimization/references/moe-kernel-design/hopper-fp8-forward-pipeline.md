# Hopper FP8 MoE Forward Pipeline

Use this page when the Hopper MoE question is about the end-to-end FP8 forward path rather than one isolated grouped-GEMM kernel.

The pipeline vocabulary is:

```text
count / gather
  -> grouped GEMM1: gate/up projections
  -> activation: SiLU(gate) * up
  -> FP8 quantization or scale preparation
  -> grouped GEMM2: down projection
  -> routing-weight scale and reduce / combine
```

## Source Basis

- Source/project docs: [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops),
  [SGLang](https://github.com/sgl-project/sglang), and
  [vLLM fused MoE](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/).
- Source/project docs: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
  for SM90 grouped FP8 GEMM and scale-layout contracts.
- Paper/blog/source: [SonicMoE](https://arxiv.org/abs/2512.14080),
  [Tri Dao SonicMoE write-up](https://tridao.me/blog/2026/sonicmoe-blackwell/).
- NVIDIA references for block-scale lineage:
  [cuBLAS 12.9 block-scaled matmul blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/)
  and [CUTLASS](https://github.com/NVIDIA/cutlass).
- Distilled idea: keep the FP8 MoE forward path staged until a measured
  boundary justifies fusion, and treat scale/layout contracts as correctness
  inputs rather than backend details.

## Pipeline Practice Cards

### Count / Gather And Expert Offsets

- Target case: top-k routing creates per-expert token groups and the operator must prepare grouped GEMM metadata.
- Rationale: HPC-Ops exposes `count_and_gather`; vLLM and SGLang build expert offsets and problem sizes before grouped GEMM.
- Pros: clear metadata boundary, easy to validate against a reference, and compatible with staged pipelines.
- Cons / guardrails: explicit gather can dominate HBM traffic; if this is hot, consider fused scatter-GEMM or static-batched metadata.

### Grouped GEMM1: Gate / Up Projection

- Target case: the first expert projection dominates or the gate/up output is the largest intermediate.
- Rationale: Hopper FP8 grouped GEMM can use TMA/WGMMA, but gate/up usually doubles output width and feeds a heavy activation/quantization stage.
- Pros: mature DeepGEMM/CUTLASS/HPC-Ops source anchors, direct tensor-core win, and clean correctness checks.
- Cons / guardrails: the output layout must match activation and GEMM2 input requirements; grouped GEMM alone may only move the bottleneck to activation or combine.

### Activation / Multiply / Re-Quantization

- Target case: `SiLU(gate) * up` plus FP8 conversion or scale generation is a visible boundary between GEMM1 and GEMM2.
- Rationale: HPC-Ops names activation-plus-multiply-plus-quantization as a separate stage, while SonicMoE motivates reducing activation IO for fine-grained MoE.
- Pros: fusing activation with quantization can remove an intermediate pass and prepare GEMM2 input layout earlier.
- Cons / guardrails: activation fusion raises register pressure and scale bookkeeping; keep a fallback split after GEMM1+activation if GEMM2 fusion spills or lowers tensor-core utilization.

### Grouped GEMM2: Down Projection

- Target case: down projection dominates after activation or the GEMM2 input has already been quantized.
- Rationale: SGLang, vLLM, and HPC-Ops all run a second grouped GEMM after activation; it is a separate optimization surface because its M/N/K and scale contracts can differ from GEMM1.
- Pros: can reuse grouped-GEMM scheduling and expert metadata; down output is closer to final token layout.
- Cons / guardrails: routing-weight application and top-k reduction are not automatically solved by GEMM2; store ownership must be explicit for top-k > 1.

### Reduce / Combine

- Target case: final token-order reconstruction, routing-weight scaling, or top-k accumulation is visible in profiles.
- Rationale: MoE outputs may have multiple expert contributions per token, so the final store is a reduction problem, not just a scatter.
- Pros: fusing scale/reduce into the store can remove a post-GEMM pass when ownership is clear.
- Cons / guardrails: atomics, partial output buffers, or token-local reductions can become bottlenecks; correctness must cover duplicate routes, ties, zero-token experts, and top-k > 1.

### Scale And Layout Contracts

- Target case: FP8 grouped GEMM, blockwise scales, or interop with DeepGEMM/CUTLASS/FlashInfer.
- Rationale: SM90 Hopper paths use FP32 scale metadata and `BLOCK_K == 128` style block contracts in the relevant source anchors; scale layout is part of correctness.
- Pros: explicit contracts prevent silent numerical/layout bugs and make backend swaps debuggable.
- Cons / guardrails: scale conversion and transposition can cost more than expected; do not import SM100 packed UE8M0, NVFP4, TMEM, or `tcgen05.mma` assumptions into an SM90 path.

## Validation

For a Hopper FP8 MoE forward plan, validate:

- top-k ids, routing weights, local expert filtering, and empty experts
- GEMM1 and GEMM2 independently against a simple reference
- activation and re-quantization scale compatibility
- duplicate top-k routes and token-local reduction semantics
- balanced, skewed, small-M decode, and large-M prefill shapes
- exact GPU model, build target, CUDA toolkit, driver, workload, and timing command
