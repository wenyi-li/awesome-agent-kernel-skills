# Hopper MoE Backend Selection

Use this page when a Hopper/H20/H100/H800 MoE request asks which backend, kernel family, or technique direction should be tried first.

The main lesson from SGLang, vLLM, DeepGEMM, FlashInfer, CUTLASS, HPC-Ops, and SonicMoE is that Hopper MoE is not one universal kernel. Classify the workload regime before choosing a backend or implementation pattern.

## Source Basis

- Source-mined public repos: [SGLang](https://github.com/sgl-project/sglang)
  and [vLLM fused MoE](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe).
- Source/project docs: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM),
  [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops), and
  [SonicMoE](https://tridao.me/blog/2026/sonicmoe-blackwell/).
- Implementation lineage: CUTLASS grouped GEMM examples, FlashInfer/TRT-LLM
  fused-MoE interfaces, and DeepGEMM SM90 grouped FP8 paths.
- Distilled idea: choose a Hopper MoE regime first. Decode, prefill,
  padding-heavy grouped GEMM, dispatch-heavy pipelines, and full FP8 forward
  fusion need different backends and failure checks.

## Decision Flow

```text
Hopper MoE request
  -> small-M decode?
  -> high-throughput grouped GEMM?
  -> padding-heavy grouped GEMM?
  -> dispatch/permute-heavy?
  -> full FP8 forward pipeline?
  -> expert-parallel communication?
```

Pick the first regime that matches measured evidence. If there is no profile, state the chosen regime as a hypothesis.

## Regime Cards

### Small-M Decode

- Target case: autoregressive decode or low-token batches where many experts receive very small M and token counts may live on GPU.
- Rationale: SGLang exposes `flashinfer_deepgemm` as an SM90 decode-oriented small-M path, while DeepGEMM's masked layout supports valid expert counts that are not host-known.
- Candidate direction: masked grouped GEMM, small-M specialized CUTLASS/FlashInfer/DeepGEMM paths, or static metadata if dispatch copies dominate.
- Pros: avoids pretending decode is the same as prefill, can preserve graph-friendly execution, and avoids host synchronization for group sizes.
- Cons / guardrails: WGMMA underutilization and descriptor/mask overhead are common; a high-throughput grouped GEMM path may be slower for tiny M even if it wins prefill.

### High-Throughput Grouped GEMM

- Target case: prefill, training, or contest-style large routed-token batches where GEMM1/GEMM2 dominate.
- Rationale: vLLM and SGLang both keep CUTLASS/DeepGEMM-style grouped expert GEMM as first-class Hopper paths; HPC-Ops chooses tile-M from expert statistics rather than treating all groups identically.
- Candidate direction: DeepGEMM contiguous grouped FP8, CUTLASS grouped FP8, FlashInfer/CUTLASS wrappers, or an equivalent repo-owned grouped-GEMM path.
- Pros: directly targets tensor-core throughput, uses mature Hopper TMA/WGMMA machinery, and is easier to validate than whole-operator fusion.
- Cons / guardrails: does not remove dispatch, activation, quantization, or combine boundaries; padding and uneven M may dominate after GEMM improves.

### Padding-Heavy Grouped GEMM

- Target case: grouped GEMM dominates but padded rows are material relative to useful routed rows.
- Rationale: TMA-Adaptive shows that Hopper FP8 grouped GEMM can waste meaningful compute/memory when every expert is padded to a fixed M multiple.
- Candidate direction: padding-free grouped GEMM, TMA descriptor pools, residual output descriptors, masked grouped GEMM, or small-M-specific layout changes.
- Pros: saves work exactly where the workload is irregular and composes with grouped GEMM and expert MLP fusion.
- Cons / guardrails: residual handling must not slow the common full-tile path; descriptor-pool mechanics are SM90-specific and may not help balanced large groups.

### Dispatch Or Permute Heavy

- Target case: profiles show dispatch, permutation, unpermutation, row gathering, or combine traffic dominates over GEMM microtiles.
- Rationale: ScatterMoE, static batching, and DashInfer-style metadata all attack token movement instead of only tuning expert GEMM.
- Candidate direction: fused scatter-GEMM, static-batched metadata, sparse dispatch/combine, or explicit pipeline decomposition.
- Pros: can remove HBM round trips and duplicate token buffers that a faster GEMM cannot fix.
- Cons / guardrails: indexed loads can hurt coalescing and complicate TMA; top-k output ownership and reduction semantics must be designed before fusing stores.

### Full FP8 Forward Pipeline

- Target case: the whole MoE operator is being planned or refactored, not just one grouped GEMM call.
- Rationale: HPC-Ops names the stages directly: count/gather, grouped GEMM1, activation/multiply/quantization, grouped GEMM2, and reduce. SonicMoE further motivates reducing activation IO and avoiding unnecessary `O(TKd)` intermediates.
- Candidate direction: keep a staged FP8 pipeline first, then selectively fuse the stages whose traffic or launch cost is measured.
- Pros: creates clear correctness boundaries and lets agents reason about scale/layout contracts between stages.
- Cons / guardrails: full fusion can explode register/shared-memory pressure; keep fallback split points after GEMM1+activation or after GEMM2 until measurements justify deeper fusion.

### Expert-Parallel Communication

- Target case: experts are sharded across GPUs and all-to-all, all-gather, reduce-scatter, or remote writes dominate.
- Rationale: Flux/COMET and distributed MoE libraries optimize communication plus layout/GEMM boundaries rather than only local GEMM throughput.
- Candidate direction: communication-fusion patterns or expert-parallel library modes. For single-GPU local experts, borrow only readiness, queueing, and staged producer/consumer ideas.
- Pros: attacks the real bottleneck in distributed MoE and can reduce idle time at synchronization barriers.
- Cons / guardrails: mostly irrelevant to local single-GPU kernels except as scheduling inspiration; topology and framework integration decide the outcome.

## Selection Output

When giving a backend-selection answer, include:

- selected regime and evidence
- rejected regimes and why they do not fit
- candidate backend or technique page
- expected win mechanism
- main risk or constraint
- fallback if measurement disproves the hypothesis
