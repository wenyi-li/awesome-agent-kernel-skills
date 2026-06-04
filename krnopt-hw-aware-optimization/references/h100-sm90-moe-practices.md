# H100 / H20 SM90 MoE Practices

Use this reference when the target is H100, H20, H800, or another SM90-family Hopper GPU and the workload is an MoE kernel, grouped expert GEMM, FP8 fused MoE forward path, or MoE dispatch/GEMM pipeline.

This page is hardware-aware guidance. Use the domain skill for choosing the MoE boundary pattern first, then use this page to keep the Hopper implementation mechanics and generation boundaries correct.

## Embedded Basis

This page embeds the relevant Hopper MoE guidance directly. The recurring patterns are grouped FP8 GEMM, masked small-M decode paths, static or persistent grouped scheduling, runtime TMA descriptor handling, FP8 scale/layout contracts, static-batched metadata, and staged MoE forward decomposition.

Local upstream checkouts can still be useful for comparison, but this reference should be sufficient to choose the Hopper-specific practice card without depending on a separate knowledge base.

## Source Basis

- Hopper grouped-GEMM baselines:
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM),
  [CUTLASS](https://github.com/NVIDIA/cutlass), and
  [PyTorch persistent cache-aware grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/).
- Serving/runtime source-mined basis:
  [SGLang](https://github.com/sgl-project/sglang),
  [vLLM fused MoE](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe),
  and [Tencent HPC-Ops](https://github.com/Tencent/hpc-ops). These are
  public source references, not required local checkouts.
- Hopper irregular-work techniques:
  [Static Batching of Irregular Workloads on GPUs](https://arxiv.org/abs/2501.16103)
  and [TMA-Adaptive FP8 Grouped GEMM](https://arxiv.org/abs/2508.16584).
- Cross-generation context:
  [SonicMoE](https://arxiv.org/abs/2512.14080) and the
  [SonicMoE write-up](https://tridao.me/blog/2026/sonicmoe-blackwell/).
- Distilled idea: keep Hopper MoE guidance centered on SM90 TMA/WGMMA,
  register-resident accumulators, FP32 scale metadata, grouped scheduling,
  and explicit decode/prefill regime splits. Do not import SM100 TMEM,
  `tcgen05.mma`, CLC, or packed Blackwell scale contracts into an SM90 plan.

## What To Establish First

- exact GPU model, compute capability, CUDA toolkit, driver, build target, and whether the path is explicitly SM90a
- whether the request is decode, prefill, training, fixed contest geometry, or an operator-library path
- token histogram per expert, local expert count, top-k, hidden/intermediate sizes, dtype, and scale layout
- time spent in routing/dispatch, grouped GEMM1, activation/quantization, grouped GEMM2, reduce/combine, and communication
- padded rows versus useful routed rows, especially residual M tiles per expert

If these facts are missing, phrase any recommendation as a hypothesis and name what must be measured.

## Practice Cards

### DeepGEMM SM90 Grouped FP8 Baseline

- Target case: H20/H100 MoE grouped GEMM where expert weights share N/K shape and M varies by routed token count.
- Rationale: DeepGEMM's SM90 path exposes compact Hopper TMA/WGMMA grouped FP8 kernels with M-grouped contiguous and masked layouts, FP32 scale metadata, and explicit scheduler heuristics.
- Use when: GEMM1/GEMM2 dominates and the workload can satisfy M alignment, scale layout, and fixed expert-shape contracts.
- Pros: strong source-backed Hopper baseline, good for high-throughput grouped GEMM, masked path covers decode/CUDA-graph cases where valid M counts live on GPU.
- Cons / guardrails: does not solve dispatch or combine by itself; contiguous layout needs alignment; SM100 Mega MoE and UE8M0 scale contracts are not Hopper implementation surfaces.

### Small-M Decode / Masked Grouped GEMM

- Target case: decode-like workloads where many experts receive tiny M and per-expert counts are known on GPU rather than the host.
- Rationale: SGLang distinguishes SM90 small-M decode from high-throughput grouped GEMM, and DeepGEMM's masked grouped layout is designed for GPU-resident valid counts.
- Use when: CPU shape discovery, per-expert launches, or graph-capture constraints are visible costs.
- Pros: avoids assuming host-known contiguous group sizes, can keep decode paths graph-friendly, and separates small-M policy from prefill policy.
- Cons / guardrails: tiny groups can underfill WGMMA; descriptor and mask overhead can dominate; high-throughput CUTLASS/DeepGEMM contiguous paths may win once M is large enough.

### Persistent L2-Aware Grouped Scheduling

- Target case: many uneven expert tiles need one resident schedule and simple grouped scheduling leaves SMs idle or loses L2 locality.
- Rationale: Hopper grouped-GEMM sources use persistent tile loops, grouped launch ordering, runtime TMA descriptor setup, and scheduler heuristics as part of the kernel design.
- Use when: launch overhead, wave quantization, scheduler imbalance, or poor expert-weight L2 reuse is measured.
- Pros: reduces one-kernel-per-expert overhead, balances uneven tiles, and can improve weight locality by revisiting expert tiles intentionally.
- Cons / guardrails: branchy mapping can erase gains; persistent residency competes with registers and shared memory; bad tile ordering can reduce cache locality.

### Runtime TMA Descriptor Mutation And Descriptor Pools

- Target case: grouped GEMM where operand base addresses, strides, expert weights, or residual output M sizes vary by group.
- Rationale: Hopper TMA uses descriptor-driven movement, so irregular grouped GEMM either mutates descriptors, builds descriptor pools, or falls back to pointer-driven movement for tails.
- Use when: descriptor work replaces larger padding, repacking, or host setup costs.
- Pros: keeps Hopper TMA/WGMMA viable for dynamic expert shapes; descriptor pools isolate residual-tile complexity from the full-tile path.
- Cons / guardrails: descriptor visibility, barriers, and phase progression become correctness-critical; descriptor churn can dominate small or jagged decode tiles.

### FP8 Scale And Alignment Contracts

- Target case: FP8 MoE grouped GEMM or fused MoE forward paths on SM90.
- Rationale: DeepGEMM and TMA-Adaptive both make scale layout part of the kernel contract: SM90 uses FP32 scale metadata, `BLOCK_K == 128` style scaling, and TMA alignment constraints.
- Use when: planning FP8 GEMM1/GEMM2, activation re-quantization, or layout interop with DeepGEMM, CUTLASS, FlashInfer, SGLang, or vLLM.
- Pros: prevents silent layout/scale mistakes and keeps tensor-core paths compatible with Hopper kernels.
- Cons / guardrails: scale transposes, alignment padding, and FP32 metadata add their own traffic; SM100 packed UE8M0 or FP4/NVFP4 contracts must not be copied into SM90 code.

### Static-Batched Metadata And Row-Index Loads

- Target case: dispatch copies are expensive but a full dynamic persistent operator is too large a change.
- Rationale: static batching and DashInfer-style implementations represent irregular expert work as compact virtual-CTA metadata plus per-expert row-index arrays, then load original token rows by index.
- Use when: per-expert M is device-resident and copying tokens into contiguous expert buffers is the measured cost.
- Pros: avoids duplicate token tensors, keeps WGMMA grouped expert GEMM viable on Hopper, and gives a middle ground between explicit dispatch and full operator fusion.
- Cons / guardrails: row-index loads can reduce coalescing; prefix/virtual-CTA metadata must stay cache-friendly; extreme skew and empty experts need explicit validation.

### FP8 Forward Pipeline Decomposition

- Target case: end-to-end Hopper FP8 MoE forward where no single boundary has been proven dominant yet.
- Rationale: HPC-Ops exposes a readable H20/SM90 pipeline: count/gather, grouped GEMM1, activation/multiply/quantization, grouped GEMM2, and reduce.
- Use when: agents need to decide whether to optimize grouped GEMM, dispatch/combine, activation quantization, or the interfaces between them.
- Pros: makes measurement boundaries explicit and gives a staged fallback for correctness.
- Cons / guardrails: staged pipelines still pay launch and HBM boundaries; deeper fusion must be justified by measured traffic or launch cost.

## SM90 / SM100 Boundary

SonicMoE and DeepGEMM Mega MoE are useful design context because they show how MoE-level ideas survive across GPU generations. The hardware realization does not transfer directly.

- On SM90, expect WGMMA, TMA, register-resident accumulators, warp specialization, and descriptor-managed shared-memory pipelines.
- On SM100, examples may use UMMA/`tcgen05.mma`, TMEM, 2-CTA MMA, CLC, and different low-precision scale formats.

For H100/H20/H800 plans, treat SM100 mechanics as non-portable unless the implementation explicitly replaces them with SM90-compatible TMA/WGMMA/register-resident alternatives.

## Output Checklist

When using this page, include:

- target GPU and build target evidence
- MoE regime and bottleneck hypothesis
- selected Hopper practice card
- rationale for why the practice matches the workload
- pros and expected win mechanism
- cons, constraints, and fallback path
- validation command, workload shape, and correctness/timing evidence required before claiming success
