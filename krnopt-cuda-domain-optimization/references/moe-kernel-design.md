# MoE Kernel Design Domain

Use this index when optimizing Mixture-of-Experts CUDA kernels: sparse routing,
dispatch, grouped expert GEMM, SwiGLU/activation fusion, combine, padding
elimination, persistent scheduling, and expert-parallel communication.

Load only the technique page that matches the measured bottleneck.

For Hopper/H20/H100/H800 MoE backend choice, start with
[hopper-backend-selection.md](moe-kernel-design/hopper-backend-selection.md).
For an end-to-end Hopper FP8 forward path, use
[hopper-fp8-forward-pipeline.md](moe-kernel-design/hopper-fp8-forward-pipeline.md).

## Domain Model

A typical top-k MoE forward path has these boundaries:

```text
router logits
  -> top-k / group-top-k selection
  -> dispatch or gather tokens by expert
  -> GEMM1: gate/up expert projections
  -> activation: SiLU(gate) * up
  -> GEMM2: down projection
  -> scale by routing weights
  -> combine / scatter-reduce to token order
```

A domain-specific optimization should state which boundary it removes,
reshapes, or overlaps.

## Measure First

Before selecting a technique, gather or estimate:

- time in top-k, dispatch/permutation, GEMM1, activation, GEMM2,
  combine/reduction, and communication
- token histogram per expert and local expert
- top-k, expert count, local expert count, hidden size, intermediate size,
  sequence length distribution, dtype, and scale format
- padded rows versus useful routed rows
- intermediate-buffer traffic between GEMM1, activation, GEMM2, and combine
- launch count and whether the implementation uses many tiny expert kernels
- L2 locality for expert weights and tile order
- tensor-core utilization, occupancy, register spills, shared-memory use, and
  barrier stalls

If there is no profile, label the recommendation as a hypothesis.

## Technique Index

| Target Case | Rationale | Technique Page | Pros | Cons / Guardrail | Primary Sources |
|---|---|---|---|---|---|
| Hopper backend or kernel-family choice is unclear | Hopper MoE is a regime-selection problem, not one universal kernel | [hopper-backend-selection.md](moe-kernel-design/hopper-backend-selection.md) | Separates small-M decode, high-throughput grouped GEMM, padding, dispatch, and pipeline cases | Serving-stack defaults must be validated on local workloads | SGLang, vLLM, DeepGEMM, FlashInfer, CUTLASS |
| Hopper FP8 forward pipeline boundaries are unclear | The full path has count/gather, GEMM1, activation/quantization, GEMM2, and reduce/combine boundaries | [hopper-fp8-forward-pipeline.md](moe-kernel-design/hopper-fp8-forward-pipeline.md) | Gives staged correctness and measurement boundaries | Full fusion can raise register, SMEM, and scheduler pressure | HPC-Ops, SonicMoE, SGLang, vLLM |
| Dispatch or combine is expensive but GEMM cannot absorb indirection | Sparse dispatch reduces dense `T * E` work to active `T * k` routes while keeping existing GEMMs | [sparse-dispatch-combine.md](moe-kernel-design/sparse-dispatch-combine.md) | Easier retrofit and useful SIMT patterns | Still materializes buffers and is often memory-bound | Tutel |
| Explicit token permutation or unpermutation dominates | Gather/scatter can move into GEMM tile load/store instead of full activation copies | [fused-scatter-gemm.md](moe-kernel-design/fused-scatter-gemm.md) | Removes HBM round trips and pads cheap index arrays | Indexed loads can hurt coalescing; top-k combine still needs ownership design | ScatterMoE, FusedXpert |
| Dropless routing with irregular expert loads matters most | Block-sparse topology represents variable expert work without token drop | [block-sparse-moe.md](moe-kernel-design/block-sparse-moe.md) | Natural dropless training formulation and forward/backward sparse GEMM coverage | Sparse infrastructure and block padding can be poor for decode | MegaBlocks |
| GEMM1, activation, and GEMM2 materialize too much data | SwiGLU and FP8 re-quantization can be fused or pipelined between grouped GEMMs | [expert-mlp-fusion.md](moe-kernel-design/expert-mlp-fusion.md) | Reduces large intermediates and launch boundaries | Register/SMEM pressure and scale layout constraints grow quickly | FusedXpert, Axe, HPC-Ops, SonicMoE |
| Grouped GEMM wastes rows on uneven expert M | Residual descriptors or masks avoid computing padded final rows | [padding-free-grouped-gemm.md](moe-kernel-design/padding-free-grouped-gemm.md) | Saves compute/memory on skewed groups and composes with grouped GEMM | Residual mechanics can slow balanced large groups | TMA-Adaptive |
| Many uneven expert tiles need one resident schedule | Persistent tile loops can balance grouped work and improve expert-weight locality | [persistent-grouped-scheduling.md](moe-kernel-design/persistent-grouped-scheduling.md) | Reduces launch overhead and scheduler imbalance | Branchy mapping, descriptors, and residency costs can erase gains | PyTorch persistent grouped GEMM, DeepGEMM |
| Dynamic scheduling is too heavy but copies should be avoided | Compact virtual-CTA metadata and row indices avoid full token-copy materialization | [static-batched-metadata.md](moe-kernel-design/static-batched-metadata.md) | Keeps Hopper WGMMA grouped GEMM viable with less dispatch traffic | Row-index access and metadata can become the bottleneck under skew | Static Batching, DashInfer |
| Launch and memory boundaries dominate a bespoke operator | A persistent task queue can keep dispatch, compute, and combine inside one operator | [whole-operator-persistent-fusion.md](moe-kernel-design/whole-operator-persistent-fusion.md) | Maximum overlap opportunity at tile granularity | Hardest to debug; deadlock/starvation and geometry overfit risk | FlashMoE, SonicMoE |
| Expert-parallel communication dominates | Communication/layout transforms can be fused with MoE sub-layer GEMMs | [communication-fusion.md](moe-kernel-design/communication-fusion.md) | Attacks distributed EP all-to-all/all-gather/reduce-scatter bottlenecks | Mostly scheduling inspiration for single-GPU local experts | NCCL EP, Flux, COMET, CCFuser |

## Method Card Shape

Every method page should give enough context for an agent to decide whether the method applies:

- target case: the workload or profile shape that makes the method relevant
- rationale: why the method addresses that bottleneck
- applicable regime: decode, prefill, training, fixed contest geometry, single-GPU local experts, or expert-parallel distributed MoE
- pros: expected win mechanism
- cons / guardrails: constraints, failure modes, and when not to use it
- primary anchors: public papers, repositories, official docs, or clearly
  labeled source-mined context to inspect when deeper detail is needed

## Decision Tree

1. Dispatch or combine dominates:
   - Prefer fused scatter-GEMM if GEMM tile load/store can absorb token
     indirection.
   - Use sparse dispatch/combine when separate dispatch/combine kernels must
     remain.
2. GEMM1 plus activation materialization dominates:
   - Use expert MLP fusion.
   - If GEMM2 can start per tile after GEMM1 activation is ready, use the Axe
     style tile-pipelining variant described in the expert MLP page.
3. Grouped GEMM dominates and expert M is irregular:
   - Use padding-free grouped GEMM, persistent grouped scheduling, or
     static-batched metadata depending on whether the cost is padding,
     scheduler imbalance, or duplicate token copies.
4. Many experts receive tiny batches:
   - Avoid one launch per expert.
   - Consider grouped, persistent, static-batched, or fused-dispatch designs.
5. A fixed capacity or block-sparse layout wastes too much work:
   - Use fused scatter-GEMM, masked M-grouped GEMM, or TMA descriptor-pool
     residual handling.
6. Top-k combine/reduction is still expensive:
   - Fuse routing-weight scaling into GEMM2 store only after defining output
     ownership and token-local accumulation for top-k > 1.
7. Expert-parallel communication dominates:
   - Use communication-fusion patterns. For single-GPU kernels, borrow only the
     scheduling ideas: queues, readiness flags, staged producer/consumer roles,
     and tile-granular overlap.

## Architecture Mapping

Domain techniques are portable ideas; implementation surfaces still depend on
the target architecture.

- SM80 / A100: use `cp.async`, WMMA/MMA, block-sparse or Triton-style grouped
  kernels. Sparse dispatch, fused scatter-GEMM, and block-sparse MoE are the
  most portable starting points.
- SM90 / H100 / H20 / H800: use TMA for regular tiles, WGMMA for expert GEMMs,
  warp specialization, descriptor pools, runtime descriptor mutation, and
  persistent grouped scheduling when they fit the kernel.
- SM100 / B200: use TMEM, `tcgen05.mma`/UMMA, 2-CTA MMA, TMA extensions, CLC,
  block-scaled FP8/MXFP8/NVFP4, and cluster scheduling only through the
  hardware-aware skill. Do not backport these mechanics to SM90.

## Output Template

When giving a MoE optimization plan, include:

- measured bottleneck or hypothesis
- selected technique page and source paper lineage
- kernel boundary changed
- data layout and metadata representation
- correctness gates
- timing workloads and accept/reject criteria
- fallback if the technique increases register pressure, synchronization, or
  scheduler overhead too much

## Validation Checklist

- Compare against a simple reference for top-k, routing weights, local expert
  filtering, SwiGLU, GEMM1/GEMM2, and combine.
- Test zero-token experts, single-token experts, skewed experts, balanced
  experts, top-k duplicates or ties, and sequence lengths around tile
  boundaries.
- Time balanced, skewed, small-batch/decode, and large-batch/prefill workloads.
- Report whether the improvement came from less memory traffic, better tensor
  core utilization, fewer launches, less padding, better scheduling, or
  communication overlap.

## Anti-Patterns

- Optimizing only GEMM tile shape when dispatch, padding, or combine dominates.
- Fusing every boundary without a register/shared-memory budget.
- Treating paper speedups as transferable without matching geometry and
  architecture.
- Using persistent scheduling for regular large GEMMs where a simple grouped
  schedule is enough.
- Moving top-k combine into a scatter store without resolving multiple expert
  contributions to the same token.
