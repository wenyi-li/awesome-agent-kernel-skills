# Persistent Grouped Scheduling

Use this when many uneven expert tiles should share one resident grouped-GEMM
kernel.

## Source Basis

- Engineering reference: [Accelerating MoE's with a Triton Persistent Cache-Aware Grouped GEMM Kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/).
- Paper: [FlashMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667).
- Source/project docs: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
  and [SGLang](https://github.com/sgl-project/sglang).
- Distilled idea: keep a fixed resident worker set and loop over grouped
  tiles when launch overhead, tail waves, or expert-weight L2 locality is the
  measured problem.

## Method Card

- Target case: many uneven expert GEMM tiles should share one resident launch, especially on Hopper when expert weights are revisited and simple grouped scheduling leaves idle waves.
- Rationale: persistent CTAs/programs can loop over grouped-GEMM tiles, use L2-aware tile order, and construct or update TMA descriptors for data-dependent expert weights instead of launching one kernel per expert.
- Applicable regime: high-throughput prefill/training grouped GEMM, skewed expert histograms, or decode batches where scheduler imbalance is larger than descriptor overhead.
- Pros: reduces launch overhead, balances uneven work across resident workers, and can improve L2 reuse for expert weights.
- Cons / guardrails: branchy tile mapping and descriptor work can erase gains; persistent residency raises register/shared-memory pressure; use it only when imbalance, launches, or L2 locality is measured.
- Primary anchors: PyTorch persistent grouped GEMM for the Triton H100 pattern, DeepGEMM SM90 practices for compact scheduler mechanics, and SGLang/vLLM backend notes for when grouped GEMM is the right regime.

## Pattern

```text
launch resident CTAs or programs
for iteration:
  tile_id = iteration * resident_tile_count + blockIdx.x
  map tile_id -> expert, tile_m, tile_n, tile_k
  configure pointers/descriptors/scales
  run tile pipeline
  write output or partial output
```

Scheduler choices:

- static round-robin, grouped launch order, or L2-aware swizzled order
- per-expert token counts from host or device
- masked invalid rows versus compacted contiguous groups
- runtime TMA descriptor construction for routed expert weights
- fixed SM count or runtime-selected SM partition

## Pros

- Reduces launch overhead and avoids one kernel per expert.
- Balances uneven expert tile counts across resident workers.
- Improves L2 reuse when tile order revisits expert weights intentionally.
- Composes with grouped GEMM and padding-free residual handling.

## Cons

- Branchy scheduler logic can erase gains.
- Persistent residency competes with register, shared-memory, and occupancy
  budgets.
- Bad tile order can destroy cache locality.
- Dynamic token counts and descriptors can complicate graph capture or
  recompilation behavior.

## Implementation Notes

- Keep tile mapping cheap enough to run per tile.
- Benchmark balanced and skewed expert histograms; persistent scheduling is
  most useful when simple grouped scheduling leaves SMs idle.
- Treat FlashMoE's persistent model as scheduler inspiration unless the kernel
  also owns dispatch and combine.
