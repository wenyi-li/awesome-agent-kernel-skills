# Sparse Dispatch / Combine

Use this when dispatch and combine are visible costs but cannot be fused into
GEMM tile load/store.

## Source Basis

- Paper: [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382).
- Source: [microsoft/tutel](https://github.com/microsoft/tutel).
- Distilled idea: use sparse encode/decode and route-index kernels when the
  GEMM path cannot absorb indirection, but treat separate dispatch/combine
  buffers as a memory-traffic cost to remeasure.

## Method Card

- Target case: dispatch and combine are measured costs, but the current GEMM path cannot absorb row indirection safely.
- Rationale: Tutel-style sparse dispatch reduces dense `T * E` work to active `T * k` routes with index assignment, scaled scatter, and weighted gather-accumulate kernels.
- Applicable regime: retrofit work where separate GEMM kernels must remain, distributed paths where dispatch/combine can overlap communication, or early-stage baselines before deeper fusion.
- Pros: simpler to integrate than fused scatter-GEMM, preserves existing expert GEMM kernels, and provides SIMT reduction/vectorization patterns.
- Cons / guardrails: still materializes buffers and is often global-memory bound; usually loses to fused scatter-GEMM when tile loads/stores can absorb indirection.
- Primary anchors: Tutel paper/source for sparse encode/decode and the
  `T * k` active-route model.

## Pattern

- Operate on `T * k` active routes instead of dense `T * E` routing entries.
- Use an index-assignment kernel for route locations.
- Use scaled scatter for dispatch and weighted gather-accumulate for combine.
- Use warp-level shuffles for reductions instead of shared-memory reductions.
- Use vectorized low-precision loads such as `half2` when alignment permits.
- Assign one warp to a token slice when it improves coalescing.

## Pros

- Reduces dispatch/combine complexity from dense `T * E` work to active
  top-k work.
- Preserves separate GEMM kernels, so it is easier to retrofit into an
  existing MoE path.
- Provides warp-level SIMT patterns that remain useful inside deeper fused
  kernels.

## Cons

- Still materializes dispatch and combine buffers.
- Often remains global-memory bound.
- Usually loses to fused scatter-GEMM when GEMM can absorb indirection.
- Most compelling in distributed systems where dispatch/combine overlap with
  communication matters.

## Implementation Notes

- Keep route-index arrays compact and aligned.
- Handle empty experts, single-token experts, repeated top-k routes, and local
  expert offsets explicitly.
- Treat Tutel's headline speedups as system-context evidence, not a local
  single-kernel guarantee.
