# Padding-Free Grouped GEMM

Use this when expert M dimensions are uneven and padded rows are visible in
profiles or traffic estimates.

## Source Basis

- Paper: [TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper](https://arxiv.org/abs/2508.16584).
- Source: [sukoncon/TMA-Adaptive-FP8-Grouped-GEMM](https://github.com/sukoncon/TMA-Adaptive-FP8-Grouped-GEMM).
- Baseline source: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).
- Distilled idea: isolate residual-M handling from the full-tile path so
  invalid padded rows are skipped only when padding waste is actually visible.

## Method Card

- Target case: grouped expert GEMM where per-expert M is uneven and padded rows are a measured fraction of compute or memory traffic.
- Rationale: Hopper FP8 grouped GEMM often pads final M tiles to a fixed block size; TMA-Adaptive shows descriptor pools and residual handling can remove that padding while keeping the full-tile path fast.
- Applicable regime: skewed MoE prefill/training, many experts with residual M, and H20/H100 grouped GEMM where padding waste remains after basic grouped scheduling.
- Pros: saves compute and memory on invalid rows, composes with persistent scheduling and expert MLP fusion, and can preserve bitwise identity for valid rows.
- Cons / guardrails: residual descriptors, masks, or overlapping stores can outweigh savings on balanced large groups; decode-style tiny M may need masked or small-M paths instead of residual cleanup.
- Primary anchors: TMA-Adaptive paper/source for descriptor-pool mechanics and Hopper MoE Grouped GEMM for when padding is the right bottleneck.

## Pattern

- Keep the full-tile path simple and fast.
- Detect residual M tiles per expert.
- Use masked grouped GEMM or residual descriptors for final tiles.
- In the TMA-Adaptive Hopper pattern, pre-build power-of-two descriptors and
  use two overlapping stores for arbitrary residual sizes.
- Preserve barrier progress even when rows are masked, skipped, or written by
  overlapping stores.

## Pros

- Saves compute and memory for skewed expert loads.
- Especially useful with many experts and small per-expert M.
- Can be bitwise identical when skipped rows are truly padding.
- Orthogonal to fused scatter-GEMM and expert MLP fusion.

## Cons

- Residual handling can slow the common full-tile path if not isolated.
- Descriptor selection, masking, or extra stores may outweigh saved rows for
  large balanced groups.
- Hopper TMA descriptor-pool mechanics are architecture-specific.
- Decode-style tiny groups may need a different layout rather than residual
  cleanup.

## Implementation Notes

- Measure padding waste as padded rows minus useful routed rows per expert.
- Use the smallest residual mechanism that covers valid rows without touching
  the main full-tile loop.
- On SM100, remap the idea through the hardware-aware skill rather than copying
  SM90 descriptor mechanics directly.
