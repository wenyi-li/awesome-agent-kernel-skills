# Block-Sparse MoE

Use this when dropless routing and variable expert sizes matter more than
maximal operator fusion.

## Source Basis

- Paper: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841).
- Source: [databricks/megablocks](https://github.com/databricks/megablocks).
- Distilled idea: represent dropless MoE as block-sparse matrix
  multiplication so variable expert loads become sparse topology instead of
  token dropping or full-capacity padding.

## Method Card

- Target case: dropless MoE training with variable expert sizes is more important than maximum single-operator fusion.
- Rationale: MegaBlocks reformulates expert work as block-sparse matrix multiplication so routing imbalance is represented by sparse block topology instead of capacity-factor token dropping.
- Applicable regime: training forward/backward paths, dropless semantics, and workloads where block-boundary padding is preferable to full capacity padding.
- Pros: natural no-token-drop formulation, covers forward and backward sparse GEMM variants, and avoids many per-expert tiny launches.
- Cons / guardrails: sparse topology, gather, activation, and scatter can remain separate boundaries; block padding can dominate decode or tiny expert batches; infrastructure cost is higher than grouped-GEMM variants.
- Primary anchors: MegaBlocks paper/source for block-sparse formulation and
  the domain technique index for where it sits relative to grouped and
  scatter-GEMM designs.

## Pattern

- Represent expert work as block-sparse matrix multiplication.
- Pad expert token rows only to block boundaries rather than fixed capacity.
- Maintain sparse topology metadata for block lookup.
- Use sparse GEMM variants for forward, activation gradient, and weight
  gradient when training.
- Rebuild routing topology each forward pass if routing is dynamic.

## Pros

- Natural dropless formulation: no capacity-factor tuning and no token drop.
- Avoids per-expert tiny launches.
- Can approach dense GEMM efficiency when block size, expert load, and sparse
  format fit the workload.
- Useful training reference because it covers forward and backward sparse GEMM
  variants.

## Cons

- Topology construction, gather, activation, and scatter often remain separate
  boundaries.
- Block padding can dominate decode or tiny expert batches.
- Sparse metadata can make the kernel less composable than dense grouped
  tensors.
- Custom sparse infrastructure is a larger engineering commitment than
  grouped-GEMM variants.

## Implementation Notes

- Check expert M histogram before choosing block size.
- Do not assume MegaBlocks' 128-row block choice is optimal outside its A100
  training regime.
- Prefer fused scatter-GEMM for single-GPU inference when dispatch buffers are
  the main bottleneck and dropless block-sparse training semantics are not
  needed.
