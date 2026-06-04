# Communication Fusion

Use this when experts are sharded across GPUs and expert-parallel communication
dominates dispatch, compute, or combine.

## Source Basis

- Paper: [NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL](https://arxiv.org/abs/2603.13606).
- Paper and source: [Flux: Fast Software-based Communication Overlap on GPUs through Kernel Fusion](https://arxiv.org/abs/2406.06858),
  [bytedance/flux](https://github.com/bytedance/flux).
- Paper: [COMET: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811).
- Paper page: [Harnessing Inter-GPU Shared Memory for Seamless MoE Communication-Computation Fusion](https://researchr.org/publication/WangXY0C25).
- Distilled idea: communication-heavy MoE should be treated as staged
  dispatch/compute/combine overlap, with separate low-latency and
  high-throughput regimes rather than one collective boundary.

## Method Card

- Target case: experts are sharded across GPUs and expert-parallel communication dominates local dispatch, compute, or combine.
- Rationale: Flux/COMET, NCCL EP, and related systems attack all-to-all/all-gather/reduce-scatter boundaries by fusing communication with scatter, grouped GEMM, gather, or reduce stages.
- Applicable regime: distributed expert-parallel inference/training; for single-GPU local experts, use only the readiness and producer/consumer scheduling ideas.
- Pros: reduces idle time behind collective barriers and can overlap communication with expert compute at sub-layer granularity.
- Cons / guardrails: topology, framework integration, completion semantics, and payload size decide the outcome; importing distributed complexity into a single-GPU kernel is usually a mistake.
- Primary anchors: Flux/COMET for fine-grained overlap, NCCL EP for
  library-mode selection, and the communication-fusion paper lineage for
  when distributed expert parallelism is the real bottleneck.

## Pattern

- Replace bulk-synchronous all-to-all boundaries with staged dispatch/combine
  primitives or fused communication+GEMM kernels.
- Use low-latency modes for decode-sized payloads and high-throughput modes for
  prefill/training payloads when the library supports both.
- Fuse all-gather with scatter/grouped-GEMM prologues or GEMM/reduce-scatter
  with epilogues.
- Use readiness signals, remote writes, or one-sided shared-memory/RDMA
  semantics when the platform exposes them.
- Overlap communication and computation at tile, token, or sub-token
  granularity depending on synchronization overhead.

## Pros

- Attacks the dominant cost in expert-parallel distributed MoE.
- Can reduce idle GPU time caused by slow peers and collective barriers.
- Provides queueing and producer/consumer patterns that transfer to
  single-GPU persistent kernels.
- Library-style APIs such as NCCL EP can hide topology details while preserving
  stream-based async execution.

## Cons

- Mostly irrelevant for a single-GPU local-expert kernel except as scheduling
  inspiration.
- Communication progress, memory ordering, and completion semantics are harder
  than local HBM scheduling.
- Small payload decode and large payload prefill usually need different
  algorithms.
- End-to-end gains depend heavily on topology, framework integration, and
  routing distribution.

## Implementation Notes

- Do not import distributed complexity into a single-GPU kernel unless the same
  queueing or readiness pattern removes a local boundary.
- For multi-GPU work, choose the communication mode from measured token count
  regime, not just from peak bandwidth claims.
- Keep explicit completion and lifetime rules for any remote or shared-memory
  buffer.
