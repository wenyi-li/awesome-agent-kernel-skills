# Structural Patterns

Use this reference when you need the deeper KB-backed pattern language rather
than generic "consider fusion" advice.

## 1. Optimize Metadata Instead Of Payload

Sometimes the right redesign is not "move data faster" but "stop moving full
payloads."

Examples from the KB:

- `Fused Scatter-GEMM` uses index padding instead of data padding
- `Static Batching` uses compact prefix maps and token index arrays
- descriptor pools avoid heavy per-shape setup or padding overhead

General lesson:

- if the hot path is paying to permute, pad, or copy full activation payloads,
  ask whether indices, prefix maps, or descriptors can carry the schedule more
  cheaply

## 2. Remove Boundaries That Only Materialize Intermediates

The KB treats fusion as boundary elimination, not as a style preference.

Examples:

- `SwiGLU Fusion` removes a full round-trip between GEMM1 and activation
- dynamic quantization systems move Q/DQ into GEMM prologue or epilogue
- MoE fusion techniques remove dispatch or combine buffers rather than merely
  merging source files

General lesson:

- fuse when the producer already owns the data in a hot form and the separate
  kernel mainly exists to materialize an intermediate

## 3. Make Irregular Schedules Explicit

Irregular workloads often want explicit schedule metadata rather than a static
grid pretending the problem is regular.

Examples:

- `Static Batching`: compact task-prefix maps for irregular grouped work
- persistent grouped GEMM: per-SM tile loops across problem lists
- queue-driven MoE kernels: device-side task progression rather than many tiny
  launches

General lesson:

- if the work shape varies heavily across experts, tasks, or tiles, redesign
  the scheduler instead of endlessly retuning block geometry

## 4. Change The Primitive, Not Just The Tuning Knobs

Sometimes the custom kernel is simply the wrong primitive.

Examples:

- weak per-expert loops should often become grouped GEMM
- some routed MoE layouts want fused scatter-GEMM or block-sparse reformulation
- near-roofline streaming loads may indicate the dense core should be split out
  and handed to cuBLAS or CUTLASS

General lesson:

- if the code is reimplementing a commodity primitive badly, primitive
  substitution is often the structural fix

## 5. Change The Internal Social Order Of The CTA

Some kernels need a new pipeline shape inside the block, not just a new tile
size.

Examples:

- FlashAttention-3 uses producer and consumer warpgroup choreography
- overlap structure, not just MMA throughput, drives the redesign
- asymmetric register budgets are part of the structure, not an afterthought

General lesson:

- when producer, compute, softmax, reduction, or epilogue work should overlap,
  think in staged roles and pipeline order rather than homogeneous warps

## 6. Treat Fusion As A Complementarity Bet

HFuse is useful because it treats fusion as a search over resource
complementarity.

General lesson:

- memory-heavy plus compute-heavy may complement each other
- compute-heavy plus compute-heavy may just compete for the same pipe
- if fusion raises issue efficiency but tanks occupancy or causes spills, the
  fused structure may still be wrong

## 7. Stop Local Tuning Near Roofline

The ADO loop has a strong structural lesson:

- once earlier inefficiencies are removed and the remaining kernel behaves like
  a real streaming-load problem near the memory roofline, more local tuning may
  be the wrong game
- at that point the next move is often decomposition, primitive substitution,
  or a tuned-library handoff
