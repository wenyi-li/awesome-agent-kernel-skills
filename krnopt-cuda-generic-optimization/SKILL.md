---
name: krnopt-cuda-generic-optimization
description: "Choose the next CUDA optimization experiment from diagnosed bottlenecks. Use when profiling has already identified the hotspot and, when needed, attributed it to a source region, and the task is to inspect that CUDA source and decide which fix family or next experiment is most justified. This includes optimization strategies that deliberately trade speed against numerical fidelity through a lower-precision path once the format contract is understood, and standard-library-style CUDA primitive substitution when a local hotspot reimplements a known primitive. Do not use this skill for hotspot finding, profiler-to-source association, running profilers, generic CUDA code review, big-picture kernel or hot-path redesign, or workload-domain pattern selection; route those to `krnopt-cuda-profiling`, `krnopt-cuda-coding`, `krnopt-cuda-structural-optimization`, or `krnopt-cuda-domain-optimization`."
---

# CUDA Kernel Optimization Direction Hunt

Use this skill after the profiling step, not before it.

This skill answers one question:

1. given this diagnosed bottleneck and this CUDA source region, what should we
   try next

This skill does not own hotspot identification or profiler-to-source
association. `krnopt-cuda-profiling` owns that handoff.

## Entry Condition

Use this skill when the input already includes most of the following:

- the target kernel
- the workload or phase context
- a bottleneck family or strong profiling diagnosis
- an attributed source region, or at least a clearly scoped code region to inspect

If any of these are missing:

- missing hotspot or missing bottleneck class:
  use `krnopt-cuda-profiling`
- missing source attribution from profile data:
  use `krnopt-cuda-profiling`
- the task is mainly "is this CUDA code sane" or "help me write CUDA":
  use `krnopt-cuda-coding`
- the task is to replan or redesign the kernel or hot path from a big-picture
  view instead of choosing one localized next experiment:
  use `krnopt-cuda-structural-optimization`
- the task is to choose a workload-domain optimization pattern such as MoE
  routing/dispatch, grouped GEMM, sparse dispatch/combine, expert MLP fusion,
  padding-free grouped GEMM, or persistent MoE scheduling:
  use `krnopt-cuda-domain-optimization`
- the task is a diagnosed local hotspot that reimplements a standard primitive
  such as GEMM, grouped GEMM, scan, reduction, histogram, sort, convolution, or
  collective communication:
  stay in this skill and first consider the standard CUDA primitive ladder
- the task is mainly "what low-precision format contract is this" or "how do
  FP8, block-scaled FP8, NVFP4, or MX formats differ":
  use `krnopt-low-precision-kernel-formats`

Once the low-precision format contract is already understood, deciding whether
to try a lower-precision path as the next optimization experiment belongs in
this skill, not in `krnopt-low-precision-kernel-formats`.

## Core Workflow

Follow this loop:

1. confirm the handoff packet
2. choose one bottleneck family to act on
3. inspect only the implicated source region
4. choose one fix family
5. propose one primary experiment
6. say what result would confirm or reject it
7. remeasure before proposing a second family of changes

In compact form:

```text
diagnosed bottleneck + attributed source
  -> choose one bottleneck family
  -> choose one fix family
  -> propose one experiment
  -> define confirmation signal
  -> remeasure
```

The main discipline is:

- one bottleneck family at a time
- one primary experiment at a time
- measured facts before hypotheses
- small, reversible changes before large rewrites

## Bottleneck Family -> Fix Family

Use this mapping to choose the next experiment.

| Bottleneck family | What to inspect in source | Typical fix family |
| --- | --- | --- |
| Launch or orchestration overhead | fragmented launch path, sync points, wrapper structure | batching, fusion, CUDA Graphs, persistent execution |
| Memory bandwidth | strided loads, redundant traffic, weak reuse, poor layout | coalescing fixes, tiling, vectorized ops, traffic reduction |
| Memory latency | short producer-consumer distance, weak locality, low reuse | shared staging, prefetch, code reordering, more independent work |
| Shared-memory or MIO pressure | bank-conflict-prone layouts, heavy scalar SMEM traffic | padding, swizzle, fewer wider SMEM ops, register reuse |
| Underfill or latency hiding | tiny grids, few eligible warps, oversized blocks, heavy dependency chains | launch-shape retuning, lower per-thread state, more parallel work |
| Low useful GPU utilization | idle Tensor Cores/SMs, wasted bytes, padding, tail waves, launch gaps, conversion-only work | utilization-targeted experiment: better MMA path, fewer bytes per useful FLOP, better scheduling, or lower launch overhead |
| Sync or divergence | barriers after uneven work, warp-misaligned branches, reduction imbalance | warp-uniform work, shorter divergent regions, warp-level primitives |
| Compute or instruction path | wrong MMA path, scalar fallback math, manual weak primitive, expensive epilogue | standard-library-style primitive substitution, tile-shape retuning, dtype or intrinsic fixes, epilogue simplification |
| Accuracy margin or dtype overhead | diagnosed compute, bandwidth, or epilogue cost that may shrink under a valid lower-precision path | lower-precision value path, lower-precision accumulation boundary, fused dequant or requant redesign, precision tradeoff experiment |

Do not turn this table into a long idea dump. Pick the one row that best
matches the diagnosis and produce one concrete next move.

If the real confusion is about the low-precision format contract itself rather
than the next experiment, route to
`krnopt-low-precision-kernel-formats`
instead of improvising format semantics here.

If the format contract is already known and valid for the workload, it is
acceptable and often important to choose a low-precision strategy as the next
optimization experiment. Treat this as a first-class optimization family rather
than as an exotic side note.

Examples of valid next-step framing:

- "The profile points to bandwidth pressure in the dequant-fed path, and the
  format contract is already understood, so the primary experiment is moving
  from the current higher-precision path to a valid block-scaled FP8 path."
- "The kernel appears compute-limited in a region whose accuracy margin may
  tolerate a lower-precision MMA path, so the next experiment is a controlled
  speed-versus-accuracy tradeoff rather than another layout tweak."

Do not assume lower precision is always acceptable. State the expected speed
benefit, the likely accuracy risk, and the exact validation signal needed to
accept or reject the tradeoff.

## How To Choose The Primary Experiment

Prefer the experiment that is:

1. most directly supported by the diagnosis
2. localized to the already-attributed hot region
3. easiest to validate or reject quickly

When the hot source region is a manual implementation of a standard CUDA
primitive, prefer the standard-library-style ladder before handcrafting:

```text
precompiled vendor/runtime libraries
  -> CCCL header-only CUDA core primitives
  -> header/template or generated-kernel libraries
  -> handcrafted CUDA kernels
```

In practice this means checking cuBLAS, cuBLASLt, cuDNN, NCCL, or available
framework/runtime kernels first when they support the operation. For standard
parallel primitives and CUDA C++ building blocks, check CCCL APIs next: CUB for
device-, block-, and warp-level algorithms, Thrust for higher-level parallel
algorithms, and libcu++/`cuda::` for CUDA-aware standard-library facilities,
atomics, barriers, pipelines, and memory utilities. CCCL is header-only and is
usually provided by the CUDA Toolkit include path used by `nvcc`; treat a
separate CCCL checkout or package as a version/include decision rather than as a
runtime library dependency. If those do not support the required feature or
perform badly on the measured workload, consider CUTLASS/CuTe, CUTLASS DSL,
TileLang/CuTile-style generators, or domain wrappers such as DeepGEMM and
FlashInfer. Only recommend handcrafted SIMT, WGMMA/TMA, or deeply fused kernels
after stating why the higher tiers do not fit. Use
[references/vendor-cuda-primitive-selection-subskill.md](references/vendor-cuda-primitive-selection-subskill.md)
when this decision matters.

Avoid proposing a full redesign unless the evidence says the current structure
is fundamentally wrong. When that broader redesign is the real task, hand it to
`krnopt-cuda-structural-optimization` instead of stretching this skill into a
rewrite planner.

Good output shape:

- "The diagnosed issue is memory bandwidth in the main load loop. The first
  experiment is changing the lane-to-address mapping to improve coalescing."
- "The diagnosed issue is register pressure in the hot reduction loop. The
  first experiment is shortening live ranges before touching memory layout."
- "The diagnosed issue is orchestration overhead. The first experiment is
  reducing launch fragmentation, not rewriting the kernel body."
- "The diagnosed issue is instruction or bandwidth cost in a path that already
  has a clear low-precision contract. The first experiment is a controlled
  lower-precision variant with explicit accuracy validation."
- "The diagnosed issue is a manual scalar GEMM in a local hot region. The first
  experiment is a cuBLAS/cuBLASLt or CUTLASS/CuTe-backed primitive path, with
  stream, handle, workspace, linker, and synchronization mechanics called out
  as implementation checks."

Bad output shape:

- ten loosely related ideas
- a rewrite plan that touches every part of the kernel
- source attribution claims that were not actually established

## Output Contract

The output of this skill should be a decision handoff, not a principle dump.

Include:

- target kernel and source region
- accepted diagnosis and evidence posture
- chosen bottleneck family
- code shape that motivates the experiment
- utilization target and likely waste source when the request or diagnosis is
  about GPU utilization
- standard primitive ladder decision when the hotspot reimplements a known
  primitive
- one primary experiment
- optional backup experiment
- confirmation signal tied to the chosen bottleneck or utilization target
- rejection signal tied to the same target
- architecture-specialized build target when the experiment depends on
  target-specific CUDA instructions or features, such as `sm_90a` or `sm_100a`
- next owner or next skill

When the chosen experiment is a low-precision tradeoff, also include:

- the precision change being proposed
- why the diagnosed bottleneck suggests it may help
- what accuracy or correctness gate must still pass
- whether `krnopt-low-precision-kernel-formats` has already resolved the
  format contract

Use
[references/optimization-direction-report-subskill.md](references/optimization-direction-report-subskill.md)
when the user wants a structured report.

## Integrated Subskills

Use these local references only when they sharpen the decision:

- Use [references/evidence-intake-subskill.md](references/evidence-intake-subskill.md)
  to check whether the handoff is complete enough to choose a direction.
- Use [references/bottleneck-to-direction-subskill.md](references/bottleneck-to-direction-subskill.md)
  to map the diagnosed bottleneck to one focused experiment.
- Use [references/gpu-utilization-enhancement-subskill.md](references/gpu-utilization-enhancement-subskill.md)
  when the user asks to enhance GPU utilization or the diagnosis mentions low
  SM/Tensor Core utilization, underfilled launches, poor useful work per byte,
  padding waste, launch-bound decode, or low-precision conversion overhead.
- Use [references/vendor-cuda-primitive-selection-subskill.md](references/vendor-cuda-primitive-selection-subskill.md)
  when a hot source region may be replaceable by a standard-library-style CUDA
  primitive.
- Use [references/optimization-direction-report-subskill.md](references/optimization-direction-report-subskill.md)
  to format the final handoff cleanly.

## Escalate When

- Escalate to `krnopt-cuda-profiling` when hotspot choice, bottleneck class, or
  profiler-to-source attribution is still uncertain.
- Escalate to `krnopt-hw-aware-optimization` when the next experiment depends
  on architecture-specific features more than generic bottleneck logic.
- Escalate to `krnopt-cuda-structural-optimization` when the diagnosis points
  to a broader kernel or hot-path redesign rather than one localized
  experiment, including a library substitution that changes boundaries,
  materialization, stage order, ownership, scheduling, or decomposition across
  multiple kernels.
- Escalate to `krnopt-cuda-domain-optimization` when the diagnosis points to a
  workload-domain pattern such as MoE routing/dispatch, grouped GEMM, sparse
  dispatch/combine, expert MLP fusion, padding-free grouped GEMM, or persistent
  MoE scheduling rather than a generic local fix family. Do not escalate only
  because the local hot region belongs to a MoE operator if the next experiment
  is a scoped standard primitive substitution.
- Escalate to `krnopt-low-precision-kernel-formats` when the blocking question
  is the meaning of a low-precision format, scale layout, or dequant contract
  rather than the next optimization experiment.
- Do not escalate to `krnopt-low-precision-kernel-formats` merely because the
  proposed next experiment uses lower precision; stay in this skill when the
  real task is deciding whether that precision tradeoff is the right
  optimization move.
- Pause for targeted docs or source research when the diagnosis is real but
  does not clearly favor one fix family.
