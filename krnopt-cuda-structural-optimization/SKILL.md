---
name: krnopt-cuda-structural-optimization
description: >-
  Replan and redesign CUDA kernels from a big-picture view. Use when the user
  wants a structural change to a kernel or hot path, such as rethinking
  boundaries, scheduling, primitives, pipeline stages, or metadata flow,
  instead of pursuing another local optimization. Do targeted idea search or
  source research before guessing when deeper knowledge is needed, and route
  workload-domain
  pattern selection such as MoE routing, grouped GEMM, dispatch/combine, expert
  MLP fusion, padding elimination, or persistent MoE scheduling to
  `krnopt-cuda-domain-optimization`. Do not use this skill for hotspot finding,
  raw profiler interpretation, or low-level CUDA coding.
---

# CUDA Structural Optimization

Use this skill when the optimization question is no longer "what small tweak
should I try next?" but "is the current kernel structure itself the problem?"

This skill owns the big-picture redesign decision:

1. what the hot path is fundamentally spending time, bandwidth, and scheduling
   capacity on
2. which part of the current structure is mismatched to the workload
3. which redesign family is most justified

This skill is grounded in the KB's profiling workflow, CUDA programmer
mindset, Nsight metric interpretation, NVIDIA's ADO loop, and case studies
around grouped GEMM, fusion, persistent scheduling, static batching, warp
specialization, and quantization fusion.

## Application Scope

Use this skill for any of these scopes:

- one CUDA kernel whose bottleneck now looks structural rather than local
- one multi-kernel hot path where the boundaries may be the real problem
- one irregular pipeline where scheduling or metadata cost dominates
- one custom kernel family that may want a different primitive entirely

Typical workload families:

- grouped GEMM and routed expert workloads
- attention and other staged producer-consumer kernels
- epilogue-heavy or postprocess-heavy kernels
- quantized paths where Q/DQ or scaling may need to move into the prologue or
  epilogue
- launch-fragmented pipelines with many tiny kernels

Use another skill instead when:

- the main question is "what is actually hot" or "which bottleneck dominates":
  use `krnopt-cuda-profiling`
- the hotspot and bottleneck are known and the task is to choose one smaller
  next experiment:
  use `krnopt-cuda-generic-optimization`
- the task is to write or rewrite CUDA source once the structural plan is
  chosen:
  use `krnopt-cuda-coding`
- the task is mainly hardware-feature-first planning:
  use `krnopt-hw-aware-optimization`
- the task is mainly choosing a domain-specific CUDA pattern for MoE,
  grouped GEMM, routed experts, dispatch/combine, expert MLP fusion, padding
  elimination, or persistent MoE scheduling:
  use `krnopt-cuda-domain-optimization`
- the task is to search for structural ideas, source material, or deeper
  knowledge before choosing a redesign:
  do targeted docs or source research before committing to the redesign

## Structural Mismatch Taxonomy

Most structural redesigns in the KB fit one of these mismatch families:

- boundary mismatch:
  launch or memory boundaries force extra copies, syncs, or materialization
- scheduler mismatch:
  the work is irregular but the kernel uses a static regular launch or tile
  schedule
- primitive mismatch:
  the custom code is acting like a weak GEMM, reduction, or routing primitive
  and should be recast or handed off, preferring standard-library-style CUDA
  primitives before handcrafted replacements when the operation contract fits
- pipeline mismatch:
  producer, consumer, reduction, and epilogue stages are serialized when the
  workload wants overlap
- metadata mismatch:
  the kernel is moving full payloads when compact indices, prefix maps, or
  descriptors would be cheaper

In compact form:

```text
Boundary mismatch  -> remove / fuse / hand off
Scheduler mismatch -> persist / queue / static-batch / retile
Primitive mismatch -> standard library / grouped GEMM / block-sparse / custom kernel
Pipeline mismatch  -> overlap / warp-specialize / epilogue fuse
Metadata mismatch  -> compact indices / prefix maps / descriptor pools
```

This taxonomy is stronger than "pick a redesign family" because it says *why*
the redesign is needed.

## Core Workflow

Follow this order:

1. define the optimization target and workload shape
2. decide whether the true unit is one kernel or a hot path of several parts
3. build a multi-metric pressure picture
4. reason across the hot path part by part
5. identify the dominant structural mismatch
6. choose one redesign family
7. define success, stop, and validation conditions

In compact form:

```text
workload + source + profile
  -> pressure picture
  -> part-by-part reasoning
  -> structural mismatch
  -> redesign family
  -> success and stop signals
```

The main discipline is:

- do not let one metric dominate the whole story
- do not treat "the kernel" as one undifferentiated blob
- do not redesign blindly when a local fix is enough
- do not stay local when the structure is clearly wrong
- choose one structural cause family at a time
- remeasure after each structural change

## Step 1: Build A Pressure Picture

Start from a small cross-check set instead of one favorite metric.

At minimum, combine:

- time share or kernel duration
- SM throughput and DRAM throughput
- occupancy or active-warps posture
- scheduler issue efficiency or eligible-warps posture
- dominant stall family
- one memory-shape signal
- one instruction-path signal when relevant

If the hot path has many kernels, also check launch count, host gaps, and bytes
spent on copies, permutes, combine, or materialization steps.

Use
[references/multi-metric-structural-scan-subskill.md](references/multi-metric-structural-scan-subskill.md)
when the main problem is synthesizing metrics into one coherent pressure
picture.

## Step 2: Reason Across Kernel Parts

The KB repeatedly shows that the bottleneck often sits at a boundary between
parts, not in the center of one loop body.

Reason across:

- launch or orchestration
- ingress, layout, and metadata
- main compute loop
- reduction and synchronization
- epilogue or postprocess
- materialize or handoff

Use
[references/hot-path-partitioning-subskill.md](references/hot-path-partitioning-subskill.md)
when the main need is to inspect the hot path part by part instead of treating
it as one blob.

## Step 3: Identify The Structural Mismatch

Treat the problem as structural when one or more of these are true:

- the same bottleneck survives the obvious local fixes
- fixing one line just moves the bottleneck to the next artificial boundary
- the measured cost comes from launch count, materialization, padding,
  dispatch, combine, descriptor churn, or staging overhead
- the kernel shape is fundamentally mismatched to the workload's irregularity
  or reuse pattern
- the profiler points to real hardware limits and the remaining win likely
  needs decomposition or a different primitive

Do not escalate to structural redesign just because the kernel is slow. Escalate
when the source and the pressure picture both say the current structure is the
bottleneck.

## Step 4: Choose One Redesign Family

Pick one redesign family that removes the measured mismatch:

- boundary elimination and fusion
- scheduler redesign
- primitive substitution, using the same preference order as localized
  optimization: precompiled vendor libraries, then header/template or
  generated-kernel libraries, then handcrafted kernels
- pipeline overlap redesign
- metadata compaction
- decomposition plus tuned-library handoff

Use
[references/redesign-family-map-subskill.md](references/redesign-family-map-subskill.md)
when the main problem is mapping a mismatch to one redesign family.

Use
[references/structural-patterns-subskill.md](references/structural-patterns-subskill.md)
when you need the deeper KB-backed pattern language for fusion, persistent
scheduling, grouped GEMM, metadata compaction, warp specialization, and
roofline-based stop conditions.

## Step 5: Validate And Stop Deliberately

Even structural changes should stay scoped.

Define:

- what mismatch family is being addressed
- what boundary, scheduler, primitive, or stage order is changing
- what source region or kernel family will change
- what architecture-specialized build target is required if the redesign
  depends on target-specific CUDA instructions or features, such as `sm_90a`
  or `sm_100a`
- what metric pattern should improve
- what correctness or numerical risk must be checked
- what result would mean "this needs an even bigger redesign"

The KB's ADO lesson matters here:

- if earlier inefficiencies are removed and the remaining kernel is behaving
  like a real streaming-load problem near a roofline, more local tuning may be
  the wrong game
- at that point decomposition or a tuned-library handoff is often the correct
  structural move

Use
[references/case-snapshots-subskill.md](references/case-snapshots-subskill.md)
when the user wants compact example cases from ADO, MoE, attention, and
quantized kernels.

## Output Contract

The output of this skill should be a structural handoff containing:

- target kernel or hot path
- workload shape and optimization objective
- evidence summary from multiple metrics or source facts
- the part-by-part reasoning across the hot path
- the structural mismatch being claimed
- the chosen redesign family
- why smaller fixes are unlikely to be enough
- success signals
- stop signals
- validation risks
- architecture-specialized build target when the redesign depends on
  target-specific CUDA instructions or features
- next owner or next skill

## Integrated References

Use these local references when they sharpen the result:

- Use [references/multi-metric-structural-scan-subskill.md](references/multi-metric-structural-scan-subskill.md)
  for metric synthesis and single-metric traps.
- Use [references/hot-path-partitioning-subskill.md](references/hot-path-partitioning-subskill.md)
  for reasoning across launch, ingress, compute, sync, epilogue, and handoff.
- Use [references/redesign-family-map-subskill.md](references/redesign-family-map-subskill.md)
  for mismatch-to-redesign mapping.
- Use [references/structural-patterns-subskill.md](references/structural-patterns-subskill.md)
  for deeper KB-backed common patterns.
- Use [references/case-snapshots-subskill.md](references/case-snapshots-subskill.md)
  for example case memories.

## Routing

- `krnopt-cuda-profiling` owns hotspot selection, first-pass classification,
  and profiler-to-source attribution.
- `krnopt-cuda-generic-optimization` owns smaller next-step experiments after
  the structure is already accepted.
- `krnopt-cuda-coding` owns implementation once the redesign family is chosen.
- `krnopt-hw-aware-optimization` owns hardware-specific feature choice when the
  branch depends on SM90 or SM100 details more than on generic structural
  reasoning.
- `krnopt-cuda-domain-optimization` owns workload-domain pattern selection when
  the redesign is best understood as MoE routing, grouped GEMM, dispatch/combine,
  expert MLP fusion, padding elimination, persistent MoE scheduling, or another
  covered domain playbook.
- Targeted docs or source research owns structural idea search, deeper CUDA
  knowledge, and the right KB, docs, or upstream sources before this skill
  commits to a redesign direction.
