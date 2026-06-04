---
name: krnopt-hw-aware-optimization
description: Reason about architecture-specific CUDA optimization. Use when a user wants to exploit hardware features on a target GPU, map features to code structure, enhance GPU utilization through hardware-specific surfaces, compare hardware-specific optimization directions, or produce a hardware-aware optimization plan. Route generic cross-evidence optimization directions to `krnopt-cuda-generic-optimization`, big-picture kernel or hot-path redesign to `krnopt-cuda-structural-optimization`, workload-domain pattern selection to `krnopt-cuda-domain-optimization`, generic CUDA coding doctrine to `krnopt-cuda-coding`, profiler-first diagnosis to `krnopt-cuda-profiling`, and low-precision format-contract questions to `krnopt-low-precision-kernel-formats`.
---

# Hardware-Aware CUDA Optimization

Use this skill when the main question is:

1. what does this target GPU want the kernel to look like
2. which hardware features are actually relevant to this workload
3. which architecture-specific path is worth planning around

This skill owns hardware-feature-first reasoning. It does not own hotspot
diagnosis, generic CUDA code review, or generic next-experiment ranking.

## Entry Condition

Use this skill when the input already includes most of the following:

- a target GPU or compute capability
- a kernel family or workload shape
- a question about architecture-specific feature choice

Good fits:

- "Should this Hopper kernel use TMA plus WGMMA?"
- "Does this B200 path want `tcgen05.mma`, TMEM, or CLC?"
- "Is this grouped GEMM better as 1SM or 2SM on SM100?"
- "What hardware-specific rewrite is even plausible on this GPU?"
- "Which SM90 or SM100 feature would improve Tensor Core or SM utilization?"

Use another skill instead when:

- the main question is "what is actually hot" or "which bottleneck dominates":
  use `krnopt-cuda-profiling`
- the main question is "what experiment should we try next from diagnosed evidence":
  use `krnopt-cuda-generic-optimization`
- the main question is to redesign the kernel or hot path from a big-picture
  view and hardware feature choice is only one part of that plan:
  use `krnopt-cuda-structural-optimization`
- the main question is choosing a workload-domain CUDA pattern such as MoE
  routing/dispatch, grouped GEMM, sparse dispatch/combine, expert MLP fusion,
  padding-free grouped GEMM, or persistent MoE scheduling before mapping it to
  hardware features:
  use `krnopt-cuda-domain-optimization`
- the main question is "is this CUDA source sane" or "how do I implement this":
  use `krnopt-cuda-coding`
- the main question is "what does this FP8, block-scale, NVFP4, or MX format
  contract mean":
  use `krnopt-low-precision-kernel-formats`

## Core Workflow

Follow this order:

1. resolve the exact hardware target
2. classify the kernel or workload shape
3. choose the architecture branch that really applies
4. identify the utilization target and likely useful-work waste source when
   utilization is the goal
5. choose the hardware surfaces that plausibly matter
6. map each candidate feature to the code shape required to use it
7. inspect contracts, tradeoffs, and fallback path
8. name the architecture-specialized build target required for the selected
   path, such as `sm_90a` or `sm_100a`
9. load the matching deeper reference instead of guessing
10. produce a hardware-aware plan

In compact form:

```text
target GPU + workload shape
  -> architecture branch
  -> candidate hardware surfaces
  -> feature-to-code mapping
  -> contracts and tradeoffs
  -> next plan
```

The main discipline is:

- resolve hardware before recommending a feature
- start from workload shape, not feature hype
- treat utilization as useful work on the limiting hardware path, not raw
  occupancy
- treat feature contracts as part of the kernel interface
- treat architecture-specialized build targeting as part of the feature
  contract; general targets such as `sm_90` or `sm_100` do not prove
  specialized paths are active
- keep likely wins separate from proven wins
- if the required hardware detail is unclear, load a reference or do targeted
  docs or source research before guessing

## Step 1: Resolve The Hardware Exactly

Do not say "Blackwell" and stop. Resolve the actual branch:

- pre-SM90
- SM90 / Hopper
- SM100 / B200 data-center Blackwell
- consumer Blackwell / CC 12.x
- unresolved or mixed guidance

This distinction is mandatory because TMEM, CLC, and some low-precision
hardware contracts do not transfer cleanly across those branches.

If the target is unresolved, stop and resolve it before recommending a
hardware-specific rewrite.

## Step 2: Classify The Workload Shape

Name the workload before choosing a feature:

- dense GEMM
- grouped GEMM
- attention
- decode
- reduction
- routing or dispatch
- epilogue-heavy kernel
- sparse or irregular workload

This matters because hardware features are only useful when the kernel shape
can satisfy their contracts.

## Step 3: Choose The Architecture Branch

### Branch A: pre-SM90

Typical questions:

- is `cp.async` the right staging path
- does the workload justify tensor-core MMA on this generation
- is the kernel really limited by register pressure or launch shape

Typical materials:

- generic CUDA source and memory reasoning:
  `krnopt-cuda-coding`
- feature selection baseline:
  [references/hw-feature-selection-subskill.md](references/hw-feature-selection-subskill.md)

Do not recommend TMA, WGMMA, TMEM, or CLC here.

### Branch B: SM90 / Hopper

Typical questions:

- should the kernel use TMA
- does WGMMA plus warp specialization fit the workload
- is grouped or irregular scheduling the real architecture-specific issue
- is register pressure still the main accumulator-lifetime problem

Treat H100, H20, H800, and other SM90-family targets as Hopper here. Resolve
the exact device before assuming every H100 resource limit or throughput ratio
applies, and do not import SM100 / Blackwell-only features such as TMEM or
`tcgen05.mma` into this branch.

Read:

- [references/h100-sm90-subskill.md](references/h100-sm90-subskill.md)
- [references/h100-sm90-moe-practices.md](references/h100-sm90-moe-practices.md)
  when the Hopper question is specifically MoE, grouped expert GEMM,
  dispatch/GEMM fusion, or FP8 fused MoE forward planning
- [references/scheduler-and-launch-control-subskill.md](references/scheduler-and-launch-control-subskill.md)
  when scheduling mode is the real question
- [references/cutlass-hw-source-map-subskill.md](references/cutlass-hw-source-map-subskill.md)
  when the choice depends on CUTLASS or CuTe internals

### Branch C: SM100 / B200 data-center Blackwell

Typical questions:

- should the kernel move to `tcgen05.mma`
- is the rewrite justified by TMEM accumulator behavior
- should the schedule be 1SM or 2SM
- does the kernel benefit from clusters, DSMEM, or CLC
- does the low-precision path satisfy SM100-scale layout contracts

Read:

- [references/b200-sm100-subskill.md](references/b200-sm100-subskill.md)
- [references/scheduler-and-launch-control-subskill.md](references/scheduler-and-launch-control-subskill.md)
  when CLC, static persistent, Stream-K, or PDL is in play
- [references/blackwell-precision-contracts-subskill.md](references/blackwell-precision-contracts-subskill.md)
  when block-scaled FP8, NVFP4, MXFP8, or scale-layout compatibility is part
  of the decision
- [references/cutlass-hw-source-map-subskill.md](references/cutlass-hw-source-map-subskill.md)
  when implementation details live in CUTLASS or CuTe

### Branch D: consumer Blackwell / CC 12.x

Typical questions:

- does public Blackwell guidance actually apply to this device
- which datacenter-Blackwell ideas transfer and which do not
- is the user accidentally asking for TMEM- or CLC-specific behavior that
  does not exist here

Do not blindly reuse the SM100 / B200 branch.

What to do:

- first use
  [references/hw-feature-selection-subskill.md](references/hw-feature-selection-subskill.md)
  to identify which surfaces are even plausible
- if the needed distinction is still unclear, do targeted docs or source
  research before recommending a TMEM-, CLC-, or SM100-specific rewrite
- if the real blocker is low-precision format semantics rather than hardware
  feature choice, route to `krnopt-low-precision-kernel-formats`

### Branch E: unresolved or mixed guidance

If the user is mixing Hopper and Blackwell advice, or mixing consumer and
datacenter Blackwell, stop and resolve the branch first.

Do not produce a hardware-aware plan until the branch is explicit.

## Step 4: Choose The Hardware Surfaces That Matter

If the request is about GPU utilization, first name what needs to be better
utilized: Tensor Core / MMA, CUDA core / SFU, HBM, L2, SMEM / MIO, registers
or TMEM, scheduler occupancy, or launch/runtime. Then name the waste source:
padding, masked lanes, low tile reuse, descriptor churn, scale/dequant traffic,
spills, tail waves, fragmented launches, or non-MMA epilogue work.

Check these surfaces in order:

1. instruction family
2. accumulator location
3. staging path
4. scheduler or launch path
5. precision and scale-layout contract

Use
[references/hw-feature-selection-subskill.md](references/hw-feature-selection-subskill.md)
when the first question is simply which surfaces are even plausible.

Use
[references/hardware-utilization-targets-subskill.md](references/hardware-utilization-targets-subskill.md)
when the hardware-aware question is specifically about enhancing GPU
utilization or explaining why a hardware feature should keep a pipeline busier.

## Step 5: Map Feature To Code Shape

For every candidate hardware feature, answer:

- what workload shape justifies it
- what code shape is required to use it well
- what contract must the kernel satisfy
- what fallback path exists if the feature is a poor fit

Do not recommend a feature just because the architecture exposes it.

## Condition -> What To Look At

Use this table when deciding what material to load next.

| Condition | What to do next | Material to load |
| --- | --- | --- |
| Need the first-pass hardware surface list | Identify plausible instruction, staging, scheduler, and precision surfaces | [references/hw-feature-selection-subskill.md](references/hw-feature-selection-subskill.md) |
| Target is Hopper / SM90 | Evaluate TMA, WGMMA, warp specialization, register-pressure implications | [references/h100-sm90-subskill.md](references/h100-sm90-subskill.md) |
| Target is Hopper / SM90 MoE | Evaluate SM90 grouped expert GEMM, FP8 scale contracts, persistent scheduling, descriptor handling, and SM90-vs-SM100 MoE boundaries | [references/h100-sm90-moe-practices.md](references/h100-sm90-moe-practices.md) |
| Target is B200 / SM100 | Evaluate `tcgen05.mma`, TMEM, 1SM vs 2SM, clusters, DSMEM, CLC | [references/b200-sm100-subskill.md](references/b200-sm100-subskill.md) |
| Main blocker is scheduler mode | Compare static, persistent, CLC, Stream-K, or PDL style choices | [references/scheduler-and-launch-control-subskill.md](references/scheduler-and-launch-control-subskill.md) |
| Main blocker is hardware-specific GPU utilization | Map the underutilized path to instruction, staging, accumulator, scheduler, launch, or precision surfaces | [references/hardware-utilization-targets-subskill.md](references/hardware-utilization-targets-subskill.md) |
| Main blocker is Blackwell low-precision compatibility | Check block-scaled FP8, NVFP4, MXFP8, or scale-layout contracts | [references/blackwell-precision-contracts-subskill.md](references/blackwell-precision-contracts-subskill.md) |
| Choice depends on CUTLASS or CuTe internals | Find where the contract lives in the source stack | [references/cutlass-hw-source-map-subskill.md](references/cutlass-hw-source-map-subskill.md) |
| Consumer Blackwell details are unclear | Stop and resolve which public Blackwell guidance transfers | Targeted docs or source research |

## Output Contract

The output of this skill should contain:

- target hardware and compute capability
- workload or kernel shape
- chosen architecture branch
- utilization target and useful-work waste source when utilization is the goal
- candidate hardware features worth considering
- feature-to-code mapping
- contracts that must be satisfied
- required architecture-specialized build target when the chosen path needs
  target-specific instructions or features
- confirmation and rejection metrics for the hardware-utilization hypothesis
- likely wins
- likely risks or traps
- fallback path
- next material or source layer to inspect
- next owner or next skill

Use
[references/hardware-aware-plan-subskill.md](references/hardware-aware-plan-subskill.md)
when the user wants a structured plan rather than free-form notes.

## Routing

- `krnopt-cuda-profiling` owns the measured answer to what is actually hot and
  which bottleneck dominates right now.
- `krnopt-cuda-generic-optimization` owns ranked next experiments once the
  question is no longer mainly about architecture-specific feature choice.
- `krnopt-cuda-structural-optimization` owns broader kernel or hot-path
  redesign once hardware feature choice is only one ingredient in the plan.
- `krnopt-cuda-domain-optimization` owns workload-domain pattern selection once
  the primary question is MoE routing, grouped GEMM, dispatch/combine, expert
  MLP fusion, padding elimination, persistent MoE scheduling, or another
  covered domain playbook rather than hardware-feature choice.
- `krnopt-cuda-coding` owns source-writing and source-review work once the
  hardware-aware decision is settled.
- `krnopt-low-precision-kernel-formats` owns basic FP8, block-scale, NVFP4,
  MXFP, and dequant-contract semantics when hardware feature choice is not yet
  the real question.
- Targeted docs or source research is the fallback when a hardware detail,
  CUTLASS contract, or consumer-vs-datacenter distinction is still unclear.
