---
name: krnopt-cuda-coding
description: "Write good, correct, efficient CUDA source code. Use when the task is authoring or modifying `.cu` code or CUDA device helpers and the main need is to choose a sound ownership model, make correctness explicit, shape memory access and reuse well, manage shared memory and register tradeoffs, keep synchronization and control flow sane, and choose dtypes or instruction paths deliberately. Do not use this skill for profiler-driven diagnosis, benchmark interpretation, ranked optimization planning, or big-picture kernel or hot-path redesign; route measured-evidence analysis to `krnopt-cuda-profiling`, next-experiment planning to `krnopt-cuda-generic-optimization`, and structural replanning to `krnopt-cuda-structural-optimization`."
---

# Mastery CUDA Coding

Use this skill when the job is to write or modify CUDA code so it is:

1. correct
2. structurally efficient
3. easy to validate and profile later

This skill is about writing CUDA well. It is not the owner of profiler-driven
diagnosis, broad source review, or ranked optimization planning.

## Entry Condition

Use this skill when the task is mainly:

- authoring a new CUDA kernel
- modifying an existing CUDA kernel or device helper
- restructuring CUDA source for correctness or better local code shape
- choosing a clean thread, warp, block, memory, or dtype design before deeper
  profiling

Use another skill instead when:

- the main question is "what is actually hot" or "which bottleneck dominates":
  use `krnopt-cuda-profiling`
- the main question is "what experiment should we try next from diagnosed evidence":
  use `krnopt-cuda-generic-optimization`
- the main question is to replan or redesign the kernel or hot path from a
  big-picture view:
  use `krnopt-cuda-structural-optimization`
- the main question is mainly about architecture-specific feature selection:
  use `krnopt-hw-aware-optimization`
- the main question is mainly about FP8, block-scale, NVFP4, MXFP, or dequant
  contracts:
  use `krnopt-low-precision-kernel-formats`

## Core Workflow

Write CUDA in this order:

1. make correctness and validation explicit
2. choose thread, warp, and block ownership clearly
3. shape memory access and reuse deliberately
4. add shared memory only when it repairs access or creates real reuse
5. manage register footprint and residency tradeoffs
6. keep synchronization and control flow legible
7. choose dtype and instruction path deliberately
8. leave runtime-dominance questions to profiling

In compact form:

```text
correctness
  -> ownership model
  -> memory and reuse
  -> shared/register tradeoffs
  -> sync and control
  -> dtype and instruction path
  -> ready for profiling
```

The main discipline is:

- correctness first
- think in mappings, not statements
- treat memory traffic as a design budget
- make performance intent visible in the source
- prefer simple, testable changes over opaque cleverness

## Step 1: Make Correctness Cheap To Recheck

Before worrying about speed, make the kernel easy to re-validate.

The code should make these things explicit:

- logical index or tile ownership
- bounds and edge handling
- synchronization semantics
- numeric or dtype semantics
- a cheap validation path

Good writing habit:

- keep a known-good reference path nearby
- re-check after each optimization family
- treat asynchronous errors and synchronization semantics as part of correctness

Use
[references/correctness-first-subskill.md](references/correctness-first-subskill.md)
when correctness posture is the main blocker.

## Step 2: Choose A Clear Ownership Model

The source should make it obvious:

- what one thread computes
- what one warp computes
- what one block computes

If those three questions are hard to answer from the code, the kernel is too
hard to optimize safely.

Good default discipline:

- keep mapping from `threadIdx` and `blockIdx` to work inspectable
- align block size with warp structure
- pick launch shape to match data shape before chasing cleverness

## Step 3: Shape Memory Access And Reuse

Before adding optimizations, make the memory plan explicit:

- do adjacent lanes touch adjacent addresses
- which values come from global memory
- where reuse lives: global, shared, or registers
- whether vector loads or stores are legal and aligned

This is where most source-level performance wins begin.

Use
[references/mapping-and-memory-subskill.md](references/mapping-and-memory-subskill.md)
when the main question is coalescing, traffic shape, or reuse.

## Step 4: Use Shared Memory And Registers Deliberately

Shared memory should have a job:

- repair access shape
- create real reuse
- reduce redundant global traffic

Registers should also be treated as a budget:

- large per-thread state is a real design cost
- unrolling and wide tiles increase live state
- spills turn private state into off-chip traffic

Do not add shared memory or larger tiles just because they sound fast.

Use
[references/shared-register-residency-subskill.md](references/shared-register-residency-subskill.md)
when the real tradeoff is shared memory, register pressure, or residency.

## Step 5: Keep Control Flow And Synchronization Legible

The code should make these things easy to inspect:

- where warps diverge
- where barriers are truly required
- whether work arrives at synchronization points uniformly
- whether control flow is fighting warp structure

Short, warp-aware, easy-to-explain control flow is usually a better starting
point than ornate branching.

Use
[references/control-flow-and-dtype-subskill.md](references/control-flow-and-dtype-subskill.md)
when divergence, barriers, or synchronization shape are the main concern.

## Step 6: Choose Dtype And Instruction Path Deliberately

Do not let the hardware path be accidental.

Check:

- literals and helper functions do not silently promote precision
- dtypes match the intended tensor-core, packed, or vector path
- alignment and tile choices satisfy the intended path's contract
- generic math is not quietly selecting a slow instruction mix
- kernel build flags target the architecture-specific feature set when the
  source depends on it; use specialized targets such as `sm_90a` or `sm_100a`
  instead of general targets such as `sm_90` or `sm_100`

Low-precision format semantics belong in
`krnopt-low-precision-kernel-formats` when the basic contract itself is still
unclear.

## Writing Rules

These are the best-practice rules this skill should reinforce:

- make correctness explicit before performance claims
- keep indexing and ownership inspectable by hand
- make lane-to-address behavior obvious in the source
- use shared memory for access repair or reuse, not as decoration
- treat occupancy as a tradeoff, not a score
- keep control flow warp-aware
- use dtypes, literals, and intrinsics deliberately
- compile architecture-sensitive kernels with the specialized target flag
  required by the intended instructions or features, such as `sm_90a` or
  `sm_100a`, not only the general target such as `sm_90` or `sm_100`
- keep the code easy to validate after each change

## Output Contract

The output of this skill should be a coding handoff containing:

- target kernel or code region
- intended ownership model
- correctness and validation posture
- memory access and reuse plan
- shared-memory and register tradeoffs
- synchronization and control-flow plan
- dtype and instruction-path plan
- architecture-specific build target when the kernel relies on target-specific
  instructions or features
- risks to validate next
- whether profiling should come next

Use
[references/writing-cuda-subskill.md](references/writing-cuda-subskill.md)
as the default output shape when the user wants practical writing guidance.

## Integrated Subskills

Use these local references as integrated subskills for this skill:

- Start with
  [references/writing-cuda-subskill.md](references/writing-cuda-subskill.md)
  when the task is to author or modify CUDA code.
- Use [references/coding-mindset-subskill.md](references/coding-mindset-subskill.md)
  when the user needs the broad programming mindset behind the writing choices.
- Use [references/correctness-first-subskill.md](references/correctness-first-subskill.md)
  when correctness, bounds, synchronization, or validation posture need to be
  established first.
- Use [references/mapping-and-memory-subskill.md](references/mapping-and-memory-subskill.md)
  when the main question is thread-to-data mapping, access shape, or reuse.
- Use [references/shared-register-residency-subskill.md](references/shared-register-residency-subskill.md)
  when shared-memory layout, register footprint, or residency risks are the
  main concern.
- Use [references/control-flow-and-dtype-subskill.md](references/control-flow-and-dtype-subskill.md)
  when control flow, synchronization shape, instruction path, or datatype
  choices need review while writing.

Load the deeper reference that matches the current writing problem instead of
dumping all CUDA lore into one prompt.
