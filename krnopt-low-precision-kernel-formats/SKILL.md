---
name: krnopt-low-precision-kernel-formats
description: "Use and understand low-precision CUDA kernel formats. Use when the task is to understand, compare, choose, or apply FP8, block-scaled FP8, NVFP4, MXFP4/6/8, dynamic quantization, or related low-precision kernel formats; interpret the scale-layout, packing, dtype-path, or dequant contract a kernel must obey; or decide which other CUDA skill should handle profiling, optimization direction, structural redesign, hardware-specific format choice, or source implementation next."
---

# Low-Precision CUDA Kernel Formats

Use this skill when the question is about the format contract itself.

This skill answers questions like:

1. what does this low-precision format mean
2. what scale layout or packing contract does the kernel need to obey
3. how is this format different from a nearby one
4. which skill should handle the next step after the format is understood

## What This Skill Owns

This skill owns format and contract understanding for topics such as:

- FP8 E4M3 or E5M2
- block-scaled FP8
- NVFP4
- MXFP4, MXFP6, and MXFP8
- dynamic activation quantization
- transparent or engine-managed quantization
- low-precision KV-cache formats

Typical questions include:

- "Is this contest tensor format ordinary FP8 or block-scaled FP8?"
- "What scale tensor shape does this kernel need?"
- "Does dequant belong in the prologue, epilogue, or MMA path?"
- "Why is NVFP4 not the same thing as the contest FP8 format?"
- "Is this a format-choice problem, a profiling problem, or a source-writing problem?"

## What This Skill Does Not Own

Keep the boundaries sharp:

- `krnopt-cuda-profiling` owns finding whether quant, dequant, scale loads, or
  the wrong low-precision path is the bottleneck in a real profile.
- `krnopt-cuda-generic-optimization` owns choosing the next optimization
  experiment after the bottleneck and source region are already diagnosed.
- `krnopt-cuda-coding` owns implementing the source changes once the format
  contract is decided.
- `krnopt-cuda-structural-optimization` owns broader kernel or hot-path
  redesign once the format contract is understood and the real question is
  stage placement, boundary redesign, or replanning the hot path.
- `krnopt-hw-aware-optimization` owns architecture-specific format choice when
  the decision depends on SM generation, Tensor Core contracts, or hardware
  feature availability more than generic format semantics.

## Core Workflow

Follow this order:

1. identify the value format
2. identify the scale format and granularity
3. identify where dequant or requant happens
4. identify the hardware or library contract involved
5. identify whether that contract requires an architecture-specialized build
   target such as `sm_90a` or `sm_100a`
6. state the practical implications for the kernel
7. route to the next skill if the task is no longer about the format itself

In compact form:

```text
format question
  -> value dtype
  -> scale dtype and granularity
  -> packing/layout contract
  -> dequant or requant location
  -> next owning skill
```

## Format Questions To Resolve

When reading or comparing a low-precision path, resolve these questions in
order:

1. what are the stored values
2. what are the stored scales
3. what block, group, token, or tensor granularity do the scales apply to
4. are scales packed separately, interleaved, or transformed for a library ABI
5. is the kernel expected to dequant before MMA, inside MMA, or in a fused
   surrounding stage
6. is the format a contest contract, a library contract, or a hardware-native
   contract

Do not jump straight from "low precision" to a specific use case. First pin
down the actual contract.

## Common Placement Rules

Use these rules to decide where the question belongs:

- If the user is asking what the format means:
  stay in this skill.
- If the user is asking why the format path is slow in practice:
  route to `krnopt-cuda-profiling`.
- If the user is asking what experiment to try next after diagnosis:
  route to `krnopt-cuda-generic-optimization`.
- If the format contract is already clear and the question becomes how to
  redesign kernel structure, move stages, or replan the hot path around Q/DQ
  or scale movement:
  route to `krnopt-cuda-structural-optimization`.
- If the user is asking how to write or patch the kernel:
  route to `krnopt-cuda-coding`.
- If the user is choosing between Blackwell-specific paths such as NVFP4,
  MXFP4, or `tcgen05.mma` contracts:
  route to `krnopt-hw-aware-optimization` after the basic format contract is
  clear.

## Contest And Blackwell Guardrail

Do not blur contest FP8 block scaling together with Blackwell-native FP4 or MX
formats.

The most important separation is:

- contest block-scaled FP8:
  FP8 E4M3 values with contest-defined scale layout and granularity
- Blackwell microscaling families:
  NVFP4 or MXFP formats with hardware- and library-specific block contracts

These families may share ideas such as block scales or fused dequant, but they
are not interchangeable contracts.

Read
[references/format-families-and-contracts.md](references/format-families-and-contracts.md)
when the task is to compare specific format families or understand their kernel
implications.

## Output Contract

The output of this skill should be a compact format handoff containing:

- named format or candidate formats
- value dtype
- scale dtype and granularity
- packing or layout contract
- where quant, dequant, or requant logically belongs
- hardware or library assumptions
- architecture-specialized build target required by the format path, if any
- why the format matters to the kernel
- next owning skill

Prefer concrete statements such as:

- "This is block-scaled FP8, not plain FP8, so scale layout is part of the
  kernel ABI."
- "This is a format-contract question, not yet an optimization-direction
  question."
- "NVFP4 is background knowledge here, but the contest path still obeys FP8
  block-scale rules."

## Integrated Reference

Use
[references/format-families-and-contracts.md](references/format-families-and-contracts.md)
as the main reference for:

- FP8 versus block-scaled FP8
- NVFP4 versus MXFP families
- dynamic or transparent quantization families
- KV-cache low-precision formats
- skill routing after the format question is resolved
