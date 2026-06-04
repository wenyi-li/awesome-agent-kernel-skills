---
name: "kernel-creator"
description: "Guides writing and refactoring GPU kernels across CUDA and DSLs. Invoke when implementing kernel files that must stay importable, independently testable, profileable, and optimizable."
---

# Kernel Creator

Use this skill whenever the user wants to write, edit, refactor, or organize GPU
kernel code in CUDA or any GPU DSL, including Triton, TileLang, CuteDSL, and
similar systems. This skill is about authoring kernels as reusable implementation
units rather than one-off notebook snippets or tightly coupled benchmark scripts.

## Core Goal

Write kernel code so each kernel implementation is:

1. Importable with a simple statement such as `from ... import some_kernel_impl`
2. Testable with a small correctness-focused script
3. Profileable with a small benchmarking script
4. Optimizable in isolation without entangling unrelated kernels or host logic

The implementation file and the scripts around it should be decoupled on purpose.
Do not bury kernel logic inside large demo programs, training loops, or ad hoc
benchmark files.

## Non-Negotiable Rules

### 1. Keep kernel implementations decoupled

- Put the kernel implementation in its own file or module whenever practical.
- Expose a small public surface: the kernel object, launcher, wrapper function,
  or factory needed by callers.
- Make imports cheap and side-effect free. Importing the module must not start
  benchmarking, allocate giant tensors, JIT dozens of variants, or parse CLI args.
- Keep framework-specific test harness code outside the implementation module.

### 2. Separate implementation from verification and profiling

For a kernel named `some_kernel_impl`, prefer a structure like:

```text
some_kernel_impl.py / .cu / .cc
test_some_kernel_impl.py
profile_some_kernel_impl.py
debug_some_kernel_impl.py      # optional
```

The exact suffix and language may vary, but the separation of concerns should not.

- The implementation file contains the kernel definition and the thinnest useful
  wrapper or launcher.
- The test script checks correctness against a trusted reference.
- The profiling script measures performance and can sweep shapes, dtypes, tile
  sizes, or launch parameters.
- The debug script is optional and only exists when the kernel needs extra
  instrumentation or reduced repro cases.

### 3. Design for isolated optimization

- Make tile sizes, block sizes, stage counts, vector widths, and related tuning
  parameters explicit.
- Prefer parameterized kernels or a thin config object over hard-coded magic.
- Keep the host wrapper minimal so the hot kernel path is easy to benchmark alone.
- If the kernel has fused behavior, document the fusion boundary clearly.

### 4. Preserve a clean operator contract

Before writing the kernel, define:

- Input and output shapes
- Dtypes and memory layout assumptions
- Aliasing and in-place behavior
- Numerically sensitive behavior such as accumulation dtype, reduction order,
  masking semantics, or tolerated error
- Static versus dynamic shape assumptions

If these constraints are unclear, clarify them before optimizing.

## Recommended Workflow

### 1. Start from the operation contract

Write down the mathematical operation first. For any nontrivial kernel, also write
a trusted reference implementation in PyTorch, NumPy, plain CUDA host code, or
another obviously correct baseline.

### 2. Pick the smallest reusable implementation unit

Choose the narrowest kernel boundary that still makes sense to import and test.
Good examples:

- a GEMM kernel
- a fused epilogue kernel
- a reduction kernel
- a layout conversion kernel
- one attention inner loop kernel

Bad examples:

- an entire training step embedded inside a single kernel file
- a benchmark script that also defines the only copy of the kernel

### 3. Write the implementation file first

The implementation file should usually contain:

- the kernel definition
- launch configuration or compile-time parameters
- a thin public entrypoint
- the minimum helpers needed only by this kernel

Avoid packing it with:

- extensive CLI code
- correctness assertions for many unrelated cases
- plotting
- benchmark tables
- autotuning report generation

### 4. Add a small correctness script

The test script should:

- import the kernel module directly
- create deterministic inputs
- compare against a trusted reference
- cover representative shapes and edge cases
- fail loudly on mismatch

Keep the script small enough that it becomes the default repro for future edits.

### 5. Add a small profiling script

The profiling script should:

- import the same implementation module
- isolate setup from timed execution
- report enough context to compare runs fairly
- make it easy to sweep shapes, dtypes, and tuning knobs

The goal is to answer "is this kernel faster?" without rewriting the harness each
time.

## File Design Guidelines

### Implementation modules

Implementation files should favor:

- explicit names such as `matmul_kernel.py`, `fused_rmsnorm.cu`,
  `attention_block_kernel.py`
- one clear exported kernel entrypoint per file when possible
- local helper functions only when they truly belong to that kernel
- comments explaining non-obvious mapping, memory movement, or synchronization

Implementation files should avoid:

- hidden global state
- mixed ownership of multiple unrelated kernels
- benchmark-only code paths as the main API
- forcing callers through heavyweight abstractions just to launch one kernel

### Host wrappers

Thin wrappers are good when they:

- validate shapes and dtypes
- choose launch parameters
- dispatch between a small number of variants
- convert layouts or strides only when necessary

Thin wrappers are bad when they:

- conceal important performance choices
- mix data loading, model logic, and kernel launch
- make it impossible to benchmark the kernel without the full application

## DSL-Agnostic Guidance

Apply the same structure whether the kernel is written in:

- CUDA C++ / PTX / inline CUDA
- Triton
- TileLang
- CuteDSL
- another GPU kernel DSL

The syntax differs, but the engineering rules stay the same:

- keep the kernel implementation importable
- keep the test harness separate
- keep the profiling harness separate
- expose tuning parameters clearly
- make optimization work local and repeatable

Do not overfit the structure to one DSL's tutorial style if it makes the module
harder to reuse elsewhere.

## Refactor Rules

When refactoring existing kernel code:

1. Extract the kernel definition out of monolithic scripts first.
2. Preserve behavior before attempting performance changes.
3. Move correctness checks into a dedicated test script.
4. Move timing code into a dedicated profiling script.
5. Reduce duplication between sibling kernels, but do not create a giant shared
   abstraction that obscures kernel-specific logic.

Prefer clear duplication over premature framework-building. Share helpers only when
the shared boundary is stable and genuinely improves maintainability.

## Review Checklist

Before finishing a kernel authoring task, check:

- Can another file import the kernel directly?
- Can I validate correctness without editing the implementation file?
- Can I benchmark performance without editing the implementation file?
- Are launch and tuning parameters visible enough to optimize later?
- Is the implementation module focused on one kernel or one tight family of
  variants?
- Did I avoid coupling the kernel to unrelated training, serving, or demo code?

If the answer to any of these is no, restructure before polishing.

## What To Avoid

- Writing the only copy of a kernel inside a benchmark file
- Coupling the kernel to a notebook, demo, or training script
- Hiding launch logic behind large application objects
- Mixing correctness testing, profiling, and kernel definition in one file
- Hard-coding tuning knobs with no easy way to sweep or override them
- Refactoring multiple unrelated kernels into one "universal" abstraction too early

## Output Style For This Skill

When using this skill to help the user:

- lead with the implementation module boundary
- propose the smallest clean file split
- keep examples short and reusable
- prefer importable kernel entrypoints over monolithic demos
- mention test and profiling harnesses as first-class deliverables, not afterthoughts
