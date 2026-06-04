---
name: "tilelang-torch-profiler"
description: "Profiles TileLang and mixed PyTorch+TileLang workloads with torch.profiler. Invoke when users want traces, operator breakdowns, memory profiling, or a lightweight first-pass performance diagnosis."
---

# TileLang Torch Profiler

Use this skill when the user wants a lightweight, always-available profiling
workflow for TileLang kernels or mixed PyTorch + TileLang programs using
`torch.profiler`.

This skill is the first-pass profiling tool. It answers:

- which operator or kernel dominates runtime
- whether the hot path is mostly GPU work, CPU work, or launch overhead
- how to generate and read a Chrome trace or Perfetto trace
- how to estimate whether a kernel looks IO-bound, CUDA-core bound, or
  Tensor-core bound from profiler timing
- how to compare a TileLang kernel against a torch reference in one trace
- how to inspect memory behavior with the profiler's allocation views

Keep the analysis grounded in the assets bundled with this skill:

- `scripts/profile_template.py`
- `references/trace-reading-guide.md`
- `references/bottleneck-classification.md`

Do not rely on outside docs unless the user explicitly asks for them.

## When To Invoke

Invoke this skill when the user asks to:

- profile a TileLang kernel or a TileLang-backed operator
- understand why a TileLang program is slow
- generate a Chrome trace, Perfetto trace, or TensorBoard profiler trace
- inspect CPU vs CUDA time in a mixed PyTorch + TileLang forward pass
- measure launch overhead in a many-small-kernel pipeline
- compare a TileLang kernel against a torch, cuBLAS, or aten reference
- inspect profiler-visible memory allocation behavior
- get a lightweight diagnosis before escalating to deeper GPU tools

Also invoke when the user uses phrases such as:

- "profile this kernel"
- "why is this slow"
- "show me the kernel timeline"
- "torch profiler"
- "chrome trace"
- "perfetto"
- "memory timeline"

## When Not To Invoke

Do not use this skill as the primary tool when the user needs:

- DSL syntax help, semantic explanation, or example lookup for TileLang itself
- code authoring or refactoring to make kernels importable, testable, or
  profileable
- low-level hardware-counter analysis such as occupancy, warp stalls, bank
  conflicts, or register pressure confirmation

In those cases, collaborate instead of forcing this skill to do everything:

- hand TileLang language or operator questions to `tilelang-wiki`
- hand kernel organization or refactoring work to `kernel-creator`
- use this skill first for attribution, then recommend deeper GPU tooling when
  profiler timing alone cannot explain underperformance

## Core Workflow

### 1. Separate compile time from profile time

TileLang JIT compilation must not contaminate the measured trace.

- build the kernel before the profile region
- run warmup calls before profiling
- call `torch.cuda.synchronize()` after warmup
- prefer at least 10 warmup iterations for autotuned or freshly compiled kernels

If the first iteration in the trace is far slower, or compile log lines appear
inside the trace, the warmup was insufficient.

### 2. Capture both CPU and CUDA activity

Use `torch.profiler.profile(...)` with:

- `ProfilerActivity.CPU`
- `ProfilerActivity.CUDA`

Without CUDA activity, the user only sees host-side launch overhead.

### 3. Label regions intentionally

Wrap candidate implementations with `record_function(...)` so the trace and
summary table contain user-meaningful names.

Typical labels:

- `tilelang_kernel`
- `torch_ref`
- `forward_pass`
- `attention_block`

This makes one-profile comparisons much easier.

### 4. Read the summary table first

Start triage from:

```python
prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
```

Use the table to answer:

- which kernel has the highest `Self CUDA`
- whether `Self CPU` is large enough to suggest host overhead
- whether `cuLaunchKernel` CPU time is suspiciously close to kernel GPU time
- whether the kernel of interest appears both as a `record_function` range and
  as the underlying CUDA kernel row

### 5. Export and inspect a trace

Always offer trace export when the user needs timing structure or overlap:

- `prof.export_chrome_trace("trace.json")`

View in:

- `chrome://tracing`
- `https://ui.perfetto.dev/`

Use the trace for:

- gaps between kernels
- CPU and GPU overlap
- stream structure
- unexpected synchronizations
- first-touch JIT or module-loading artifacts that escaped warmup

### 6. Convert timing into a diagnosis

Use `references/bottleneck-classification.md` for the decision flow.

High-level rules:

- achieved bandwidth near HBM peak suggests IO-bound behavior
- achieved TFLOPS near scalar peak suggests CUDA-core bound behavior
- achieved TFLOPS near tensor-core peak suggests Tensor-core bound behavior
- far below every plausible peak means timing alone is insufficient; the next
  step is source inspection or deeper GPU tooling

### 7. Corroborate with generated source when needed

When the user asks whether TileLang actually lowered to tensor-core code, use
the generated CUDA source as a quick corroboration step.

Important caution:

- TileLang often hides MMA emission behind included template headers
- raw `wgmma` or `mma.sync` tokens may not appear in the top-level `.cu` text
- look for TileLang MMA template includes, `CUtensorMap`, `mbarrier`, and
  related async-path indicators before concluding the kernel is scalar-only

If the source suggests tensor-core lowering but achieved performance is still
poor, timing attribution has reached its limit and the user should escalate to
deeper GPU analysis.

## Practical Patterns

### Single kernel profile

Use `scripts/profile_template.py` as the starting point when the user wants a
minimal script that:

- constructs a kernel
- warms it up
- profiles both TileLang and a reference path
- prints a summary table
- writes a trace file
- optionally computes achieved TFLOPS from measured time

### Long-running loop or model profile

Prefer scheduled profiling when the user is profiling:

- a training loop
- repeated inference steps
- a multi-step forward pass

The scheduled profiler avoids over-weighting cold startup behavior and bounds
profiling overhead.

### Memory profiling

Turn on memory profiling when the user is debugging:

- unexpected memory growth
- OOM conditions
- allocator churn

Use `profile_memory=True` and export a memory timeline, or advise the newer
CUDA memory snapshot path when allocator-level inspection is required.

## Interoperability

This skill should cooperate cleanly with other skills instead of duplicating
their roles.

### With `kernel-creator`

Use `kernel-creator` when the user needs to restructure code so profiling is
easy and reusable. This skill assumes there is a kernel entrypoint that can be
imported and called in a small harness. If that is not true yet:

- first ask for or propose an importable kernel boundary
- separate implementation from correctness and profiling harnesses
- then return here to profile the result

### With `tilelang-wiki`

Use `tilelang-wiki` when the user's blocker is not "where is the time going?"
but instead:

- how a TileLang construct works
- which local example or operator family matches the target kernel
- whether a lowering pattern is supported
- how `T.gemm`, `T.Pipelined`, memory scopes, or target features behave

A common collaboration pattern is:

1. use `tilelang-wiki` to ground semantics or operator choice
2. use `kernel-creator` to make the kernel importable and profileable
3. use this skill to attribute runtime and inspect traces

## Answering Rules

When helping the user with this skill:

- lead with the profiling objective and what will be measured
- explicitly separate compile cost from steady-state runtime
- prefer a minimal runnable harness over abstract advice
- name the exact profiler columns used for the conclusion
- distinguish attribution from root-cause proof
- state when a conclusion is strong, weak, or only provisional
- recommend escalation only when profiler-visible evidence runs out

## Output Style

A strong answer from this skill usually includes:

- a short profiling harness or a targeted edit plan
- the exact `torch.profiler` options to use
- what table sort key to inspect first
- what the trace should reveal
- a concise bottleneck classification
- the most likely next step if the result is inconclusive
