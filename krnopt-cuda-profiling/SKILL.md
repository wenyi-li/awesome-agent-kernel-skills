---
name: krnopt-cuda-profiling
description: "Identify CUDA bottlenecks from source code and profiling evidence. Use when the task is to inspect CUDA source, analyze `nsys` or `ncu` profiling data, determine which kernel or code region is actually limiting performance, and attribute the bottleneck back to specific source blocks, loops, memory paths, or source lines when the evidence supports it. Do not use this skill for choosing optimization directions or rewriting the code; route localized source-plus-evidence optimization planning to `krnopt-cuda-generic-optimization`, broader redesign handoffs to `krnopt-cuda-structural-optimization`, and diagnosed MoE or other covered domain-pattern choices to `krnopt-cuda-domain-optimization`."
---

# Identify CUDA Bottlenecks From Profiling

Use this skill when the job is to answer:

1. where time is going
2. what bottleneck family limits the hot kernel
3. which source region most likely causes that bottleneck

This skill owns profiling-driven diagnosis and profiler-to-source attribution.
It does not own rewrite planning.

## Entry Condition

Use this skill when the input includes one or more of:

- CUDA source code
- `nsys` traces or summaries
- `ncu` reports or counters
- NVTX phase context
- a suspected hot kernel that still needs confirmation

If the hotspot is already known, the bottleneck is already classified, and the
source region is already attributed, route to
`krnopt-cuda-generic-optimization` for a localized next experiment or to
`krnopt-cuda-structural-optimization` when the diagnosis points to a broader
kernel or hot-path redesign. If the diagnosed bottleneck is best addressed by a
MoE or other covered workload-domain pattern, such as routing/dispatch,
grouped GEMM, padding elimination, expert MLP fusion, combine/reduction, or
persistent MoE scheduling, route to `krnopt-cuda-domain-optimization`.

## Core Workflow

Follow this order:

1. identify the workload, phase, and target being profiled
2. when multiple GPUs are available, pick the most idle GPU before profiling
3. use `nsys` to find where time actually goes
4. select the dominant kernel or phase
5. use focused `ncu` to classify the bottleneck coarsely
6. choose one drill-down question based on the bottleneck family
7. use source views, source counters, kernel names, and NVTX only as needed to
   attribute the bottleneck back to source
8. produce a diagnosis handoff with measured facts separated from inference

In compact form:

```text
source + profiler evidence
  -> hotspot
  -> bottleneck class
  -> focused drill-down
  -> source attribution
  -> diagnosis handoff
```

The main discipline is:

- prefer the most idle visible GPU for profiling runs
- `nsys` before deep `ncu`
- when profiling this repository through `mlsys-cli eval timing`, pass
  `--runner persistent`; the default isolated runner can hang, miss
  child-process CUDA work, or produce inconclusive `nsys`/`ncu` reports
- when reducing benchmark volume for profiling, pass timing-count flags
  explicitly, such as `--warmup-runs 1 --iterations 1 --num-trials 1`, and
  label the result as a profiling or smoke override rather than official-style
  timing
- coarse classification before detailed counters
- one bottleneck family at a time
- measured facts before inference
- source attribution before rewrite planning

## The Three Diagnosis Questions

Ask these in order:

1. where is time going
2. what resource limits the chosen kernel
3. which source region should be blamed next

Do not skip ahead to stall details before hotspot selection and first-pass
classification are done.

## Bottleneck Branches

Use one branch at a time.

| Profile symptom | Next diagnosis question | Typical source focus |
| --- | --- | --- |
| Many tiny launches, host gaps, sync-heavy timeline | Is the real problem above the kernel body? | wrappers, launch path, orchestration boundaries |
| SM throughput low and DRAM throughput low | Is this underfill or poor latency hiding? | launch shape, dependency-heavy loop bodies, oversized blocks |
| Roofline or SOL says memory-bound | Is this bandwidth pressure or memory-latency pressure? | load/store structure, reuse, tiling, staging |
| `long_scoreboard` high | Which global-memory path is creating the wait? | producer loads, coalescing, locality, reuse |
| `short_scoreboard` or `mio_throttle` high | Is shared-memory or MIO pressure the limiter? | SMEM layout, bank conflicts, scalar SMEM traffic |
| Barrier or branch stalls high | Is divergence or synchronization imbalance dominant? | barrier-heavy reductions, warp-misaligned branches |
| Tensor utilization low on GEMM-like code | Is the kernel missing the intended MMA path? | dtype path, tile shape, instruction selection |

Use the branch to decide the next profiler-backed question, not to jump
straight to a rewrite plan.

## Source Attribution Rules

- Treat NVTX as phase correlation, not exact source attribution.
- Treat the stalled line as the waiting site, not automatically the root cause.
- Prefer `-lineinfo` builds and focused source counters when exact attribution
  is needed.
- When profiling architecture-sensitive kernels, keep the same specialized
  architecture target used by the intended implementation, such as `sm_90a` or
  `sm_100a`; do not switch to a general `sm_90` or `sm_100` build just to get
  profiler source attribution.
- Separate measured evidence from inferred blame.

Exact source attribution should come from:

- source views or source counters
- kernel names and launch context
- line-attributed profiler evidence
- disciplined reasoning from waiting site back to the likely producer or
  barrier source

## Output Contract

The output of this skill should contain:

- target workload or phase
- dominant kernel or phase
- bottleneck classification
- measured evidence that supports the classification
- attributed source region or explicit attribution uncertainty
- measured facts versus inference
- profiling build target and whether architecture-specialized flags were
  preserved when relevant
- next owner or next skill

Use [references/evidence-report-subskill.md](references/evidence-report-subskill.md)
when the user wants a structured report.

## Routing

- `krnopt-cuda-generic-optimization` owns the next experiment after diagnosis.
- `krnopt-cuda-structural-optimization` owns the downstream handoff when the
  diagnosis says the current kernel or hot-path structure is the real problem.
- `krnopt-cuda-domain-optimization` owns downstream handoff when the measured
  bottleneck points to a covered workload-domain pattern such as MoE routing,
  grouped GEMM, dispatch/combine, padding, expert MLP fusion, or persistent
  MoE scheduling.
- `krnopt-cuda-coding` owns source-writing and source-review work once the hot
  region is known.
- `krnopt-hw-aware-optimization` owns architecture-specific follow-on
  reasoning.
- Targeted docs or source research is the fallback when profiler behavior or
  hardware meaning is still unclear.

## Integrated Subskills

Use these local references as integrated subskills for this skill:

- Use [references/profiling-mindset-subskill.md](references/profiling-mindset-subskill.md)
  when the main need is diagnosis order and evidence discipline.
- Use [references/nsys-hotspot-subskill.md](references/nsys-hotspot-subskill.md)
  when the first question is where runtime actually goes.
- Use [references/ncu-bottleneck-subskill.md](references/ncu-bottleneck-subskill.md)
  when the hot kernel is known and the next question is which bottleneck family
  actually limits it.
- Use [references/nvtx-instrumentation-subskill.md](references/nvtx-instrumentation-subskill.md)
  when phase names, wrappers, or launch context are needed to correlate
  profiling evidence.
- Use [references/source-attribution-subskill.md](references/source-attribution-subskill.md)
  when profiler evidence needs to be narrowed to source regions, hot blocks, or
  likely source lines.
- Use [references/evidence-report-subskill.md](references/evidence-report-subskill.md)
  when the user wants a structured diagnosis handoff instead of free-form
  profiler notes.
