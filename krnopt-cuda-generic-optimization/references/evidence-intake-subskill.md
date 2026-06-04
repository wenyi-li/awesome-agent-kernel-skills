# Evidence Intake Subskill

Use this integrated subskill at the start of an optimization-direction task.

## First Classify The Evidence Posture

Sort the request into one of four buckets:

- **source only**: no profiler evidence, only static inspection
- **source plus phase context**: source plus NVTX, wrapper, or kernel-name clues
- **source plus summary metrics**: source plus `nsys` or `ncu` headline data
- **source plus line attribution**: source plus source counters or line-level
  hotspot evidence

State the bucket explicitly because it changes what confidence is warranted.

## Separate Facts From Hypotheses

Measured facts include:

- hotspot ranking from `nsys`
- throughput, occupancy, stall, cache, or sector metrics from `ncu`
- NVTX phase timing and correlation
- source-line attribution from profiler source views

Hypotheses include:

- likely uncoalesced access from a specific index pattern
- likely bank conflicts from a particular shared-memory layout
- likely register pressure from wide live ranges or per-thread arrays
- likely missing tensor path from dtype or tile choices

Do not blur those together.

## Intake Questions

Resolve these before suggesting directions:

- what kernel or code region is in scope
- what workload or phase is being discussed
- what evidence is measured versus inferred
- whether the user wants generic directions only or actual implementation next

## The Profile-To-Change Feedback Loop

Every generic CUDA optimization effort is a variant of the same closed loop:

```
profile -> compress evidence -> classify bottleneck -> select direction
       -> generate change -> verify correctness -> remeasure -> accept/reject
```

The loop breaks most often between "compress evidence" and "select direction."
The core discipline at intake is to collect only the counters that help choose
the next action, not every counter the profiler can emit. Filtered profiles can
preserve optimization quality at materially lower context cost; overloading the
reasoning with irrelevant metrics tends to dilute the decision rather than
sharpen it. When multiple reasonable evidence compressions exist, prefer the
smallest set that still supports the classification above it.

A mature intake packet distinguishes four layers:

1. `nsys` or timeline evidence (where time goes at the application level)
2. `ncu` throughput classification (compute, memory-bandwidth, memory-latency)
3. resource and stall detail (occupancy, stall mix, cache hit rate, coalescing)
4. line-level attribution (which source line carries the stall or sector mass)

Name the highest layer available. Everything below that layer is inference.

## Confidence And Evidence Posture

Confidence rises with layer, not with verbosity:

- **source only**: confidence is low. Claims should be phrased "probably",
  "suspected", or "likely". The intake must say what measurement would
  disambiguate the top candidates.
- **source plus phase context**: confidence stays low for the internal cause
  but rises for *which* kernel group deserves attention. Phase names do not
  prove uncoalesced loads or bank conflicts.
- **source plus summary metrics**: confidence is medium. The bottleneck family
  (compute vs bandwidth vs latency) is evidence-backed, but the specific
  failing code line is still a hypothesis.
- **source plus line attribution**: confidence is high for the responsible
  region but not yet for the best fix. Even line-attributed stalls name the
  waiting instruction, not necessarily the instruction that caused the wait.

Treat the sampled stall site as a suspect, not a verdict: the producing load,
the barrier, or an earlier predicate may be the actual cause. When line-level
attribution is available, ask explicitly whether the producer-consumer distance
is short enough that the stall is a latency-hiding problem, or whether the
stall family really implies spills, sync pressure, shared-memory pressure, or
global-memory latency before selecting a change family.

## SOL-Gap Hypothesis Triage

When a Speed-of-Light (SOL) bound is available for the workload, use it to
calibrate how ambitious the next experiment should be. Let `t_best` be the
current best measured time and `t_SOL` the first-principles bound derived from
FLOPs, bytes, and the arithmetic-intensity roofline. The gap

```
g = t_best / t_SOL
```

governs triage:

- **large `g` (far from SOL)**: there is room for structural rewrites. Dtype
  changes, layout transforms, better tiling, epilogue fusion, pipeline
  restructuring, and other ambitious hypotheses are worth trying. Do not
  waste the budget polishing instruction mix.
- **moderate `g`**: incremental improvements (coalescing fixes, shared-memory
  staging, small tile tuning) are the right scale. Avoid risky rewrites.
- **small `g` (near the ceiling)**: the kernel is near the modeled limit.
  Stop spending iterations on it unless a tighter SOL model is available, or
  consider whether the workload itself can be reformulated to move less data
  or perform less work.

SOL is also a guardrail against overreading local counters. A kernel can show
moderate SM throughput because it is doing unnecessary work. SOL keeps the
question "could a different formulation reduce the work?" on the table even
when `ncu` says "SM utilization looks fine."

Finally, an unexpectedly fast candidate (more than ~10% below the physical
ceiling computed from the tightest dtype-appropriate SOL) is suspicious. Treat
it as potentially gaming the benchmark (skipped work, shape shortcuts,
correctness regressions) until the intake includes a sanity check against the
reference.

## Metric-Meaning Cheat Sheet At Intake

Deep `ncu` metric interpretation belongs to profiling doctrine; at intake the
following light-touch mapping is enough to decide what the evidence packet is
really saying. Exact metric names and lookup tables are a profiling-skill
concern.

| Headline signal | What it tends to mean at intake |
|---|---|
| SM throughput high, DRAM low | compute-bound candidate |
| SM throughput low, DRAM high | memory-bandwidth-bound candidate |
| Both throughputs low, few eligible warps | latency-bound or underfill |
| Both high | rare; near optimal or hybrid |
| Low L1 hit + low L2 hit + high long-scoreboard stall | streaming from DRAM with no reuse |
| Short-scoreboard or MIO-throttle dominant | shared-memory or special-math pressure |
| Barrier / branch-resolving / no-instruction high | sync, divergence, or I-cache pressure |
| `not_selected` dominant | enough warps; occupancy is not the limiter |
| FP64 pipe non-zero in an FP16/FP32 kernel | accidental double-precision path |
| Tensor-core pipe near zero in a GEMM-shaped kernel | wrong instruction path |

Two rules travel with this table:

- high occupancy is not automatically good; chasing occupancy blindly can hurt
  cache coherence and register locality
- stall-reason breakdown is only meaningful when the scheduler is under-issuing;
  if issue slots are mostly busy, stall mix is noise

Flag the intake explicitly when the posture is "source only" and the cheat
sheet is being used to describe what *would* be measured, rather than what is
measured.
