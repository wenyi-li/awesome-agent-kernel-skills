# Profiling Mindset Subskill

Use this integrated subskill when the first question is not "what code change
should I make?" but "how should I profile this CUDA workload without getting
lost?"

## Core Rules

- Profile to eliminate hypotheses, not to collect every counter.
- Start with `nsys` to answer where time is going.
- Move to focused `ncu` only after choosing the kernels that matter.
- Keep one bottleneck family in focus at a time: launch overhead, underfill,
  memory bandwidth, memory latency, divergence/synchronization, or compute
  saturation.
- Distinguish measured findings from unvalidated optimization ideas.
- If a metric, section name, or profiler behavior is unclear, do targeted docs
  or source research instead of guessing.

## Ordered Questions

1. Where is time going?
2. Which kernels or phases dominate?
3. What resource limits those kernels?
4. Which source regions or code structures deserve inspection next?

## What Profiling Should Prevent

Profiling should stop the user from:

- optimizing a kernel that barely matters end-to-end
- rewriting math when `nsys` shows launch overhead or host gaps
- chasing occupancy before checking transaction efficiency
- adding shared memory before confirming reuse or access repair is needed

## The Core Attitude

Carry these attitudes into every profiling session:

- **Profile to eliminate hypotheses, not to admire numbers.** A good profile
  tells you what *not* to work on.
- **Classify coarsely before drilling down.** A small targeted diagnosis set is
  usually more useful than a giant metric dump.
- **Treat throughput and occupancy as context, not trophies.** High occupancy
  is not always good, and high memory activity does not always mean
  bandwidth-bound.
- **Assume each real fix changes the next bottleneck.** After every coherent
  optimization, reprofile. The next limiting factor is often different.
- **Keep one bottleneck family in focus at a time.** Memory bandwidth, memory
  latency, underfill, launch overhead, divergence, and compute saturation are
  different problems. Mixing them produces fuzzy edits and noisy conclusions.

## The Stall Site Is Not Always The Cause

A raw stall sample reports the instruction that waited, not necessarily the
instruction that created the wait. The GPA instruction-blamer line of work
formalizes this: the sampled site is the waiting consumer; the real root cause
may be an earlier producer load, a barrier source, a spill-producing access, or
an arithmetic dependency. Keep that separation explicit when interpreting
source-level stall evidence:

1. treat the sampled stall site as the waiting consumer, not automatically the
   culprit
2. identify the producing load, arithmetic op, or barrier source
3. ask whether the producer-consumer distance is too short to hide latency
4. ask whether the stall family really implies spills, sync pressure,
   shared-memory pressure, or global-memory latency
5. only then choose the code change family

This avoids "fixing" the hot line while the real producer remains unchanged.

## Profiler Symptom To Next Diagnosis Question

The useful bridge is counter to next diagnosis question, not counter to
immediate code rewrite:

| Profiler symptom | Next diagnosis question |
|---|---|
| Many tiny launches, host gaps, sync-heavy timeline | Is the problem above the kernel body? Are launches fragmented across small ops? |
| SM throughput low and DRAM throughput low | Is the grid too small, or are warps present but rarely eligible? |
| `stall_long_scoreboard` high | Is this real memory-latency pressure, and do cache/locality metrics agree? |
| Sectors/request high or source shows excessive global sectors | Are memory transactions inflated relative to requested data? |
| `stall_short_scoreboard` or `mio_throttle` high | Is shared-memory or MIO pressure the actual limiter? |
| Barrier or branch stalls high | Is divergence or synchronization imbalance dominating issue slots? |
| Tensor utilization low on GEMM-like code | Is the kernel missing the intended MMA path? |
| `not_selected` high | Are there already enough eligible warps, making more occupancy irrelevant or harmful? |

## SOL First, Fix Underfill First

NVIDIA's Analysis-Driven Optimization guidance puts `GPU Speed Of Light` at
the top of the reading order for a reason: underfill outranks subtlety. If the
top rule in the SOL section says the grid is too small to fill the device,
drop into `Launch Statistics` and fix launch geometry before looking at cache
counters, stalls, or math pipes. There is little value in studying coalescing
on a one-block launch.

Durable hygiene that carries across sessions:

- keep runtime and correctness checks alongside the optimization loop
- build with `-lineinfo` so source attribution stays usable in later stages
- preserve architecture-specialized targets such as `sm_90a` or `sm_100a`
  when the kernel depends on specialized instructions or features; do not
  collect profiler evidence from a generic `sm_90` or `sm_100` build and treat
  it as equivalent
- treat profiler rules as breadcrumbs, not as final explanations
- use profiler baselines so each run is compared against the previous best

## APOD Loop And Speedup Bounds

The broader optimization loop that wraps profiling is APOD: Assess,
Parallelize, Optimize, Deploy. The ordering matters more than the name:

1. assess the application on realistic workloads
2. identify the true hotspots
3. parallelize the parts that matter
4. optimize one pass at a time
5. deploy partial gains instead of waiting for perfection

A hotspot is a property of the real workload, not of the kernel that happens
to look interesting. Tiny or artificial inputs can shift which code path
dominates and can distort the transfer/compute balance.

Before committing to deep work, estimate the upside. Amdahl's Law bounds
speedup on fixed-size problems: `S = 1 / ((1-P) + P/N)`, where `P` is the
parallelizable fraction and `N` is the parallel hardware. A small `P` caps
speedup no matter how powerful the GPU is. Gustafson's Law applies when
problems grow with available compute: `S = N + (1-P)(1-N)`. The practical
distinction is strong scaling (fixed total problem size) versus weak scaling
(fixed work per processor, growing total problem size).

The working loop within a single profiling session is:

1. isolate the dominant phase with `nsys`
2. classify the hottest kernel coarsely with `ncu`
3. choose one bottleneck family to investigate
4. patch one coherent cause
5. remeasure correctness and runtime
6. assume the bottleneck moved, and repeat

## Metric Selection And Overhead Management

`ncu` overhead is the primary cost driver on any iterative profiling loop.
Multi-pass replay for software-patched metrics can make "more metrics"
dramatically more expensive. Start narrow and expand only when the initial
classification is ambiguous.

Tiered metric menu for disciplined collection:

- **Tier 1 (always collect):** `gpu__time_duration.avg`,
  `sm__cycles_elapsed.avg`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`, and
  `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`. These four answer
  the fundamental compute-vs-memory question.
- **Tier 2 (occupancy and launch config):** `launch__registers_per_thread`,
  `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`,
  `launch__block_size`, `launch__grid_size`,
  `launch__waves_per_multiprocessor`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`.
- **Tier 3 (stall analysis):** long-scoreboard, short-scoreboard,
  no-instruction, and not-selected warp-issue stall metrics. Only meaningful
  when issue slots are under-issuing.
- **Tier 4 (cache and divergence):** `l1tex__t_sector_hit_rate.pct`,
  `lts__t_sector_hit_rate.pct`, divergent-vs-uniform branch counters.
- **Tier 5 (pipe utilization):** FP64, FMA, FP16, and tensor-op pipe
  utilization metrics.

Overhead controls to apply by default:

- section selection, not `--set full`
- kernel filtering with `--kernel-name regex:...`
- launch filtering with `--launch-skip` and `--launch-count` to skip warmup
- per-kernel invocations for multi-kernel programs so launch-count limits do
  not silently drop kernels
- timeouts on each profiling pass so a single hang does not stall the loop

## Hierarchical Roofline As A First Branch

When locality across L1, L2, and HBM matters, a hierarchical roofline view
gives a better first-step than a flat compute-vs-DRAM classification. The
measurement reduces to three raw quantities per kernel: runtime, FLOPs
executed, and bytes moved at each memory level. Arithmetic intensity is
`AI_level = FLOPs / Bytes_level`, and performance is `FLOPs / Runtime`.

Keep distinct intensities for L1, L2, and HBM so the plot serves as a
locality diagnostic, not just bandwidth-vs-compute classification. If a kernel
is truly HBM-bound, the performance point moves along the HBM ceiling. If it
is not near any bandwidth ceiling, roofline has already told you to look
elsewhere (launch, latency, divergence). A framework can silently switch
algorithmic paths or dtypes, so verify the executed precision path when the
point moves strangely.

Record at minimum: kernel name, problem shape, dtype and actual execution
path, runtime, FLOPs, L1/L2/HBM bytes (or proxies), inferred bottleneck
hypothesis, and the next code change planned. Without that ledger, roofline
turns into a plot rather than an instrument.

## What Profiling Should Prevent

Profiling should stop a programmer from making the wrong kind of clever
change:

- rewriting math when the timeline says launch overhead dominates
- chasing occupancy when transactions-per-request are obviously poor
- studying branch details when the grid is too small to fill the device
- adding shared memory before confirming reuse or access repair is needed
- micro-tuning one kernel before proving it matters in end-to-end runtime

## Relationship To Other Guidance

- Use `nsys-hotspot-subskill.md` for hotspot selection and timeline reasoning.
- Use `ncu-bottleneck-subskill.md` for focused kernel diagnosis.
- Use `source-attribution-subskill.md` after bottleneck narrowing points toward
  code.
- Do targeted docs or source research when the local subskills are not enough
  and you need the right conceptual backing source.
