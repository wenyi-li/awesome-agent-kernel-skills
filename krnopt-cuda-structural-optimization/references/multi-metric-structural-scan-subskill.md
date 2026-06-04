# Multi-Metric Structural Scan

Use this reference when a CUDA kernel needs a big-picture read instead of a
single-metric diagnosis.

The goal is to compress several metrics into one structural claim and decide
whether the problem lives in the kernel body, a boundary, or the schedule.

## Cross-Check Set

Start with this compact set:

- kernel duration or time share
- SM throughput
- DRAM throughput
- occupancy or active-warps posture
- scheduler issue posture
- dominant stall family
- one memory-shape signal
- one instruction-path signal when relevant

If the hot path has many kernels, also include:

- launch count or timeline fragmentation
- host gaps or explicit sync posture
- bytes spent on copies, permutes, combine, or materialization steps

## Read Patterns Together

- low SM plus low DRAM:
  likely underfill, dependency, launch fragmentation, or poor latency hiding
- high DRAM plus poor transactions per request:
  bandwidth waste from layout or lane mapping
- high `long_scoreboard` plus weak reuse:
  data dependency problem that may need tiling, staging, or decomposition
- high `short_scoreboard`, `mio_throttle`, or barrier pressure together:
  shared-memory or synchronization structure problem
- low tensor utilization on GEMM-like work:
  wrong kernel family or wrong instruction path
- high issue underuse with many tiny launches:
  orchestration or fusion problem above the kernel body
- high data movement in dispatch, permute, or combine stages:
  boundary or metadata problem, not just main-loop inefficiency
- high setup churn for descriptors or shape metadata on irregular work:
  scheduling or metadata representation problem

## Structural Questions

After the metric read, ask:

1. is the machine underused because the launch shape is wrong
2. is traffic wasted because the boundary or layout is wrong
3. is waiting caused by a specific instruction line or by the whole schedule
4. is the kernel structurally streaming data that should be reused or fused
5. is the custom kernel imitating a tuned primitive badly
6. is the irregularity really a scheduler problem rather than a math problem
7. are we already close enough to a roofline that local tuning should stop

Do not claim a redesign family until the metrics and the code shape tell the
same story.

## Single-Metric Traps

Do not overreact to any one of these alone:

- high `long_scoreboard` without checking whether the real fix is layout,
  staging, fusion, or decomposition
- low occupancy without checking whether the workload is actually blocked by a
  boundary or by poor eligible-warps posture
- high DRAM throughput without checking whether payload movement itself is
  unnecessary
- low tensor utilization without checking whether the kernel should be recast as
  a different primitive
