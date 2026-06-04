# ncu Bottleneck Subskill

Use this integrated subskill when `nsys` has already identified the kernels
that matter and the next question is "why is this kernel slow?"

## ncu Owns Focused Kernel Diagnosis

Start narrow. Do not run `ncu --set full` over the whole application unless the
user explicitly asks for that kind of heavy capture.

Typical first-pass shapes:

```bash
ncu --set basic --kernel-name "myKernel" ./program

ncu --section SpeedOfLight \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStatistics \
    --section WarpStateStatistics \
    --kernel-name "myKernel" ./program
```

For this repository, when the profiled program is `mlsys-cli eval timing`, use
the persistent runner:

```bash
ncu --set basic --kernel-name "myKernel" \
  pixi run mlsys-cli eval timing --workload-set moe-medium --runner persistent
```

Do not profile the default isolated runner for this command surface. It may
hang, miss child-process CUDA work, or produce inconclusive reports.

Use launch filters when overhead is high:

```bash
ncu --kernel-name "myKernel" --launch-skip 10 --launch-count 5 ./program
```

## Bottleneck Classification Questions

Use the first pass to classify the kernel into one dominant family:

- compute throughput pressure
- memory bandwidth pressure
- memory latency pressure
- underfill / poor latency hiding
- divergence / synchronization / MIO pressure

Keep the next question narrow. Do not mix every section into one conclusion.

## Common Decision Pattern

- High SM throughput, low memory pressure -> compute-bound suspicion
- High memory throughput, low SM throughput -> memory-bandwidth suspicion
- Low both -> latency, underfill, or dependency-hiding problem
- High `long_scoreboard` -> global-memory latency / locality / reuse question
- High `short_scoreboard` or `mio_throttle` -> shared-memory or MIO pressure
- High barrier or branch stalls -> synchronization imbalance or divergence

## Next-Step Discipline

The useful bridge is:

```text
counter -> next diagnosis question or broad follow-on area
```

not:

```text
counter -> immediate source-level optimization plan
```

Use `source-attribution-subskill.md` once the bottleneck family is clear enough
to localize the hot code with intent.

## Practical Reading Order Inside The NCU Report

Read the report top-down, in this order, and do not skip levels:

1. `GPU Speed Of Light`
2. the exact section named by the top rule
3. `Compute Workload Analysis` or `Memory Workload Analysis`
4. `Scheduler Statistics` and `Warp State Statistics`
5. `Source` page or `Source Counters`

Stopping early gives wrong diagnoses. A classic trap from NVIDIA's
Analysis-Driven Optimization walkthrough: "compute is more utilized than
memory" did not mean "the kernel is math bound." The pipe-level view in
`Compute Workload Analysis` showed that the LSU pipe was still the dominant
consumer of cycles, which redirected attention back to memory operations.

## Symptom to Section to Cause Playbook

### Grid too small in `GPU Speed Of Light`

Jump directly to `Launch Statistics`. Likely cause is machine underfill.
Confirm launch grid, waves per SM, and obvious under-subscription. Typical
fix is to increase blocks or parallelize over more independent work.
Underfill outranks subtlety: there is little value studying cache counters on
a one-block launch.

### Compute low and memory low

Do not immediately label the kernel "memory bound" or "compute weak." This is
a latency or under-issuing clue. Inspect `Scheduler Statistics` and
`Warp State Statistics`, and ask:

- are there enough active warps?
- are there enough eligible warps?
- which stall family dominates the gap between active and eligible?

If active warps are plentiful but eligible warps are scarce, the issue is
stalled work, not residency.

### `Stall LG Throttle` high

The warp is waiting for a free entry in the local/global instruction queue.
Likely cause is too many memory operations too frequently, often combined
with inefficient transaction patterns. Next sections are
`Memory Workload Analysis`, then `Source`. Fix family: reduce
memory-instruction pressure, widen operations where reasonable, and especially
test coalescing. Queue pressure is not the same as peak bandwidth saturation.

### Actual transactions much larger than ideal (Source page)

Compare per-line `Memory Ideal L2 Transactions Global` versus actual global
transactions. A large actual-to-ideal ratio indicates uncoalesced access or
thread-to-data mismatch. Typical fix is remapping threads so adjacent lanes
read adjacent addresses, using warp-stride or block-stride traversal aligned
with memory layout. This is the most concrete "pin it down" move in the
entire ADO series: the profiler shows a specific line whose actual
transactions exceed ideal by a large ratio.

### Compute looks stronger than memory, but LSU pipe is hottest

`Compute Workload Analysis` exposes the pipe mix. If LSU dominates, the
bottleneck is still memory-operation pressure, not global bandwidth. Next
section is `Memory Workload Analysis`, and the likely shift is from
global-memory to shared-memory diagnosis.

### Shared memory hot, reduction line dominates

If the memory chart shows high shared-memory activity and the source page
points to a reduction line, the likely culprit is the reduction structure
itself. Typical pattern: sweep-style shared-memory reduction with repeated
load/store traffic. Fix family: warp-shuffle reduction or a structure that
keeps more of the reduction in registers and only uses shared memory for
cross-warp exchange.

### `Stall Long Scoreboard` after earlier fixes

Late in an optimization loop, long-scoreboard often becomes dominant. This
means the kernel is waiting on L1TEX-backed data dependencies, which in
practice usually means global-memory load latency. Inspect
`Warp State Statistics`, then navigate to source on the long-scoreboard
metric. Fix family: verify access pattern quality, improve locality if there
is still room, or accept that the kernel has become a true streaming load
problem. A hot source line here may no longer mean "bug"; it may mean "this
is the cost center that remains once the easy mistakes are gone."

## Stopping Check Against Device Roofline

A more useful stopping criterion than "the counters look good" is: isolate
the load-dominated phase and compare its achieved bandwidth to a realistic
device-level proxy such as CUDA `bandwidthTest`. If they are close, the
kernel is near the memory roofline for that phase. Local micro-tuning is
unlikely to buy more. The next owner is algorithmic refactor, phase split,
or library substitution (e.g., handing a matrix-multiply phase to cuBLAS as
a GEMM), not another round of kernel-body tuning.

## Metric Cheat Sheet

### First-pass classification

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM throughput as percent
  of peak. High = closer to compute-bound.
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: DRAM bandwidth
  utilization as percent of peak. High = memory-bandwidth-bound.
- SM high, DRAM low: compute-bound suspicion.
- SM low, DRAM high: memory-bandwidth suspicion.
- Both low: latency-bound or underfill.
- Both high: rare, likely near-optimal or hybrid.

Confirm with SOL and roofline. A kernel can show moderate SM throughput
because it is doing unnecessary compute (poor algorithm), not because it is
compute-bound.

### Occupancy and launch config

- `sm__warps_active.avg.pct_of_peak_sustained_active`: achieved occupancy.
- `launch__registers_per_thread`: primary occupancy limiter on most kernels.
- `launch__occupancy_limit_registers` and `launch__occupancy_limit_shared_mem`:
  which resource binds occupancy.
- `launch__block_size`, `launch__grid_size`: underlying geometry.
- `launch__waves_per_multiprocessor`: low waves implies tail effect.

High occupancy is not automatically good. It can increase register spilling
(more local-memory traffic), reduce per-warp L1/L2 residency (thrashing), and
increase SMEM bank-conflict pressure. If `not_selected` stall is high,
consider *reducing* occupancy to improve cache coherence.

### Stall reasons

Only meaningful when scheduler issue efficiency shows under-issuing.

- `long_scoreboard`: L1TEX-backed ops, classic "waiting on global memory"
  signal. Fix family: coalescing, reuse, tiling, SMEM staging, prefetch.
- `short_scoreboard`: MIO-backed ops, shared memory, special math, often bank
  conflicts. Fix family: pad or swizzle SMEM, use fewer wider ops.
- `mio_throttle`: MIO pipeline oversubscribed, too many SMEM or special-math
  ops in flight. Fix family: fewer wider loads (`float4` over four `float`),
  batch special math.
- `barrier`: waiting at `__syncthreads()` or cooperative group barrier. Fix
  family: reduce divergence before barriers, more uniform per-warp work,
  smaller blocks when large CTAs amplify waits.
- `branch_resolving`: heavy divergent branching. Fix family: simplify control
  flow, use predication for short conditional sequences.
- `no_instruction`: instruction-cache miss or fetch stall. Fix family: reduce
  code size, less aggressive unrolling, restructure hot loops for I-cache.
- `not_selected`: warp was eligible but scheduler picked another. Usually
  means enough warps for latency hiding. Do not chase more occupancy; may
  justify reducing active warps.
- `math_pipe_throttle`: math pipeline fully occupied, genuinely compute-bound
  at the pipe level.

Key rule: combined long-scoreboard plus short-scoreboard is the
memory-dependency stall ratio. High stalls with low DRAM throughput is
memory-latency-bound; high stalls with high DRAM throughput is
memory-bandwidth-bound.

### Cache and locality

- `l1tex__t_sector_hit_rate.pct`: L1 hit rate.
- `lts__t_sector_hit_rate.pct`: L2 hit rate.

Low L1 plus low L2 plus high long-scoreboard: data is streaming from DRAM
with no reuse. Tiling or shared-memory staging is the primary direction.

### Global coalescing via transactions-per-request

Replaces the deprecated `gld_efficiency`:

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`

Ratio = sectors / requests. About 4 sectors/request is perfect coalescing
for warp-wide 32-bit loads (32 threads x 4 bytes = 128 bytes = 4 x 32-byte
sectors). Ratios far above 4 indicate strided or scattered access; about 32
is the worst case (every thread triggers a separate cache-line fetch).

### Pipe utilization

- `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`: FP32 FMA.
- `smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active`: FP16.
- `smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active`: FP64. On
  consumer or datacenter GPUs FP64 throughput is 1/32 to 1/64 of FP32, so
  even small amounts of accidental FP64 (e.g., literal `1.0` instead of
  `1.0f`) can dominate.
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`:
  Tensor Core (HMMA). Zero on a GEMM kernel means the kernel is not using
  `wmma`, `mma.sync`, or WGMMA instructions.

## Two Subtle Rules

1. High occupancy is not automatically good. It can hurt cache coherence and
   register locality.
2. Stall reasons only matter when the machine is under-issuing. If issue
   slots are mostly busy, stall breakdown is noise.

## Hands-On Habits

- build with `-lineinfo`, not `-G`, so source attribution stays useful
- preserve specialized build targets such as `sm_90a` or `sm_100a` when the
  kernel relies on architecture-specific features; a generic `sm_90` or
  `sm_100` profiling build is not evidence for the specialized path
- keep correctness and timing checks next to the optimization loop
- use profiler baselines so each run is compared against the previous best
- make one fix family at a time; every real fix changes the next bottleneck
- treat rules as navigation, not infallible diagnosis
