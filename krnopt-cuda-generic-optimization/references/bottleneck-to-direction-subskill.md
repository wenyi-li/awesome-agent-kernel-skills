# Bottleneck To Direction Subskill

Use this integrated subskill when the goal is to turn evidence into one or more
testable optimization directions.

## Direction Mapping

- **launch or orchestration overhead**:
  fuse stages, batch work, reduce syncs, use grouped dispatch, or consider
  persistent execution
- **memory-bandwidth pressure**:
  improve coalescing, align or vectorize memory ops, reduce redundant traffic,
  change layout, or cache reused operands
- **memory-latency pressure**:
  increase reuse, stage through shared memory, prefetch, improve locality, or
  increase independent work per load
- **shared-memory pressure / bank conflicts**:
  pad or swizzle layout, reduce scalar SMEM instruction count, move hot reuse
  into registers, or change tile arrangement
- **occupancy / register pressure**:
  shorten live ranges, reduce per-thread state, retune unrolling, change block
  shape, or split a monolithic kernel
- **divergence / barrier cost**:
  make work more warp-uniform, shorten divergent regions, restructure
  reductions, or use warp-level primitives
- **compute / instruction-path limit**:
  remove redundant math, simplify epilogues, select the intended Tensor Core or
  vector path, and audit dtype or intrinsic choices

## Ranking Rule

Rank directions by:

1. how directly they match the evidence
2. how localized the likely code change is
3. how reversible and testable the experiment is

Prefer one to three directions. A short ranked list is better than ten vague
ideas.

## Required Pairing

Every direction should include:

- suspected cause
- likely source region
- change family to try
- what result would confirm it
- key risk or counter-signal

## Metric-Family To Direction Table

The following table is the compact version of the profiler-driven decision
tree. Read it left-to-right: symptom family, what it classifies the kernel as,
and which change family most directly addresses it.

| Symptom | Classification | Direction |
|---|---|---|
| Big host gaps, frequent syncs, expensive memcpies in the timeline | Orchestration-limited | Fix overlap and transfer first; do not touch the kernel body yet. Consider batching transfers, pinning memory, async streams, or keeping intermediates on device. |
| SM and DRAM throughput both low, few eligible warps | Latency-hiding or underfill | Increase eligible warps, reduce dependency chains, fix grid/block decomposition. Consider raising threads per block or launching more blocks. |
| Roofline or SOL says memory-bound, DRAM throughput high | Memory-bandwidth-limited | Reduce bytes/FLOP, improve tiling and reuse, vectorize loads, stage through shared memory, consider asynchronous copy or tensor-memory transfers. |
| Long-scoreboard stall high, DRAM throughput moderate | Global-memory latency | Coalescing, reuse, shared-memory staging, cache friendliness, software prefetch. |
| Short-scoreboard or MIO-throttle stall high | Shared-memory or MIO pressure | Fix bank conflicts, use fewer and wider SMEM ops, batch special math, lift hot values into registers. |
| Barrier, branch-resolving, or no-instruction stall high | Sync / divergence / I-cache | Reduce divergence before barriers, make per-warp work more uniform, simplify control flow, consider smaller blocks or less aggressive unrolling. |
| `not_selected` stall dominant | Enough warps, maybe over-subscribed | Consider *reducing* active warps for better locality; do not chase more occupancy. |
| Compute throughput high, math-pipe-throttle dominant | Compute pipeline | Audit instruction mix, compare against and prefer a standard-library-style primitive when the operation fits, then tune tile sizes or the intended MMA/tensor path. |
| Tensor-core pipe near zero on a GEMM-shaped kernel | Wrong instruction path | First consider cuBLAS/cuBLASLt or CUTLASS/CuTe-style substitution when the contract fits; otherwise align dtype, tile shape, and accumulator to the target MMA and verify compile flags. |
| Non-zero FP64 pipe in an inference kernel | Accidental double precision | Audit for `double` literals, implicit promotions, generic math paths. |

## Symptom-to-Cause-to-Fix Chains From Best-Practice Doctrine

The CUDA best-practices guidance maps real symptom patterns into a ten-point
map. Condensed form:

1. **Host-device transfer dominates application time.** Keep intermediates on
   device, batch small transfers, pin important buffers, overlap with compute,
   consider recomputing instead of transferring.
2. **Effective bandwidth is far below theoretical.** Inspect alignment,
   strided access, requested vs actual throughput. Fix coalescing, reorder
   layout, stage through shared memory for stride repair or reuse.
3. **Requested bandwidth much lower than actual bandwidth.** The kernel is
   over-fetching. Align data, preserve warp alignment via block-size choice,
   avoid large-stride access patterns.
4. **Repeated global loads of the same data.** Stage the tile into shared
   memory as a user-managed cache; on suitable hardware consider L2
   residency windows; asynchronous copy can help overlap and reduce register
   pressure.
5. **Shared memory present but throughput poor.** Bank conflicts are
   serializing. Change layout, add padding (the classic `+1` column trick for
   transpose-like tiles), or apply a swizzle.
6. **Low occupancy and exposed latency.** Reduce register pressure and shared
   footprint; try 128-256 threads per block first; prefer several smaller
   blocks over one giant block when syncs are frequent. Measure occupancy
   sensitivity rather than assuming "100% is required."
7. **Register spill into local memory.** Per-thread state is being pushed
   off-chip. Reduce per-thread state, check `-Xptxas=-v`, use `__launch_bounds__`
   or `-maxrregcount` carefully. Raising occupancy at the cost of spills
   usually loses overall.
8. **Expensive math dominates after memory is clean.** Use intrinsics where
   acceptable, prefer single precision where valid, replace division or modulo
   by powers of two with shifts and masks, prefer specialized transcendentals.
9. **Warp-divergent control flow.** Align conditions with warp boundaries
   where possible, keep small branches short enough for predication, use
   `__syncwarp()` when later code depends on reconvergence.
10. **Over-optimization in the wrong place.** Stop. Go back to realistic
    profiling. Fix the dominant hotspot first. Low-level math tuning without
    a proven hotspot is a process violation, not progress.

## Coarse Throughput-Class Playbooks

When only the three throughput headlines (`SM`, `DRAM`, `Memory`) are
available, a coarse classification is still usable. The CUDAMaster playbook
uses approximately:

- **compute bound**: SM throughput above ~30% of peak.
- **memory-latency bound**: all three throughputs below ~30% of peak (the
  machine is waiting, not saturated).
- **memory-bandwidth bound**: SM throughput low but DRAM or memory throughput
  above ~30% (interface is active).

Inside each class, prefer the change families below.

### Compute-bound

Metrics to consult next: issue-slots-busy, executed-IPC-active, SM-busy.

Common causes: instruction-delivery inefficiency, branch divergence,
insufficient instruction-level efficiency.

Preferred directions: loop unrolling, instruction-stream cleanup, more
efficient reduction structure, vectorization when the kernel is regular,
warp-shuffle reductions instead of atomics, `float4` / `__ldg` style access
when aligned, packed SIMD (for example `half2`), fast-math intrinsics, and
explicit tensor-core usage (`wmma`, `mma.sync`, WGMMA, etc) if the shape fits.

### Memory-latency-bound

Metrics to consult next: L2 hit rate, L1/TEX hit rate, warp-cycles-per-
executed-instruction, executed-IPC-elapsed, mem-busy.

Common causes: scattered access with low reuse, too many passes over memory,
insufficient parallelism to cover latency.

Preferred directions: algorithmic tiling, shared memory as a managed cache,
fusion to reduce intermediate traffic, increase arithmetic intensity, improve
thread-level parallelism, software prefetch, register blocking, texture or
read-only cache for streaming read-only patterns.

### Memory-bandwidth-bound

Metrics to consult next: DRAM throughput, memory throughput, max bandwidth,
mem-pipes-busy.

Common causes: uncoalesced or inefficient transfers, repeated loading of
reusable operands, low arithmetic intensity over large streams.

Preferred directions: improve coalescing, cache reused inputs, compress or
reduce data-movement volume, use vectorized memory operations, change data
layout.

## Line-Attributed Stall Families And Change Families

When line-level attribution is available, the stall family at the blamed
*source* instruction (not the waiting consumer) maps to a specific intervention
family. This is the GPA-style mapping:

### Stall-elimination family

| Matched signal at source | Root cause idea | Typical change |
|---|---|---|
| Local-memory read/write dependency | spills or local traffic | reduce register pressure, split loops, keep values live in registers |
| Long-latency arithmetic dependency | expensive math choice | strength reduction (reciprocal multiply, bit tricks, cheaper equivalents) |
| Instruction-fetch stalls | code too large | split cold or bulky logic into separate functions |
| Stalls inside CUDA math functions | expensive precise transcendentals | use fast-math variants when acceptable |
| Warp synchronization stalls | excess or imbalanced syncs | remove syncs, use `__syncwarp()`, rebalance reduction |
| Global memory throttling | too many transactions / wrong memory type | reduce transactions, use constant memory when shared and read-only |

### Latency-hiding family

| Matched signal | Root cause idea | Typical change |
|---|---|---|
| Memory or execution dependency inside a loop | too little independent work between producer and consumer | unroll, widen software pipeline |
| Short producer-consumer distance in straight-line code | no room to hide latency | reorder code, hoist loads and address math earlier |
| Stalls around device-function calls | call structure blocks scheduling | inline small hot helpers |

Latency hiding is bounded; GPA's upper-bound analysis shows its ceiling is
around 2x even in the best case. Prefer stall-elimination fixes when available.

### Parallel family

| Matched signal | Root cause idea | Typical change |
|---|---|---|
| Few blocks relative to SM count | machine underfill | reduce threads per block or restructure to launch more blocks |
| Occupancy limited by threads per block | too little in-block parallelism | increase threads per block |

If the hotspot mass is diffuse (no line carries a large share), prefer a
broader structural change (launch geometry, tile restructure, fusion) over
line-by-line edits.

## Named Bottleneck States For Quick Lookup

Some taxonomies expose more granular states than the three coarse classes.
These are useful as a quick lookup when a short description of the kernel's
state is available:

- `memory_bandwidth_saturated` — vectorize, coalesce, tile into SMEM,
  transform layout, consider compression or lower precision.
- `memory_latency_bound` — SMEM blocking, register blocking, software
  prefetch, cache-aware algorithms, texture for read-only patterns.
- `memory_bank_conflicts` — padding, swizzling, broadcast-friendly layout,
  permutation.
- `cache_inefficient` — tile to L2 size, improve spatial and temporal
  locality, consider L2-residency windows on hardware that supports them.
- `compute_throughput_saturated` — tensor-core use, ILP, fast-math,
  specialized instructions, vectorized ops, packed SIMD.
- `instruction_mix_suboptimal` — audit dtype path, intrinsic choices,
  accidental FP64, generic `pow` vs specialized math.
- `thread_divergence_high` — predication, branch reduction, warp-level
  primitives, stream compaction, work redistribution.
- `low_occupancy_register_pressure` — register-pressure reduction, variable
  scoping, spill elimination, `__launch_bounds__` tuning, algorithmic
  register reduction.
- `low_occupancy_shared_memory` — reduce SMEM per block, dynamic SMEM with
  smaller carveout, refactor tile sizes.
- `insufficient_parallelism` — increase blocks or threads per block, persistent
  kernels, rework grid geometry.
- `api_overhead_dominant` — kernel fusion, CUDA Graphs, persistent kernels,
  asynchronous execution; the fix is not inside one kernel body.
- `memory_compute_balanced` / `latency_memory_bound` / `hybrid_bound` —
  memory-compute overlap, adaptive tiling or block sizing, fused operations,
  occupancy-memory co-design, workload restructuring.

These states are operational labels, not measurements. Use them to organize
candidate change families, then rank per the rule at the top of this page.

## Worked Optimization Ladder (SGEMM)

An instructive worked ladder, from naive kernel to near-cuBLAS FP32
performance on a single data size, illustrates how each step is justified by a
concrete resource symptom rather than by fashion.

For production optimization, do not treat this ladder as permission to
handcraft GEMM before checking standard CUDA primitives. The preferred order is
precompiled vendor/runtime libraries first, CCCL header-only CUDA core
primitives second when the hotspot is a scan/reduction/sort/collective building
block or CUDA C++ utility, header/template or generated-kernel libraries third,
and handcrafted kernels only when the higher tiers fail the feature,
integration, fusion, or measured-performance requirements.

1. **Baseline one-thread-per-output**: warp-adjacent lanes walk strided
   addresses; most outputs re-load the same global bytes. Before optimizing,
   compute FLOPs and the minimum memory traffic to identify the eventual
   bottleneck class (a well-tiled FP32 SGEMM should be compute-bound).
2. **Fix coalescing**: remap C coordinates so consecutive warp lanes read
   consecutive addresses. This alone moved the worklog's memory throughput
   from ~15 GB/s to ~110 GB/s and compute from ~300 GFLOP/s to ~2000 GFLOP/s,
   with no arithmetic change. *Rule*: before adding caches, fix the lane-to-
   address mapping.
3. **Stage reused tiles in shared memory**: load an A-tile and B-tile once
   per K step, synchronize, consume. Gains are moderate because each thread
   still computes one output. *Rule*: shared memory is not magic; its value
   comes from increased reuse per global byte.
4. **Compute many results per thread (1D strip)**: one loaded operand feeds
   several FMAs. This is the first large arithmetic-intensity jump; the
   worklog reports ~8.6 TFLOP/s. *Rule*: when memory stalls persist after
   basic tiling, ask whether each thread does enough math per loaded operand.
5. **2D register blocking (8x8 per thread)**: explicit outer-product into
   per-thread accumulators makes the hot loop FMA-heavy instead of load-heavy.
   The worklog reports ~16 TFLOP/s.
6. **Vectorize and tune layout**: transpose the A-tile in SMEM so the compiler
   can emit 128-bit SMEM loads; use `float4` global loads for a stronger
   alignment contract. Smaller jump, but reduces load-instruction pressure.
7. **Autotune tile sizes and add warp-tiling**: parameters differ across GPU
   models; some combinations break vectorization or exceed resource limits.
   Adding an explicit warp-tile between block-tile and thread-tile aligns the
   source structure with hardware scheduling (blocks to SMs, warps to warp
   schedulers, threads to register tiles). The final kernel reports ~21.7
   TFLOP/s (roughly 93.7% of cuBLAS FP32 in the top-line table).

The generalizable takeaway is the *order*: correct the memory shape first,
then raise arithmetic intensity, then fix instruction mix, then autotune.
Skipping ahead tends to produce false positives (polishing a kernel whose real
problem is upstream).

## Hierarchical Roofline Reasoning Patterns

When multi-level cache intensities are available, these inference patterns
sharpen the direction choice:

- **Performance tracks the HBM ceiling**: kernel is HBM-bandwidth-bound. Ask
  whether arithmetic intensity dropped, whether byte traffic increased while
  FLOPs held, or whether cache locality broke.
- **L2 intensity close to HBM intensity**: L2 is not filtering. Focus on reuse
  that catches before HBM; stacking more HBM traffic will just slide the point
  down the same ceiling.
- **L1 intensity roughly constant across sizes while outer levels move**:
  block-local access is stable; the real effect is in inter-block reuse,
  cache residency duration, ghost zones, or outer scheduling. Stop blaming
  the inner loop.
- **Intensity collapses at outer levels for the "better-looking" variant**:
  the implementation is failing to capture locality in L1, flooding L2, and
  increasing HBM traffic. This is a sharper diagnosis than "variant X is
  slower."
- **Performance saturates below all visible bandwidth ceilings**: roofline
  has done its job. Move to instruction mix, occupancy, predication, and
  scheduler-side investigation.
- **Anomalous points may be runtime path changes**: different library
  subkernel, FFT fallback, FP16 input with FP32 execution and conversion
  overhead, or housekeeping kernels polluting the measurement. Ask whether
  the library chose a different kernel family before blaming hardware.
- **Roofline gives the next question, not the exact fix**: pair the roofline
  result with source counters, scheduler and stall views, launch stats, and
  code review. The tool is a branching heuristic, not a complete optimizer.

## Evidence Discipline When Selecting The Next Direction

Impose the same gate doctrine used by well-designed optimization agents:

- cite at least one numeric metric or one source-structure clue for each
  proposed direction
- cite at least one analysis-section conclusion (bottleneck class, stall
  family, cache pattern, SOL gap) that supports the direction
- explicitly rule out at least one alternative bottleneck the evidence could
  plausibly imply
- propose exactly one incremental change at a time when the evidence supports
  it; avoid full-redesign laundry lists

If the evidence does not justify a definite direction, name the missing
measurement (a specific counter, a specific section, an NVTX mark) instead of
guessing.
