# Scheduler And Launch Control Subskill

Use this integrated subskill when the main hardware-aware question is not
"which MMA path?" but "which work-distribution model should this kernel use?"

This subskill keeps four ideas separate:

- static tile ownership
- persistent scheduling inside one kernel
- Cluster Launch Control on SM100
- Programmatic Dependent Launch across kernels

Do not collapse those into one vague "scheduling" story.

## The Four Scheduler Modes

### Static Tile Ownership

Use when:

- tile costs are regular
- load balance is already acceptable
- persistent logic would add complexity without helping much

Typical fit:

- dense GEMM or attention with fairly uniform tiles
- kernels where launch overhead is not the main issue

Likely wins:

- simplest kernel, easiest to debug, minimum register and SMEM overhead for
  scheduler state
- predictable tile-to-CTA mapping that is friendly to cache-residency tricks

Likely traps:

- tail effects when the grid shape is not a nice multiple of the SM count
- load imbalance across tiles with variable K or variable per-tile cost
  (e.g. grouped GEMM with expert-count skew)

### Persistent Scheduling

Use when:

- many small or irregular tiles need one resident kernel
- launch overhead or imbalance across tiles is meaningful
- tile ordering and cache reuse matter

Typical fit:

- grouped GEMM
- routed expert kernels
- irregular decode or expert-batch shapes

One well-known static-persistent pattern worth naming: DeepGEMM's scheduler
(works on both SM90 and SM100) uses a `(current_iter * kNumSMs + blockIdx.x)`
round-robin with L2 swizzle grouping. This has no CLC dependency and is a
reasonable reference point when CLC is not available or is not yet
justified.

A related explicit-scheduling pattern is static batching for irregular MoE
work: build compact task metadata (inclusive prefix over per-task tile
counts) before the main kernel, then have each CTA decode its assigned
task/tile with a warp-level prefix search:

```
p = (B >= TilePrefix[lane])
mask = ballot(p)
task = popc(mask)
prev = task > 0 ? TilePrefix[task - 1] : 0
tile = B - prev
```

For MoE, empty experts need a second indirection (virtual CTA -> non-empty
expert index -> real expert id). Static batching gives per-expert tile shape
variation, empty-expert skipping, and avoids the per-tile metadata blow-up of
grouped GEMM's device-side dynamic tile scheduling, but it is a
one-warp-of-tasks pattern and needs extensions for very large task counts.

Likely wins:

- amortized launch overhead across many small tiles
- better tile ordering for cache reuse
- clean way to handle per-group variable tile counts

Likely traps:

- assuming persistence is free
- ignoring the residency hit from larger state, registers, or shared memory
- expecting persistence alone to fix load imbalance; it changes the
  scheduling mechanism but not the underlying per-tile cost variance

Fallback path:

- static tile ownership with tighter grid shape, or split-K / Stream-K when
  the imbalance is long-K tail rather than irregular tile count

### Cluster Launch Control

Use when:

- the hardware is SM100 / Blackwell
- the kernel is already persistent or clearly should be
- cluster-level load balancing is important

Key mechanics:

- CLC is a hardware-assisted dynamic work distribution path on SM100,
  replacing SM90's static tile schedulers for persistent kernels
- grid is launched with as many threadblocks as output tiles (like
  non-persistent), and CLC dynamically assigns work to available clusters
- the scheduler warp calls `clusterlaunchcontrol.try_cancel` to query the
  next unprocessed ClcID; returns success with coordinates, or decline when
  all tiles are done
- the first ClcID equals `blockIdx`, so the first tile requires no query
- transaction bytes for the CLC response are 16; one elected thread from the
  scheduler warp issues the query
- work is assigned at cluster granularity: a 2x2 cluster consumes 4 ClcIDs
  at once
- CUTLASS dense-GEMM examples commonly use pipeline depth 3 for latency
  hiding

Canonical warp-role assignment in the CLC persistent pattern (8 warps/block):

| Warp | Role |
|------|------|
| 0 | MMA |
| 1 | Scheduler (CLC producer/consumer) |
| 2 | Mainloop Load |
| 3 | Epilogue Load |
| 4-7 | Epilogue |

The scheduler warp is a first-class role, not an afterthought; the kernel is
designed around it. CUTLASS example 95 (Blackwell GEMM with Green Context)
shows why CLC-backed scheduling is useful in practice: under partial-SM
availability, CLC adapts with less manual retuning than older static
persistent strategies.

Preferred dynamic clusters deserve a mention: large clusters can improve
multicast behavior but strand SMs when the device SM count is not a neat
multiple of the preferred cluster size. Use a preferred cluster for the fast
path plus a fallback cluster that evenly divides it. This is the right
remedy when the issue is cluster-size quantization rather than tile math or
copy bandwidth.

Likely wins:

- dynamic tile load balancing across clusters
- clean handling of irregular grid shapes without manual work-queue code
- integrates naturally with 2-CTA MMA and cluster-level kernels

Likely traps:

- recommending CLC on Hopper, where it does not exist
- assuming CLC is a generic speedup rather than a response to irregular work
- adopting CLC before the kernel is persistent; CLC is a scheduling
  mechanism for persistent kernels, not a replacement for kernel redesign

Fallback path:

- DeepGEMM-style static persistent scheduling on SM100, or Hopper-style
  static persistent on SM90

### Programmatic Dependent Launch

Use when:

- the question is overlap between back-to-back kernels
- a downstream kernel can start while the upstream kernel tail drains
- CUDA Graphs or a fixed launch pipeline already exist

PDL lets a downstream kernel begin executing while the upstream kernel's
tail is still draining, as long as the dependency graph permits. Combined
with CUDA Graphs, it addresses MoE decoder blocks that fire many small
kernels (router, top-k, permute, grouped GEMM1, SwiGLU, grouped GEMM2,
unpermute, reduce) where per-kernel launch latency dominates. The public
TensorRT-LLM DeepSeek-R1 result on 8xB200 used CUDA Graphs plus PDL to go
from 67 to 253 tokens/s, which is largely a launch-overhead story.

CUTLASS exposes PDL through `include/cutlass/arch/grid_dependency_control.h`
with `launch_dependent_grids()` and `wait_on_dependent_grids()`. The
weight-prefetch example (CUTLASS example 63 and the
`dependent_kernel_launch` doc) makes the key operational knobs explicit:

- `overlap_ratio`: how much of the downstream kernel begins before the
  upstream is fully drained
- `prefetch_ratio`: how much of the downstream's operand can be
  opportunistically prefetched into L2

Both are runtime knobs, not compile-time template choices.

Typical fit:

- MoE FC1 -> FC2 style pipelines
- overlap of prologue work in kernel K2 with epilogue drain in kernel K1
- CUDA Graph-captured transformer blocks

Likely wins:

- reduced tail latency between dependent kernels when the downstream kernel
  can start work that does not depend on the last upstream outputs
- meaningful speedups on launch-bound MoE paths

Likely traps:

- treating PDL as an in-kernel scheduler
- using PDL language when the real issue is grouped work inside one kernel
- forgetting that PDL requires the upstream kernel to have written the bytes
  the downstream reads; the dependency graph must be real

## Decision Pivots

Ask these before recommending a scheduling model:

1. Are tile costs regular enough that static scheduling is fine?
2. Is launch overhead or tile imbalance large enough to justify persistence?
3. Is the target hardware actually SM100 if CLC is under consideration?
4. Is the scheduling problem within one kernel or across two dependent
   kernels?
5. What state, occupancy, or cluster cost comes with the chosen scheduler?
6. If the irregularity is MoE-shaped, is static batching a cleaner answer
   than generic persistent scheduling?

## Where To Escalate

- Use `krnopt-cuda-profiling` when you still need measurement to prove load
  imbalance, launch overhead, or scheduler underutilization.
- Use `references/cutlass-hw-source-map-subskill.md` when the user needs the
  exact CUTLASS layer that implements the chosen scheduler model.
