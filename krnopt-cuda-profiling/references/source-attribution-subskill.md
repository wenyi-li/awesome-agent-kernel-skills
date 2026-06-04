# Source Attribution Subskill

Use this integrated subskill when profiler evidence has narrowed the problem to
one kernel or one bottleneck family and the next question is "where in the code
should I inspect?"

## Exact Source Attribution Requires Build Metadata

Build with `-lineinfo` so profiler source views remain useful without the heavy
semantic distortion of a debug-only device build.

Then use focused profiler source views such as:

- NCU Source page
- NCU SourceCounters
- line-level stall or memory-transaction counters when available

## What To Associate

Use profiler evidence to connect:

- the kernel name
- the launch context or wrapper
- the line or region with high memory inefficiency, divergence, bank conflicts,
  or stall contribution

Then stop at profiler-backed localization plus broad follow-on areas such as:

- inspect lane-to-address mapping in this load block
- inspect reuse structure in this tiled loop
- inspect shared-memory layout in this transpose or reduction block
- inspect register footprint in this accumulator or epilogue region
- inspect synchronization shape in this reduction or barrier-heavy region

Do not turn this subskill into a source-rewrite planner. Detailed experiment
selection belongs to `krnopt-cuda-generic-optimization`.

The next owner may then hand off to programmer-style source reasoning:

- lane-to-address mapping
- reuse structure
- shared-memory layout
- register footprint
- synchronization shape
- instruction and datatype path

## Important Boundary

Teach this explicitly:

- NVTX helps find the phase or wrapper
- `ncu` plus `-lineinfo` helps find the source region inside the kernel

Those are complementary, not interchangeable.

## Typical Questions

- Which lines inflate memory transactions relative to requested work?
- Which reduction or synchronization regions dominate stalls?
- Which code region appears to trigger bank conflicts or divergence?
- Does the kernel shape actually match the intended tensor/vector path?
- Which code block should be inspected next, and in what broad area?

## Source-Level Metrics To Read

Available via `ncu --page details --csv` or `--section SourceCounters`:

- **Warp Stall Sampling (All Samples):** hotspot lines where warps spend the
  most stall time.
- **Divergent Branches:** specific source lines causing warp divergence.
- **L2 Theoretical Sectors Global Excessive:** lines with uncoalesced global
  memory access. Compare ideal versus actual transactions per line.
- **L1 Wavefronts Shared Excessive:** lines causing shared-memory bank
  conflicts.
- **Memory Ideal L2 Transactions Global** versus actual global transactions:
  the most concrete pin-down move; a large actual-to-ideal ratio on one line
  usually means uncoalesced access or thread-to-data mismatch.

Build must use `-lineinfo`, not `-G`. The `-G` flag disables device
optimization and distorts both timing and counter attribution.

For architecture-sensitive kernels, the source-attribution build must also keep
the specialized target such as `sm_90a` or `sm_100a`. Do not collect source
counters from a generic `sm_90` or `sm_100` build and use them as evidence for
the specialized implementation path.

## The Stall Site Is Not The Cause

A raw stall report names the instruction that waited, not the instruction
that created the wait. This separation is usually the whole diagnosis.

Common failure modes when interpreting source-level stall evidence:

- the sampled instruction is the *consumer* of a value loaded earlier; the
  load is the real bottleneck source
- the sampled instruction is blocked by a barrier or predicate-dependent
  path created elsewhere; fixing the sample site changes nothing

Keep backward-tracing logic explicit when reading a Source page:

1. identify the producing load, arithmetic op, or barrier source behind the
   sampled stall
2. ask whether the producer-consumer distance is too short to hide latency
3. refine the stall family: memory dependency may be local (spill/register
   pressure), constant, or global; execution dependency may be shared-memory,
   arithmetic, or write-after-read hazard
4. only then choose the code change family

Examples that shift the fix:

- local-memory dependency at the blamed source points toward register
  pressure or spills, not "memory is slow"
- global-memory dependency with short producer-consumer distance points
  toward a latency-hiding opportunity such as reordering or unrolling
- shared-memory execution dependency points toward SMEM traffic or reuse
  structure, not HBM
- write-after-read dependency points toward register reuse hazards from
  variable-latency instructions

The value of this reasoning is portable even without a dedicated tool: treat
the sampled line as the waiting consumer, find the producer, then ask which
stall subfamily actually explains the wait.

## Interpreting A Hot Line Late In Optimization

Once coalescing, reduction structure, and obvious divergence are cleaned up,
a hot source line may reflect a real hardware limit rather than a coding
mistake. At that stage:

- compare achieved bandwidth on the load-dominated phase against a
  device-level proxy such as `bandwidthTest`
- if close to that proxy, local source-level tuning is unlikely to buy more
- the next owner is algorithmic refactor or library substitution, not
  another round of line-level tweaks

## Typical Mappings From Profiler Evidence To Broad Follow-on Area

- high ratio of actual-to-ideal global transactions on one load line:
  inspect lane-to-address mapping in that block
- high shared-memory activity concentrated on a reduction line:
  inspect the reduction structure; warp-shuffle may replace the sweep
- `stall_long_scoreboard` concentrated on a load line after earlier fixes:
  inspect reuse or accept the streaming-load cost and plan an algorithmic
  change
- `stall_short_scoreboard` or `mio_throttle` on SMEM-heavy lines: inspect
  SMEM layout for bank conflicts, consider wider loads
- barrier stall clustered near `__syncthreads()`: inspect divergence and
  work balance just before the barrier, not the barrier itself

These are diagnosis handoff categories. Concrete rewrite experiments belong
to the optimization-direction skill downstream.
