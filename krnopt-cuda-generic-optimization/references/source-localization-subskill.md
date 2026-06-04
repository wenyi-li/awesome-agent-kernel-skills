# Source Localization Subskill

Use this integrated subskill when the task asks which part of the code likely
causes the problem.

## Localize By Code Shape

Look for the source region whose structure best matches the observed issue:

- **coalescing problem**:
  lane-to-address mapping, strided indexing, gather-style loads, badly aligned
  vector access
- **shared-memory pressure or bank conflicts**:
  transpose-like layouts, column-wise access, repeated scalar SMEM ops, missing
  padding or swizzle
- **register pressure / low occupancy**:
  large per-thread temporaries, wide accumulator tiles, long live ranges,
  aggressive unrolling
- **divergence / sync cost**:
  warp-misaligned branches, barriers after uneven work, reductions with many
  inactive lanes
- **wrong instruction path**:
  dtype conversions, scalar fallback code, missing MMA-friendly tile shapes,
  unexpected FP64 or generic math path
- **epilogue overhead**:
  extra conversions, reduction, normalization, or postprocessing fused into the
  hot loop

## Be Concrete

Do not stop at "memory issue near the main loop." Name the likely region:

- the index expression
- the load/store loop
- the reduction body
- the shared-memory tile declaration and access pattern
- the epilogue block
- the launch-shape or template parameter choice

## Confidence Rule

If localization comes from static inspection only, say "likely" or "probable".
If localization is backed by line-attributed profiler evidence, say so
explicitly.

## Blamer Discipline: Waiting Site vs Causing Site

Raw line-level profiler data reports where a warp *stalled*, not necessarily
the instruction that *created* the stall. Localization that ignores this
distinction tends to fix symptoms instead of causes.

Two common failure modes:

- the sampled instruction is the consumer of a value loaded earlier; the real
  culprit is the producing load and the insufficient producer-consumer
  distance
- the sampled instruction is blocked by a barrier or predicate-dependent path
  created elsewhere; the real culprit is an earlier sync or predicate choice

Disciplined localization therefore walks backward through dependency sources
before claiming a root cause. In practice this means:

1. treat the sampled stall site as the waiting consumer, not automatically the
   culprit
2. identify the producing load, arithmetic op, or barrier source
3. ask whether the producer-consumer distance is too short to hide latency
4. ask whether the stall family really implies spills, sync pressure,
   shared-memory pressure, or global-memory latency
5. only then choose a code change family

Useful pruning rules when reasoning about candidate producers:

- **opcode alignment**: only memory instructions can explain a memory
  dependency stall; only sync instructions can explain a synchronization
  stall; reject candidates whose opcode class cannot produce the observed
  family
- **dominator suppression**: if another non-predicate use of the same
  dependency appears on every path from the candidate source to the stalled
  instruction, the closer use would have stalled first; blame the closer use,
  not the distant one
- **latency coverage**: if every path from the candidate source to the use is
  longer than the source instruction's latency, the dependency is already
  covered; reject the candidate

After pruning, if two or more sources remain plausible, weight blame by issued
samples at each source and penalize long source-to-use paths. Most cases then
collapse to a single suspect; the residual ambiguity gets a principled split
instead of a forced one-hot guess.

## Reclassify The Blamed Family

Once blame lands on a source instruction, refine the stall family before
proposing a fix. The same raw family covers different source-level causes:

- **memory dependency** splits into local (spills / register pressure),
  constant (constant-memory traffic), or global (HBM/L2 latency). Local is
  often misread as "memory is slow" when it is really "per-thread state is
  spilling."
- **execution dependency** splits into shared-memory dependency (SMEM traffic
  or reuse structure), arithmetic dependency (latency of expensive math), or
  WAR dependency (register reuse hazard caused by variable-latency
  instructions).

Concrete examples:

- local-memory dependency at the blamed source points toward register pressure
  and spills, not global bandwidth
- global-memory dependency with a short producer-consumer distance points
  toward latency hiding (reordering, unrolling) rather than "more bandwidth"
- shared-memory execution dependency points at SMEM traffic or reuse, not
  HBM
- WAR dependency points at register reuse hazards around variable-latency ops

## Scope Matters

The same stall family implies different code-level actions depending on the
scope where it concentrates:

- **inside a loop**: unrolling, software pipelining, or widening the loop
  body to expose more independent work
- **in a straight-line region** with short producer-consumer gaps: reorder
  code, hoist loads and address math earlier
- **around device-function calls**: inline small hot helpers, or split cold
  or bulky logic to relieve instruction-cache pressure
- **at launch configuration**: change block size or grid geometry

"Fix memory stalls" without scope is usually too vague to act on; name the
scope before naming the change.

## Matching Code Structures To Bottleneck Families

The following structural fingerprints are typical for each family. Use them as
starting hypotheses when static inspection is the only evidence.

- **launch or orchestration overhead**: many back-to-back small kernel
  launches, frequent `cudaMemcpy` calls or implicit syncs in the host path,
  per-iteration allocation, missing CUDA Graphs or persistent execution.
- **memory-bandwidth pressure (saturated)**: streaming loops over large
  tensors with low reuse, scalar loads where vector loads would align,
  layout that forces warp lanes to stride, frequent re-reads of the same
  operand.
- **memory-latency pressure (scattered)**: gather/scatter by non-contiguous
  indices, linked-list or pointer-chasing access, random indirection into
  large arrays, per-thread loops with no reuse.
- **shared-memory pressure / bank conflicts**: column-wise access to a tile
  whose leading dimension is a multiple of 32 bytes without `+1` padding,
  transpose-like writes, many scalar `LDS` ops in the inner loop.
- **occupancy / register pressure**: per-thread accumulator arrays, large
  template-unrolled tiles, wide live state across long loops, aggressive
  `#pragma unroll` on deep loops, many inline helper functions with large
  temporaries.
- **divergence / barrier cost**: `if (threadIdx.x < K)` where K is not a
  multiple of 32, early exits inside a warp, reductions with many inactive
  lanes, barriers immediately after uneven per-warp work.
- **compute / instruction-path limit**: GEMM-shaped kernels with no
  `wmma`, `mma.sync`, or WGMMA use; dtype-mixed epilogues doing implicit
  promotions to FP32 or FP64; scalar fallback paths where a packed SIMD
  intrinsic exists; manual implementations of standard primitives where
  cuBLAS, cuBLASLt, cuDNN, NCCL, framework/runtime primitives, CCCL APIs
  such as CUB/Thrust/libcu++, or CUTLASS/CuTe may fit; generic `pow`/`exp`
  where a specialized variant fits.

## Source-Localized Hotspots To Collect First

When line-attributed profiler views are available, four source-level signals
are the most useful for localization:

- **warp-stall sampling (all samples)**: hotspot lines where warps spend the
  most stall time — the waiting consumers
- **divergent branches**: source lines causing warp divergence
- **ideal-vs-actual global transactions (or equivalent sector excess)**: lines
  with uncoalesced global memory access
- **shared wavefront excess**: lines causing shared-memory bank conflicts

These four together let the skill say "this region of code is the probable
cause of this specific bottleneck family" rather than "somewhere in the hot
loop."

Line-attributed collection requires the kernel to be built with `-lineinfo`
(not `-G`, which disables device optimization and distorts the profile).

## Kernel-Level Classification Drives Line-Level Search

The order of reasoning is: classify the kernel first, then look at specific
lines only for the suspected family.

- **memory-bandwidth-saturated state**: drive line search with sector excess
  and coalescing metrics; look at the load/store loops and index expressions
- **memory-latency-bound state**: drive line search with cache-miss and
  long-scoreboard stall signals; look at scattered-access regions and
  pointer-chasing loops
- **shared-memory bank-conflict state**: drive line search with shared
  wavefront excess; look at tile declarations and column-wise access
- **divergence-high state**: drive line search with divergent-branch counts;
  look at control flow near warp-misaligned boundaries
- **register-pressure / low-occupancy state**: static inspection carries
  more weight here than line-level profiler evidence; look at per-thread
  arrays, long live ranges, large inline functions
- **API-overhead-dominant state**: the body of any single kernel is not the
  right scope; localize at the host side (launch pattern, graph capture
  opportunity, fusion opportunity)

## Worked Examples From SGEMM Profiling

A few source-localization lessons generalize well from the SGEMM worklog:

- **Very low global-memory throughput on a naive kernel** (around 15 GB/s
  on the worklog's machine) is typically not a shared-memory problem. Draw
  the lane-to-address mapping for A, B, and C; make the contiguous matrix
  dimension correspond to consecutive warp lanes; preserve alignment so the
  hardware can merge lane requests. Fix this before adding caches.
- **Good memory bandwidth but low FLOP/s** after adding shared-memory
  staging means insufficient arithmetic intensity, not insufficient
  bandwidth. The suspect region is the inner loop: each thread is doing too
  little math per loaded operand. Increase output elements per thread;
  reuse shared operands across more FMAs; move hot operands into registers;
  prefer 2D output tiles when register budget allows.
- **Instruction mix dominated by shared-memory loads** (many `LDS` per
  `FMA`) means the hot loop alternates SMEM loads with arithmetic. The
  suspect region is again the inner loop. Restructure so one loaded value
  feeds multiple FMAs; cache repeated values in registers; use compile-time
  loop bounds so the compiler can unroll and eliminate repeated loads.
- **High `Stall MIO Throttle` with no special math or dynamic branches**
  likely means shared-memory instruction pressure. Reduce SMEM instructions
  per result; increase register blocking; vectorize SMEM loads (for example
  scalar `LDS` becoming `LDS.128` via a layout change); inspect SASS to
  confirm wide loads actually emerged.
- **Occupancy looks reasonable but runtime is poor** (for example around
  66% occupancy on a GEMM kernel) argues against occupancy as the cause.
  There are enough warps for latency hiding; the actual limiter is
  instruction mix. Use warp-state and instruction-mix views rather than
  chasing higher occupancy.
- **Optimization helps on one GPU but not another** is normal: best tile
  shapes depend on per-GPU shared memory, registers, and vector load
  divisibility. Encode validity constraints before benchmarking; test real
  problem sizes, not only canonical squares; expect production libraries to
  dispatch across many specialized kernels.
- **Plausible locality optimization does nothing** (for example thread
  swizzling failing to improve runtime because L2 hit rate was already
  high) is a reminder to confirm the metric an optimization is *supposed*
  to move is actually bad before adding complexity. Check cache hit rate
  and memory-traffic counters first; drop locality transforms without
  measured benefit.

The generalizable discipline from these cases: localize using the specific
metric family that matched the classification; confirm that the metric is
actually unhealthy before proposing a transformation that targets it.

## Output Discipline For Localization

Every localization statement should carry:

- the region (function, loop, tile declaration, epilogue, launch config)
- the suspected family (coalescing, bank conflict, register pressure, etc.)
- the evidence type (static inspection vs line-attributed vs summary-metric
  inference)
- the confidence (probable / likely / measured)
- one or two alternative regions that are plausible but lower-ranked,
  especially when hotspot mass is diffuse

If the hotspot mass is diffuse (no single region carries a large share of the
blame), prefer a structural change (launch geometry, tile restructure, fusion)
over heroic line-by-line edits.
