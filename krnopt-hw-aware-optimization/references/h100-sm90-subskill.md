# H100 / H20 SM90 Subskill

Use this integrated subskill when the target hardware is NVIDIA H100, H20,
H800, or another SM90-family Hopper GPU and the question is which
Hopper-native features should change how the kernel is written or
restructured.

This subskill is deliberately self-contained. It should be enough to make a
first hardware-aware design decision without external prerequisite reading.
Source paths near the end are optional implementation anchors, not required
reading.

For Hopper MoE, grouped expert GEMM, FP8 fused MoE, or dispatch/GEMM fusion,
also read [h100-sm90-moe-practices.md](h100-sm90-moe-practices.md). That page
adds MoE-specific method cards for DeepGEMM-style grouped FP8, masked decode,
persistent scheduling, descriptor handling, static metadata, and FP8 forward
pipeline boundaries.

The main Hopper lesson is that TMA and WGMMA are the center of gravity, but
the accumulator still lives in registers. That means Hopper optimization is
often a balance between better async overlap and surviving register pressure.

## SM90 Facts That Change Kernel Decisions

Treat these as first-order constraints, not trivia:

- `wgmma` is the defining tensor-core instruction family on SM90 Hopper
- TMA (Tensor Memory Accelerator) is the preferred async multidimensional data
  mover for regular global/shared tiles
- accumulators stay in registers; Hopper has no TMEM
- `228 KB` shared memory per SM makes multi-stage pipelines practical
- H100 exposes up to `227 KB` shared memory per thread block because CUDA
  reserves `1 KB` per block
- static shared memory above `48 KB` still needs explicit dynamic-SMEM opt-in
- H100 keeps `64` resident warps per SM, up to `32` thread blocks per SM, a
  `64K` 32-bit register file per SM, and `255` max registers per thread
- H100 has a `50 MB` L2 cache and keeps Ampere-style L2 persistence controls
- thread-block clusters and distributed shared memory exist on Hopper, but
  cluster launch shape needs explicit occupancy checks through
  `cudaOccupancyMaxActiveClusters`
- Programmatic Dependent Launch (PDL) exists on Hopper, but is a launch-latency
  and prefetch-overlap tool, not a default kernel-speed knob

The key asymmetry:

- Hopper gives stronger overlap tools than Ampere
- but it still makes large kernels pay directly in registers, shared memory,
  barriers, and live state

So on SM90, optimization usually means:

- using TMA only when descriptor-driven tile movement is regular enough
- using WGMMA only when staging and warp roles can keep tensor cores fed
- searching tile shape, stage count, copy path, warp-role split, and epilogue
  together
- treating scale handling, activation, softmax, reductions, stores, and
  routing as real bottlenecks once GEMM gets faster
- verifying H20/H800 resource limits and throughput behavior locally instead
  of copying every H100 assumption

Do not port SM100 / Blackwell ideas mechanically to Hopper. TMEM,
`tcgen05.mma`, 2-CTA MMA, Cluster Launch Control, and Blackwell block-scaled
examples are design context only; they are not implementation surfaces for
H100, H20, or H800.

## Compile Targeting Is Part Of The Plan

For Hopper-specific claims, require an explicit SM90a target path.

Useful target spellings depend on the build system, but the important markers
are `sm_90a`, `compute_90a`, or an equivalent `9.0a` target. Plain `sm_90` or
`compute_90` is a fallback target for this subskill, not enough evidence for
SM90a-specific Hopper claims such as WGMMA/TMA-heavy fast paths.

Practical rule:

- keep a portable fallback path when the project needs it
- build the Hopper fast path with explicit SM90a targeting
- record the CUDA toolkit, driver, GPU model, build flags, and actual target
  in every timing or correctness claim
- never use `sm100a`, `compute_100*`, TMEM, `tcgen05.mma`, or CLC flags for an
  H100/H20 implementation path

## Most Useful SM90-Specific Features

Treat these as the first Hopper feature set to consider:

- TMA for regular multidimensional global/shared tile movement
- WGMMA for warpgroup asynchronous tensor-core compute
- warp specialization for producer / consumer decomposition
- register redistribution between producer and consumer warpgroups
- persistent grouped scheduling when many small expert GEMMs share one launch
- runtime TMA descriptor mutation or descriptor pools for grouped irregularity
- L2-aware grouped tile ordering or persistent tile ordering
- thread-block clusters and DSMEM only when cross-CTA sharing is worth the
  launch/occupancy cost
- PDL only when launch coupling, latency hiding, or data-prefetch overlap is
  measurable end-to-end

## Feature To Code Mapping

### TMA On Hopper

Use when:

- tiles are multidimensional and regular enough for descriptor-driven movement
- global-to-shared traffic is a major structural concern
- the kernel can afford explicit producer/barrier choreography
- shared-memory staging is reused enough to justify descriptor setup

What TMA changes:

- addressing moves from byte/pointer offsets to descriptor coordinates
- TMA descriptors describe the tensor map, rank, strides, swizzles, and base
  address; they are not ordinary pointers
- a TMA op is usually issued by a small producer role while other warps compute
- shared memory stage reuse is guarded by barriers, not by assuming the copy is
  complete when the issuing thread advances
- in CuTe/CUTLASS-style code, TMA-coordinate tensors and coordinate tuples
  replace much of the raw pointer arithmetic

Implications for code:

- maintain a stage index or circular `PipelineState`
- pair full/empty stage barriers with each shared-memory stage
- separate descriptor construction/update from descriptor use
- keep the producer path short enough that consumers do not starve
- benchmark descriptor construction/mutation cost against the padding or
  repacking work it replaces

Likely wins:

- reduced register-staging traffic versus ordinary global-to-register-to-SMEM
  copies
- cleaner multi-stage pipelines for regular tiles
- better overlap when producer and consumer roles are split
- multicast and cluster-aware movement when the cluster design is justified

Likely traps:

- using TMA for tiny, jagged, or decode-side operands where descriptor churn is
  larger than the copy benefit
- treating TMA as a drop-in faster load rather than a different addressing and
  synchronization contract
- forgetting that extra stage count increases SMEM footprint and may reduce
  residency
- placing too much descriptor logic or predicate work in the producer path

Fallback path:

- `cp.async` / `cuda::memcpy_async` with `cuda::pipeline` for small or irregular
  residual pieces
- mixed kernels are legitimate: TMA for regular bulk operands, `cp.async` or
  ordinary cooperative loads for irregular tails

### WGMMA

Use when:

- operands are staged into shared memory in a tensor-core-friendly layout
- warpgroup-level cooperation is acceptable
- the kernel can overlap staging, WGMMA issue, and epilogue work
- the tile shape does not create unmanageable accumulator register pressure

What WGMMA changes:

- compute is issued at warpgroup scale, not as isolated per-warp scalarized MMA
- shared-memory layout, alignment, swizzle, and operand order become part of
  correctness
- WGMMA can be asynchronous, so commit/wait structure matters
- accumulator fragments remain register-resident on SM90

Implications for code:

- assign consumer warpgroups explicitly
- keep producer and consumer barrier state coherent
- use WGMMA commit/wait groups consistently
- budget epilogue, scale, activation, and store logic against live accumulator
  registers
- search smaller tiles if spills or low occupancy erase tensor-core gains

Likely wins:

- higher tensor-core throughput than scalarized MMA paths
- good match to TMA-staged operands and warp-specialized schedules
- natural foundation for FP16/BF16/FP8 grouped GEMM and attention kernels on
  Hopper

Likely traps:

- treating WGMMA as a drop-in replacement for an existing scalarized MMA loop
- picking a large tile because arithmetic intensity looks good while register
  pressure collapses occupancy
- adding fused epilogue or dequant code after the fact without re-budgeting
  registers
- ignoring layout conversions, transposes, or permutation costs required to
  feed the WGMMA operand contract

Fallback path:

- smaller WGMMA tile
- fewer live accumulators and more K stages
- a cooperative non-WGMMA path when the schedule cannot keep a warpgroup fed

### Warp Specialization

Use when:

- data movement and compute can be decoupled
- TMA and WGMMA overlap is a first-order goal
- producer work is predictable enough that dedicated warps are not wasted
- consumer work has enough tensor-core or scalar side work to hide TMA latency

Canonical Hopper split:

- producer warpgroup: TMA loads, descriptor updates, stage barrier arrival
- consumer warpgroups: WGMMA, softmax or activation, reductions, epilogue
- shared memory stages plus barriers connect the roles

Register redistribution is part of the pattern. Producer warps should carry a
small register footprint; consumer warps need more registers for accumulator
fragments and fused math. DeepGEMM and FlashAttention-style Hopper kernels both
use this idea: deallocate registers from producer roles and allocate more to
math roles where the compiler/runtime path supports it.

Likely wins:

- real overlap between global/shared movement and tensor-core compute
- clearer resource budgets for producer and consumer roles
- a natural path to overlapping scalar work, such as softmax or activation,
  with asynchronous WGMMA

Likely traps:

- splitting roles when the producer has too little work and just burns warps
- letting producer predicate/descriptor work become the bottleneck
- creating deadlocks through mismatched full/empty barrier progression
- designing warp specialization independently from tile shape and stage count

Fallback path:

- symmetric CTA where all warps cooperate on load/compute/store
- one producer warp rather than a full producer warpgroup
- a simpler double-buffered `cp.async` pipeline for small or irregular tiles

### Persistent Grouped Scheduling

Use when:

- many small expert GEMMs would otherwise launch inefficiently
- routed MoE work is irregular enough that one kernel should own many tiles
- expert token counts vary across batches
- grouped launch ordering or L2 locality is visible in the profile
- padding, permutation, per-expert launch overhead, or scheduler overhead is a
  material part of total time

The basic persistent pattern:

1. launch a grid sized for resident work, often around the chosen SM count
2. each CTA repeatedly claims tile work from a logical grouped problem space
3. tile order is swizzled or grouped to improve L2 locality for expert weights
4. per-expert token counts, K offsets, and residual M shape feed the scheduler
5. the kernel exits only after all grouped tiles are processed

DeepGEMM's SM90 scheduler is the compact pattern to emulate conceptually:

- tile ownership follows a persistent loop like `current_iter * num_sms +
  blockIdx.x`
- M/N block ordering is swizzled for locality
- invalid TMA multicast choices are disabled dynamically
- masked expert token counts are walked on device
- K-grouped cumulative offsets drive descriptor mutation between groups
- the heuristic model filters candidates by SMEM, swizzle, stage count,
  register pressure, wave efficiency, and estimated L1/L2 traffic

Implications for code:

- scheduler state is part of the kernel, not a wrapper detail
- tile order affects cache behavior
- descriptor construction or mutation may happen at runtime
- expert token-count histograms matter as much as GEMM dimensions
- persistent residency and register footprint bound throughput

Likely wins:

- fewer tiny launches and better SM utilization
- better load balance across uneven expert groups
- improved L2 reuse when tile ordering revisits expert weights coherently

Likely traps:

- adopting persistence when the workload is already regular enough
- expecting persistence to replace correct load balancing
- adding a scheduler whose registers and branches outweigh the saved launches
- copying a grouped schedule without checking the target expert distribution

Fallback path:

- static tile ownership inside one grouped kernel
- one launch per expert batch when group count is small
- Stream-K decomposition when the real problem is long-K imbalance rather than
  per-expert irregularity

### Runtime TMA Descriptors And Descriptor Pools

Use when:

- grouped problems have variable residual sizes
- padding to tile multiples wastes measurable work or memory
- expert weights or output tiles are selected dynamically
- host-side descriptor construction cannot keep up with device-side routing

Two useful Hopper patterns:

- runtime descriptor mutation: update tensor-map base address or stride between
  groups, as in K-grouped or pointer-array grouped GEMM
- descriptor pools: prebuild a small set of descriptors for residual row
  shapes and select among them on device, especially for padding-free final
  tiles

Padding-free residual tile handling:

- keep the regular full-tile path simple
- detect residual M rows explicitly
- map padded coordinates to unpadded output rows
- use prebuilt descriptors or overlapping stores to cover arbitrary final tile
  heights
- compare descriptor/predicate overhead against the saved padded work

Implications for code:

- descriptor lifetime and ownership need a clear policy
- descriptor updates must be synchronized with the stage using them
- residual-tile logic becomes part of correctness
- barrier use must account for skipped or masked rows

Likely wins:

- less wasted compute on padded expert rows
- less memory traffic for residual outputs
- one persistent kernel can serve many dynamic expert shapes

Likely traps:

- assuming TMA alone solves grouped irregularity
- mutating descriptors too often on small decode paths
- forgetting that descriptor update and validation cost is real
- allowing residual-path complexity to slow the common full-tile path

Fallback path:

- padded full-tile grouped GEMM when waste is small
- `cp.async` or pointer-driven copies for irregular residuals
- split regular bulk and irregular tail into separate kernels when that is
  simpler and faster

### Thread-Block Clusters And DSMEM

Use when:

- multiple CTAs need to cooperate on shared staged data
- one CTA's shared memory is too small for the desired tile or reduction
- TMA multicast or cluster-wide shared-memory movement has a clear data-reuse
  benefit
- the cluster shape still leaves enough active clusters to occupy the GPU

Implications for code:

- cluster dimensions become launch-contract state
- use `cudaOccupancyMaxActiveClusters` to reason about residency
- DSMEM access patterns need coalesced, aligned behavior
- cluster barriers and memory ordering are now part of correctness

Likely wins:

- cross-CTA reuse for large tiles or reductions
- multicast staging for operands shared across CTAs
- more flexible decomposition when one CTA cannot own the whole tile

Likely traps:

- using clusters as an early optimization before a single-CTA tile is clean
- losing occupancy through oversized clusters
- adding cluster synchronization cost without enough reuse
- copying SM100 2-CTA MMA or CLC thinking into Hopper; those are not SM90
  features

Fallback path:

- single-CTA WGMMA/TMA kernel
- split the operation into independent CTAs with global-memory reduction
- smaller tile or fewer stages

### Programmatic Dependent Launch

Use when:

- two kernels are already coupled in a producer/consumer sequence
- the second kernel can prelaunch or prefetch before the first fully completes
- end-to-end latency matters, not just single-kernel time
- the dependency can be expressed clearly without making debugging brittle

Implications for code:

- PDL belongs to the launch plan and measurement plan
- correctness must be validated with the launch coupling enabled
- timing should include end-to-end sequence latency and not only isolated
  kernel duration

Likely wins:

- reduced gap between dependent kernels
- earlier prefetch or setup for the consumer kernel

Likely traps:

- using PDL to avoid fusing kernels that should simply be fused
- adding launch complexity when the bottleneck is inside one kernel
- treating PDL as a scheduler for persistent grouped work; it is not

Fallback path:

- ordinary sequential launches
- CUDA graphs
- direct kernel fusion when the dataflow is tight enough

## Hopper Attention Pattern

FlashAttention-style Hopper kernels are the cleanest example of TMA/WGMMA
overlap with scalar side work.

The SM90 attention pattern:

1. producer warpgroup issues TMA loads for Q/K/V or dQ/dO/K/V tiles
2. consumer warpgroups issue WGMMA for QK and PV-like GEMMs
3. softmax, masking, rescaling, and reductions are interleaved with async WGMMA
4. some operands or results may bypass shared memory through register-source
   paths when layout and register budgets allow it
5. stage barriers and WGMMA wait groups keep tiles live only as long as needed

The central technique is interleaving, not just faster GEMM. A phase-ordered
kernel that does QK, then all softmax, then all PV leaves Hopper hardware
underused. A better SM90 design overlaps softmax or rescaling from one tile
with WGMMA from another tile.

FP8 attention adds extra layout obligations:

- operand layout may be K-major for WGMMA throughput
- V tiles may need in-kernel transpose or permutation
- quantization can be blockwise rather than per-tensor
- accumulator and operand layout mismatches may require explicit register
  permutation
- local or sliding-window masks can become scalar predicate bottlenecks even
  when causal-only masks remain WGMMA-dominated

When to apply this pattern outside attention:

- there is a tensor-core mainloop plus visible scalar side work
- the scalar work can be scheduled on different tiles than the current WGMMA
- the kernel can tolerate the extra barrier and register complexity

When not to apply it:

- tiles are too small to amortize TMA/WGMMA setup
- scalar work is tiny or memory-bound elsewhere
- register pressure already limits occupancy

## Hopper MoE And Grouped GEMM Pattern

For MoE on H100/H20, do not start by asking only "which GEMM tile is fastest."
First measure routing, permutation, grouped GEMM1, activation, grouped GEMM2,
combine, padding overhead, expert token-count distribution, L2 hit rate,
tensor-core utilization, register spills, shared-memory use, and occupancy.

The useful SM90 MoE design space:

- pointer-array grouped GEMM: each group has its own problem shape, pointer
  layouts, strides, and optional scale or zero-point metadata
- FP8 grouped GEMM: scale metadata and block/group scale layout are part of
  the kernel contract; on Hopper these remain Hopper-compatible scale paths,
  not Blackwell MXFP8/NVFP4/TMEM paths
- M-grouped contiguous GEMM: experts are grouped along M/token rows, good for
  prefill or batches where per-expert rows are contiguous enough
- masked M-grouped GEMM: per-expert M lives on GPU; invalid rows are skipped
  or masked while preserving barrier correctness
- K-grouped GEMM: K is partitioned by group, and runtime TMA descriptor address
  or stride mutation changes which K slice is staged
- static-batched metadata: per-expert row-index arrays and compact virtual-CTA
  metadata let WGMMA operate without materializing fully contiguous token
  tensors per expert
- fused dispatch plus GEMM: token movement, row-block readiness, persistent
  compute CTAs, and expert GEMM are designed together when routing overhead is
  visible

Concrete technique summaries:

- CUTLASS-style pointer-array grouped GEMM uses `GroupProblemShape`-like
  problem arrays, pointer/stride arrays, device-side scheduling, and
  on-the-fly TMA descriptor modification between groups.
- CUTLASS-style FP8 grouped GEMM adds block/group scale metadata and schedule
  choices such as warp-specialized cooperative FP8 blockwise paths.
- Mixed-input grouped GEMM adds group count, problem-shape arrays, stride
  arrays, and scale/zero-point stride arrays; this is useful when A/B/input
  dtypes or quantization metadata differ by group.
- DeepGEMM SM90 1D2D FP8 keeps FP32 scale metadata, supports normal, batched,
  M-grouped contiguous, psum-layout, and masked layouts, and emits BF16 output.
- DeepGEMM SM90 1D1D FP8 covers K-grouped contiguous GEMM and shows runtime
  TMA descriptor address/stride mutation between K groups.
- ThunderKittens-style dispatch/GEMM fusion pulls routed tokens from input
  tensors, writes post-dispatch local tokens, signals per-row-block readiness,
  and runs a four-stage H100 WGMMA GEMM over per-expert padded token ranges.
- TMA-adaptive grouped GEMM attacks wasted residual rows by selecting
  descriptor-pool entries for final tile sizes instead of padding every group
  to a full tile.

Decision rules:

- if routing/permutation/combine dominates, consider fusing dispatch,
  static-batched metadata, or row-index based expert access before retuning
  GEMM tiles
- if padding dominates, consider residual descriptor pools or masked grouped
  GEMM
- if expert weights are revisited, tune tile order for L2 locality
- if many experts are small, consider persistent grouped scheduling before
  per-expert launches
- if scale metadata or activation dominates, budget it against registers and
  pipeline slots rather than treating it as an epilogue afterthought

DeepGEMM's SM100 Mega MoE and other Blackwell kernels are design context only
for Hopper MoE. They rely on TMEM and `tcgen05.mma`, so do not treat them as
direct H20 ports.

## Hopper Pipeline Sketch

A useful Hopper mainloop should be able to answer these questions:

1. Which tile is the producer loading now?
2. Which tile are the consumers computing now?
3. Which barrier proves shared memory is safe to read?
4. Which barrier proves shared memory is safe to overwrite?
5. How many WGMMA groups are outstanding?
6. Where do epilogue, store, activation, scale, or softmax work overlap with
   subsequent loads?
7. Which registers are live across each stage boundary?

Minimal conceptual shape:

```text
initialize stage barriers
prefetch descriptor state

for k_tile in pipeline:
  producer role:
    wait for empty stage
    issue TMA load into shared stage
    arrive full barrier

  consumer role:
    wait for full stage
    issue WGMMA on shared stage
    commit / wait WGMMA group as required
    run fused scalar side work for an older or current tile
    release empty barrier when shared stage can be reused

epilogue:
  drain remaining WGMMA groups
  convert / scale / activate / store
```

This is not a literal template. It is the dependency checklist. Production
code may split producer and consumer into different warpgroups, use multiple
consumer warpgroups, add stores or reductions to the producer side, or use
different barriers for loads and stores.

## Common Failure Modes

- **Overweight CTA:** too many stages or too large a tile reduces residency
  enough to lose the benefit of overlap.
- **Register spills:** WGMMA consumers need accumulator registers; spills erase
  tensor-core gains.
- **Layout mismatch:** WGMMA and FP8 paths may require operand layouts that
  force extra transpose, permutation, or shared-memory traffic.
- **Producer starvation:** descriptor updates, predicates, or irregular
  address math keep consumers waiting.
- **Consumer starvation:** TMA cannot deliver regular tiles fast enough or the
  producer role has too few warps.
- **Scalar side bottleneck:** softmax, activation, scale handling, reductions,
  masking, or output combine dominate once GEMM improves.
- **Barrier mismatch:** producer/consumer stages deadlock or reuse shared
  memory too early.
- **Cluster overreach:** DSMEM and inter-SM TMA add occupancy and
  synchronization cost without enough reuse.
- **Blackwell leakage:** TMEM, `tcgen05.mma`, 2-CTA MMA, CLC, or SM100
  block-scaled examples enter an SM90 plan.

## Hopper Profiling Checklist

Before recommending a Hopper rewrite, produce or request a short note with:

- target GPU and whether it is H100, H20, H800, or another SM90-family device
- CUDA toolkit, driver, build flags, and whether SM90a targeting is explicit
- correctness baseline, tolerance, and trial count
- warmed-up timing baseline with workload shape and repetition count
- top kernel timeline evidence
- Nsight Compute evidence for tensor-core utilization, memory stalls, register
  spills, achieved occupancy, shared-memory use, barrier stalls, and L2 locality
- for MoE: expert-token histogram, padding overhead, and time in routing,
  permutation, grouped GEMMs, activation, and combine
- for attention or softmax-like kernels: time in exp/MUFU, reductions, masking,
  rescaling, and non-MMA instructions

If this evidence is absent, the hardware-aware output should name hypotheses
and the measurements needed, not present a Hopper feature as proven.

## SM90 Decision Pivots

Use these as fast questions before recommending a Hopper rewrite:

1. Is this truly an SM90a/Hopper target, or did Blackwell guidance leak in?
2. Is the operand movement regular enough that TMA really pays?
3. Is WGMMA starvation or register pressure more likely to dominate?
4. Does the workload benefit from warp-specialized producer / consumer roles?
5. Is the scalar side work large enough to require overlap, fusion, or
   redesign?
6. Is grouped irregularity large enough to justify persistent scheduling?
7. Would runtime descriptor handling beat padding or repacking?
8. Would clusters or DSMEM increase reuse enough to pay for occupancy and
   synchronization cost?
9. What is the clean fallback if the Hopper-native path is too state-heavy?
10. For H20/H800, have local device resource limits and throughput assumptions
    been checked rather than copied from H100?

## Local Source Pattern Pack

When code shape matters, use the local companion note
`references/h100-sm90-source-patterns.md`. It packages the useful SM90 source
patterns inside this skill: TMA/WGMMA producer-consumer structure, barrier
checklists, register redistribution, attention interleaving, persistent
grouped GEMM, runtime descriptor mutation, descriptor-pool residual handling,
static-batched MoE metadata, fused dispatch plus GEMM, and CUTLASS-style
grouped argument layout.

## When SM90 Knowledge Is Not Enough

If a Hopper feature detail is still unclear after this subskill:

- do targeted docs or source research to find the specific source or paper
- prefer source-backed implementation details over improvising descriptor,
  barrier, or scheduler semantics from memory
