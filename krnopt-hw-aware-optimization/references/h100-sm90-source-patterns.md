# H100 / SM90 Source Pattern Pack

Use this local pattern pack when the H100 / H20 SM90 subskill needs
implementation-shaped guidance. It is intentionally stored inside the skill so
the subskill does not depend on exact source paths elsewhere in the repo.

These are source-pattern summaries and pseudocode skeletons, not drop-in
source files. Adapt them to the local kernel, binding contract, and workload
shape.

## TMA + WGMMA Producer / Consumer Skeleton

Use this shape when regular tiles can be staged through TMA and consumed by
WGMMA:

```text
role assignment:
  producer warp or warpgroup
  one or more consumer warpgroups

shared state:
  smem stage ring
  full barriers, one per stage
  empty barriers, one per stage
  tensor-map descriptors for regular operands
  pipeline state: stage index, phase, k tile

prologue:
  initialize barriers
  prefetch descriptor state
  optionally prefill one or more stages

mainloop:
  producer:
    wait(empty[stage])
    issue TMA load for operand tile into smem[stage]
    arrive(full[stage])
    advance producer stage

  consumer:
    wait(full[stage])
    issue WGMMA from smem[stage]
    commit/wait WGMMA group according to pipeline distance
    run fused side work for the current or older tile
    arrive(empty[stage]) when smem can be reused
    advance consumer stage

epilogue:
  drain outstanding WGMMA groups
  convert / scale / activate / store
```

Design rules:

- Keep producer descriptor and predicate work short enough that consumers do
  not starve.
- Treat stage count as a resource decision. More stages improve overlap only
  until SMEM, registers, or barrier pressure reduce residency too much.
- Consumer register pressure is usually the limiting resource because Hopper
  accumulators remain in registers.
- If a tile is irregular or too small, use a pointer-driven fallback for that
  tile rather than forcing it through TMA.

## Minimal Barrier Checklist

For every shared-memory stage, answer:

- Who is allowed to write the stage?
- Which event proves the TMA load is complete?
- Which event proves WGMMA consumers are done reading the stage?
- What happens when a tile is masked or skipped?
- Are producer and consumer phase bits advanced in the same order?

Deadlocks usually come from a skipped tile that fails to advance the matching
barrier, or from a consumer releasing a stage before every WGMMA path has
finished reading it.

## Register Redistribution Pattern

Use warp specialization only when role-specific register budgets make sense:

```text
producer role:
  small live state
  descriptors, coordinates, stage counters
  no accumulator fragments

consumer role:
  accumulator fragments
  WGMMA descriptors / operands
  fused side-work temporaries
  epilogue or scale temporaries
```

Practical guidance:

- Deallocate or minimize producer registers when the toolchain supports it.
- Give consumer warpgroups the register budget needed for accumulator tiles.
- If a fused epilogue causes spills, shrink the WGMMA tile or split epilogue
  work rather than only reducing stage count.

## Attention Interleaving Pattern

Use for attention-like kernels where GEMM and scalar softmax work both matter:

```text
loop over K/V blocks:
  producer:
    TMA load K/V block for future work

  consumer group A:
    WGMMA Q x K for current block

  consumer group B or interleaved phase:
    update max / sum / scale for previous block
    run mask predicates
    issue WGMMA P x V when probabilities are ready

  store path:
    delay output write until all rescale corrections are applied
```

Technique notes:

- The goal is not just faster QK or PV. The goal is overlapping tensor-core
  work, softmax/rescale, masking, and memory movement.
- FP8 attention often needs operand-layout fixes: V transpose, register
  permutation, block quantization, or k-major staging.
- Sliding-window and local masks can become scalar predicate bottlenecks; do
  not assume causal-mask behavior transfers.

## Persistent Grouped GEMM Scheduler Pattern

Use for MoE or grouped GEMM when many uneven groups share one kernel:

```text
grid:
  launch around chosen resident SM count

scheduler state:
  group descriptors or pointer arrays
  per-group M/N/K shape
  per-group stride and scale metadata
  per-group tile offsets
  optional expert token counts on device

persistent loop:
  logical_tile = iteration * num_sms + blockIdx.x
  group, tile_m, tile_n, tile_k = map(logical_tile)
  if tile is valid:
    configure descriptor / pointer state
    run TMA + WGMMA tile pipeline
    write output or partial output
  iteration += 1
```

Design rules:

- Swizzle M/N tile order when it improves L2 reuse for expert weights.
- Treat masked or invalid expert rows as part of the scheduler contract.
- If per-expert M is device-resident, avoid host-only scheduling assumptions.
- The scheduler must be benchmarked; branchy scheduler logic can erase launch
  savings when groups are not actually irregular.

## Runtime TMA Descriptor Mutation Pattern

Use when a grouped kernel changes base address, stride, or K/M slice between
groups:

```text
before using descriptor for group g:
  compute group base pointer or stride
  update tensor-map field for group g
  synchronize descriptor visibility before TMA issue
  issue TMA for group g tile
```

Guardrails:

- Mutate descriptors at group or tile granularity only when the saved padding,
  repacking, or host setup cost is larger than the mutation cost.
- Keep descriptor mutation off the critical consumer path when possible.
- Do not reuse a descriptor for a new group until all TMA operations that
  depended on its old contents are complete.

## Descriptor-Pool Residual Tile Pattern

Use when grouped GEMM wastes work by padding final M tiles:

```text
descriptor pool:
  full tile descriptor
  residual descriptors for selected row counts or powers of two

for each output tile:
  if tile_m is full:
    use full tile descriptor
  else:
    choose residual descriptor
    map padded coordinates to real output rows
    store only valid rows
```

Design rules:

- Keep the full-tile path simple and fast.
- Put residual complexity behind a branch that only final tiles take.
- Compare descriptor-pool overhead against the padded rows it removes.
- Preserve barrier progression even when residual rows are skipped.

## Static-Batched MoE Metadata Pattern

Use when dispatch/permutation overhead is visible and fully materializing
contiguous per-expert token tensors is too expensive:

```text
metadata:
  per-expert row index arrays
  compact virtual-CTA descriptors
  token count per expert
  optional output combine indices

compute:
  CTA maps to virtual expert tile
  load input rows through row indices
  run WGMMA expert tile
  write to output rows or partial output slots
```

Design rules:

- Row-index access can save dispatch traffic but may reduce memory coalescing.
- Use it when permutation/unpermutation is a real measured cost.
- Keep metadata compact enough to fit cache and avoid adding a new bottleneck.

## Fused Dispatch + GEMM Pattern

Use when token dispatch or combine dominates enough that a pure grouped GEMM
cannot solve the bottleneck:

```text
dispatch phase:
  read routed token assignment
  write local expert-token buffer or direct staged rows
  signal row-block readiness

compute phase:
  persistent compute CTAs walk ready row blocks
  run staged WGMMA pipeline per expert tile
  preserve output mapping for combine

combine phase:
  write final rows or partial rows according to route metadata
```

Design rules:

- Fusing dispatch and GEMM increases synchronization and bookkeeping.
- Use readiness flags or barriers only when compute can overlap dispatch.
- Preserve a simple unfused fallback for debugging correctness.

## CUTLASS-Style Grouped Argument Pattern

Use this mental model when adapting a template-library grouped GEMM design:

```text
arguments:
  group_count
  problem_shapes[group]
  A_ptrs[group], B_ptrs[group], C_ptrs[group], D_ptrs[group]
  A_strides[group], B_strides[group], C_strides[group], D_strides[group]
  scale_ptrs[group] or scale_strides[group] when quantized
  scheduler parameters
  optional descriptor workspace
```

Design rules:

- Grouped GEMM is not just a loop over pointers. Shape arrays, stride arrays,
  scale metadata, and scheduler policy are all part of correctness.
- Mixed-input or quantized grouped GEMM needs explicit metadata layout.
- If group descriptors are updated on device, the update path must be included
  in timing.

## SM90 Do-Not-Port List

Do not import these SM100 assumptions into this pattern pack:

- TMEM accumulator storage
- `tcgen05.mma`
- 2-CTA MMA / `cta_group::2` MMA
- Cluster Launch Control
- Blackwell MXFP8 / NVFP4 block-scale layout contracts as hardware-native SM90
  instructions

On Hopper, accumulators remain in registers and WGMMA/TMA/warp specialization
are the main hardware-specific surfaces.
