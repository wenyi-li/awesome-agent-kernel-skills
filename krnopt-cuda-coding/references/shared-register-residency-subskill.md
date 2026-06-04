# Shared Register Residency Subskill

Use this integrated subskill when the main question is how shared memory,
register footprint, and occupancy trade against each other in the source.

## Shared-Memory Questions

Inspect:

- whether shared memory repairs a bad global-memory access pattern
- whether it eliminates redundant global loads
- whether its layout risks bank conflicts
- whether the footprint is large enough to reduce residency materially

Shared memory is a reuse and layout tool, not automatically a win.

## Register Questions

Look for:

- large per-thread temporary arrays
- wide accumulator tiles
- aggressive unrolling
- values that stay live longer than needed
- dynamic indexing that can defeat scalar replacement

## Residency Questions

Ask:

- Does the private state per thread look too large for healthy latency hiding?
- Is occupancy being traded away for real reuse, or just for accidental bloat?
- Could spills push supposedly private state into local memory?

## Common Anti-Patterns

- adding shared memory without eliminating enough traffic to justify it
- maximizing tile size until registers or shared memory quietly wreck residency
- accepting large live ranges because the code structure never narrows them
- treating occupancy as a goal instead of one knob in a tradeoff

## Doctrine From CUDA Best Practices

### Shared memory has three jobs

Shared memory is not simply "faster memory." It is a reuse and layout tool.
The best practices guide frames its use narrowly:

1. to make global reads coalesced (by staging in one pattern and reading in
   another)
2. to eliminate redundant global loads across threads in a block
3. to avoid wasting bandwidth

If a shared-memory tile does not do at least one of these jobs, adding it
will make the kernel slower, not faster. The footprint costs occupancy, and
the staging introduces a `__syncthreads()`.

### Bank conflicts are the next trap

When threads in a warp hit the same shared-memory bank with different
addresses, the accesses serialize. The canonical fix is padding: declaring a
tile as `[TILE][TILE + 1]` so a stride of 32 becomes a stride of 33 modulo
the bank count. Other fixes include swizzled indexing or changing the
iteration order so lane access walks across banks rather than down one.

Practical lesson:

- shared memory is not automatically fast
- its access pattern has to be designed too
- check bank behavior after introducing shared-memory tiling

### Occupancy is a latency-hiding metric, not an end state

Occupancy is the ratio of active warps on an SM to the maximum possible.
Low occupancy is dangerous because memory and dependency latency become
harder to hide. Higher occupancy is not automatically better; it can force
smaller register budgets (which spill), weaker tile shapes, or give up
reuse. Three resource limiters drive occupancy:

- registers per thread
- shared memory per block
- threads per block

Registers are allocated at block granularity, and allocation granularity
itself matters. Two kernels that both use 37 registers per thread can land
at different occupancies depending on block size because of rounding.
Occupancy cannot be reasoned about from one number alone; the question is
always "which resource limits residency, and does raising it actually help?"

A useful diagnostic: vary dynamic shared-memory allocation without changing
kernel logic, which experimentally lowers occupancy. If performance barely
moves, occupancy is not the dominant limiter and the tuning knob is
elsewhere.

### Launch bounds and register control

Declaring `__launch_bounds__(maxThreadsPerBlock)` tells the compiler the
intended launch envelope so future architecture changes do not trip
"too many resources requested" errors. The two-argument form
`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` plus
`-maxrregcount` gives explicit control over the spill/residency tradeoff:

- fewer registers per thread -> potentially more resident warps
- too few registers -> spills into local (off-chip) memory

Clamping registers blindly just converts register pressure into global-memory
latency. That tradeoff must be measured, not guessed.

### Block and thread heuristics

Starting points from the best practices guide:

- thread counts are multiples of 32
- at least 64 threads per block, and only if multiple blocks can still
  reside on an SM
- start experimentation in the 128-256 threads-per-block range
- prefer several smaller blocks over one giant synchronized block per SM
  when barriers are frequent
- launch enough blocks to keep the whole GPU busy; grids in the thousands
  are preferred for forward-facing code

These are starting points, not laws. The right shape depends on register
pressure, shared-memory usage, memory layout, and architecture.

### Register dependencies and arithmetic pipelining

Arithmetic results take several cycles before they can be consumed. Other
warps fill those gaps. That is why occupancy and instruction throughput are
linked: when register dependencies stall threads, the fix is often to keep
enough active warps to cover the gap, not to rewrite the instruction.

## Async Staging And Tensor-Core Tile Contracts

Faster matrix instructions do not remove memory pressure; they amplify it.
Once compute throughput gets high enough, staging policy becomes part of the
algorithm. The global-to-shared async-copy path (`cp.async` on Ampere and
later, or TMA on Hopper/Blackwell) lets a copy overlap with compute and
skips the intermediate register step, potentially reducing register pressure
and increasing occupancy.

Typical shapes that appear in well-designed tiled kernels:

- naive load/store baseline
- single-stage `cuda::memcpy_async` into a shared-memory tile
- multi-stage pipeline with rotating shared-memory buffers
- arrive/wait barrier versions coordinating the stages

### What a tensor-core kernel actually contracts for

Tensor-core samples expose the "boring but decisive" parts. When a
tensor-core kernel underperforms, the problem is usually not the MMA
instruction choice; it is the surrounding tile residency contract:

- A/B tiles must be aligned for the load primitive (for WMMA, often 256-bit
  alignment)
- shared-memory rows may need a small skew (not aesthetic; it reduces bank
  conflicts while preserving the load alignment)
- the load/store path must be vectorized enough to feed the MMA rate
- the kernel must actually reserve enough SMEM via
  `cudaFuncAttributeMaxDynamicSharedMemorySize` to realize the intended tile
  shape; the default dynamic-SMEM budget is often too small

Across dtype variants (FP16, BF16, TF32, FP8), the surrounding schedule
changes less than the MMA primitive. The stable performance logic is:

- keep operands aligned and staged
- reuse B fragments where possible
- separate load roles from MMA roles where the schedule benefits
- spend SMEM to reduce pipeline bubbles and bank conflicts

## Practical Residency Rules

- Do not maximize occupancy blindly; find the limiting resource first.
- Pick warp-aligned block sizes; test 128-256 early.
- Watch register pressure and shared-memory footprint together, not in
  isolation.
- Add shared memory only when it coalesces, eliminates reloads, or repairs
  a bad layout.
- If tensor cores or vector loads are the goal, verify alignment, skew, and
  dynamic SMEM budget before blaming the MMA choice.
- When spills appear, reducing `-maxrregcount` often makes things worse;
  prefer narrower live ranges or smaller accumulator tiles.
