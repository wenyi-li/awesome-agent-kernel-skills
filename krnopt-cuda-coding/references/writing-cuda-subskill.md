# Writing CUDA Subskill

Use this integrated subskill when the main question is "how should I write this
CUDA code so it is both correct and performance-aware?"

## Writing Priorities

Write in this order:

1. choose a clear ownership model for thread, warp, and block work
2. make bounds and edge handling explicit
3. make synchronization requirements explicit
4. make memory access shape and reuse visible in the structure
5. keep register footprint and residency tradeoffs in mind while choosing tiles,
   unrolling, and private state
6. choose dtypes, literals, and instruction paths deliberately
7. use the architecture-specialized build target when the kernel relies on
   target-specific instructions or features

## What Good CUDA Source Should Make Obvious

A reader should be able to see:

- what work each thread owns
- how adjacent lanes move through memory
- where reuse lives: global memory, shared memory, or registers
- where synchronization is required
- which datatype and hardware path the code is trying to use
- which architecture-specific build target is required when the hardware path
  depends on features gated by targets such as `sm_90a` or `sm_100a`

## Good Writing Discipline

- keep a cheap correctness path alive
- make one optimization family at a time
- prefer explicit structure over magical helper layers in the hot path
- only add shared memory when it repairs access or creates real reuse
- only increase tile or accumulator size when the register cost still makes sense
- build architecture-sensitive kernels with specialized targets such as
  `sm_90a` or `sm_100a`; do not rely on general targets such as `sm_90` or
  `sm_100` when the code needs architecture-specific instructions or features

## Bad Writing Signals

- indexing logic too tangled to inspect by hand
- implicit assumptions about launch shape or tensor alignment
- synchronization whose purpose is not obvious
- broad abstractions that hide lane-to-address behavior
- code that looks tensor-core- or vector-ready but does not satisfy the real
  dtype, shape, or alignment requirements

## The Execution Model, As A Writing Contract

A CUDA kernel is a function launched across a grid of thread blocks, with
each thread computing its own slice by inspecting `threadIdx`, `blockIdx`,
`blockDim`, and `gridDim`. The canonical skeleton every kernel should make
visible:

```cuda
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

Three things should always be recoverable from the top of the kernel:

1. a logical index computed from block and thread coordinates
2. an explicit edge guard (`if (i < N)`) when the problem is not a perfect
   multiple of the launch shape
3. one unit of work per thread (or per tile, for advanced kernels)

Advanced kernels map *tiles*, *warps*, or *pipeline stages* to indices
rather than one scalar output per thread, but the skeleton is the same.
When an advanced kernel hides the index calculation behind helpers, the
reader can no longer verify the mapping by inspection. That is usually a
writing bug, not an abstraction win.

### Pick launch shape to match data shape

- 1D blocks/grids for vectors and linear buffers
- 2D blocks/grids for matrices and images
- 3D shapes only when the data is genuinely 3D

Good default starting points:

- 128 or 256 threads per block for simple 1D kernels
- `16x16` or `32x8` style blocks for simple 2D kernels

These are starting points. The right shape depends on register pressure,
shared-memory usage, memory layout, and architecture. But a clean mapping
is more important than cleverness at the beginning, and a clean mapping is
much easier to optimize later.

### Thread, block, and warp hierarchy

- **thread**: smallest execution unit
- **warp**: hardware execution group of 32 threads moving through
  instructions together; not launched directly but decisive for performance
- **thread block**: cooperation unit with shared memory and `__syncthreads`
- **grid**: all blocks launched for one kernel

Two practical writing rules follow:

- choose block sizes as multiples of 32 so warps are fully populated
- avoid heavy divergence where neighboring threads take very different paths

Blocks should be largely independent so the scheduler can place them freely.
Anything that assumes a particular block execution order is a writing bug
unless explicit cooperation primitives (clusters, cooperative groups) are
used.

### Beginner checklist that still applies to advanced kernels

Ask these before worrying about performance:

- Can each thread compute its own logical index correctly?
- Is the launch shape aligned with the data shape?
- Is the block size a sensible multiple of 32?
- Are blocks independent enough for normal scheduling?
- Is the kernel simple enough to validate before optimizing?

If any answer is no, fix that before adding shared memory, tiling, or
vectorization.

## Writing Idioms That Pay Off Later

- declare shared-memory tiles with an explicit shape that shows where
  padding lives (`__shared__ float tile[TILE][TILE + 1];`)
- keep the tile ownership formula on one or two lines near the top of the
  kernel so it can be inspected by hand
- place each `__syncthreads()` next to the producer/consumer transition it
  protects, with a one-line comment identifying the boundary
- use vector types (`float4`, `int4`) for global loads when alignment
  allows, so the compiler can emit wide loads
- declare `__launch_bounds__(maxThreadsPerBlock)` when a specific launch
  envelope is intended, so the compiler's register budget matches the
  design
- separate load roles from MMA roles when targeting tensor cores; helpers
  that elect a single thread (or warp) for bulk copies make the schedule
  legible

### When modern features should enter

The classic blocks-and-grids model is enough for most kernels. Thread-block
clusters, distributed shared memory, TMA, and warp specialization are
extensions, not replacements. The right order is:

1. learn ordinary blocks and grids first
2. become comfortable mapping data to threads
3. only then add clusters, pipelines, and architecture-specific launch
   tricks

Reaching for clusters or TMA before a kernel has a clean thread-to-data
mapping tends to produce code that is hard to validate and hard to profile,
because correctness bugs and performance bugs are entangled.
