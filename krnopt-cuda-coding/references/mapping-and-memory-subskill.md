# Mapping And Memory Subskill

Use this integrated subskill when the main question is whether the source has a
sane thread-to-data mapping and memory behavior.

## Mapping Questions

Ask:

- What does one thread compute?
- What does one warp compute?
- What does one block compute?
- Do adjacent lanes touch adjacent addresses?
- Are stores as well as loads shaped sensibly?

This is the fastest way to spot coalescing trouble.

## Memory And Reuse Questions

Then ask:

- Which operands come from global memory?
- How many times is each loaded value reused?
- Is reuse in registers or shared memory visible from the code?
- Is the kernel streaming once, or revisiting the same values many times?
- Are vectorized loads and stores aligned and legal?

## Structural Warnings

- strided or scattered lane access without an explicit reason
- repeated global loads of the same data without staging
- over-fetch that exceeds the useful work per transaction
- complicated indexing that hides the real lane-to-address map
- memory movement that grows faster than the arithmetic done on the data

## Next-Step Discipline

If the source looks bandwidth-heavy, say that explicitly as a hypothesis. Do not
claim it is the dominant runtime bottleneck until profiling confirms it.

## Doctrine From CUDA Best Practices

### Coalescing is the first rule

Threads in a warp do not fetch memory independently if the hardware can combine
their requests. The useful definition is transaction-based: for a warp, the
fraction of bytes actually used over bytes transferred should be as close to
1.0 as possible, and accesses should stay within as few 32-byte segments as
possible. Consecutive lanes reading consecutive aligned words is the common
way to achieve that, but the deeper rule is broader:

- make adjacent threads touch adjacent words
- use block sizes that preserve warp alignment
- avoid misalignment and non-unit stride where possible

Misaligned sequential access increases the number of memory transactions.
Strided access is much worse because cache lines are fetched and barely used.
Non-unit stride is a classic bandwidth killer. A kernel can be "correctly
parallel" and still waste most of its bandwidth on badly mapped traffic.

### Thread-to-data mapping decides coalescing at the source level

CUDA source is written from a single-thread perspective, but global-memory
efficiency is decided at warp granularity. The most common coalescing bug is
mapping `threadIdx.x` to the *wrong* matrix dimension: if adjacent lanes end
up loading different rows of a row-major matrix, the stride across a warp is
the row length rather than one element. Remapping the index (so lane 0 and
lane 1 land on elements 0 and 1 of the contiguous axis) is usually a small
source change, and the SASS instruction mix does not have to change for
coalescing to improve. Coalescing is a runtime memory-system behavior, not an
instruction selection.

Practical rule: if a profiler shows very low memory throughput and poor
requested-vs-actual efficiency, inspect lane-address diagrams before reaching
for shared memory.

### Shared memory as a repair tool

The programming guide's canonical pattern shows shared memory used to
decouple compute layout from global-memory layout:

1. read global memory in a coalesced pattern
2. stage the tile in shared memory
3. `__syncthreads()`
4. write logically transformed data back in a coalesced pattern

Minimal transpose shape (with bank-conflict padding):

```cpp
template <int TILE>
__global__ void transpose_tiled(float* out, const float* in, int ld) {
  __shared__ float tile[TILE][TILE + 1];  // +1 avoids a classic bank conflict

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = in[y * ld + x];
  __syncthreads();

  int ox = blockIdx.y * TILE + threadIdx.x;
  int oy = blockIdx.x * TILE + threadIdx.y;
  out[oy * ld + ox] = tile[threadIdx.x][threadIdx.y];
}
```

The pattern is not about transpose specifically; it is about using shared
memory to repair access shape when the compute layout and storage layout
disagree.

### Vectorization requires layout and alignment

Wider global loads (`LDG.E.128` via `float4`, `int4`, or equivalent vector
types) reduce instruction count and improve throughput, but the compiler
cannot emit them unless the code supplies an alignment contract. Raw pointers
passed as kernel arguments do not carry 128-bit alignment guarantees; using a
vector type (or a properly aligned struct) gives the compiler the contract it
needs.

For shared memory, addresses are compiler-controlled, so a layout change
(such as transposing the shared-memory tile for A in GEMM so the reuse
direction lands on contiguous banks) can let the compiler issue `LDS.128`
loads without any pointer-type change.

### Local memory usually means a hidden tax

"Local" describes scope, not physical location. Local memory lives in device
memory, so it carries global-memory latency and bandwidth. Kernels end up
there through:

- large automatic arrays
- large structs
- non-constant indexing that prevents scalarization
- register spilling

A kernel that spills aggressively can look memory-bound even when the source
feels compute-heavy.

## Worked Example: SGEMM's Three Memory Problems

Distinct "memory bottlenecks" require distinct fixes. A single SGEMM
optimization worklog shows three, in order:

1. **Non-coalesced global loads.** Naive SGEMM maps `threadIdx.x` to a
   dimension with large stride, so adjacent lanes load distant rows of A.
   Fix: remap the index so lanes walk the contiguous axis. No instruction
   change needed.

2. **Insufficient reuse.** After coalescing, adjacent threads still pull the
   same operands from global memory repeatedly. Fix: shared-memory tiling of
   A and B. Load tile, `__syncthreads()`, run the K-slice of dot products,
   `__syncthreads()`, advance. Sizing the tile trades global traffic against
   shared-memory footprint and occupancy.

3. **Too many shared-memory instructions per FMA.** Once SMEM caching is in,
   profiling shows the instruction mix dominated by SMEM loads and stalls
   such as `Stall MIO Throttle`. The next move is *not* more shared memory;
   it is register blocking plus vectorized SMEM loads. Compute more outputs
   per thread, cache reused operands in registers, and transpose the A tile
   in shared memory so the compiler can issue `LDS.128`.

The diagnostic mapping this produces:

- low global-memory throughput -> check coalescing
- good global bandwidth but low FLOP/s -> check arithmetic intensity
- high MIO / shared-memory stalls -> reduce SMEM instructions per FMA
- already-high cache hit rate -> be skeptical of cache-locality swizzles

That same worklog reports a thread-swizzling attempt for L2 locality that did
not help because L2 hit rate was already about 80%. Structurally valid
optimizations can still be irrelevant when the measured bottleneck is
elsewhere.

## Priority Order For Memory Work

When a kernel is slower than expected, walk this sequence:

1. Are neighboring threads reading neighboring addresses?
2. Is the kernel reloading the same data from global memory repeatedly?
3. Would a shared-memory tile reduce traffic or repair stride?
4. Did register pressure push data into local memory?
5. Is the architecture offering a better movement primitive such as
   `cp.async` or TMA that can overlap the copy with compute?

Only after those should narrower tricks (constant memory for small read-only
data, L2 persistence windows for reused regions) enter the conversation.
