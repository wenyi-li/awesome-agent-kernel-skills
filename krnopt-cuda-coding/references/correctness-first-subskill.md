# Correctness First Subskill

Use this integrated subskill when the code may be fast-looking but the first
real question is whether it is semantically trustworthy.

## Correctness Questions

Check these before performance claims:

- Is the thread or tile ownership formula explicit and inspectable?
- Are edge conditions guarded when dimensions are not perfect multiples of the
  launch shape?
- Are asynchronous launches, copies, and dependencies synchronized where the
  semantics require it?
- Are barriers and warp-sync calls placed correctly for the data-sharing shape?
- Are dtype choices, literal types, and reduction order deliberate?
- Is there a cheap reference path or regression input to re-check after changes?

## Common Failure Patterns

- off-by-one or swapped-index bugs in tile ownership
- missing edge guards for partial tiles
- relying on implicit ordering across async operations
- using synchronization as decoration instead of semantic necessity
- accidental FP64 or unexpected promotion from literals and helper functions
- changing precision or accumulation order without re-validating tolerances

## Good Discipline

- keep a known-good baseline or reference implementation nearby
- re-check after each optimization family
- separate semantic correctness from performance suspicion in written notes
- treat "works on one shape" as insufficient

## Methodology From The CUDA Programming Guide

The programming guide's introductory examples repeatedly compare GPU output
against a serial CPU result for a reason: many CUDA optimizations change
execution order, memory layout, synchronization structure, or precision
behavior. If each change forces building a new ad hoc validation path,
iteration stalls and changes feel too dangerous to make. A small CPU or
library reference is part of the performance workflow, not a beginner crutch.

### Treat asynchronous errors as first-class

Kernel launches and many runtime calls are asynchronous. That has three
consequences:

- a launch can succeed even though the kernel later faults
- a later CUDA runtime call may surface an earlier asynchronous error
- error state persists until cleared with `cudaGetLastError()`

"No crash" is not meaningful evidence that a kernel is correct. The minimal
discipline is to separate launch-time error checking from completion-time
error checking:

```cpp
kernel<<<grid, block, smem, stream>>>(...);
CUDA_CHECK(cudaGetLastError());              // launch parameter / setup errors
CUDA_CHECK(cudaStreamSynchronize(stream));   // execution errors for that stream
```

A `cudaSuccess` return from the launch path does not prove kernel execution
succeeded. Both checks are needed before trusting a result.

### Prefer scoped synchronization over global stalls

`cudaDeviceSynchronize()` waits for all prior device work and is useful for
debugging and early bring-up, but it is usually too blunt for tuned code.
Performance work generally wants the narrowest synchronization scope that
still preserves correctness:

- `cudaStreamSynchronize(stream)` waits only for one stream
- events let you wait on exactly the dependency you need

Leaving device-wide synchronization in a hot path masks ordering bugs and
prevents overlap opportunities.

### Pinned host buffers are part of the overlap contract

`cudaMemcpyAsync()` only overlaps with compute when host buffers are pinned
and page-locked. Pageable buffers may still work functionally, but the
transfer can fall back to effectively synchronous behavior. If a program
relies on transfer/compute overlap, pinned buffers (via `cudaMallocHost` or
`cudaHostRegister`) are not an optional micro-optimization; they are part of
the intended execution model.

### Barriers and warp sync are semantic, not decorative

A barrier that exists "just in case" is a performance cost pretending to be
safety. The right question for every `__syncthreads()` or `__syncwarp()` is
which producers and consumers it covers and whether the data-sharing shape
actually requires it there. On Volta and later, independent thread scheduling
can leave warps diverged beyond a conditional region, so `__syncwarp()` may
be needed for later code that assumes lane-uniform state even when no data
race is obvious.

### Unified Memory is convenient, not magic

Managed memory removes explicit movement code, but optimal performance still
comes from predictable locality. If the working set bounces unpredictably
between CPU and GPU, Unified Memory can lose badly. Use `cudaMemAdvise` and
`cudaMemPrefetchAsync` when access locality is known; do not assume managed
memory automatically means optimal placement.

### Precision and accumulation order

Changes to dtype, literal types, reduction order, fast-math intrinsics, or
block size can all shift numerical results. Each such change should be
re-validated against the reference path, not just against "does it still
run?" Accidental FP64 from a literal (`1.0` instead of `1.0f`) or an
unintended promotion inside a helper function can change both correctness
and speed without any visible source-level signal.

## Do And Don't

- Do keep a small CPU or library oracle for regression checks.
- Do check both launch-time and execution-time CUDA errors.
- Do use stream- or event-scoped synchronization when possible.
- Do allocate pinned host buffers when overlap matters.
- Don't treat `cudaSuccess` from a launch path as proof that kernel execution succeeded.
- Don't keep `cudaDeviceSynchronize()` in the hot path once finer-grained synchronization is available.
- Don't change precision or accumulation order without re-running the reference.
