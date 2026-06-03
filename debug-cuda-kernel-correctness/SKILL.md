# Skill: Debug CUDA Kernel Correctness

## Purpose
Guide the agent through a systematic process of isolating, reproducing, and diagnosing correctness errors in CUDA kernels — covering indexing bugs, layout mismatches, synchronization races, reduction errors, numerical drift, and out-of-bounds memory access.

## Use this when
- A CUDA kernel produces wrong output compared to a CPU or high-precision reference.
- The kernel output differs between runs (non-determinism indicating a race condition or uninitialized memory read).
- The kernel produces correct results for small inputs but fails for large inputs or non-power-of-two sizes.
- Correctness errors appear only on specific hardware (sm_80 vs sm_86) or with specific compilation flags.
- The kernel produces NaN, Inf, or suspiciously exact-zero outputs.
- A previously correct kernel starts failing after a layout, dtype, or tiling change.

## Do not use this when
- The kernel is numerically close to the reference but not bit-exact — evaluate whether the error is within acceptable floating-point tolerance before treating it as a correctness bug.
- The issue is performance, not correctness (use profiling skills instead).
- The bug is in the host-side launch configuration wiring, not the kernel body itself — check grid/block dimensions, stream assignments, and argument passing before diving into kernel internals.

## Inputs the agent should gather first
- **Reference implementation**: what is the expected correct result? Is there a CPU fp64 reference, a PyTorch equivalent, or a known-correct CUDA baseline?
- **Input shapes and dtypes**: exact M, N, K (or equivalent dimensions), dtype (fp32, fp16, bf16, int8), memory layout (row-major, column-major, strided, non-contiguous).
- **Hardware and driver**: CUDA compute capability, CUDA toolkit version, driver version. Some bugs are toolkit-specific (e.g., compiler optimizations that break on certain SM targets).
- **Reproducibility**: does the error appear on every run with the same inputs, or is it intermittent? Intermittent = likely race condition. Deterministic = likely logic or indexing bug.
- **Error characterization**: is the error localized (specific output elements wrong), global (all elements wrong by a constant factor), or structured (wrong values along a row/column/diagonal boundary)?
- **Last known-good change**: what changed between the last correct version and the current buggy version?

## Required reasoning process

1. **Reproduce the error on a minimal input.** Before anything else, reduce the input size to the smallest case that still fails. A 4x4 matrix multiplication failure is much easier to diagnose than a 4096x4096 one. Powers of two are not representative of all failure modes — test sizes like 31, 63, 65, 127, and 129 to exercise partial-tile boundary conditions.

2. **Classify the error pattern.** Print the output tensor (for small failing inputs) and compare element-by-element against the reference:
   - Errors on all elements, uniformly scaled: likely an incorrect scale, dtype cast, or accumulated factor in the epilogue.
   - Errors on specific rows or columns: likely an indexing bug where the wrong scale vector, bias, or address is used.
   - Errors only in the last few rows or columns: almost certainly a boundary condition handling bug (partial tile, out-of-bounds load treated as zero, wrong loop trip count).
   - Random-looking errors in a few elements: likely a race condition (missing `__syncthreads`, incorrect warp shuffle mask, read-after-write without barrier).
   - NaN or Inf at specific positions: look for division by zero (uninitialized or zero-valued denominator), sqrt of a negative value, or an overflowing accumulator.

3. **Check indexing arithmetic explicitly.** Index bugs are the most common class of CUDA kernel correctness errors. For every global memory access in the kernel:
   - Write out the full flattened index formula: `row * stride_row + col * stride_col + offset`.
   - Check that strides are passed correctly from the host and match the actual tensor layout.
   - Verify that `threadIdx`, `blockIdx`, and `blockDim` are combined correctly. A common bug: using `blockIdx.x * blockDim.x + threadIdx.x` when the intended mapping uses `blockIdx.y`.
   - Check broadcast dimensions: a scale vector of length N applied to an MxN output should index with `col`, not `row`.
   - Check transpose logic: if A is passed as transposed, its k and m dimensions are swapped.

4. **Audit shared memory usage and synchronization.** For every shared memory read/write pair:
   - Is there a `__syncthreads()` between the write phase and the read phase?
   - Is there a `__syncthreads()` at the end of each tile loop iteration before the next tile is loaded?
   - Are all threads in the block guaranteed to reach every `__syncthreads()` call? A `__syncthreads()` inside a conditional that only some threads enter is a deadlock or race condition.
   - Check shared memory bank conflicts as a secondary concern, but only after correctness is confirmed.

5. **Check reduction correctness.** For kernels that perform parallel reductions:
   - Verify the reduction tree covers all elements, including the last partial tile.
   - Verify that the initial accumulator value is correct (`0.0` for sum, `-FLT_MAX` for max, `+FLT_MAX` for min, `1.0` for product).
   - Verify warp-level reductions use the correct `__shfl_xor_sync` mask and offset sequence.
   - For multi-block reductions: verify the final combination step reads from the correct global buffer positions.

6. **Check dtype and precision.** For every intermediate value in the compute path:
   - Is fp16 or bf16 arithmetic used where fp32 is required for accurate accumulation?
   - Are INT8 values accidentally sign-extended to INT16 instead of INT32 before multiplication?
   - Is there a loss of precision from a narrow accumulator (e.g., using `half` as the accumulator in a dot product loop)?
   - Are casts explicit? Implicit narrowing casts (fp32 → fp16) can silently lose information in epilogue code.

7. **Use `cuda-memcheck` / `compute-sanitizer` for memory safety.** Run the kernel under `compute-sanitizer --tool memcheck` to detect:
   - Out-of-bounds global memory accesses.
   - Out-of-bounds shared memory accesses.
   - Uninitialized memory reads.
   - Misaligned memory accesses.
   Any of these can produce wrong results silently without sanitizer tooling. This step is mandatory for any kernel that fails with apparently random or intermittently wrong outputs.

8. **Isolate to a single thread block.** For non-trivial correctness bugs, add a guard `if (blockIdx.x != 0 || blockIdx.y != 0) return;` to run only one block, print its outputs, and verify them in isolation. This eliminates global memory aliasing, ordering-dependent writes, and inter-block synchronization issues from the investigation.

9. **Check the host-side launch configuration.** Verify that:
   - `gridDim` and `blockDim` are computed correctly for the actual input dimensions (check rounding: `(N + BLOCK - 1) / BLOCK`).
   - Kernel arguments match the kernel parameter list in order and type.
   - Pitched memory strides (if using `cudaMallocPitch`) are passed in bytes, not elements.
   - Any host-device synchronization required before reading results is in place (`cudaDeviceSynchronize()` or stream synchronization).
   - The correct device is selected when using multi-GPU code.

10. **Binary search with assertion kernels.** For large, complex kernels, insert assertion checks at intermediate stages using `assert()` inside the kernel (enabled under `CUDART_DEVICE_ASSERTION`) or by writing intermediate results to debug buffers and validating them on the host. Narrow the failing region to the smallest unit of code that first introduces wrong values.

## Kernel design rules
- Always initialize output buffers to a sentinel value (e.g., `NaN`, `-1`, or a known constant) before launching the kernel being debugged. A zeroed output buffer can mask a bug where a kernel writes zero (or fails to write at all).
- Print the first failing element's value, expected value, and flattened index whenever a mismatch is detected. Raw index decomposition (row, col, tile_row, warp_lane) is more useful than a flattened index alone.
- For race condition investigation: disable `--use_fast_math` compilation and test with a single thread block. If the bug disappears, it is likely related to fast-math transformations or reduced parallelism exposing a race.
- Do not use `printf` inside warps with conditional printing as the primary debugging tool for correctness bugs — output ordering is non-deterministic. Write to a debug buffer indexed by thread and read it out on the host.
- Synchronization bugs often manifest differently depending on GPU occupancy. Test with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` and adjusted launch configs to exercise different occupancy levels.

## Correctness requirements
- The kernel must produce results within the expected numerical tolerance of the reference. For fp32 reductions over large arrays, tolerance should account for floating-point non-associativity (typically < 1e-5 relative error for well-conditioned inputs). For INT8 kernels, tolerance must be bounded by the quantization error (0.5 * scale per element).
- The kernel must produce correct results for all input sizes in the valid range, including sizes that are not multiples of the tile or warp size.
- The kernel must produce consistent results across multiple runs with the same input (if determinism is required). Non-determinism indicates a synchronization bug or reads from uninitialized memory.
- The kernel must produce correct results regardless of the number of blocks launched. A correctness dependency on launch configuration is a design flaw.
- Out-of-bounds memory accesses must not occur for any valid input. Accesses caused by incorrect boundary handling that happen to not corrupt visible output are still bugs.

## Performance requirements
Correctness debugging does not require performance optimization. However, do not introduce debugging infrastructure (debug buffers, assertion checks, per-thread output printing) that disables the performance features being debugged, unless the goal is to simplify the problem space temporarily. Validate correctness at the full target configuration after the bug is fixed.

## Output format
The final response must include:
1. **Error classification**: what category of bug is this (indexing, synchronization, dtype, boundary, launch config)?
2. **Minimal reproduction**: the smallest input size and configuration that reproduces the bug.
3. **Root cause**: the specific line(s) of code responsible, with an explanation of why they produce wrong output.
4. **Fix**: the corrected code with a clear explanation of what changed and why.
5. **Verification plan**: how to confirm the fix is correct (reference comparison, input size sweep, sanitizer run).
6. **Regression tests to add**: what test cases should be added to prevent this class of bug from regressing.

## Common failure modes
- **Transposed stride bug**: passing the wrong stride to a tiled kernel after transposing an input tensor. The kernel computes the right logical index but accesses the wrong physical memory location.
- **Off-by-one in grid dimension**: computing `gridDim.x = N / BLOCK` instead of `(N + BLOCK - 1) / BLOCK`. The last tile is silently not processed, leaving the corresponding output elements at their initial (zero or garbage) values.
- **Missing `__syncthreads()` after shared memory write**: a classic race condition where some threads advance to the read phase before all threads have completed the write phase of the same tile. The bug often appears only under certain occupancy conditions where thread scheduling happens to reveal the race.
- **Wrong accumulator type**: using `float` instead of `double` for a reduction that requires high precision, or using `half` as the accumulator in an inner product loop.
- **Scale broadcast dimension mismatch**: a per-channel weight scale vector of shape [N] is applied with the row index instead of the column index when writing to an MxN output.
- **`__shfl_sync` with wrong mask**: using `0xffffffff` for a partial warp at the tail of the input, causing lanes outside the valid range to participate in the shuffle, producing garbage values in edge positions.
- **Non-contiguous tensor passed as contiguous**: the kernel assumes row-major contiguous layout but the input tensor has non-unit strides from a previous transpose or slice operation. The tensor appears to have the right shape but the data is not laid out as expected.
- **Uninitialized shared memory read**: a thread reads a shared memory location that was not written by any thread in the current tile iteration (e.g., reading padding positions during a reduction without masking them).
- **Atomics without global memory fence**: using `atomicAdd` for a multi-block reduction without ensuring all blocks have completed before the final read. On discrete GPUs, there is no inter-block synchronization guarantee without explicit synchronization primitives or kernel split.

## Review checklist
- [ ] Has the smallest failing input been identified?
- [ ] Has the output been compared element-by-element against a fp64 CPU or known-correct reference?
- [ ] Has the error pattern been classified (all elements, structured, boundary-only, non-deterministic)?
- [ ] Has every global memory index formula been written out and verified explicitly?
- [ ] Has every `__syncthreads()` placement been audited against the shared memory read/write sequence?
- [ ] Has the kernel been run under `compute-sanitizer --tool memcheck`?
- [ ] Has the kernel been tested with non-power-of-two input sizes including sizes smaller than the tile?
- [ ] Has the host-side launch configuration (grid, block dims, argument order) been verified?
- [ ] For reductions: has the initial accumulator value and the final combination step been verified?
- [ ] After the fix: does the kernel pass for a full sweep of input sizes and shapes?
