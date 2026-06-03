# Skill: Handle Boundary Conditions in Tiled Kernels

## Purpose
Guide the agent through correctly handling partial tiles — cases where a problem dimension does not evenly divide the tile size — in CUDA and Triton kernels, without introducing out-of-bounds accesses, incorrect output values, or silent data corruption.

## Use this when
- Writing any tiled kernel (GEMM, reduction, softmax, elementwise, attention) where the tile size may not divide the problem dimension evenly.
- The problem shape is determined at runtime and cannot be guaranteed to be a multiple of the tile size.
- A kernel processes batched inputs where batch elements may have different sizes.
- Writing an attention kernel with variable-length sequences.
- Reviewing a kernel for correctness and needing to audit boundary handling explicitly.

## Do not use this when
- The kernel is exclusively called with shapes that are statically known and guaranteed to be multiples of the tile size (e.g., a highly constrained production kernel with a fixed shape contract). Even then, document this invariant explicitly and add an assertion.
- The problem dimension is padded externally to a tile multiple before the kernel is called. In this case, verify the padding strategy is correct and document that boundary handling is delegated to the caller.

## Inputs the agent should gather first
- The tile size (or candidate tile sizes) in each dimension.
- The problem dimensions (M, N, K, or sequence length, batch size, etc.) and whether they are known statically or determined at runtime.
- The memory layout of the input and output tensors: row-major, column-major, strided.
- Whether the input is padded externally, and if so, what value the padding contains (zero, -inf, arbitrary).
- For batched ops: whether all batch elements have the same size or variable sizes.
- For attention with variable sequence length: whether the sequence lengths are stored in a separate tensor and how they are passed to the kernel.
- Whether the kernel reads, writes, or both for each tensor involved in the boundary tile.

## Required reasoning process
1. **Identify all tiled dimensions.** For each dimension tiled by the kernel, compute the number of tiles as `ceil(N / TILE_SIZE)`. The last tile covers indices `[(num_tiles-1)*TILE_SIZE, N)` and has `N % TILE_SIZE` valid elements (when `N % TILE_SIZE != 0`).

2. **Classify each tensor access in the last tile as read or write.** Reads and writes have different safety implications:
   - Out-of-bounds global memory reads in CUDA are undefined behavior. They may return garbage, cause silent corruption, or fault depending on the driver and hardware configuration.
   - Out-of-bounds global memory writes in CUDA may corrupt adjacent allocations or fault. Even if the pointer arithmetic happens to land within the allocated buffer, writing to unowned memory is undefined behavior.
   - In Triton, out-of-bounds loads with a mask return `other` (a safe default); out-of-bounds stores with a mask are no-ops. This is defined behavior.

3. **Choose a boundary strategy for each dimension.** The three options are:
   - **Masking in-kernel (preferred for most cases)**: compute a per-thread or per-element validity predicate and conditionally skip the load/store. Zero overhead for full tiles; a small branch or predicate for partial tiles.
   - **External padding (valid when caller controls allocation)**: pad the input tensor to the next tile multiple before calling the kernel. The kernel then sees only full tiles. Requires extra memory and a separate padding step. Caller must document padding value semantics.
   - **Loop-level guard in CUDA (simple alternative)**: use `if (global_idx < N)` around the entire thread's computation. Safe but may waste warp efficiency if many threads in a warp are masked.

4. **For CUDA kernels: write the boundary guard explicitly.** For a 1D kernel:
   ```
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= N) return;  // or mask out store only, keeping load safe
   ```
   For a 2D tiled kernel loading a tile into shared memory, use a per-element guard:
   ```
   int row = tile_row * TILE_M + threadIdx.y;
   int col = tile_col * TILE_N + threadIdx.x;
   if (row < M && col < N) {
       smem[threadIdx.y][threadIdx.x] = input[row * N + col];
   } else {
       smem[threadIdx.y][threadIdx.x] = 0.0f;  // safe padding value
   }
   ```

5. **For Triton kernels: use masks on every tl.load and tl.store.** Triton's tile-based model makes this natural:
   ```python
   offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   mask = offs < N
   x = tl.load(ptr + offs, mask=mask, other=0.0)
   # ... computation ...
   tl.store(out_ptr + offs, result, mask=mask)
   ```
   For 2D tiles:
   ```python
   row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
   col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
   mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
   ```

6. **Assess reduction over partial tiles separately.** When the last tile is partial, the reduction must only include the valid elements:
   - In Triton: mask out invalid elements by setting them to the reduction identity (0.0 for sum, -inf for max) before the reduction.
   - In CUDA: in the warp/block reduction loop, inactive threads must contribute the identity element, not arbitrary values from their load.

7. **Handle variable-length batched inputs.** When batch elements have different sizes:
   - Store per-element lengths in a separate int32 tensor.
   - In the kernel, load the length for the current batch element and use it as the boundary `N` for that element.
   - Ensure each program instance (Triton) or block (CUDA) handles at most one batch element, or explicitly track which batch element each tile belongs to.

8. **Verify that padding value semantics are consistent with the computation.** Zero padding is appropriate for sums and dot products. For softmax, padding should be `-inf` so masked positions contribute zero to the softmax output. For max reductions, padding should be `-inf`. Using zero padding in a softmax kernel is a correctness bug.

## Kernel design rules
- Never load from a global memory address that may be out of bounds in CUDA. Compute the validity predicate before loading.
- For shared memory loads in CUDA tiled kernels: pad out-of-bounds positions with the computation's identity element, not with an uninitialized value.
- In Triton, never call `tl.load` without a `mask` parameter on any tile that could be a boundary tile. Unconditional loads are only safe for the non-boundary tiles, and only if you have a static guarantee about the shape.
- For reductions over partial tiles, explicitly set out-of-bounds elements to the identity value before the reduction. Do not rely on the load returning zero — that is only guaranteed when using the `other=0.0` parameter.
- When the tile size is larger than the problem dimension (e.g., BLOCK_SIZE=128 on a tensor of size 64), the entire first tile is also the only tile and all of it is potentially a boundary. The masking logic must handle this case.
- For 2D tiled kernels (GEMM), the K dimension (reduction dimension) also has a boundary case. Partial K tiles must zero-pad the shared memory load so that the dot product is not corrupted.
- Write boundary checks as predicates on global indices, not on thread indices. A thread-index-based check does not generalize correctly across different launch configurations.

## Correctness requirements
- The output for every valid output position `[0, M) x [0, N)` must be correct, regardless of whether that position falls in a full tile or a partial tile.
- No write is made to any output position outside `[0, M) x [0, N)`. Confirm this by inspection of the store guard.
- For reductions, the result for the last partial tile must be identical to the result of reducing only the valid elements. Out-of-bounds elements must not affect the reduction output.
- Softmax over a partial tile must normalize only over the valid elements. A softmax that includes zeros from out-of-bounds positions produces outputs that do not sum to 1.0 over the valid positions.
- Boundary behavior must be tested with shapes that are: exactly one tile, one tile minus one element, one element, a large prime number, and the maximum expected size.

## Performance requirements
- Masking adds predicated instructions, not branches, in most cases. For Triton, masks are handled at the compiler level and do not introduce warp divergence for boundary tiles. For CUDA, `if (idx < N) return;` causes divergence only in the last block — accept this cost.
- For kernels where the overwhelming majority of tiles are full (large tensors, small tile size), the cost of boundary masking is negligible. Do not pad the tensor externally to avoid masking unless profiling shows a measurable bottleneck.
- If external padding is chosen for performance reasons, quantify the memory overhead: padding a non-power-of-2 tensor to the next tile multiple adds `(TILE_SIZE - N % TILE_SIZE) * element_size` bytes per row/dimension. For large tensors with small remainder, this is small. For small tensors with TILE_SIZE >> N, it can be substantial.
- Persistent kernels and multi-tile programs must ensure boundary masking applies correctly to all tiles, not just the launch's final tile.

## Output format
The agent should produce:

1. **Boundary analysis**: for each tiled dimension, state the tile count, the size of the last partial tile, and which tensors are read or written in that tile.
2. **Masking strategy**: explicit statement of which strategy (in-kernel mask, external padding, loop guard) is used for each dimension and why.
3. **Kernel implementation**: complete, compilable kernel code with boundary guards clearly annotated with comments.
4. **Correctness test**: a test suite that includes shapes that trigger the partial tile path — specifically: `N = TILE_SIZE - 1`, `N = TILE_SIZE + 1`, and one prime-sized N. Each test compares output to a CPU reference.
5. **Edge case documentation**: explicit documentation of any shapes or configurations the kernel does not support (e.g., "requires N >= 1", "undefined for empty tensors").

## Common failure modes
- **Reading past the end of the input array in CUDA**: computing `input[row * stride + col]` without checking `col < N` and `row < M`. On most hardware this is undefined behavior and may silently return stale cache data.
- **Writing garbage to valid output positions**: loading an out-of-bounds element into shared memory without padding it to the identity value, then using that garbage value in a dot product or reduction that writes to a valid output position.
- **Off-by-one in the boundary predicate**: using `if (idx <= N)` instead of `if (idx < N)`, accessing one extra element.
- **Masking the wrong dimension**: in a 2D tile, masking on `col < N` but forgetting to also mask on `row < M`.
- **Softmax with zero-padded boundary**: padding out-of-bounds positions with 0.0 instead of -inf causes those positions to contribute `exp(0) = 1` to the softmax denominator, producing wrong probabilities for the valid positions.
- **Reduction identity mismatch**: using 0.0 as the identity for a max reduction (should be -inf), or -inf as the identity for a sum reduction (should be 0.0).
- **Assuming padding is zero when it is arbitrary**: when the caller pads the input, the kernel cannot assume padding values are zero unless that is explicitly documented and enforced.
- **Variable-length batches with incorrect stride arithmetic**: computing the offset into the input tensor for batch element `b` without accounting for the actual length of each prior batch element, producing reads into wrong batch elements.
- **Last tile in the K (reduction) dimension of GEMM**: if K is not a multiple of the tile size, the partial K tile's smem load must zero-pad the out-of-bounds positions. Failing to do so adds the contribution of garbage values to every output row.

## Review checklist
- [ ] Every tiled dimension has been analyzed for partial tile cases.
- [ ] All global memory reads in CUDA are guarded against out-of-bounds access.
- [ ] All global memory writes in CUDA are guarded to prevent writing outside the valid output range.
- [ ] All `tl.load` and `tl.store` calls in Triton include a mask for dimensions that may have boundary tiles.
- [ ] Out-of-bounds elements loaded into shared memory or Triton tensors are set to the correct identity value for the computation (0 for sum, -inf for max, etc.).
- [ ] The boundary condition is tested with shapes that are: exactly one tile, one tile minus one, a prime number, and one element.
- [ ] For softmax and attention over partial tiles: out-of-bounds positions are masked with -inf before the softmax, not with 0.
- [ ] For variable-length batched inputs: the per-element length is loaded from the length tensor and used as the boundary for that batch element.
- [ ] The K-dimension boundary is handled in any GEMM-style kernel with a reduction over tiles.
- [ ] Padding value semantics are documented if external padding is used.
