# TileLang Error Catalog

Comprehensive catalog of error messages, their causes, and fixes. Organized by failure category.

## Category A: Build/Codegen Errors

### `T.gemm K shape check failed: K_A = X, K_B = Y`
**Cause**: The shared buffers passed to T.gemm have incompatible K dimensions.
**Rule**: A must be (BM, BK), B must be (BK, BN), C must be (BM, BN). The inner dimension (K) must match.
**Fix**: Check the shape passed to `T.alloc_shared` for the B buffer. Common mistake: using `(block_M, block_N)` instead of `(block_K, block_N)`.
```python
# Wrong
B_shared = T.alloc_shared((block_M, block_N), dtype)
# Right
B_shared = T.alloc_shared((block_K, block_N), dtype)
```

### `Check failed: lanes <= 4 (8 vs. 4): Ramp of more than 4 lanes is not allowed`
**Cause**: The compiler attempted to vectorize a memory access with 8 lanes, but the target only supports 4.
**Common trigger**: 1D T.copy with sizes that lead to wide vectorization.
**Fix**: Use power-of-2 tile sizes. If the issue is a 1D copy, try a 2D reshape.

### `ValueError` during lowering / `TVMError` in pass
**Cause**: The TileLang IR contains a construct that a lowering pass cannot handle.
**Fix**: Inspect which pass failed (error message includes pass name). Common issues:
- Non-constant loop bounds where constants are required
- Invalid T.copy source/dest scope combinations
- Mixing Level 2 and Level 3 constructs incorrectly

### nvcc compilation error
**Cause**: The generated CUDA code has syntax or type errors.
**Fix**: Inspect the generated CUDA:
```python
cuda_source = kernel.get_kernel_source()
print(cuda_source)
```
Or use the callback to intercept before compilation:
```python
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, target):
    with open("/tmp/generated.cu", "w") as f:
        f.write(code)
    return code
```

### `Failed to build prim_func from kernel`
**Cause**: The `@T.prim_func` function body has a Python-level error during tracing.
**Fix**: The error message includes a `source=` field showing the traced Python code. Look at the line numbers in the traceback to find which TileLang construct is wrong.

## Category B: Runtime Errors

### `Invalid TMA descriptor arguments: effective cuda globalStrides[0] must be a multiple of 16 bytes`
**Cause**: On Blackwell (sm_120) and Hopper (sm_90), TMA hardware requires 16-byte aligned strides.
**Rule**: For fp16 (2 bytes), the inner dimension must be a multiple of 8 (8 * 2 = 16). For fp32 (4 bytes), the inner dimension must be a multiple of 4.
**Fix**: Pad tensor dimensions to meet alignment, or use dynamic shapes that guarantee alignment. Note: the outer dimension (M in M x N) does NOT need alignment -- only the stride-bearing inner dimension.
```python
# For fp16 tensors, N must be a multiple of 8
N = ((raw_N + 7) // 8) * 8  # pad to multiple of 8
```

### `CUDA error: an illegal memory access was encountered`
**Cause**: Kernel accessed memory outside allocated bounds.
**Debug**:
1. Run with: `CUDA_LAUNCH_BLOCKING=1 python script.py`
2. Run with: `compute-sanitizer --tool memcheck python script.py`
3. Reduce problem to single tile (M=block_M, N=block_N)
4. Check index expressions: `by * block_M + local_idx < M`

### `CUDA error: too many resources requested for launch`
**Cause**: Block requires more registers or shared memory than available.
**Fix**: Reduce `threads`, tile sizes, or `num_stages`. Each pipeline stage doubles shared memory usage for the pipelined buffers.

### Kernel hangs / timeout
**Cause**: Infinite loop, deadlock, or very large iteration count.
**Fix**: Check `T.Pipelined(T.ceildiv(K, block_K))` -- make sure you're dividing by the tile size, not using `K` directly. Also check `T.serial` loop bounds.

## Category C: Wrong Results

### Output is all NaN or contains NaN
**Likely cause**: Missing `T.clear()` on accumulator fragment, or division by zero in epilogue.
**Fix**: Always `T.clear(accumulator)` before the first use. Check any division operations for zero-denominator cases.

### Output has garbage values (large random numbers)
**Likely cause**: Missing `T.clear()` or missing `T.copy` writeback.
**Distinguish**: Missing `T.clear` → partial garbage (some correct elements mixed with wrong ones, ~50% mismatch). Missing writeback → all garbage (100% mismatch, values are whatever was in global memory before).

### Output is all zeros
**Likely cause**: Kernel didn't write to output, or wrote to wrong offset, or `out_idx` mismatch.
**Fix**: Verify the final `T.copy(result, Output[offset])` offset matches your grid indexing.

### Output is correct except at tile boundaries
**Likely cause**: Off-by-one in tile indexing. For example, using `by * block_M` when the grid dimension was computed differently.
**Fix**: Ensure grid dims and offsets are consistent: if grid is `T.ceildiv(N, block_N)`, then offset should be `bx * block_N`.

### Output matches reference but is transposed
**Likely cause**: Grid dimensions (bx, by) are swapped relative to the T.copy offsets.
**Fix**: Convention: `T.Kernel(grid_x, grid_y)` where `bx` is the first output, `by` is the second. If your output is (M, N), use `C[by * block_M, bx * block_N]` with grid `(ceildiv(N, block_N), ceildiv(M, block_M))`.

## Category D: Numerical Drift

### `assert_close` fails but max error is small (< 0.1)
**Likely cause**: Normal floating-point precision differences. This is expected for fp16.
**Fix**: Use appropriate tolerances: `rtol=1e-2, atol=1e-2` for fp16 GEMM, `rtol=5e-2, atol=5e-2` for reductions with transcendentals.

### Results vary slightly between runs
**Likely cause**: Non-deterministic reduction order due to thread scheduling.
**Fix**: This is usually acceptable. If not, use `profiler.assert_consistent(repeat=10)` to verify the variation is within tolerance.

### `assert_close` passes with small inputs but fails with large inputs
**Likely cause**: Accumulation error grows with problem size. For fp16 GEMM with K=4096, individual products can accumulate significant rounding error.
**Fix**: Use `accum_dtype=T.float32` for the accumulator. If already using float32 accum, loosen tolerances slightly or use `max_mismatched_ratio`.
