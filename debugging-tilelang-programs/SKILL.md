---
name: debugging-tilelang-programs
description: >
  How to diagnose and fix errors in TileLang programs. Use this skill whenever a
  TileLang kernel fails to compile, crashes at runtime, produces incorrect results,
  or shows numerical mismatches. Also use when the user encounters TVM/TIR errors,
  CUDA codegen failures, shape mismatches, assertion failures, NaN/inf outputs,
  tilelang.compile errors, or needs to inspect generated CUDA code, use T.print
  for debugging, run AutoDD for minimal reproduction, or compare against a PyTorch
  reference. Trigger even for vague complaints like "my kernel doesn't work" or
  "wrong output" when TileLang is involved. Also trigger for TMA descriptor errors,
  illegal memory access, or kernel launch failures.
---

# Debugging TileLang Programs

## Step 1: Classify the Failure

Before debugging, identify which category the error falls into:

| Category | Symptoms | When it happens |
|----------|----------|-----------------|
| **A: Build/codegen** | Python traceback during `@tilelang.jit` or `tilelang.compile`. TVM/TIR errors, nvcc errors. | At compilation time (first call to the kernel function) |
| **B: Runtime crash** | CUDA errors, illegal memory access, TMA descriptor errors, kernel timeout | After compilation, when calling `kernel(a, b)` |
| **C: Wrong results** | Output is garbage, all zeros, NaN/inf, or completely wrong values. `assert_close` fails with large error. | Kernel runs without crash but output is wrong |
| **D: Numerical drift** | Output is close but not within tolerance. `assert_close` fails with small relative error. | Kernel runs and is approximately correct |

## Step 2: Reproduce

Always start by reproducing the error with the exact command:
```bash
python script.py
```

Capture the full traceback. If the error is intermittent, fix the random seed:
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

## Category A: Build/Codegen Failures

### Common Error Messages

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `T.gemm K shape check failed: K_A = X, K_B = Y` | B_shared has wrong shape | A must be (BM,BK), B must be (BK,BN), C must be (BM,BN) |
| `Check failed: lanes <= 4` | Unsupported vectorization (e.g., 1D copy with odd sizes) | Use power-of-2 tile sizes; check T.copy source shape |
| `Dimension mismatch` | Buffer shapes don't match in T.gemm or T.copy | Verify all shared/fragment dimensions are consistent |
| `ValueError` during lowering | Invalid TileLang construct | Check that loop bounds, buffer shapes are compile-time constants or valid T.dynamic |
| nvcc compilation error | Generated CUDA has issues | Inspect with `kernel.get_kernel_source()` or callback |

### Debugging Steps

1. **Read the error message carefully.** TileLang errors often contain the exact buffer names and dimensions involved.

2. **Inspect generated IR** by compiling explicitly:
   ```python
   kernel = tilelang.compile(func, target="cuda")
   src = kernel.get_kernel_source()
   print(src)
   ```

3. **Use the CUDA post-processing callback** to intercept generated code:
   ```python
   from tilelang.engine.callback import register_cuda_postproc_callback

   @register_cuda_postproc_callback
   def tilelang_callback_cuda_postproc(code, target):
       print(code)  # print full generated CUDA
       return code

   kernel = tilelang.compile(func, target="cuda")
   ```

4. **Minimize with AutoDD** (see below).

## Category B: Runtime Crashes

### Common Error Messages

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `Invalid TMA descriptor: globalStrides must be multiple of 16 bytes` | Non-aligned inner dimension on Blackwell/Hopper | For fp16: inner dims must be multiples of 8. For fp32: multiples of 4. |
| `illegal memory access` | Out-of-bounds read/write | Reduce problem size to tile size; check index math |
| `misaligned address` | Alignment violation | Check that tensor strides are contiguous |
| `CUDA launch failure` | Too many resources requested | Reduce threads, tile sizes, or num_stages (shared memory) |

### Debugging Steps

1. **Run with CUDA launch blocking** for synchronous errors:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python script.py
   ```

2. **Use compute-sanitizer** for memory errors:
   ```bash
   compute-sanitizer --tool memcheck python script.py
   ```
   **What to look for**: `Invalid __global__ read/write` errors include the thread and block IDs that caused the violation. If only the last block triggers the error, you likely have a boundary issue (tile size doesn't divide problem size). If all blocks trigger, it's a systematic indexing bug. See `references/interpreting-debug-output.md` §4 for details.

3. **Reduce problem size** to exactly one tile (M=block_M, N=block_N, K=block_K for GEMM). This eliminates multi-block interactions.

4. **Check shared memory usage**: `(block_M * block_K + block_K * block_N) * dtype_bytes * num_stages` must fit within GPU shared memory limits.

## Category C: Wrong Results

This is the most common and trickiest category. The kernel compiles and runs but produces incorrect output.

### Quick Checks (do these first)

1. **Missing `T.clear`?** Check that accumulators are zeroed before use:
   ```python
   C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
   T.clear(C_local)  # ← REQUIRED before accumulation
   ```
   Without this, the fragment contains uninitialized memory (garbage, NaN).

2. **Missing writeback?** Check that results are copied back to global memory:
   ```python
   T.copy(C_local, C[by * block_M, bx * block_N])  # ← REQUIRED at end
   ```
   Without this, output contains whatever was in the buffer before the kernel ran (NOT zeros).

3. **Shape mismatch in T.gemm?** A must be (BM,BK), B must be (BK,BN), C must be (BM,BN). This is the most common GEMM bug.

### Using T.print for Debugging

`T.print` prints from thread 0 of each block. It shows buffer contents element by element:

```python
with T.Kernel(...) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    T.copy(A[by * block_M, 0], A_shared)

    # Print the shared buffer contents
    T.print(A_shared, msg="A_shared after copy:")

    # Print a scalar value
    T.print(bx, msg="block index x:")
```

Output format:
```
msg='A_shared after copy:' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): buffer=A_shared, index=0, dtype=half_t value=1.234
```

**Interpreting T.print output**: Each line shows `index=N` which maps to buffer coordinates — for a 2D buffer (M, N), `index=k` → `buffer[k // N, k % N]` (row-major). Compare element-by-element against a PyTorch reference at the same small size. If all values are NaN → missing `T.clear`. If values match for the first tile but not subsequent ones → off-by-one in tile iteration. See `references/interpreting-debug-output.md` §1 for the full pattern interpretation table.

**Tips for effective T.print debugging:**
- Shrink to minimal sizes (e.g., M=N=K=block_M for GEMM) so output is manageable
- Print after each major step: after copy, after gemm, after epilogue
- Compare T.print output against the PyTorch reference computed at the same small size
- Use `TensorSupplyType.Integer` inputs for predictable values

### Checking Intermediate Values

For multi-step kernels, verify each step produces correct results:

```python
# Step 1: Verify copy works
T.copy(A[by * block_M, ko * block_K], A_shared)
T.print(A_shared, msg="after copy:")

# Step 2: Verify gemm works
T.gemm(A_shared, B_shared, C_local)
T.print(C_local, msg="after gemm:")

# Step 3: Verify epilogue works
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.sigmoid(C_local[i, j])
T.print(C_local, msg="after sigmoid:")
```

### Inspecting Generated CUDA

```python
cuda_source = kernel.get_kernel_source()
print(cuda_source)
```

Look for:
- Shared memory declarations and sizes
- Loop bounds (match your expected tile counts?)
- Index expressions (any off-by-one?)
- Memory access patterns (coalesced?)

## Category D: Numerical Drift

Results are close but `assert_close` fails with small relative error.

### Understanding Precision by Dtype

| Dtype | Decimal digits | Typical GEMM rtol | Typical reduction rtol |
|-------|---------------|-------------------|----------------------|
| float16 | ~3.3 | 1e-2 | 5e-2 |
| bfloat16 | ~2.4 | 2e-2 | 5e-2 |
| float32 | ~7.2 | 1e-5 | 1e-4 |

### Fixes

1. **Use float32 accumulation**: `accum_dtype=T.float32` in your kernel (most examples do this).

2. **Loosen tolerances appropriately**:
   ```python
   torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)  # fp16
   torch.testing.assert_close(c, ref_c, rtol=2e-2, atol=2e-2)  # bf16
   ```

3. **Use integer inputs for debugging**: Eliminates precision issues from large random values:
   ```python
   profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Integer)
   profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
   ```

4. **Check computation order**: Floating-point addition is not associative. Different tiling or reduction orders can produce slightly different results. This is expected behavior, not a bug.

5. **Use `max_mismatched_ratio`** for large tensors where a few elements may drift:
   ```python
   profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)
   ```

## AutoDD: Automatic Delta Debugging

When a kernel fails and the source is large, AutoDD automatically minimizes it to a minimal reproduction:

```bash
python -m tilelang.autodd script.py --err-msg "error substring" -o minimized.py -j 4 --timeout 60
```

| Parameter | Description |
|-----------|-------------|
| `source` | Path to the buggy Python file |
| `--err-msg` | Unique substring from the error output |
| `-o` | Output path for minimized file |
| `--backend` | `runner` (fast, default) or `subproc` (more stable) |
| `--timeout` | Seconds per trial (default 60) |
| `-j` | Parallel jobs (default 1, use 4+ for speed) |

**Tips:**
- Use a unique error substring (not generic like "Error")
- Increase `--timeout` for slow compilations
- Try `--backend subproc` if runner mode crashes
- The minimized output is a valid Python file you can run directly

Example: A 200-line buggy GEMM with wrong B_shared shape gets minimized to ~30 lines exposing the exact bug.

## TVM Logging

For deeper debugging of the compilation pipeline:

```bash
TVM_LOG_DEBUG=1 python script.py
```

This enables verbose TVM/TileLang logging showing each pass and transformation. For specific passes:
```bash
TVM_LOG_DEBUG=DEFAULT=0,target/codegen_cuda.cc=1 python script.py
```

## Race Condition Detection

If results vary between runs:
```python
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
profiler.assert_consistent(repeat=10)
```

This runs the kernel 10 times and checks all results are identical. If it fails, the kernel has a race condition (likely missing synchronization or incorrect atomic usage).

## Common Pitfalls Summary

| Rank | Pitfall | How it manifests | How to spot it |
|------|---------|-----------------|----------------|
| 1 | Missing `T.clear(accumulator)` | Garbage/NaN in output, 50%+ elements wrong | Check fragment allocations before accumulation loops |
| 2 | Wrong T.gemm buffer shapes | `AssertionError: T.gemm K shape check failed` | A=(BM,BK), B=(BK,BN), C=(BM,BN) |
| 3 | Missing T.copy writeback | Output has random values (not zeros) | Check fragment→global copy at kernel end |
| 4 | Wrong `out_idx` or arg count | `Kernel expected N inputs, but M provided` | out_idx=[-1] means last arg is output, pass N-1 args |
| 5 | TMA alignment (Blackwell/Hopper) | `globalStrides must be multiple of 16 bytes` | Inner dims must be 8-aligned for fp16 |
| 6 | Tile size exceeds shared memory | Compile error or silent perf loss | Check total shared memory: tiles * dtype * stages |
| 7 | Wrong loop bound in Pipelined | Hang or wrong results | Use `T.ceildiv(K, block_K)` not `K` |
| 8 | Indexing off-by-one | Partial corruption at tile boundaries | Reduce to single tile, compare element by element |

## Escalation

- If the bug appears to be in TileLang itself (not user code), minimize with AutoDD and file a GitHub issue at https://github.com/tile-ai/tilelang
- For gradient/backward bugs, use the **testing-fwd-bwd-kernels** skill
- For performance issues (not bugs), use the **profiling-tilelang-programs** skill

For a comprehensive error catalog with more examples, read `references/error-catalog.md`.
For interpreting T.print output, compute-sanitizer errors, and AutoDD results, read `references/interpreting-debug-output.md`.
For the full debug tools documentation (T.print, AutoDD, callbacks), read `references/debug-tools-doc.md`.
For performance analysis with ncu/nsys (not debugging), use the **profiling-tilelang-programs** skill.
