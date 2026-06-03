# Interpreting Debug Output

How to read and act on output from TileLang debugging tools.

## Table of Contents

1. [T.print Output](#1-tprint-output)
2. [compute-sanitizer Output](#2-compute-sanitizer-output)
3. [AutoDD Minimized Output](#3-autodd-minimized-output)

For performance profiling output (ncu reports, nsys timelines), see `profiling-tilelang-programs/references/ncu-bottleneck-guide.md`.

---

## 1. T.print Output

### Output Format

Each `T.print(buffer, msg='label:')` call produces one line per element, from thread 0 of each block:

```
msg='label:' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): buffer=A_shared, index=0, dtype=half_t value=1.234
msg='label:' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): buffer=A_shared, index=1, dtype=half_t value=5.678
```

### Mapping index to Tensor Coordinates

For a 1D buffer of size N, `index=i` maps directly to element `buffer[i]`.

For a 2D buffer of shape (M, N), `index=k` maps to `buffer[k // N, k % N]` (row-major). For example, with a (4, 8) buffer:
- `index=0` → `[0, 0]`
- `index=7` → `[0, 7]`
- `index=8` → `[1, 0]`
- `index=15` → `[1, 7]`

### Comparison Workflow

1. Shrink to minimal size (e.g., M=N=K=block_M for GEMM) so output is manageable
2. Use `TensorSupplyType.Integer` for predictable values
3. Compute reference in PyTorch at the same small size
4. Print the reference tensor: `print(ref_tensor)`
5. Compare element-by-element against T.print output

### Pattern Interpretation

| What you see | Likely cause |
|-------------|-------------|
| All values are NaN | Missing `T.clear()` on accumulator fragment |
| Mix of correct values and NaN/garbage | Missing `T.clear()` — some elements happen to be finite from uninitialized memory |
| All values are large random numbers | Missing `T.copy` writeback — reading uninitialized global memory |
| Values match reference for first tile but not subsequent tiles | Off-by-one in tile iteration or indexing |
| Values are transposed vs reference | Grid dimensions (bx, by) swapped relative to T.copy offsets |
| Values correct except a few scattered elements | Possible race condition or boundary issue |

## 2. compute-sanitizer Output

### Running

```bash
compute-sanitizer --tool memcheck python script.py
```

### Reading Error Messages

Typical output:

```
========= Invalid __global__ read of size 8 at 0x... by thread (31, 0, 0) in block (3, 2, 0)
=========     Address 0x... is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
```

### Interpreting Errors

| Error | Meaning | Common cause |
|-------|---------|-------------|
| `Invalid __global__ read of size N` | Read from address outside allocated buffer | Index calculation goes past tensor bounds |
| `Invalid __global__ write of size N` | Write to address outside allocated buffer | T.copy destination offset wrong |
| `Misaligned address` | Access not aligned to element size | Non-contiguous tensor passed to kernel |
| `Address is out of bounds` | Address completely outside any allocation | Likely large indexing error (wrong buffer or dimension) |

### Mapping to Kernel Code

The error includes `thread (tx, ty, tz) in block (bx, by, bz)`. Map these back to your kernel:

1. `block (bx, by, bz)` corresponds to `T.Kernel(grid_x, grid_y, grid_z)` — identifies which tile
2. The thread indices help narrow down which element within the tile caused the error
3. Reduce to single tile (M=block_M, N=block_N) to eliminate multi-block interactions

### Common Patterns

- Error in last block only → tile size doesn't divide problem size, boundary not handled
- Error in all blocks → systematic indexing bug
- Error only with large problem sizes → off-by-one that only triggers at boundaries

## 3. AutoDD Minimized Output

### What AutoDD Produces

AutoDD takes a failing script and systematically removes code until it finds the minimal reproduction. The output is a valid Python file that still triggers the same error.

```bash
python -m tilelang.autodd script.py --err-msg "T.gemm K shape check" -o minimized.py -j 4
```

### Using the Minimized Output

1. **Run it** to confirm it still fails: `python minimized.py`
2. **Read the kernel** — it will be much shorter than the original (often 20-40 lines)
3. **Identify the bug** — in a minimal kernel, the problematic construct is usually obvious:
   - Wrong buffer shape passed to `T.gemm`
   - Missing `T.clear` or `T.copy`
   - Wrong loop bounds
4. **Fix in the minimal version first**, verify it works, then apply the fix to your original code

### Tips

- Use a specific error substring (`--err-msg`), not a generic one like "Error"
- If AutoDD itself crashes, try `--backend subproc` for more isolation
- Increase `--timeout` for kernels that take longer to compile
- Use `-j 4` or higher for faster minimization (runs trials in parallel)
