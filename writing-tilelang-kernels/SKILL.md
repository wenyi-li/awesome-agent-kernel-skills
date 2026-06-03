---
name: writing-tilelang-kernels
description: >
  How to write TileLang GPU kernels from scratch or by adapting existing patterns.
  Use this skill whenever the user wants to create a new TileLang kernel, implement
  a GPU operator in TileLang, write a prim_func, set up tiling/blocking, define
  shared memory and fragment allocations, use T.gemm or T.copy, create a JIT-compiled
  kernel, or port an algorithm (GEMM, elementwise, reduction, attention, normalization,
  softmax, convolution) to TileLang. Also use when the user asks about tilelang.jit,
  T.prim_func, T.Kernel, T.Tensor, T.alloc_shared, T.alloc_fragment, T.Pipelined,
  T.Parallel, or any TileLang DSL construct for kernel authoring. Even for questions
  like "how do I start with TileLang" or "show me a basic TileLang example", this
  skill provides the right templates and workflow.
---

# Writing TileLang Kernels

## Pre-flight Checklist

Before writing any kernel:

1. **Verify the environment**: `python -c "import tilelang; print(tilelang.__version__)"`
2. **Identify the target operation**: What are the input/output shapes, dtypes, and the mathematical operation?
3. **Write a PyTorch reference first**: You will need this for correctness validation.
4. **Find the closest pattern** — read `references/tilelang-examples.md` for complete working examples:

| Pattern | What to look for |
|---------|-----------------|
| GEMM | T.gemm, shared memory, pipelining |
| GEMM + fusion | ReLU/sigmoid epilogue after GEMM |
| Elementwise | T.Parallel, shared memory staging |
| Reduction/Norm | T.reduce_sum, two-pass pattern |
| Online Softmax | Running max, exp, rescaling |
| Flash Attention | Multi-fragment, softmax in loop |
| Linear Attention | Chunked, running state |

For the complete TileLang language reference (primitives, control flow, instructions), read `references/language-docs.md`.

## Kernel Anatomy

Every TileLang kernel has a two-level structure:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])  # last arg is the output, returned by kernel call
def my_kernel(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    # Outer function: compile-time parameters (problem sizes, tile sizes, dtypes)

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Inner function: the actual GPU kernel
        # T.Tensor declares buffer shapes and dtypes

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Grid: (blocks_x, blocks_y), threads per block
            # bx, by are block indices (like blockIdx.x, blockIdx.y)

            # 1. Allocate on-chip memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)      # shared memory (block-visible)
            B_shared = T.alloc_shared((block_K, block_N), dtype)      # shared memory
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # registers (per-thread)

            # 2. Initialize
            T.clear(C_local)

            # 3. Compute (pipelined loop)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)  # global -> shared
                T.copy(B[ko * block_K, bx * block_N], B_shared)  # global -> shared
                T.gemm(A_shared, B_shared, C_local)               # shared -> fragment

            # 4. Write back
            T.copy(C_local, C[by * block_M, bx * block_N])       # fragment -> global

    return kernel  # must return the inner function
```

### Key Concepts

**Memory scopes** (3 tiers):
- `T.alloc_shared(shape, dtype)` -- on-chip SRAM, visible to all threads in a block. Use for staging tiles from global memory.
- `T.alloc_fragment(shape, dtype)` -- registers. Despite looking like full tile shapes, the compiler distributes across threads via layout inference. Use for accumulators and intermediate results.
- `T.alloc_local(shape, dtype)` -- explicitly thread-private. Use for scalar accumulators.

**Data movement**: `T.copy(src, dst)` handles global<->shared<->fragment transfers. Extents are inferred from the destination buffer shape.

**`@tilelang.jit` arguments**:
- `out_idx=[-1]` -- the last parameter is the output; kernel call returns it. `out_idx=[2, 3]` means params 2 and 3 are outputs.
- Without `out_idx` -- caller must pre-allocate and pass all buffers; kernel returns None.
- `target="cuda"` (optional) -- inferred automatically from input tensors if omitted.

**`T.Kernel` grid**:
- `T.Kernel(grid_x, grid_y, threads=N)` -- 2D grid. `T.Kernel(grid_x, grid_y, grid_z, threads=N)` for 3D.
- `threads` sets threads per block (typically 128 or 256).
- Use `T.ceildiv(dim, block_dim)` for grid dimensions.

## Step-by-Step Workflow

1. **Write a PyTorch reference**:
   ```python
   def ref_program(A, B):
       return A @ B
   ```

2. **Choose the kernel pattern**: elementwise, reduction, GEMM, or fused.

3. **Pick tile sizes** (start with these defaults):
   - GEMM: `block_M=128, block_N=128, block_K=32, num_stages=2, threads=128`
   - Elementwise: `block_M=64, block_N=64, threads=128`
   - Reduction: `block_M=64, block_N=128, threads=128` (reduce along N)

4. **Write the kernel** following the appropriate template (see below or read `references/kernel-templates.md` for full versions).

5. **Compile and run**:
   ```python
   kernel = my_kernel(M, N, K, block_M, block_N, block_K)
   c = kernel(a, b)  # with out_idx=[-1]
   # OR
   kernel(a, b, c)   # without out_idx
   ```

6. **Validate**:
   ```python
   ref_c = ref_program(a, b)
   torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
   ```

7. **Inspect generated CUDA** (optional):
   ```python
   print(kernel.get_kernel_source())
   ```

## Quick Templates

### Elementwise (2D)

```python
@tilelang.jit(out_idx=[-1])
def softplus(M, N, block_M, block_N, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.copy(X[by * block_M, bx * block_N], X_shared)
            for i, j in T.Parallel(block_M, block_N):
                val = T.cast(X_shared[i, j], accum_dtype)
                Y_local[i, j] = T.log(T.cast(1.0, accum_dtype) + T.exp(val))
            T.copy(Y_local, Y[by * block_M, bx * block_N])
    return kernel
```

### GEMM (Minimal)

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

For more templates (1D elementwise, row reduction, dynamic shape GEMM) with complete imports, host code, and validation, read `references/kernel-templates.md`. For full working examples, read `references/tilelang-examples.md`.

## Key API Patterns

- `T.ceildiv(a, b)` -- ceiling division for grid dimensions
- `T.clear(buf)` / `T.fill(buf, value)` -- initialize buffers
- `T.reduce_sum(src, dst, dim=N)`, `T.reduce_max(...)`, `T.reduce_min(...)` -- tile-level reductions
- `T.exp`, `T.log`, `T.exp2`, `T.log2`, `T.rsqrt`, `T.sigmoid`, `T.max`, `T.min` -- elementwise math
- `T.cast(value, dtype)` -- explicit type casting
- `T.if_then_else(cond, true_val, false_val)` -- conditional expression (for masking)
- `T.atomic_add(dst, value)` -- atomic addition to global memory
- `T.infinity(dtype)` -- typed infinity constant (use `-T.infinity(dtype)` for init)
- `T.gemm(A_shared, B_shared, C_fragment, transpose_A=False, transpose_B=False)` -- tensor core GEMM
- `T.Pipelined(iters, num_stages=N)` -- software pipelining (overlaps copy + compute)
- `T.Parallel(dim0, dim1)` -- parallel loops mapped to threads
- `T.serial(start, stop)` -- sequential loop
- `T.use_swizzle(panel_size=10, enable=True)` -- L2 cache locality optimization

For the complete instruction reference, read `references/api-quick-reference.md`.

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Garbage/NaN in output | Missing `T.clear(C_local)` before accumulation | Add `T.clear(C_local)` before the pipelined loop |
| `T.gemm K shape check failed: K_A = X, K_B = Y` | Shape mismatch in T.gemm args | A must be (BM,BK), B must be (BK,BN), C must be (BM,BN) |
| Output has random-looking values (not zeros) | Missing `T.copy(result, Output[...])` writeback | Add the final T.copy from fragment to global memory |
| `Kernel expected N inputs, but M are provided` | Wrong arg count due to `out_idx` | With `out_idx=[-1]`: pass N-1 args, kernel returns output. Without: pass all N args. |
| `Invalid TMA descriptor: globalStrides must be multiple of 16 bytes` | Non-aligned inner dimension on Blackwell/Hopper | For fp16: inner dims must be multiples of 8. For fp32: multiples of 4. |
| Kernel compiles but hangs | Infinite loop or deadlock | Check `T.Pipelined` iteration count is correct: `T.ceildiv(K, block_K)` not `K` |

## Escalation

- Kernel compiles but produces wrong results → use the **debugging-tilelang-programs** skill
- Kernel is correct but slow → use the **profiling-tilelang-programs** skill, then **optimizing-tilelang-programs**
- Need forward + backward passes → use the **testing-fwd-bwd-kernels** skill
