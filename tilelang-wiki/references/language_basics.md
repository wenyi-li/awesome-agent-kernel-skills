# Language Basics TLDR

This page is a quick usage cheatsheet for writing ordinary TileLang kernels.
Detailed API behavior lives under `tilelang_language/`; use this page to pick
the right primitive and then open the matching basic or advanced API page when
you need exact semantics.

Conventional imports:

```python
import tilelang
import tilelang.language as T
```

## Kernel Shape

Most examples use either a direct `@tilelang.jit` function or a JIT factory that
returns a nested `@T.prim_func`.

```python
@tilelang.jit
def add(A, B, block_M: int, block_N: int, dtype=T.float32):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), dtype)
    B: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            row = by * block_M + i
            col = bx * block_N + j
            C[row, col] = A[row, col] + B[row, col]

    return C
```

Use `T.Tensor(...)` for input/output annotations, `T.empty(...)` for returned
outputs, and `T.const("M, N")` for shape symbols inferred from concrete input
tensors. Use `T.dynamic("m")` only when a dimension must remain dynamic in the
generated kernel. Dtypes are usually `T.float16`, `T.float32`, or string forms
such as `"float16"` and `"float32"`.

## Launch And Work Mapping

`T.Kernel(...)` defines the GPU launch region. Positional extents map to block
indices; `threads=` chooses the block thread shape.

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

Use `T.Parallel(...)` for tile-local elementwise work and `T.Pipelined(...)`
for repeated copy/compute loops. Ordinary `T.serial`, `T.grid`, `T.unroll`, and
`T.vectorized` loops are available when you need scalar or explicit loop forms.

```python
for i, j in T.Parallel(block_M, block_N):
    ...

for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    ...
```

## Memory And Movement

Typical kernels move data through global, shared, and fragment/register storage.

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_frag = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((4,), dtype)
scale = T.alloc_var("float32", init=1.0)
```

Use `T.copy(src, dst)` as the default data movement primitive. It covers common
global-to-shared, shared-to-fragment, fragment-to-shared, and fragment-to-global
copies. Use explicit async copy, TMA, cluster copy, or gather/scatter variants
only when you are deliberately managing a lower-level path.

```python
T.copy(A[by * block_M, k * block_K], A_shared)
T.copy(B[k * block_K, bx * block_N], B_shared)
T.copy(C_frag, C[by * block_M, bx * block_N])
```

Initialize temporary buffers before use:

```python
T.clear(C_frag)
T.fill(scores_max, -T.infinity("float32"))
```

## Compute

Start from tile operators and reductions before reaching for target-specific
instructions.

```python
T.gemm(A_shared, B_shared, C_frag)
T.reduce_max(scores, scores_max, dim=1)
T.reduce_sum(scores, scores_sum, dim=1)
T.exp2(x)
T.max(a, b)
T.if_then_else(cond, true_value, false_value)
```

Use Python `if`/`else` for control flow. Use `T.if_then_else(...)` when the
conditional must produce a value inside an expression, such as a causal mask.

## Minimal Tiled GEMM Pattern

This is the basic tiled copy/compute/writeback structure used by many examples.

```python
@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_frag)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_frag)

        T.copy(C_frag, C[by * block_M, bx * block_N])

    return C
```

Use this as the baseline for tiled matrix work. Add layout annotations,
swizzling, async/TMA copies, warpgroup instructions, or autotuning only when
the simple tile-level version is correct and profiling shows a need.

## Put Together: FlashAttention 2 Forward

The bundled FlashAttention 2 example is a good "put together" demo because it
uses the same basic primitives from this page in a real fused kernel:
`@tilelang.jit`, `@T.prim_func`, `T.Kernel`, shared-memory staging,
fragment accumulators, `T.copy`, `T.Pipelined`, `T.gemm`, reductions, math
intrinsics, and value-producing masks.

This is the core kernel from `examples/flash_attention/example_mha_fwd_bshd.py`,
with comments added around the TileLang language pieces:

```python
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    # Host-side Python runs while building the kernel specialization.
    # The scale uses exp2 below, so multiply by log2(e).
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        # Grid axes: query block, attention head, batch item.
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            # Shared memory stages global Q/K/V tiles for tile GEMM.
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            # Fragment/register state stays live across the K/V tile loop.
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            # Load the Q tile once. K and V are streamed through the loop.
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Causal attention only visits K/V blocks up to the current Q block.
            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # Initialize the score tile. `T.if_then_else` produces a value
                # in the expression, which is useful for causal and boundary masks.
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype),
                            0,
                        )

                # Q @ K^T adds into the pre-initialized score tile.
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Online softmax update:
                # 1. keep the previous row max,
                # 2. reduce the new row max,
                # 3. rescale the previous output accumulator,
                # 4. exponentiate and sum the current score tile.
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                # Convert scores to the input dtype for the second GEMM.
                T.copy(acc_s, acc_s_cast)

                # Rescale the old output accumulator before adding P @ V.
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Normalize by the final row sum and write the output tile.
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main
```

The main reading path is:

- `out_idx=[3]` marks the fourth argument, `Output`, as the output buffer.
- The outer Python function is a specialization factory; its Python values
  choose shapes, block sizes, pipeline stages, and pass configs.
- `@T.prim_func` defines the device kernel signature and body.
- `T.Kernel(...)` maps work to query blocks, heads, and batch items.
- `T.copy(...)` moves tiles between global, shared, and fragment storage.
- `T.Pipelined(...)` streams K/V tiles while keeping online-softmax state in
  fragments.
- `T.gemm(...)` expresses both `Q @ K^T` and `softmax(QK^T) @ V`.
- `T.reduce_max(...)`, `T.reduce_sum(...)`, `T.exp2(...)`, and fragment
  assignments implement the numerically stable online softmax.

The important lesson is that a complete fused attention kernel is still built
from the same basic language pieces as the minimal GEMM pattern; it just keeps
more fragment state alive across the pipelined loop.

## Where To Go Next

- `tilelang_language/loop/`: loop helpers and pipeline loops.
- `tilelang_language/allocate/`: tensors, shared/local/fragment storage, and
  scalar variables.
- `tilelang_language/copy_op/`: `T.copy` and lower-level movement operations.
- `tilelang_language/gemm_op/`: GEMM operators.
- `tilelang_language/basic_operations/`: tensor annotations, fill/clear,
  elementwise helpers, and math.
- `tilelang_language/reduce_op/`: reductions.
- `tilelang_language/kernel_warpgroup_cluster_builtins/`: launch frames,
  thread/block bindings, and low-level hardware builtins.
- `tilelang_language/annotations/`: layout, swizzle, and compile annotations.
- `tilelang_language/misc/`: atomics, debug helpers, dynamic symbols, and raw
  TIR helpers.
