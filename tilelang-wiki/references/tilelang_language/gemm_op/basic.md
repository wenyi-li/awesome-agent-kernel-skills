# GEMM Operations: Basic

Use `T.gemm` for tile-level matrix multiply-accumulate. The common pattern is:
stage input tiles into shared memory, accumulate into a fragment buffer, then
copy the accumulator to the output.

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), T.float32)

T.clear(C_local)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)

T.copy(C_local, C[by * block_M, bx * block_N])
```

## Dense GEMM

```python
T.gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    k_pack=1,
    mbar=None,
)
```

`C` is the accumulator tile and must be rank 2 with shape `(M, N)`. `A` and
`B` provide the `M x K` and `K x N` input tiles. Most examples use shared-memory
operands for `A` and `B`, plus a `local.fragment` accumulator for `C`.

Use `transpose_B=True` when the staged `B` tile is laid out as `(block_N,
block_K)`:

```python
B_shared = T.alloc_shared((block_N, block_K), dtype)
T.copy(B[bx * block_N, k * block_K], B_shared)
T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

`policy` controls how work is partitioned across warps. `Square` is the default
for ordinary matmul. Attention kernels commonly use `FullRow`:

```python
T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
T.gemm(scores_cast, V_shared, out_accum, policy=T.GemmWarpPolicy.FullRow)
```

`clear_accum=True` asks the GEMM op to initialize the accumulator as part of the
operation. The usual example style is clearer: call `T.clear(C_local)` once
before the K loop and keep `clear_accum=False` inside the loop.

## Sparse GEMM

Use `T.gemm_sp` for the compressed 2:4 sparse GEMM path:

```python
T.gemm_sp(
    A_sparse,
    E,
    B,
    C,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    k_pack=1,
)
```

`A_sparse` stores the compressed nonzero values, `E` stores the sparsity
metadata, `B` is dense, and `C` is the rank-2 accumulator. The logical K
dimension is twice the compressed K dimension of `A_sparse`.

```python
A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
B_shared = T.alloc_shared((block_K, block_N), T.float16)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.clear(C_local)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
    T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm_sp(A_shared, E_shared, B_shared, C_local, policy=policy)
```

Keep sparse GEMM tiles and metadata shapes matched to the example family you are
using; the metadata packing is part of the sparse operand format, not a generic
dense tensor.
