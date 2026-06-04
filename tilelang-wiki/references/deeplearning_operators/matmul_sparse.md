# Sparse Matrix-Matrix Multiplication with Tile Library

*Author: [botbw](https://github.com/botbw)*

:::{warning}
   This document is still **experimental** and may be incomplete.

   This feature is still **experimental** and need further optimization.

   Suggestions and improvements are highly encouraged—please submit a PR!
:::

:::{tip}
It's suggested to go through `matmul.md` first.

The current working example can be found at `../../examples/gemm_sp/example_gemm_sp.py`.
:::

## Structured sparsity in the NVIDIA Ampere architecture

Since the Ampere architecture (sm80 and above), sparsity support has been integrated into Tensor Cores. This allows a 2:4 (or 1:2 for 32-bit data types) semi-structured matrix to be compressed into its non-zero values along with associated metadata, which can then be fed into the Tensor Core. This enables up to **2x throughput** compared to the equivalent dense computation.

:::{warning}
   This tutorial primarily focuses on CUDA, as this feature is not yet supported on ROCm. However, AMD provides a similar capability in the matrix cores of GPUs such as the MI300X.
:::

The sparse MMA storage figure from the original blog post is omitted here
because the referenced `_static` asset is not included in this self-contained
skill package.

## Compress a dense tensor

To utilize sparse Tensor Cores, a dense tensor must first be **compressed** into its non-zero values along with the corresponding metadata.

Both `PyTorch` and `vLLM` use `CUTLASS` as their computation backend (see references [here](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredOps.cu#L47) and [here](https://github.com/vllm-project/vllm/blob/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/csrc/sparse/cutlass/sparse_scaled_mm_c3x.cuh#L116)), leveraging `CUTLASS`’s built-in compressor (or reimplementing it in `PyTorch`).

A compressor is provided in `tilelang.utils.sparse`. Pass in a dense 2:4-sparse tensor and optionally a metadata dtype to get back the compressed values and metadata. The current example uses an explicit metadata dtype so that the compressor and kernel agree on layout:

```python
from tilelang.utils.sparse import compress
A_sparse, E = compress(A, meta_dtype=e_dtype.as_torch())
```

Here, `A_sparse` contains all the non-zero elements of `A`, while `E` stores the corresponding metadata (indexing information) required to reconstruct the original sparse pattern. The metadata uses a natural row-major layout that `T.gemm_sp` consumes directly — no additional layout annotation is needed.

## `T.gemm_sp`

A 2:4 sparse GEMM kernel is similar to its dense counterpart, except that it also requires loading the metadata into shared memory and passing it to `T.gemm_sp`.

The default metadata dtype for fp16/bf16 is `int16` with an E-factor of 16 (one `int16` value covers 16 K-elements). For int8/float8 the default is `int32` with E-factor 32.

```python
import tilelang
import tilelang.language as T
from tilelang.utils.sparse import get_e_factor


@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16(M, N, K, accum_dtype, e_dtype, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasterization):
    e_factor = get_e_factor(T.float16, e_dtype)

    @T.prim_func
    def gemm_sp_fp16(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), e_dtype),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.disable_warp_group_reg_alloc()
            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local,
                          transpose_A=False, transpose_E=False, transpose_B=False,
                          policy=policy)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_sp_fp16
```

For the surrounding host-side setup, correctness check, and benchmarking flow,
see `../../examples/gemm_sp/example_gemm_sp.py`.

For the maintained background on the underlying primitives, also see
[Language Basics](../language_basics.md),
[GEMM APIs](../tilelang_language/gemm_op/basic.md), and
[loop APIs](../tilelang_language/loop/basic.md).
