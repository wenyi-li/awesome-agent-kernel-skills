# Canonical TileLang Examples

These are canonical TileLang examples integrated directly from the TileLang repository.
They serve as authoritative reference implementations for common GPU kernel patterns
including GEMM, elementwise operations, reductions, and attention mechanisms.

---

## 1. GEMM (example_gemm.py)

Minimal tiled GEMM with shared memory and pipelining.

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")


if __name__ == "__main__":
    main()
```

---

## 2. GEMM + ReLU Fusion (quickstart.py)

GEMM with fused ReLU epilogue, demonstrates `T.Parallel` for elementwise operations.

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # relu
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M = 1024
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

matmul_relu_kernel = matmul(M, N, K, block_M, block_N, block_K)
import torch

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)

matmul_relu_kernel(a, b, c)

ref_c = torch.relu(a @ b)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

profiler = matmul_relu_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
```

---

## 3. Elementwise Add (example_elementwise_add.py)

2D elementwise operation with shared memory staging.

```python
import torch
import tilelang
import tilelang.language as T


def ref_program(x, y):
    return x + y


@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.prim_func
    def elem_add(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add


def main(M=1024, N=1024):
    a = torch.randn(M, N, dtype=torch.float32, device="cuda")
    b = torch.randn(M, N, dtype=torch.float32, device="cuda")

    kernel = elementwise_add(M, N, block_M=32, block_N=32, threads=128, in_dtype=T.float32, out_dtype=T.float32)

    out = kernel(a, b)
    torch.testing.assert_close(out, ref_program(a, b), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
```

---

## 4. RMS Norm (rms_norm.py)

Reduction pattern with two-pass normalization.

```python
import torch
import tilelang
import tilelang.language as T


def rms_norm_splitk(M, N, blk_m, blk_k):
    dtype = T.float

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

            for k in range(num_k_step):
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


@tilelang.jit(out_idx=[-1])
def rms_norm(M, N, blk_m):
    dtype = T.float

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return main


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)


if __name__ == "__main__":
    M, N, blk_m, blk_k = 8192, 8192, 1, 512
    kernel = rms_norm(M, N, blk_m)
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
```

---

## 5. Online Softmax (online_softmax.py)

Online softmax with running max/sum, demonstrates `T.Pipelined` with non-GEMM compute.

```python
import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench


@tl.jit(out_idx=[1])
def softmax_kernel(
    M,
    N,
    dtype: T.dtype = T.float16,
):
    BN = min(tl.next_power_of_2(N), 8192)
    NN = tl.cdiv(N, BN)

    accum_dtype = T.float32

    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),
        Y: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                T.reduce_max(x, max_x, dim=0, clear=True)

                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)

                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)

                lse[0] = max_x[0] * scale + T.log2(T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])

                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main


M = 8192
N = 8192
kernel = softmax_kernel(M, N)
dtype = torch.float16
X = torch.randn(M, N, dtype=dtype, device="cuda")
Y = kernel(X)
Y_ref = X.softmax(dim=1)

torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)

t1 = do_bench(lambda: X.softmax(dim=1), warmup=25, rep=100)
t2 = do_bench(lambda: kernel(X), warmup=25, rep=100)
print(f"torch latency: {t1:.3f} ms")
print(f"TileLang latency: {t2:.3f} ms")
print(f"Speedup: {t1 / t2:.3f}x")
```

---

## 6. Flash Attention Forward (extracted pattern)

Multi-head attention forward with online softmax (from `example_mha_fwd_bhsd.py`).

The kernel function signature and decorator:

```python
@tilelang.jit(out_idx=[3], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    # ... kernel body follows
```

Key patterns used in the kernel body:

- **Q@K^T via GEMM with transpose:** `T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)` computes attention scores as a matrix multiply with transposed K.

- **Online softmax with running max and rescaling:**
  ```python
  T.reduce_max(acc_s, scores_max_prev, dim=1)
  # compute scores_scale = exp2(old_max - new_max) for rescaling
  for i, j in T.Parallel(block_M, dim):
      acc_o[i, j] *= scores_scale[i]
  ```

- **Mixed-precision softmax weights for V gemm:** The softmax output `acc_s` (in float32 accumulator) is cast to an intermediate `acc_s_cast` fragment in the input dtype before being used in `T.gemm(acc_s_cast, V_shared, acc_o)`.

- **Causal masking via `T.if_then_else`:**
  ```python
  for i, j in T.Parallel(block_M, block_N):
      acc_s[i, j] = T.if_then_else(
          by * block_M + i >= bx * block_N + j,
          acc_s[i, j],
          -T.infinity(accum_dtype),
      )
  ```

- **Final normalization:** After all K/V blocks are processed, the output accumulator is divided by the running sum of exponents to produce correctly normalized attention output.

Note: The full example also includes a `torch.autograd.Function` wrapper (`_attention` class) that calls the TileLang kernel in its `forward` method and can be used as a drop-in replacement for standard attention in PyTorch models.

---

## 7. Linear Attention Forward (extracted pattern)

Chunked linear attention with running state accumulation (from the linear attention example, ~157 lines).

Key patterns used in this kernel:

- **Running state accumulation across chunks:** Instead of softmax, linear attention maintains a running KV state matrix that accumulates across chunks of the sequence. Each chunk updates the state and uses it to compute the output.

- **`T.serial` loop for K-dimension chunks:** The K-dimension is iterated with `T.serial` to enforce sequential processing order, which is required because the running state depends on all previous chunks.

- **No softmax -- direct Q*K*V pattern:** Linear attention replaces the softmax(Q@K^T)@V computation with a direct linear combination. The kernel computes feature-mapped Q and K, then accumulates K^T@V into a running state, and finally multiplies Q by the accumulated state to produce output.

- **State management pattern:**
  ```python
  # Accumulate K^T @ V into running state
  T.gemm(K_shared, V_shared, state, transpose_A=True)
  # Compute output as Q @ state
  T.gemm(Q_shared, state, output)
  ```

- **Chunk-level parallelism:** The batch and head dimensions are parallelized across thread blocks, while the sequence dimension is processed chunk-by-chunk within each block to maintain the sequential state dependency.
