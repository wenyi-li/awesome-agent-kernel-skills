# Forward+Backward Kernel Examples

Canonical examples from the TileLang repository showing complete forward+backward kernel implementations.

## 1. Flash Attention Forward+Backward (Full Source)

The most instructive reference for fwd+bwd TileLang kernels. Contains all five phases: forward, backward preprocess (Delta), backward main, backward postprocess, and torch.autograd.Function wrapper.

Source: TileLang repository `examples/flash_attention/example_mha_bwd_bhsd.py`

```python
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse


@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


def make_dq_layout(dQ):
    return T.Layout(dQ.shape, lambda b, h, l, d: [b, h, l // 8, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),
        dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, by, bx * blk : (bx + 1) * blk, :],
                dQ_out[bz, by, bx * blk : (bx + 1) * blk, :],
            )

    return flash_bwd_post


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),
        dK: T.Tensor(shape, dtype),
        dV: T.Tensor(shape, dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_M, dim], dtype)
            dk_shared = T.alloc_shared([block_M, dim], dtype)

            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                }
            )
            T.copy(K[bz, bx, by * block_M : (by + 1) * block_M, :], K_shared)
            T.copy(V[bz, bx, by * block_M : (by + 1) * block_M, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                T.copy(Q[bz, bx, k * block_N : (k + 1) * block_N, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j], 0)
                T.copy(dO[bz, bx, k * block_N : (k + 1) * block_N, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, bx, k * block_N + i, j], dq[i, j])
            T.copy(dv, dv_shared)
            T.copy(dk, dk_shared)
            T.copy(dv_shared, dV[bz, bx, by * block_M : (by + 1) * block_M, :])
            T.copy(dk_shared, dK[bz, bx, by * block_M : (by + 1) * block_M, :])

    return flash_bwd


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        BATCH, H, N_CTX, D_HEAD = q.shape
        block_M = 64
        block_N = 64 if D_HEAD <= 128 else 32
        o, lse = flashattn_fwd(BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N)(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, H, N_CTX, D_HEAD = q.shape

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 64
        block_N = 64 if D_HEAD <= 64 else 32
        kernel_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)
        kernel_post = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)
        delta = kernel_prep(o, do)
        kernel = flashattn_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M, block_N)
        shape = [BATCH, H, N_CTX, D_HEAD]
        dq = torch.zeros(shape, dtype=torch.float32, device=q.device)
        dk = torch.empty(shape, dtype=torch.float16, device=q.device)
        dv = torch.empty(shape, dtype=torch.float16, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = kernel_post(dq)
        return dq, dk, dv, None


attention = _attention.apply


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main(
    BATCH: int = 8,
    H: int = 32,
    N_CTX: int = 1024,
    D_HEAD: int = 64,
    causal: bool = False,
):
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 5 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    Q = torch.empty(BATCH, H, N_CTX, D_HEAD, dtype=torch.half, device="cuda").normal_().requires_grad_()
    K = torch.empty_like(Q).normal_().requires_grad_()
    V = torch.empty_like(Q).normal_().requires_grad_()
    dO = torch.randn_like(Q)
    O = attention(Q, K, V, causal)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    O_ref = ref_program(Q, K, V, causal)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2)

    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--h", type=int, default=32, help="Number of heads")
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--d_head", type=int, default=64, help="Head dimension")
    parser.add_argument("--causal", type=bool, default=False, help="Causal flag")
    args = parser.parse_args()
    main(args.batch, args.h, args.n_ctx, args.d_head, args.causal)
```

## 2. Linear Attention Backward (Extracted Pattern)

Source: TileLang repository `examples/linear_attention/example_linear_attn_bwd.py`

Linear attention replaces softmax attention with a kernel feature map, so the backward has no softmax, no lse, and a simpler gradient flow. The key structural differences from flash attention backward are described below.

### Single monolithic backward kernel

Unlike flash attention which uses four separate kernels (fwd, preprocess, bwd, postprocess), the linear attention backward is a single `@tilelang.jit` kernel with no `out_idx`. All gradient buffers are passed as pre-allocated mutable arguments.

### All three gradients use atomicAdd

Because the backward iterates over chunks and each chunk contributes to overlapping gradient regions, all three gradient buffers (dQ, dK, dV) must be pre-zeroed for atomic accumulation:

```python
# In the autograd.Function backward method:
dQ = torch.zeros_like(Q, dtype=torch.float32)
dK = torch.zeros_like(K, dtype=torch.float32)
dV = torch.zeros_like(V, dtype=torch.float32)
kernel(Q, K, V, dO, dQ, dK, dV)
# Simple host-side cast instead of a postprocess kernel:
return dQ.to(torch.float16), dK.to(torch.float16), dV.to(torch.float16)
```

### 2D grid tiling across key and value dimensions

The backward kernel uses a 2D grid `(NV, NK, B*H)` that tiles across both the key dimension and the value dimension, rather than the single sequence-dimension grid used in flash attention.

### Chunk-based iteration with running state

The backward iterates over sequence chunks. A running state accumulator is maintained across chunks:

- For the dQ pass: a running state `h` is accumulated in the forward direction across chunks. At each chunk, `dQ_chunk = dO_chunk @ h` plus intra-chunk contributions.
- For the dK/dV pass: a running state `dh` is accumulated in the **reverse** direction. The loop iterates chunks from last to first.

```python
# Forward-direction pass (dQ computation):
# h accumulates K^T @ V across preceding chunks
for chunk_idx in range(num_chunks):
    # intra-chunk: lower-triangular causal mask for dQ
    # inter-chunk: dQ_chunk += dO_chunk @ h
    # update: h += K_chunk^T @ V_chunk

# Reverse-direction pass (dK/dV computation):
# dh accumulates Q^T @ dO across succeeding chunks
for chunk_idx in reversed(range(num_chunks)):
    # intra-chunk: upper-triangular mask for dK/dV
    # inter-chunk: dK_chunk += V_chunk @ dh^T, dV_chunk += K_chunk @ dh
    # update: dh += Q_chunk^T @ dO_chunk
```

### Intra-chunk causal masking

Within each chunk, the backward applies directional masks:

- dQ uses a **lower-triangular** mask (causal: position i depends on positions <= i within the chunk)
- dK and dV use an **upper-triangular** mask (each key position receives gradient from query positions >= it within the chunk)

```python
# dQ intra-chunk masking pattern:
for i, j in T.Parallel(block_M, block_M):
    qk_local[i, j] = T.if_then_else(i >= j, qk_local[i, j], 0)

# dK/dV intra-chunk masking pattern:
for i, j in T.Parallel(block_M, block_M):
    qk_local[i, j] = T.if_then_else(i <= j, qk_local[i, j], 0)
```

### No softmax correction terms

Since linear attention has no softmax, there is no lse (log-sum-exp) saved from forward, no Delta preprocess step, and no `P * (dP - Delta)` correction in the backward. The gradient flow is direct matrix multiplication without the exponential reweighting.

## 3. Attention Sink Backward (Extracted Key Patterns)

Source: TileLang repository `examples/attention_sink/example_mha_sink_bwd_bhsd.py`

Attention sink extends flash attention with a learnable `sinks` parameter that provides dedicated attention sink tokens. The backward must compute gradients for this extra parameter in addition to dQ, dK, dV.

### Extra learnable parameter: sinks

The forward takes an additional tensor `sinks` with shape `[heads, num_sink, dim]` representing learnable sink tokens. These tokens are prepended to the key/value sequence so that every query can attend to them, preventing attention score collapse.

```python
# Forward signature includes sinks:
@T.prim_func
def flash_fwd(
    Q: T.Tensor([batch, heads, seq_len, dim], dtype),
    K: T.Tensor([batch, heads, seq_len, dim], dtype),
    V: T.Tensor([batch, heads, seq_len, dim], dtype),
    sinks: T.Tensor([heads, num_sink, dim], dtype),  # extra learnable parameter
    Output: T.Tensor([batch, heads, seq_len, dim], dtype),
    lse: T.Tensor([batch, heads, seq_len], accum_dtype),
):
    # ...
```

### Sliding window masking in backward

The attention sink backward uses a sliding window mask instead of a simple causal mask. Each query position attends to:
1. The sink tokens (always visible)
2. A local window of key positions around the query

The backward kernel applies this same windowed mask when recomputing attention weights:

```python
# Sliding window condition in backward:
for i, j in T.Parallel(block_M, block_N):
    qkT[i, j] = T.if_then_else(
        # key position is within the sliding window of the query position
        (by * block_M + i - (k * block_N + j) >= 0) and
        (by * block_M + i - (k * block_N + j) < window_size),
        qkT[i, j],
        0,
    )
```

### Separate kernel for dsinks gradient

The gradient of the sink parameter requires a dedicated kernel because sinks are shared across the batch dimension and their gradient must be summed over batch and sequence:

```python
# Dedicated dsinks backward kernel:
@tilelang.jit(out_idx=[...])
def flashattn_bwd_dsink(batch, heads, seq_len, dim, num_sink, ...):
    @T.prim_func
    def flash_bwd_dsink(
        sinks: T.Tensor([heads, num_sink, dim], dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        # ... other inputs ...
        dsinks: T.Tensor([batch, heads, num_sink, dim], dtype),  # per-batch gradient
    ):
        # Compute dsinks contribution from each query block
        # Each query tile contributes to the sink gradient via:
        #   P_sink = exp(Q @ sinks^T - lse)  (recomputed attention to sinks)
        #   dsinks += P_sink^T @ dO - P_sink^T * Delta @ sinks  (softmax correction)
        # ...
```

### autograd.Function with extra parameter gradient

The wrapper saves the sinks tensor and returns its gradient. The dsinks kernel produces a per-batch-per-head result that is then summed to match the parameter shape:

```python
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, window_size):
        BATCH, H, N_CTX, D_HEAD = q.shape
        o, lse = flashattn_fwd(BATCH, H, N_CTX, D_HEAD, ...)(q, k, v, sinks)
        ctx.save_for_backward(q, k, v, sinks, o, lse)
        ctx.window_size = window_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, lse = ctx.saved_tensors
        BATCH, H, N_CTX, D_HEAD = q.shape

        # Standard phases: preprocess, main backward, postprocess
        delta = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)(o, do)
        dq = torch.zeros([BATCH, H, N_CTX, D_HEAD], dtype=torch.float32, device=q.device)
        dk = torch.empty([BATCH, H, N_CTX, D_HEAD], dtype=q.dtype, device=q.device)
        dv = torch.empty([BATCH, H, N_CTX, D_HEAD], dtype=q.dtype, device=q.device)
        flashattn_bwd(BATCH, H, N_CTX, D_HEAD, ctx.window_size, ...)(
            q, k, v, do, lse, delta, dq, dk, dv)
        dq = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)(dq)

        # Extra phase: compute dsinks gradient
        # Result shape is [batch, heads, num_sink, dim], sum over batch and reduce heads
        dsinks = flashattn_bwd_dsink(BATCH, H, N_CTX, D_HEAD, ...)(
            sinks, delta, lse, q, do)
        dsinks = dsinks.sum(0).sum(1)  # reduce batch -> [heads, num_sink, dim] (example)

        return dq, dk, dv, dsinks, None  # None for window_size (not differentiable)
```

### Dtype-parameterized tolerances

The attention sink example parameterizes both kernels and tests over dtype, setting tolerances accordingly:

```python
rtol, atol = {
    T.float16: (1e-2, 1e-2),
    T.bfloat16: (2e-2, 2e-2),
}[dtype]
assert torch.allclose(dQ, dQ_ref, rtol=rtol, atol=atol)
assert torch.allclose(dsinks, dsinks_ref, rtol=rtol, atol=atol)
```
