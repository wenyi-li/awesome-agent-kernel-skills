---
name: testing-fwd-bwd-kernels
description: >
  How to write and test TileLang kernels that need both forward and backward passes.
  Use this skill whenever the user is implementing custom operators with gradients,
  writing attention forward+backward, linear attention fwd+bwd, any op used inside
  torch.autograd.Function, or debugging gradient mismatches. Also trigger when the
  user mentions backward pass, gradients, dQ/dK/dV, autograd, ctx.save_for_backward,
  atomic_add for gradient accumulation, or gradient testing in TileLang context.
  Use even for questions like "how do I test my TileLang backward kernel" or
  "my gradients don't match the reference".
---

# Testing TileLang Kernels with Forward and Backward Passes

## Architecture Overview

TileLang attention and linear-attention examples follow a standard multi-phase architecture
for implementing differentiable GPU operators. The pattern consists of five components that
work together through a `torch.autograd.Function` wrapper.

### 1. Forward Kernel: Output + Intermediates

The forward kernel computes the main output and any intermediates needed by the backward pass.
Use `@tilelang.jit(out_idx=[N-2, N-1])` to return multiple outputs.

For flash attention, the forward returns both the output tensor `O` and the log-sum-exp `lse`:

```python
@tilelang.jit(out_idx=[3, 4])
def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    # ...
    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),          # out_idx=3
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # out_idx=4
    ):
        # ... computes O and lse ...
    return flash_fwd
```

The forward must save everything the backward needs: the inputs (Q, K, V), the output (O),
and any intermediates (lse). These are saved via `ctx.save_for_backward`.

### 2. Backward Preprocess: Delta = rowsum(O * dO)

Before the main backward kernel, a lightweight preprocessing step computes per-row dot products
between the forward output and the upstream gradient. For attention, this is
`Delta[i] = sum_j(O[i,j] * dO[i,j])`, used later to correct the softmax gradient.

```python
@tilelang.jit(out_idx=[2])
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    @T.prim_func
    def flash_bwd_prep(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            # elementwise O*dO then reduce_sum along dim dimension
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                # load O tile and dO tile
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])
    return flash_bwd_prep
```

### 3. Main Backward Kernel: Compute Gradients (dQ, dK, dV)

The main backward kernel is the most complex phase. It receives all forward inputs, the
upstream gradient dO, the lse from forward, the Delta from preprocessing, and the pre-allocated
gradient buffers. The kernel has NO `out_idx` -- it writes into pre-allocated mutable arguments.

```python
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    # NOTE: no out_idx -- dQ/dK/dV are passed as mutable pre-allocated buffers
)
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    @T.prim_func
    def flash_bwd(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ: T.Tensor(shape, accum_dtype),   # float32, pre-zeroed for atomicAdd
        dK: T.Tensor(shape, dtype),          # fp16, written directly
        dV: T.Tensor(shape, dtype),          # fp16, written directly
    ):
        # ... computes and writes dQ (via atomic_add), dK, dV ...
    return flash_bwd
```

### 4. Backward Postprocess: Cast float32 Accumulators Back to fp16/bf16

Gradients accumulated via `T.atomic_add` must use float32 targets (atomicAdd on fp16 is
not supported or loses too much precision). A postprocessing kernel casts them back:

```python
@tilelang.jit(out_idx=[1])
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    @T.prim_func
    def flash_bwd_post(
        dQ: T.Tensor(shape, accum_dtype),    # float32 input
        dQ_out: T.Tensor(shape, dtype),      # float16 output
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(dQ[bz, by, bx * blk : (bx + 1) * blk, :],
                   dQ_out[bz, by, bx * blk : (bx + 1) * blk, :])
    return flash_bwd_post
```

Note: `T.annotate_layout` with a custom layout function is used when the atomicAdd write
pattern produces a non-standard memory layout (e.g., matching the 8x8 GEMM fragment layout).
The postprocess kernel reads from that layout and writes back in standard row-major order.

### 5. torch.autograd.Function Wrapper

The autograd Function connects the kernels and manages tensor lifecycle:

```python
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        BATCH, H, N_CTX, D_HEAD = q.shape
        o, lse = flashattn_fwd(BATCH, H, N_CTX, D_HEAD, causal, block_M, block_N)(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, H, N_CTX, D_HEAD = q.shape
        # Phase 1: preprocess
        delta = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD)(o, do)
        # Phase 2: main backward (pre-allocate gradient buffers)
        dq = torch.zeros([BATCH, H, N_CTX, D_HEAD], dtype=torch.float32, device=q.device)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        flashattn_bwd(BATCH, H, N_CTX, D_HEAD, ctx.causal, block_M, block_N)(
            q, k, v, do, lse, delta, dq, dk, dv)
        # Phase 3: postprocess (cast dQ float32 -> float16)
        dq = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD)(dq)
        return dq, dk, dv, None

attention = _attention.apply
```

## Key Patterns

### Atomic Accumulation for Scattered Gradients

In attention backward, each K-block iterates over all Q-blocks and scatters gradient
contributions into dQ. Multiple thread blocks write to the same dQ positions, so you
must use `T.atomic_add`:

```python
# Inside the backward kernel -- multiple blocks write to overlapping dQ regions
T.clear(dq)
T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
for i, j in T.Parallel(block_N, dim):
    T.atomic_add(dQ[bz, bx, k * block_N + i, j], dq[i, j])
```

Rules for atomic accumulation:
- **Pre-allocate with `torch.zeros`**: The buffer MUST be zeroed because atomicAdd adds to existing values. Using `torch.empty` gives garbage.
- **Use float32 dtype**: `dq = torch.zeros(shape, dtype=torch.float32, device=q.device)`. AtomicAdd on fp16 is either unsupported or loses precision.
- **Cast back after the kernel**: Use a postprocess kernel to convert float32 back to the working dtype.

For dK and dV in attention backward, there is a one-to-one mapping between K-blocks and
output positions, so no atomics are needed -- use regular `T.copy` to write them:

```python
T.copy(dv, dv_shared)
T.copy(dk, dk_shared)
T.copy(dv_shared, dV[bz, bx, by * block_M : (by + 1) * block_M, :])
T.copy(dk_shared, dK[bz, bx, by * block_M : (by + 1) * block_M, :])
```

### `out_idx` Strategy

**Forward kernel**: Use `out_idx` to return the output and intermediates:
- `out_idx=[3, 4]` means parameters at index 3 and 4 are outputs; the kernel allocates them and returns them.

**Backward kernel**: Typically NO `out_idx`. The gradient buffers (dQ, dK, dV) are passed
as pre-allocated mutable arguments. This is necessary because:
- dQ must be pre-zeroed for atomicAdd (the kernel cannot know the initial state).
- The caller controls memory allocation and dtype (float32 for atomic targets, fp16 for direct writes).

**Postprocess kernel**: `out_idx=[1]` to return the cast result.

### Multiple Kernel Phases

The backward pass typically requires 2-3 separate TileLang kernels called in sequence:

1. **Preprocess** (simple elementwise): `Delta = rowsum(O * dO)`. Fast, simple kernel with `out_idx`.
2. **Main backward** (GEMM-like with online operations): Computes dQ/dK/dV using GEMM, softmax corrections, and conditional masking. No `out_idx`; writes to mutable buffers.
3. **Postprocess** (type casting): Converts float32 atomic accumulator to fp16/bf16 output. Uses `out_idx`.

Each phase is a separate `@tilelang.jit`-decorated function that compiles to its own CUDA kernel.
They are called sequentially in the `backward` method.

### torch.autograd.Function Template

Here is the complete template for any differentiable TileLang operator:

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        BATCH, H, N_CTX, D = q.shape
        # Call forward kernel -- returns output + intermediates
        o, lse = fwd_kernel(BATCH, H, N_CTX, D, block_M, block_N)(q, k, v)
        # Save everything backward needs
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, H, N_CTX, D = q.shape

        # Phase 1: preprocess
        delta = preprocess_kernel(BATCH, H, N_CTX, D)(o, do)

        # Phase 2: main backward with pre-allocated buffers
        dq = torch.zeros_like(q, dtype=torch.float32)  # MUST be zeros for atomicAdd
        dk = torch.empty_like(k)                         # can be empty (direct write)
        dv = torch.empty_like(v)                         # can be empty (direct write)
        bwd_kernel(BATCH, H, N_CTX, D, block_M, block_N)(
            q, k, v, do, lse, delta, dq, dk, dv)

        # Phase 3: postprocess (float32 -> fp16)
        dq = postprocess_kernel(BATCH, H, N_CTX, D)(dq)

        return dq, dk, dv
```

For operators with extra learnable parameters (e.g., attention sinks), add them to
`save_for_backward` and return their gradients from `backward`:

```python
# From attention_sink example
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, window_size):
        o, lse = flashattn_fwd(...)(q, k, v, sinks)
        ctx.save_for_backward(q, k, v, sinks, o, lse)
        ctx.window_size = window_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, lse = ctx.saved_tensors
        # ... compute dq, dk, dv as before ...
        # Additional kernel for gradient of learnable sinks parameter
        dsinks = flashattn_bwd_dsink(...)(sinks, delta, lse).sum(0).sum(1)
        return dq, dk, dv, dsinks, None  # None for window_size (not differentiable)
```

## Testing Workflow

Follow these steps exactly. This matches the pattern used in every TileLang fwd+bwd example.

### Step 1: Write a PyTorch Reference

Write a pure PyTorch reference for the forward pass. Autograd handles the backward
automatically. The reference must be mathematically equivalent to your TileLang kernel:

```python
def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output
```

### Step 2: Create the autograd.Function Wrapper

Wrap your TileLang forward and backward kernels as shown in the template above.

### Step 3: Create Input Tensors with requires_grad_()

```python
Q = torch.randn(BATCH, H, N_CTX, D_HEAD, dtype=torch.half, device="cuda").requires_grad_()
K = torch.randn_like(Q).requires_grad_()
V = torch.randn_like(Q).requires_grad_()
dO = torch.randn_like(Q)  # upstream gradient (no requires_grad needed)
```

### Step 4: Run Custom Path and Collect Gradients

```python
O = attention(Q, K, V, causal)    # calls MyOp.apply
O.backward(dO, retain_graph=True)
dQ, Q.grad = Q.grad.clone(), None  # clone then clear
dK, K.grad = K.grad.clone(), None
dV, V.grad = V.grad.clone(), None
```

IMPORTANT: You must clone the gradients and then set `.grad = None` to clear them before
running the reference. If you skip clearing, the reference backward will ADD to the existing
gradients, corrupting the comparison.

### Step 5: Run Reference Path and Collect Gradients

```python
O_ref = ref_program(Q, K, V, causal)
O_ref.backward(dO, retain_graph=True)
dQ_ref, Q.grad = Q.grad.clone(), None
dK_ref, K.grad = K.grad.clone(), None
dV_ref, V.grad = V.grad.clone(), None
```

### Step 6: Compare Each Gradient Separately

```python
assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2), f"O max err: {(O - O_ref).abs().max()}"
assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2), f"dV max err: {(dV - dV_ref).abs().max()}"
assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2), f"dK max err: {(dK - dK_ref).abs().max()}"
assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2), f"dQ max err: {(dQ - dQ_ref).abs().max()}"
```

Check the forward output FIRST, then each gradient independently. If the forward is wrong,
the backward will also be wrong (garbage in, garbage out). Check dV and dK before dQ because
dQ uses atomicAdd and is more likely to have issues.

IMPORTANT: Do NOT use `torch.autograd.gradcheck` or `torch.autograd.gradgradcheck` for
TileLang kernels. Experimental testing across fp16, bf16, fp32, and mixed-precision (fp16/bf16
forward with fp32 gradients) shows gradcheck **misses all tested bug types at every dtype** —
swapped gradients, 2x-scaled gradients, and zeroed gradients all pass. This happens because
gradcheck with `fast_mode=True` only checks a random Jacobian projection, and the loose
tolerances needed for low-precision forward passes mask the errors. gradgradcheck OOMs even at
256x256. Instead, use **`compare_backward`** — run forward+backward through both your kernel
and a PyTorch reference, then compare gradients directly. This catches all bug types reliably
and is significantly faster. See `references/gradient-testing-patterns.md` §3 for the utility and §6
for the full experimental evidence.

## Gradient Debugging Techniques

### Check Each Gradient Independently

One gradient may pass while others fail. This tells you which GEMM in the backward is wrong:

```python
print(f"dQ max err: {(dQ - dQ_ref).abs().max()}")  # often the hardest (atomicAdd)
print(f"dK max err: {(dK - dK_ref).abs().max()}")
print(f"dV max err: {(dV - dV_ref).abs().max()}")
```

### Cosine Similarity Diagnosis

Cosine similarity between your gradient and the reference reveals the TYPE of bug:

```python
def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)

sim = cos_sim(dQ, dQ_ref)
```

- **cos_sim near 0**: Wrong transform entirely. You are computing an unrelated matrix. Common cause: wrong transpose in a GEMM (e.g., `dC @ B` instead of `dC @ B^T` for dA).
- **cos_sim near 1 but magnitude off**: Correct direction, wrong scale. Common cause: missing or doubled `sm_scale`, or dividing by wrong normalizer.
- **cos_sim near -1**: Sign flip. The gradient is negated. Common cause: subtracting where you should add, or swapped operand order in a subtraction.

### Verify Intermediates First

Debug the pipeline in order. If an earlier stage is wrong, later stages inherit the error:

1. **Forward output**: Compare O vs O_ref. If this fails, fix the forward kernel first.
2. **lse (log-sum-exp)**: Print and compare `lse` vs a reference computation. Off-by-one in log2/ln conversion is common.
3. **Delta (preprocess)**: Compare `delta` vs `(O * dO).sum(dim=-1)` computed in PyTorch.
4. **Gradients**: Only debug dQ/dK/dV after confirming the above are correct.

### Test atomicAdd Correctness

For buffers accumulated via `T.atomic_add`, a quick sanity check:

```python
# Run backward on the SAME pre-zeroed buffer twice -- values should double
dq = torch.zeros(shape, dtype=torch.float32, device="cuda")
bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)
dq_once = dq.clone()
bwd_kernel(q, k, v, do, lse, delta, dq, dk, dv)  # accumulates again
assert torch.allclose(dq, 2 * dq_once, rtol=1e-5, atol=1e-5), "atomicAdd is not accumulating correctly"
```

If this fails, the atomic_add is writing to wrong positions or the buffer was not properly zeroed.

### Shrink Problem for Debugging

Set problem dimensions to a single tile size so you can use `T.print` to inspect values:

```python
# Use tiny dimensions for debugging
BATCH, H, N_CTX, D_HEAD = 1, 1, 64, 32  # single tile
block_M, block_N = 64, 32
```

At this scale, you can print fragment contents with `T.print` inside the kernel and
manually verify each intermediate computation step.

### Use Identity/Simple Inputs

Replace random inputs with structured inputs to isolate which transform the kernel computes:

```python
# Identity test: Q=K=I, V=I should give O=softmax(I)*I for attention
Q = torch.eye(D, device="cuda", dtype=torch.half).unsqueeze(0).unsqueeze(0)
K = Q.clone()
V = Q.clone()
```

## Dtype-Specific Tolerances

Different dtypes require different comparison tolerances. The codebase uses these conventions:

| Dtype | rtol | atol | Notes |
|-------|------|------|-------|
| float16 | 1e-2 | 1e-2 | Standard for most operators |
| bfloat16 | 2e-2 | 2e-2 | Lower mantissa precision (8 vs 11 bits) |
| Complex ops (attention + softmax) | 5e-2 | 5e-2 | Error compounds through exp/log/division |

The attention sink example sets tolerances based on dtype at runtime:

```python
rtol, atol = {
    T.float16: (1e-2, 1e-2),
    T.bfloat16: (2e-2, 2e-2),
}[dtype]
assert torch.allclose(dQ, dQ_ref, rtol=rtol, atol=atol), f"dQ max err: {(dQ - dQ_ref).abs().max()}"
```

Always print the max absolute error in the assertion message. When a test fails, the error
magnitude immediately tells you whether the issue is numerical precision (max err ~ 0.03)
or a logic bug (max err ~ 1.0 or larger).

## Common Pitfalls

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | Forgetting to pre-zero dQ buffer for atomicAdd | Garbage gradients, non-deterministic values | Use `torch.zeros(shape, dtype=torch.float32, ...)` not `torch.empty` |
| 2 | Wrong transpose in backward GEMM | cos_sim near 0; gradient is unrelated matrix | Check GEMM operand order: for `C = A @ B`, `dA = dC @ B^T` and `dB = A^T @ dC` |
| 3 | Not clearing `.grad` between custom and reference runs | Gradients are doubled or contaminated | Always do `dQ, Q.grad = Q.grad.clone(), None` after each backward |
| 4 | Missing `retain_graph=True` | RuntimeError: graph already freed | Add `retain_graph=True` to `.backward()` calls when running backward twice |
| 5 | Mixed-precision atomicAdd (using fp16 target) | Silent precision loss or CUDA error | Always use float32 buffer for `T.atomic_add`, cast back in postprocess |
| 6 | Forgetting `ctx.save_for_backward` | AttributeError in backward: tensors not available | Save all tensors needed by backward: inputs, output, and intermediates like lse |
| 7 | Forward modifies input in-place | Backward sees modified values, wrong gradients | Never modify inputs in-place in forward; use copies if needed |
| 8 | Using `torch.empty_like` for atomic target | Same as #1 -- empty does not zero the memory | Use `torch.zeros` or `torch.zeros_like` explicitly |
| 9 | Forgetting to `maybe_contiguous` inputs | Wrong results or crash from non-contiguous strides | Add `x = x.contiguous() if x.stride(-1) != 1 else x` before kernel calls |
| 10 | Wrong `out_idx` on backward kernel | Kernel tries to allocate outputs that should be mutable inputs | Backward kernel should have NO `out_idx` when taking pre-allocated gradient buffers |

## Linear Attention Backward: A Different Pattern

Linear attention backward uses a different approach where ALL gradient buffers (dQ, dK, dV)
use atomicAdd in float32, and the backward is a single monolithic kernel instead of multiple
phases. The float32-to-float16 cast is done with `.to()` on the host rather than a separate
postprocess kernel. It also uses reverse iteration for dK/dV and running state accumulation.

For the full implementation details, read `references/fwd-bwd-examples.md` §Linear Attention Backward.

## Reference Examples

For complete fwd+bwd implementations, read `references/fwd-bwd-examples.md` which contains:

| Example | Pattern | Key Features |
|---------|---------|-------------|
| Flash Attention Bwd | Canonical fwd+bwd | Complete: fwd, preprocess, bwd, postprocess, autograd.Function, ref, test |
| Linear Attention Bwd | Monolithic backward | All-atomic gradients, reverse iteration, running state |
| Attention Sink Bwd | Fwd+bwd with extra params | Extra learnable parameter, dtype-aware tolerances |

The flash attention backward example is the most instructive starting point — it contains
every phase plus the autograd wrapper and testing code in a single self-contained file.

For gradient testing templates (matmul fwd+bwd, test harness, debugging techniques), read `references/gradient-testing-patterns.md`.

## Escalation

- **Forward kernel has bugs** (wrong output before even testing backward): Use the
  **debugging-tilelang-programs** skill to diagnose tile-level issues, print intermediate
  values, and validate the forward kernel in isolation.

- **Gradients are correct but backward is slow**: Use the **profiling-tilelang-programs** skill
  to identify bottlenecks, then the **optimizing-tilelang-programs** skill for block size
  tuning, pipelining, and memory optimization.

- **Complex backward patterns** (warp specialization, TMA-based reductions, persistent kernels):
  These are advanced techniques beyond the standard pattern described here. Read the full
  examples in `references/fwd-bwd-examples.md` for implementation guidance.
