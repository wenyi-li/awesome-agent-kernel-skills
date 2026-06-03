# Gradient Testing Patterns

Full templates for testing TileLang kernels with forward and backward passes.

## Table of Contents

1. [Simple Matmul Fwd+Bwd](#1-simple-matmul-fwdbwd)
2. [torch.autograd.Function Wrapper](#2-torchautogradfunction-wrapper)
3. [compare_backward: Reference-Based Gradient Testing](#3-compare_backward-reference-based-gradient-testing)
4. [Gradient Comparison Test Harness](#4-gradient-comparison-test-harness)
5. [Attention Fwd+Bwd Architecture](#5-attention-fwdbwd-architecture)
6. [Why Not gradcheck/gradgradcheck?](#6-why-not-gradcheckgradgradcheck)
7. [Debugging Gradient Mismatches](#7-debugging-gradient-mismatches)

---

## 1. Simple Matmul Fwd+Bwd

Minimal complete example: a matmul forward kernel and its backward kernel, wrapped in `torch.autograd.Function`, with gradient validation.

### Forward Kernel

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul_fwd(M, N, K, block_M=128, block_N=128, block_K=32,
               dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

### Backward Kernel

For C = A @ B, the gradients are:
- dA = dC @ B^T  (shape: M x K)
- dB = A^T @ dC  (shape: K x N)

```python
@tilelang.jit(out_idx=[-1])
def matmul_dA(M, K, N, block_M=128, block_N=128, block_K=32,
              dtype=T.float16, accum_dtype=T.float32):
    """dA = dC @ B^T. dC is (M, N), B is (K, N), dA is (M, K)."""
    @T.prim_func
    def kernel(
        dC: T.Tensor((M, N), dtype),
        B: T.Tensor((K, N), dtype),
        dA: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(T.ceildiv(K, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            dC_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            dA_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(dA_local)
            for ko in T.Pipelined(T.ceildiv(N, block_K), num_stages=2):
                T.copy(dC[by * block_M, ko * block_K], dC_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(dC_shared, B_shared, dA_local, transpose_B=True)
            T.copy(dA_local, dA[by * block_M, bx * block_N])
    return kernel

@tilelang.jit(out_idx=[-1])
def matmul_dB(K, N, M, block_M=128, block_N=128, block_K=32,
              dtype=T.float16, accum_dtype=T.float32):
    """dB = A^T @ dC. A is (M, K), dC is (M, N), dB is (K, N)."""
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        dC: T.Tensor((M, N), dtype),
        dB: T.Tensor((K, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(K, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_K, block_M), dtype)
            dC_shared = T.alloc_shared((block_K, block_N), dtype)
            dB_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(dB_local)
            for ko in T.Pipelined(T.ceildiv(M, block_K), num_stages=2):
                T.copy(A[ko * block_K, by * block_M], A_shared)
                T.copy(dC[ko * block_K, bx * block_N], dC_shared)
                T.gemm(A_shared, dC_shared, dB_local, transpose_A=True)
            T.copy(dB_local, dB[by * block_M, bx * block_N])
    return kernel
```

### autograd.Function Wrapper

```python
class TileLangMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        fwd_kernel = matmul_fwd(M, N, K)
        C = fwd_kernel(A, B)

        ctx.save_for_backward(A, B)
        ctx.M, ctx.N, ctx.K = M, N, K
        return C

    @staticmethod
    def backward(ctx, dC):
        A, B = ctx.saved_tensors
        M, N, K = ctx.M, ctx.N, ctx.K

        dA_kernel = matmul_dA(M, K, N)
        dB_kernel = matmul_dB(K, N, M)

        dA = dA_kernel(dC, B)
        dB = dB_kernel(A, dC)
        return dA, dB
```

### Test

```python
M, N, K = 256, 256, 256
A = torch.randn(M, K, device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn(K, N, device="cuda", dtype=torch.float16, requires_grad=True)
dO = torch.randn(M, N, device="cuda", dtype=torch.float16)

# Reference
A_ref = A.detach().clone().requires_grad_(True)
B_ref = B.detach().clone().requires_grad_(True)
C_ref = A_ref @ B_ref
C_ref.backward(dO)

# Custom
C_custom = TileLangMatmul.apply(A, B)
C_custom.backward(dO)

# Compare
torch.testing.assert_close(C_custom, C_ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(A.grad, A_ref.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(B.grad, B_ref.grad, rtol=1e-2, atol=1e-2)
print("All gradients match!")
```

## 2. torch.autograd.Function Wrapper

### General Template

```python
class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        # 1. Unpack inputs
        A, B = inputs
        M, K = A.shape
        K2, N = B.shape

        # 2. Run forward kernel
        fwd_kernel = my_fwd(M, N, K)
        output = fwd_kernel(A, B)
        # For multi-output: output, intermediate = fwd_kernel(A, B)

        # 3. Save for backward
        ctx.save_for_backward(A, B)
        # For attention: ctx.save_for_backward(Q, K, V, output, lse)
        ctx.M, ctx.N, ctx.K = M, N, K

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve saved tensors
        A, B = ctx.saved_tensors
        M, N, K = ctx.M, ctx.N, ctx.K

        # 2. Run backward kernel(s)
        bwd_kernel = my_bwd(M, N, K)

        # Option A: kernel returns gradients (simpler ops)
        dA, dB = bwd_kernel(grad_output, A, B)

        # Option B: pre-allocate for atomicAdd (attention-style)
        # dA = torch.zeros(M, K, device=A.device, dtype=torch.float32)
        # bwd_kernel(grad_output, A, B, dA, dB)  # mutates dA, dB in-place
        # dA = dA.to(A.dtype)

        # 3. Return one gradient per forward input
        return dA, dB
```

### Key Rules

1. **Return one gradient per input** -- if `forward` takes 3 inputs, `backward` must return 3 values. Return `None` for inputs that don't need gradients.

2. **Save everything backward needs** -- forward intermediates (lse, running state) must be explicitly saved via `ctx.save_for_backward`. They won't be available otherwise.

3. **Pre-zero atomic targets** -- when using `T.atomic_add` for gradient accumulation (common for dQ in attention), pre-allocate with `torch.zeros(..., dtype=torch.float32)`. The float32 dtype is required for atomic precision.

4. **Cast back after atomicAdd** -- atomic accumulation happens in float32. Cast back to the input dtype before returning: `dQ = dQ.to(Q.dtype)`.

5. **Clear gradients between runs** -- when comparing custom vs reference:
   ```python
   A.grad = None  # clear before reference backward
   ```

## 3. compare_backward: Reference-Based Gradient Testing

The recommended approach for testing TileLang backward passes. Runs the same inputs through both
the fused kernel and a PyTorch reference, triggers backward with the same random upstream gradient,
then compares the resulting input gradients.

### Utility

```python
def clone_args(args):
    """Deep-clone tensor arguments so fused and reference don't share state."""
    out = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.detach().clone().requires_grad_(a.requires_grad))
        else:
            out.append(a)
    return tuple(out)

def compare_backward(fused, ref, args, rtol=1e-2, atol=1e-2):
    """
    Compare gradients between a fused kernel and a reference implementation.

    fused: callable wrapping the TileLang autograd.Function
    ref: callable using pure PyTorch ops (autograd handles backward)
    args: tuple of input tensors (with requires_grad=True on differentiable inputs)
    """
    args_f = clone_args(args)
    args_r = clone_args(args)

    y_f = fused(*args_f)
    y_r = ref(*args_r)

    if isinstance(y_f, torch.Tensor):
        grad_out = torch.randn_like(y_f)
        torch.autograd.backward(y_f, grad_out)
        torch.autograd.backward(y_r, grad_out)
    else:
        grad_out = tuple(torch.randn_like(y) for y in y_f)
        torch.autograd.backward(y_f, grad_out)
        torch.autograd.backward(y_r, grad_out)

    for i, (af, ar) in enumerate(zip(args_f, args_r)):
        if torch.is_tensor(af) and af.requires_grad:
            torch.testing.assert_close(af.grad, ar.grad, rtol=rtol, atol=atol)
```

### Usage

```python
M, N, K = 256, 256, 256
A = torch.randn(M, K, device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn(K, N, device="cuda", dtype=torch.float16, requires_grad=True)

compare_backward(
    lambda a, b: TileLangMatmul.apply(a, b),
    lambda a, b: a @ b,
    (A, B),
    rtol=1e-2, atol=1e-2,
)
print("Backward: PASS")
```

### What It Catches

Experimentally verified on a 256x256 matmul with TileLang forward kernel across fp16, bf16,
fp32, and mixed-precision configurations:

| Bug type | Example | Detected? |
|----------|---------|-----------|
| Swapped gradients | `return dB, dA` instead of `dA, dB` | Yes |
| Scaled gradient | `return dA * 2, dB` | Yes |
| Zeroed gradient | `return dA, torch.zeros_like(B)` | Yes |
| Wrong transpose | `dA = dC @ B` instead of `dC @ B^T` | Yes |

Full results across configurations (256x256 matmul):

| Configuration | Correct | Swapped | Scaled (dA*2) | Zeroed dB |
|---------------|:-------:|:-------:|:-------------:|:---------:|
| fp16 | pass | caught | caught | caught |
| fp32 | pass | caught | caught | caught |
| mixed fp16 (fp32 inputs, fp16 fwd, fp32 grad) | pass | caught | caught | caught |
| mixed bf16 (fp32 inputs, bf16 fwd, fp32 grad) | pass | caught | caught | caught |

### Mixed Precision (fp16/bf16 forward, fp32 gradients)

For the common training setup where weights and activations are fp16/bf16 but gradients
are accumulated in fp32:

```python
class MyOpMixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        A16 = A.half(); B16 = B.half()
        M, K = A16.shape; K2, N = B16.shape
        C = matmul_fwd(M, N, K)(A16, B16)
        ctx.save_for_backward(A16, B16)
        return C.float()

    @staticmethod
    def backward(ctx, dC):
        A16, B16 = ctx.saved_tensors
        dC16 = dC.half()
        dA = torch.mm(dC16, B16.t()).float()
        dB = torch.mm(A16.t(), dC16).float()
        return dA, dB

def ref_mixed(A, B):
    return (A.half() @ B.half()).float()

A = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
B = torch.randn(K, N, device="cuda", dtype=torch.float32, requires_grad=True)

compare_backward(
    lambda a, b: MyOpMixed.apply(a, b),
    ref_mixed, (A, B),
    rtol=1e-2, atol=1e-2,
)
```

The reference function must match the precision behavior — cast to the same low-precision
dtype before the matmul and cast back to fp32, so the numerical error profile matches.

### For Multi-Output / Attention Kernels

```python
def ref_attention(Q, K, V):
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(Q.size(-1))
    P = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", P, V)

Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
K = torch.randn_like(Q).requires_grad_(True)
V = torch.randn_like(Q).requires_grad_(True)

compare_backward(
    lambda q, k, v: my_attention(q, k, v),
    lambda q, k, v: ref_attention(q, k, v),
    (Q, K, V),
    rtol=5e-2, atol=5e-2,  # looser for attention (softmax amplifies error)
)
```

## 4. Gradient Comparison Test Harness

A more detailed manual approach with diagnostic reporting when gradients don't match.

### Full Template

```python
import torch

def test_fwd_bwd(custom_fn, ref_fn, input_shapes, dtype=torch.float16,
                 rtol=1e-2, atol=1e-2):
    """
    Test a custom autograd function against a reference.

    custom_fn: callable that takes tensors and returns output (wraps autograd.Function)
    ref_fn: callable that takes tensors and returns output (e.g., torch ops)
    input_shapes: list of (shape, requires_grad) tuples
    """
    # Create inputs
    inputs_custom = []
    inputs_ref = []
    for shape, needs_grad in input_shapes:
        t = torch.randn(*shape, device="cuda", dtype=dtype)
        inputs_custom.append(t.clone().requires_grad_(needs_grad))
        inputs_ref.append(t.clone().requires_grad_(needs_grad))

    # Forward
    out_custom = custom_fn(*inputs_custom)
    out_ref = ref_fn(*inputs_ref)

    # Check forward
    torch.testing.assert_close(out_custom, out_ref, rtol=rtol, atol=atol)
    print("Forward: PASS")

    # Backward
    dO = torch.randn_like(out_ref)
    out_custom.backward(dO)
    out_ref.backward(dO)

    # Check each gradient
    for i, (tc, tr) in enumerate(zip(inputs_custom, inputs_ref)):
        if tc.grad is not None and tr.grad is not None:
            try:
                torch.testing.assert_close(tc.grad, tr.grad, rtol=rtol, atol=atol)
                print(f"Gradient {i}: PASS")
            except AssertionError as e:
                cos = torch.nn.functional.cosine_similarity(
                    tc.grad.flatten().float(), tr.grad.flatten().float(), dim=0
                )
                mismatched = (
                    ~torch.isclose(tc.grad, tr.grad, rtol=rtol, atol=atol)
                ).float().mean()
                print(f"Gradient {i}: FAIL")
                print(f"  Cosine similarity: {cos.item():.4f}")
                print(f"  Mismatched elements: {mismatched.item()*100:.1f}%")
                print(f"  Max abs error: {(tc.grad - tr.grad).abs().max().item():.6f}")
                raise

# Usage
test_fwd_bwd(
    custom_fn=lambda A, B: TileLangMatmul.apply(A, B),
    ref_fn=lambda A, B: A @ B,
    input_shapes=[((256, 256), True), ((256, 256), True)],
    dtype=torch.float16,
)
```

### Diagnostics When Gradients Don't Match

The test harness reports three metrics for failed gradients:

| Metric | What it means | Likely cause |
|--------|--------------|--------------|
| Cosine similarity ~0.0 | Gradient direction is completely wrong | Wrong transpose, wrong matrix in GEMM |
| Cosine similarity ~1.0, high error | Right direction, wrong magnitude | Scaling bug, missing factor |
| High mismatch % (>50%) | Most elements wrong | Missing T.clear, wrong buffer shape |
| Low mismatch % (<5%) | Most elements correct | Boundary issue, precision |

### Using Simple Inputs for Diagnosis

When cosine similarity is near 0, use identity matrices to determine what transform the kernel is computing:

```python
A = torch.eye(K, device="cuda", dtype=torch.float16)
B = torch.eye(K, device="cuda", dtype=torch.float16)
dO = torch.eye(K, device="cuda", dtype=torch.float16)

C = TileLangMatmul.apply(A, B)
C.backward(dO)

# With identity inputs:
# dA should be dO @ B^T = I @ I^T = I
# dB should be A^T @ dO = I^T @ I = I
print("dA:", A.grad)  # should be identity
print("dB:", B.grad)  # should be identity
```

If dA looks like a transposed version of what you expect, the backward kernel has a transpose bug.

## 5. Attention Fwd+Bwd Architecture

This section outlines the architecture without full code -- for complete implementations, read `references/fwd-bwd-examples.md`.

### Forward Phase

```
Inputs: Q (B,H,M,D), K (B,H,N,D), V (B,H,N,D)
Outputs: O (B,H,M,D), lse (B,H,M)

Grid: (ceildiv(M, block_M), B*H)  -- one block per query tile per batch*head

For each query tile [by * block_M : (by+1) * block_M]:
    Load Q_tile
    Initialize: m_prev = -inf, l_prev = 0, O_local = 0

    For each KV tile (pipelined):
        Load K_tile, V_tile
        S = Q_tile @ K_tile^T                    # attention scores
        m_new = max(m_prev, rowmax(S))            # online softmax: max
        P = exp(S - m_new)                         # attention weights
        l_new = exp(m_prev - m_new) * l_prev + rowsum(P)  # online softmax: sum
        O_local = exp(m_prev - m_new) * O_local + P @ V_tile  # rescale + accumulate
        m_prev, l_prev = m_new, l_new

    O = O_local / l_new                           # normalize
    lse = m_prev + log(l_prev)                    # log-sum-exp for backward

    Write O, lse to global memory
```

### Backward Preprocess

```
Inputs: O (B,H,M,D), dO (B,H,M,D)
Output: Delta (B,H,M)  where Delta = rowsum(O * dO)

Simple elementwise-then-reduce kernel.
```

### Backward Main

```
Inputs: Q, K, V, O, dO, lse, Delta
Outputs: dQ (float32, pre-zeroed), dK, dV

Grid: (ceildiv(N, block_N), B*H)  -- one block per KV tile per batch*head

For each KV tile:
    Load K_tile, V_tile, dK_local=0, dV_local=0

    For each Q tile:
        Load Q_tile, dO_tile, lse_tile, Delta_tile
        S = Q_tile @ K_tile^T
        P = exp(S - lse_tile)                      # recompute attention weights
        dV_local += P^T @ dO_tile                   # accumulate dV
        dP = dO_tile @ V_tile^T
        dS = P * (dP - Delta_tile)                  # softmax backward
        dQ_tile = dS @ K_tile                       # partial dQ
        T.atomic_add(dQ[...], dQ_tile)              # scatter into global dQ (float32)
        dK_local += dS^T @ Q_tile                   # accumulate dK

    Write dK_local, dV_local to global memory
```

### Key Implementation Details

**dQ uses atomicAdd because:** The backward grid iterates over KV tiles. Each KV tile contributes a partial gradient to ALL query positions. Multiple blocks write to the same dQ locations, requiring atomics.

**dK/dV don't need atomics because:** Each KV tile's gradients are fully computed within one block. The block accumulates locally and writes once.

**Layout annotation for dQ:**
```python
T.annotate_layout({dQ: make_dq_layout(dQ)})
```
This reorganizes the dQ buffer for efficient per-element atomic writes matching the GEMM fragment layout.

**Backward postprocess:** Cast dQ from float32 (used for atomic precision) back to fp16:
```python
dQ_fp16 = dQ_fp32.to(torch.float16)
```

## 6. Why Not gradcheck/gradgradcheck?

`torch.autograd.gradcheck` uses finite differences to numerically verify gradients. This does
not work for TileLang kernels. Testing across fp16, bf16, fp32, and mixed-precision (fp16/bf16
forward with fp32 gradients) shows gradcheck **misses all tested bug types at every dtype
and precision combination** — swapped, 2x-scaled, and zeroed gradients all pass as correct:

| Configuration | Correct | Swapped | Scaled (dA*2) | Zeroed dB |
|---------------|:-------:|:-------:|:-------------:|:---------:|
| gradcheck fp16 | pass | **MISSED** | **MISSED** | **MISSED** |
| gradcheck fp32 | pass | **MISSED** | **MISSED** | **MISSED** |
| gradcheck mixed fp16 | pass | **MISSED** | **MISSED** | **MISSED** |
| gradcheck mixed bf16 | pass | **MISSED** | **MISSED** | **MISSED** |
| gradgradcheck fp32 | OOM | — | — | — |

Compare with `compare_backward` (§3):

| Configuration | Correct | Swapped | Scaled (dA*2) | Zeroed dB |
|---------------|:-------:|:-------:|:-------------:|:---------:|
| compare_backward fp16 | pass | caught | caught | caught |
| compare_backward fp32 | pass | caught | caught | caught |
| compare_backward mixed fp16 | pass | caught | caught | caught |
| compare_backward mixed bf16 | pass | caught | caught | caught |

### Why gradcheck fails

With `fast_mode=True`, gradcheck only checks a random projection of the Jacobian rather than
the full matrix. The loose tolerances required for low-precision forward passes (atol/rtol ~
0.1-0.5) mask the errors in these projections. Even with a native fp32 TileLang kernel and
tighter tolerances (atol/rtol = 1e-3), gradcheck still misses all three bug types.

Without `fast_mode`, gradcheck computes the full Jacobian — one forward pass per input element.
For a 256x256 matrix that is 65,536 forward passes, making it impractical for GPU kernel sizes.

`gradgradcheck` allocates O(N^2) tensors for the Hessian and OOMs even at 256x256.

## 7. Debugging Gradient Mismatches

### Step-by-Step

1. **Isolate which gradient fails**: Test dQ, dK, dV independently
2. **Check cosine similarity**: near 0 = wrong transform; near 1 = magnitude issue
3. **Try identity inputs**: reveals what transform the kernel computes
4. **Check atomicAdd accumulation**: run backward twice on same pre-zeroed buffer; values should double
5. **Verify forward intermediates**: check lse is correct before debugging backward
6. **Reduce problem size**: use M=block_M, N=block_N for single-tile debugging

### Typical Bugs and Signatures

| Bug | Cosine sim | Mismatch % | Diagnosis |
|-----|-----------|-----------|-----------|
| Missing transpose in dA GEMM | ~0.02 | 99%+ | Wrong matrix multiply entirely |
| Wrong scaling factor | ~1.0 | varies | Correct direction, wrong magnitude |
| Missing T.clear on accumulator | N/A | 50%+ | Garbage mixed with correct values |
| Forgot to pre-zero dQ for atomicAdd | N/A | 100% | dQ has garbage from prior allocations |
| Wrong lse in backward | varies | varies | Fix forward lse first, then retest |
| Off-by-one in tile indexing | ~0.95+ | <10% | Boundary elements wrong |

### Verifying atomicAdd

```python
# Run backward twice on the same pre-zeroed dQ buffer
dQ = torch.zeros(M, K, device="cuda", dtype=torch.float32)
bwd_kernel(dO, A, B, dQ)       # first accumulation
dQ_once = dQ.clone()
bwd_kernel(dO, A, B, dQ)       # second accumulation (into same buffer)

# dQ should now be 2x the single-run value
torch.testing.assert_close(dQ, 2 * dQ_once, rtol=1e-5, atol=1e-5)
```

### Dtype-Specific Tolerances

| Dtype | Forward rtol/atol | Gradient rtol/atol | Notes |
|-------|------------------|-------------------|-------|
| float16 | 1e-2 / 1e-2 | 1e-2 / 1e-2 | Standard for GEMM |
| bfloat16 | 2e-2 / 2e-2 | 2e-2 / 2e-2 | Lower precision mantissa |
| float16 attention | 1e-2 / 1e-2 | 5e-2 / 5e-2 | Softmax amplifies error |
