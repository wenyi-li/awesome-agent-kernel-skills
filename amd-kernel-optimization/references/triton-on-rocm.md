# Writing Triton Kernels for ROCm

## When to Consider Triton on ROCm

Triton is well-suited for **elementwise ops, reductions, and normalizations** where fusing multiple operations into a single kernel saves kernel launch overhead and memory traffic.

For **GEMM**, vendor libraries (rocBLAS, hipBLASLt, aiter) are often strong competitors on AMD hardware — benchmark Triton GEMM against vendor alternatives for your shapes before committing.

## ROCm-Specific Gotchas

### 1. tanh intrinsic may be unavailable

Some ROCm Triton builds do not expose `tl.math.tanh`. Implement manually via `exp` with clamping to avoid `inf/inf → NaN`:

```python
# May fail or produce NaN on some ROCm builds:
# tanh_val = tl.math.tanh(x)

# Safe alternative: manual tanh with overflow protection
x_clamped = tl.maximum(tl.minimum(x, 10.0), -10.0)
exp_2x = tl.math.exp(2.0 * x_clamped)
tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
```

### 2. Cast output to target dtype explicitly

Compute in float32 for accuracy, cast back to bfloat16 on store:

```python
x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)  # load as f32
y = some_computation(x)                                   # compute in f32
tl.store(out_ptr, y.to(tl.bfloat16), mask=mask)           # store as bf16
```

### 3. Wavefront size is 64 (not 32)

AMD GPUs use wavefront size 64 (NVIDIA uses warp size 32). This affects Triton kernel tuning:
- `num_warps` in Triton autotuning configs corresponds to wavefronts of 64 threads
- BLOCK_SIZE values that are multiples of 64 align better with hardware
- Occupancy calculations differ from NVIDIA — more threads per wavefront means fewer wavefronts needed

### 4. BLOCK_SIZE selection

Use `triton.next_power_of_2(N)` for the hidden dimension. Launch one block per row:

```python
BLOCK_SIZE = triton.next_power_of_2(N)
grid = (M,)  # M = number of rows
kernel[grid](... , BLOCK_SIZE=BLOCK_SIZE)
```

## Common Kernel Pattern

Most elementwise/reduction Triton kernels on ROCm follow this template:

```python
@triton.jit
def _kernel(X_ptr, Y_ptr, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    y = compute(x)
    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)

def wrapper(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _kernel[(M,)](x_2d, y, x_2d.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(orig_shape)
```

## Fused Kernel Examples

### RMSNorm

```python
@triton.jit
def _rms_norm_kernel(X, W, Y, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + row * stride_y + cols, (x * rrms * w).to(tl.bfloat16), mask=mask)
```

### Fused SiLU + Mul (for gated MLP)

Input `[*, 2N]` → output `[*, N]`. Gate is first half, up-projection is second half.

```python
@triton.jit
def _silu_mul_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (gate * tl.sigmoid(gate) * up).to(tl.bfloat16), mask=mask)
```

### Fused GELU(tanh) + Mul

Same structure as SiLU+Mul but with GELU tanh approximation (uses manual tanh for ROCm safety):

```python
@triton.jit
def _gelu_mul_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    # GELU tanh approx with safe manual tanh
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (gate + 0.044715 * gate * gate * gate)
    inner = tl.maximum(tl.minimum(inner, 10.0), -10.0)
    exp2 = tl.math.exp(2.0 * inner)
    tanh_v = (exp2 - 1.0) / (exp2 + 1.0)
    gelu = 0.5 * gate * (1.0 + tanh_v)
    tl.store(Y + row * stride_y + cols, (gelu * up).to(tl.bfloat16), mask=mask)
```

### Fused Residual Add + RMSNorm

Combines `hidden = x + residual` and `output = RMSNorm(hidden)` in one kernel (saves a memory round-trip):

```python
@triton.jit
def _add_rms_norm_kernel(X, R, W, Y, RS, sx, sr, sy, srs, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * sx + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * sr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(RS + row * srs + cols, h.to(tl.bfloat16), mask=mask)  # residual sum
    var = tl.sum(h * h, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    tl.store(Y + row * sy + cols, (h * rrms * w).to(tl.bfloat16), mask=mask)
```

## Fusion Targets

High-value fusion targets (each saves kernel launch overhead + memory round-trip):

| Fusion | Ops Combined |
|--------|-------------|
| SiLU + Mul | `silu(gate) * up` |
| GELU + Mul | `gelu(gate) * up` |
| RMSNorm | `x * rsqrt(mean(x²) + eps) * w` |
| Add + RMSNorm | `rmsnorm(x + residual)` |
| Add + LayerNorm | Same idea for LayerNorm models |
| Residual + Dropout + Add | Common in training |
