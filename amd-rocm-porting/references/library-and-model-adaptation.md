# Library & Model Adaptation

## Library Mapping

### Drop-in via HIPIFY (C/C++ only)
cuBLAS→hipBLAS+rocBLAS, cuSPARSE→hipSPARSE, cuFFT→hipFFT, cuRAND→hipRAND, NCCL→RCCL, Thrust→rocThrust

### Requires manual adaptation
| NVIDIA | AMD | Notes |
|---|---|---|
| flash-attn | aiter | Different Python API; see below |
| cuDNN | MIOpen | Different C++ descriptor API |
| CUTLASS | Composable Kernel | Full manual rewrite |
| TensorRT | MIGraphX | Different optimization pipeline |

## Three-Tier Fallback (always implement all three)

```python
AITER_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    pass

def attention_forward(q, k, v, softmax_scale, is_causal=False):
    # Tier 1: AMD-optimized
    if AITER_AVAILABLE and os.environ.get("USE_AITER_ATTENTION") == "1":
        out, *_ = torch.ops.aiter.mha_fwd(
            q, k, v, 0.0, softmax_scale, is_causal,
            -1, -1, False, False
        )
        return out
    # Tier 2: PyTorch SDPA (works on both NVIDIA and AMD)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=softmax_scale, is_causal=is_causal
        )
    # Tier 3: Pure PyTorch eager
    attn = (q @ k.transpose(-2, -1)) * softmax_scale
    if is_causal:
        attn = attn.masked_fill(torch.tril(torch.ones_like(attn)) == 0, float("-inf"))
    return torch.softmax(attn, dim=-1) @ v
```

## aiter Attention API

```python
# torch.ops.aiter.mha_fwd signature:
out, lse, p, rng_state = torch.ops.aiter.mha_fwd(
    query,           # [B, Sq, H, D]
    key,             # [B, Sk, H, D]  (MQA/GQA: H can differ from query)
    value,           # [B, Sk, H, D]
    dropout_p,       # float, 0.0 for inference
    softmax_scale,   # float, typically 1/sqrt(D)
    is_causal,       # bool
    window_size_left,   # int, -1 = no sliding window
    window_size_right,  # int, -1 = no sliding window
    return_softmax_lse,   # bool, False normally
    return_dropout_randval,  # bool, False normally
)
# Do NOT expand K/V heads for MQA/GQA — aiter handles natively
```

## Python-Level AMD Fixes

### pynvml — guard all usages

```python
# BAD: top-level import breaks on AMD
import pynvml

# GOOD: use torch.cuda.is_available() as primary check
try:
    import torch
    if torch.cuda.is_available():
        return  # works for both NVIDIA and AMD ROCm
except ImportError:
    pass
try:
    import pynvml
    pynvml.nvmlInit(); pynvml.nvmlShutdown()
    return
except Exception:
    pass
```

### PYTORCH_CUDA_ALLOC_CONF — remove max_split_size_mb on ROCm

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
if not is_rocm:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### torch.backends flags — guard on ROCm

`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.benchmark`, and
`torch.backends.cudnn.allow_tf32` are CUDA-only. Wrap in `if not is_rocm:`.
