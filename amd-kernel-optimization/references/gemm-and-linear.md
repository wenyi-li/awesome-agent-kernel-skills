# GEMM and Linear Layer Optimization on AMD ROCm

## Available GEMM Backends

On AMD GPUs, several GEMM backends are available. Performance varies by (M, N, K) shape, dtype, and batch size — **always benchmark on your specific workload**.

| Backend | Description | How to Use |
|---------|-------------|------------|
| **rocBLAS** | AMD's vendor BLAS; default ATen GEMM backend on ROCm | Used automatically by `F.linear` / `torch.mm` |
| **hipBLASLt** | Lightweight BLAS with fused epilogue support | Set via `TORCH_ROCM_USE_HIPBLASLT=1` or inductor config |
| **aiter tuned GEMM** | Auto-dispatches best kernel per shape from tuned configs | `from aiter.tuned_gemm import gemm_a16w16` |
| **Triton GEMM** | Triton-generated matmul kernels | Used by inductor with `max-autotune`; or write custom |
| **CK GEMM** | AMD Composable Kernel library | Via CK Python bindings or through aiter |
| **FP8 GEMM** (MI300+) | Quantized GEMM using E4M3/E5M2 formats | `aiter.tuned_gemm.gemm_a8w8` or via `torch.float8_e4m3fnuz` dtype |

## aiter Tuned GEMM

AMD's `aiter` library provides a tuned GEMM dispatcher that selects the best kernel (asm / hipBLASLt / skinny / torch) per (M, N, K) shape based on pre-tuned config files.

### Usage

```python
from aiter.tuned_gemm import gemm_a16w16

# x: [*, K] input, weight: [N, K] (nn.Linear layout)
output = gemm_a16w16(x, weight, bias=None, otype=x.dtype)
```

### Per-Shape Config CSV

aiter reads tuning configs from CSV files specified by `AITER_CONFIG_GEMM_BF16` env var. Format:

```csv
cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw
304,64,4096,4096,0,bf16,bf16,0,0,ASM,0,0,10.2,asm_kernel,0.001,180.5,1200
304,128,4096,4096,0,bf16,bf16,0,0,hipBLASLt,3,0,18.1,hipblaslt_kernel,0.001,385.2,2100
```

To add your own tuned shapes, append your CSV path:
```bash
export AITER_CONFIG_GEMM_BF16="/path/to/aiter/configs/bf16_tuned_gemm.csv:/path/to/your/tuned_gemm.csv"
```

### When Configs Are Missing (Silent Failure Mode)

**The default tuned config CSV ships empty.** Without populated configs, `gemm_a16w16` silently falls back to plain `F.linear` for most shapes — **no error, no crash, no benefit**. This is the most common failure mode when applying aiter tuned GEMM: the monkey-patch appears to work, performance doesn't improve, and there's no obvious signal why.

**Diagnose first:**
```bash
AITER_LOG_TUNED_CONFIG=1 python3 your_script.py 2>&1 | grep -E "not found tuned|using torch"
```
If you see `"not found tuned config ... will use default config!"` or `"using torch solution:0"`, configs are missing for those shapes. If nothing prints, check that `AITER_CONFIG_GEMM_BF16` points to a non-empty CSV.

**Fix option 1 — Generate configs with aiter's built-in tuner (optimal):**

aiter ships a `GemmTuner` in its `gradlib/` directory that benchmarks all available kernels per shape and selects the winner. Two-step workflow:

```bash
# Step 1: collect GEMM shapes from your actual workload (one inference pass is enough)
AITER_TUNE_GEMM=1 python3 your_inference_script.py
# → writes seen shapes to bf16_untuned_gemm.csv in aiter's configs/ directory

# Find where aiter is installed and locate both the config file and tuner script.
# In the standard docker environment, aiter is at /sgl-workspace/aiter:
#   Untuned shapes: /sgl-workspace/aiter/aiter/configs/bf16_untuned_gemm.csv
#   Tuner script:   /sgl-workspace/aiter/gradlib/gradlib/gemm_tuner.py
# To locate dynamically in other environments:
AITER_DIR=$(python3 -c "import aiter, os; print(os.path.dirname(aiter.__file__))")
TUNER=$(find "$AITER_DIR/.." -name "gemm_tuner.py" 2>/dev/null | head -1)

# Step 2: benchmark all kernels for collected shapes (5–30 min, requires GPU)
python3 "$TUNER" \
    --untune_file "$AITER_DIR/configs/bf16_untuned_gemm.csv" \
    --out_file /tmp/my_tuned_gemm.csv

# Step 3: apply results
export AITER_CONFIG_GEMM_BF16=/tmp/my_tuned_gemm.csv
```

**Fix option 2 — Use TunableOp instead (simpler, no separate tuning run):**

`PYTORCH_TUNABLEOP_ENABLED=1` (already in Level 1 env vars) auto-tunes GEMM via hipBLASLt on first run and caches results. It covers fewer kernel types than aiter's full tuner (no ASM/skinny kernels) but requires zero extra steps and is a solid fallback when tuning time is not available.

## Routing nn.Linear Through a Custom GEMM Backend

### Monkey-patch approach (zero model code changes)

```python
from aiter.tuned_gemm import gemm_a16w16
import torch.nn as nn
import torch.nn.functional as F

_original_linear_forward = nn.Linear.forward

def _patched_forward(self, x):
    if x.dtype in (torch.bfloat16, torch.float16) and self.weight.dtype == x.dtype:
        try:
            return gemm_a16w16(x, self.weight, bias=self.bias, otype=x.dtype)
        except Exception:
            pass
    return _original_linear_forward(self, x)

nn.Linear.forward = _patched_forward
```

Call this **after** loading model weights but **before** `torch.compile`.

## Projection Fusion

Fuse multiple linear projections into one GEMM to reduce kernel launch count:

### QKV Fusion (3 GEMMs → 1)

```python
# Before: 3 separate matmuls
q = self.q_proj(x)  # [B, S, D] @ [D, D]
k = self.k_proj(x)  # [B, S, D] @ [D, Dkv]
v = self.v_proj(x)  # [B, S, D] @ [D, Dkv]

# After: 1 fused matmul
# Fuse weights once at init time:
fused_qkv_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
model.register_buffer("_fused_qkv_weight", fused_qkv_weight)

# At forward time:
qkv = F.linear(x, self._fused_qkv_weight)  # single GEMM
q, k, v = qkv.split([D, Dkv, Dkv], dim=-1)
```

### Gate+Up Fusion (MLP, 2 GEMMs → 1)

```python
fused_gate_up_weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0)
model.register_buffer("_fused_gate_up_weight", fused_gate_up_weight)

# Forward:
gate_up = F.linear(x, self._fused_gate_up_weight)  # single GEMM
gate, up = gate_up.chunk(2, dim=-1)
output = activation(gate) * up
```

Fuse weights **after** loading but **before** `torch.compile`.

### Routing Fused Projections Through aiter Tuned GEMM

Fusing projections changes the GEMM shape: N becomes the sum of the individual projection sizes (e.g., QKV fusion yields `N = D_q + D_k + D_v`). **Existing tuned configs for the individual projections do not cover the new fused shape** — generate configs for it using `AITER_TUNE_GEMM=1` with your workload after applying fusion (see "When Configs Are Missing" above).

Also note: the `nn.Linear` monkey-patch only intercepts `nn.Linear.forward`. Fused weights stored as buffers and called via `F.linear` bypass it. Call `gemm_a16w16` directly for those:

```python
from aiter.tuned_gemm import gemm_a16w16

# In forward(), replace F.linear(x, self._fused_qkv_weight) with:
inp = x.view(-1, x.size(-1))
qkv = gemm_a16w16(inp, self._fused_qkv_weight, bias=None, otype=x.dtype)
qkv = qkv.view(*x.shape[:-1], qkv.shape[-1])
```

## Weight Preshuffling (asm Kernel Fast Path)

aiter's asm GEMM kernels on gfx950+ can use a pre-shuffled weight layout. Whether this helps or hurts depends on GEMM shape — benchmark with and without.

```python
from aiter.ops.shuffle import shuffle_weight

def preshuffle_weights(model, require_multiple=256, layout=(16, 16)):
    """Pre-shuffle eligible Linear weights for aiter asm kernels."""
    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        if w.dtype != torch.bfloat16 or w.ndim != 2:
            continue
        n, k = w.shape
        if n % require_multiple != 0 or k % require_multiple != 0:
            continue
        if module.bias is not None:
            continue  # asm path needs bias=None (or use bias splitting)
        w_shuffled = shuffle_weight(w, layout=layout)
        module._preshuffled_weight = w_shuffled
        count += 1
    return count
```

### M-threshold gating

Preshuffled weights may help for some GEMM shapes but hurt for others. A common pattern is to gate by input size (M = total rows):

```python
def patched_forward(self, x):
    w = self.weight
    w_shuf = getattr(self, "_preshuffled_weight", None)
    if w_shuf is not None:
        m = x.numel() // x.shape[-1]  # total rows
        if m >= M_THRESHOLD:  # tune this threshold for your workload
            w = w_shuf
    return gemm_a16w16(x, w, bias=self.bias, otype=x.dtype)
```

## Bias Splitting

Some aiter fast paths (asm kernels) require `bias=None`. Split bias into a separate add to unlock these paths:

```python
def linear_with_split_bias(x, weight, bias):
    out = gemm_a16w16(x, weight, bias=None, otype=x.dtype)
    if bias is not None:
        out = out + bias
    return out
```

## Attention Backend Selection

Several attention backends are available on AMD. Performance depends on sequence length, head dimensions, and whether you're in prefill or decode. **Benchmark all relevant options.**

| Backend | API | Notes |
|---------|-----|-------|
| **aiter flash attention** | `torch.ops.aiter.mha_fwd.default(q, k, v, ...)` | AMD-optimized; supports GQA natively; `torch.ops` path is compile-friendly |
| **SDPA** | `F.scaled_dot_product_attention(q, k, v, ...)` | PyTorch built-in; dispatches to available backends; supports `is_causal` |
| **CK flash attention** | Via CK bindings or `ck_flash_attn` | AMD Composable Kernel implementation |

### aiter flash attention (torch.ops path)

```python
# Direct torch.ops path (compile-friendly, avoids graph breaks):
output = torch.ops.aiter.mha_fwd.default(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=scale,
    is_causal=is_causal,
    window_size_left=-1,
    window_size_right=-1,
    return_softmax_lse=False,
    return_dropout_randval=False,
)[0]
```

### SDPA fast-path for KV-cache decode

When `q_len != k_len` (decode step with cached K/V), SDPA can avoid explicit mask construction:

```python
if query.shape[2] != key.shape[2]:  # decode step
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
    )
```

### GQA/MQA tip

Avoid expanding K/V heads to match Q heads with `repeat_kv` — this creates zero-stride tensors that can trigger expensive `.contiguous()` calls under `torch.compile`. Both aiter FA and SDPA support different Q/KV head counts natively.
