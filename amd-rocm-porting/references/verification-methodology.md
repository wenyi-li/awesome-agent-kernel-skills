# Verification Methodology

## 4-Level Pyramid (run in order)

**L1 Static** → **L2 Build** → **L3 Load** → **L4 Numerical**

### L1: Static Analysis (grep checks)

```bash
# Untranslated CUDA headers/types
grep -rn "cuda_runtime\.h\|cublas_v2\.h\|cudaStream_t\|cudaError_t" src/

# 32-bit warp masks (should be 64-bit on AMD)
grep -rn "0xFFFFFFFF\|__ballot_sync\|__shfl_sync" --include="*.cu" --include="*.hip" src/

# Inline PTX (cannot be auto-ported)
grep -rn "asm\s*(" --include="*.cu" --include="*.hip" src/

# Python-level NVIDIA-specific patterns
grep -rn "reduce.overhead\|max_split_size_mb\|allow_tf32\|import pynvml" --include="*.py" src/
```

### L2: Build Test (C/C++ only)
```bash
GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-f]*' | head -1)  # e.g., gfx942, gfx950
hipcc -c src/kernel.hip -o /tmp/kernel.o --offload-arch=$GPU_ARCH
```

### L3: Load Test
```python
import torch
x = torch.randn(4, 32, device="cuda")
y = model(x)
assert not torch.isnan(y).any(), "NaN detected"
assert not torch.isinf(y).any(), "Inf detected"
assert y.device.type == "cuda", f"Wrong device: {y.device}"
```

### L4: Numerical Correctness

Compare against NVIDIA reference (golden vectors generated on CUDA GPU):

```python
ref = torch.load("golden.pt").float()
got = model(test_input).float()
torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
```

**Recommended tolerances:**

| dtype | rtol | atol |
|---|---|---|
| float32 | 1e-4 | 1e-4 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 5e-2 | 5e-2 |
| bfloat16 multi-layer | 1e-1 | 1e-1 |

**Golden vector shapes:** `(1,32,512)`, `(4,128,2048)`, `(16,512,2048)`, `(1,1,2048)`, `(1,2048,2048)`

## Sync Helper

```python
def sync():
    """Prefer stream-level sync on ROCm (lower overhead than device-wide)."""
    try:
        torch.cuda.current_stream().synchronize()
    except Exception:
        torch.cuda.synchronize()
```

## Stability Test (100 iterations)

```python
for i in range(100):
    out = model(input)
    assert not torch.isnan(out).any(), f"NaN at iteration {i}"
    if i % 10 == 0:
        print(f"[{i}] mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```
