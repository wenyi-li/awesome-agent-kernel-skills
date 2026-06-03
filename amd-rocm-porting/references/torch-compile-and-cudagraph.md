# torch.compile Safety on ROCm (Porting)

This reference covers **making existing torch.compile calls safe on ROCm**.
For performance optimization with torch.compile (graph breaks, manual CUDAGraph, profiling),
see the `amd-kernel-optimization` skill.

## Step 0: Audit Environment Variables BEFORE Compiling

**Always run this before your first torch.compile attempt:**
```bash
env | grep -iE 'TORCH|INDUCTOR|AUTOTUNE|COMPILE|TRITON|OMP|MKL' | sort
```

**Red flags that silently break compilation:**

| Variable | Problem | Fix |
|---|---|---|
| `TORCHINDUCTOR_MAX_AUTOTUNE=1` | Forces expensive kernel search on EVERY mode, 2-min → 15+ min hang | `unset TORCHINDUCTOR_MAX_AUTOTUNE` |
| `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1` | Same for pointwise ops | `unset TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE` |

AMD Docker images often set `max_autotune` globally — the #1 cause of "compilation hangs."

## Compile Mode Gating (critical)

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
compile_mode = "default" if is_rocm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

`reduce-overhead` triggers Inductor's CUDAGraph capture, which is **broken on ROCm**
(up to 65x slowdown or hang). Gate with `is_rocm` — NVIDIA path unchanged.

## Inductor Safety Configuration (mandatory on ROCm)

```python
import torch._inductor.config as inductor_config

# Override container default (True → False) to prevent autotuning hangs
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.coordinate_descent_tuning = False

# CUDAGraphs: OFF on ROCm
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Memory planning: OFF (deep recursion crash on ROCm)
inductor_config.memory_planning = False

# Fusion: enable (this is where mode="default" gets its speedup)
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.reorder_for_locality = True
```

Apply before any `torch.compile()` call. Gate the whole block behind `if is_rocm:`.

## Compile Safety Monkey-Patch

Intercept any code path that hardcodes `reduce-overhead`:

```python
_orig_compile = torch.compile
def _safe_compile(model=None, **kwargs):
    if is_rocm and kwargs.get("mode") in (None, "reduce-overhead"):
        kwargs["mode"] = "default"
    return _orig_compile(model, **kwargs)
torch.compile = _safe_compile
```

## Triton on ROCm (quick notes)

- Prefer block sizes that are multiples of **64** (AMD wavefront width)
- Always accumulate in `float32`, store back in `bfloat16`/`float16`
- Clamp inputs to `[-10, 10]` before `exp` to avoid tanh overflow NaN
