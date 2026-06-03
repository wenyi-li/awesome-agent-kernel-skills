# torch.compile and CUDAGraph on ROCm (Performance)

## API: use `torch.compile`, NOT `torch._dynamo.optimize`

The public API is `torch.compile(fn, mode="default")`. Older PyTorch
tutorials and some internal PyTorch code reference `torch._dynamo.optimize`
— that is the **legacy name for the same machinery** and is no longer the
recommended entry point. Both accept similar arguments (e.g. `mode=`),
which makes the confusion easy. To avoid silently invoking the wrong
code path:

- Always wrap with `torch.compile(fn, mode="default")`.
- **Never** call `torch._dynamo.optimize(fn, ...)` to "compile" a function
  in new code; it is a private symbol and its observable behaviour can
  diverge from `torch.compile`.
- It is fine to import `torch._dynamo.config` to tune dynamo settings
  (`dynamo_config.cache_size_limit = 128` etc.) — that is the documented
  configuration surface, not the compile entry point.

If you find yourself writing `torch._dynamo.optimize(...)` you have
reached for the wrong API; switch to `torch.compile(...)` before
benchmarking.

## Step 0: Audit Environment Variables

**Run this before your first torch.compile attempt:**
```bash
env | grep -iE 'TORCH|INDUCTOR|AUTOTUNE|COMPILE|TRITON|OMP|MKL' | sort
```

| Red Flag | Problem | Fix |
|---|---|---|
| `TORCHINDUCTOR_MAX_AUTOTUNE=1` | Forces kernel search on every mode; 2-min → 15+ min hang | `unset TORCHINDUCTOR_MAX_AUTOTUNE` |
| `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1` | Same for pointwise | `unset TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE` |
| `OMP_NUM_THREADS` too high | Compile workers contend for threads | Set to `nproc / 4` |

AMD Docker images often set `max_autotune` globally — the #1 cause of "compilation hangs."

## Inductor Configuration (mandatory)

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# Override container default (True → False) to prevent autotuning hangs
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.coordinate_descent_tuning = False

# CUDAGraphs: OFF (HIP graph support unstable)
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Memory planning: OFF (recursion errors on ROCm with large graphs)
inductor_config.memory_planning = False

# Fusion: enable (this is where mode="default" gets its speedup)
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.reorder_for_locality = True

# Dynamo cache (prevents recompilation storms with dynamic shapes)
dynamo_config.cache_size_limit = 128
```

Apply before any `torch.compile()` call.

## Mode Selection

Use `mode="default"` on ROCm. This is the recommended mode.

| Mode | ROCm Status |
|---|---|
| `default` | **Recommended.** Stable with correct inductor config. Enables Triton fusion for elementwise ops. |
| `reduce-overhead` | CUDAGraphs disabled by config above → equivalent to `default`. Only attempt if you re-enable inductor CUDAGraphs. |
| `max-autotune` | **Avoid.** Triggers Triton GEMM autotuning that hangs on ROCm. |

```python
model = torch.compile(model, mode="default")
```

## Graph Breaks: Find and Fix

Graph breaks split the compiled graph → extra kernel launch overhead, prevents fusion. **Fix all breaks before proceeding with other optimizations.**

### Finding breaks
```bash
TORCH_LOGS="graph_breaks" python3 your_script.py
```

### Common patterns and fixes

| Pattern | Fix |
|---|---|
| `a, b, c, d = tensor.shape` (UNPACK_SEQUENCE — most common) | `h, w = tensor.shape[2], tensor.shape[3]` |
| `if tensor.item() > 0:` (data-dependent control flow) | `torch.where(tensor > 0, true_val, false_val)` |
| Class defined inside compiled region | Move class outside |
| `[tensor[i] for i in range(n)]` | `tensor.unbind(0)` or `torch.split(...)` |
| `while condition:` (dynamic loop) | `for step in range(fixed_count):` |
| `tensor.item()` / `tensor.numpy()` | Keep on GPU, use tensor directly in torch ops |

**After fixing, always verify:** `TORCH_LOGS="graph_breaks" python3 your_script.py`

Search the codebase for the most common break: `grep -rn "= .*\.shape$" --include="*.py" src/`

## Compiling Through Vendor Ops

Use `torch.ops.*` path so Dynamo can trace through without graph-breaking:

```python
# BAD — Python wrapper may cause graph break:
from aiter import flash_attn_func
out = flash_attn_func(q, k, v)

# GOOD — compile-friendly:
out = torch.ops.aiter.mha_fwd.default(q, k, v, ...)[0]
```

## Manual CUDAGraph Capture

Since Inductor CUDAGraphs are disabled, capture manually when kernel launch overhead is high.
Manual capture wraps the **entire inference pipeline** (not just `model.forward`) in one graph — often 2x+ speedup.

### The Dynamo RNG Patch (required before capture on ROCm)

ROCm forbids `torch.cuda.get_rng_state()` during stream capture:
```
RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture
```

**Fix:** Patch `preserve_global_state` to skip CUDA RNG during capture:
```python
import functools, torch._dynamo.convert_frame as _cf

def patch_dynamo_for_rocm_capture():
    if getattr(_cf, "_rocm_patched", False):
        return
    _cf._rocm_patched = True

    def _safe_preserve(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            rng = None
            if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                try: rng = torch.cuda.get_rng_state()
                except Exception: pass
            try: return fn(*args, **kwargs)
            finally:
                if rng is not None:
                    try: torch.cuda.set_rng_state(rng)
                    except Exception: pass
        return wrapper

    _cf.preserve_global_state = _safe_preserve
```

### Capture Pattern

```python
patch_dynamo_for_rocm_capture()
compiled_model = torch.compile(model, mode="default")

# 1. Warmup (triggers Inductor compilation)
with torch.no_grad():
    for _ in range(5): _ = compiled_model(static_input)
torch.cuda.current_stream().synchronize()

# 2. Capture full inference as one graph
pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    static_output = compiled_model(static_input)

# 3. Replay — near-zero CPU overhead
graph.replay()
# Result updated in-place in static_output
```

### Fixing Dynamic Tensor Creation During Capture

Error: `hipErrorStreamCaptureUnsupported` — `torch.tensor(...)`, `torch.ones(...)`, etc. inside forward.

**Fix 1 — `register_buffer()` for constant shapes:**
```python
# In __init__:
self.register_buffer("_static_mask", torch.ones(1, max_seq_len, dtype=torch.bool), persistent=False)
# In forward — view, no allocation:
mask = self._static_mask.expand(batch_size, -1)
```

**Fix 2 — `setattr()` caching for runtime-dependent tensors:**
```python
cache_key = (str(device), int(num_steps))
cached = getattr(self, "_cached_schedule", None)
if not isinstance(cached, dict) or cached.get("key") != cache_key:
    t = torch.arange(num_steps, device=device, dtype=torch.float32)
    setattr(self, "_cached_schedule", {"key": cache_key, "t": t})
t_schedule = self._cached_schedule["t"]
```

**Fix 3 — Warm up before capture** so lazy allocations happen first.

### Capture Rules
- All inputs/outputs must be pre-allocated
- No data-dependent control flow (use `for`, not `while`)
- No CPU-GPU sync during capture
- **HIP does NOT raise errors for illegal ops** — it silently produces wrong results. Always validate outputs.
- Large graphs can segfault on HIP — try `@torch.compiler.disable` on largest submodule, or piecewise compilation
- Memory growth with dynamic shapes: pad inputs to fixed shape before capture

## Caching

```bash
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
```

First compile is slow (2-15 min on AMD). Subsequent runs reuse cache. Clear only if behavior is wrong after code changes.

## HIP Environment Variables

```bash
export HIP_LAUNCH_BLOCKING=0       # Keep 0 for production; 1 for debugging only
export AMD_LOG_LEVEL=0             # Suppress logging
export HIP_CACHE_ENABLED=1        # Avoid recompilation on restart
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```
