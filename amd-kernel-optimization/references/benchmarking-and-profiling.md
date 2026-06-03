# Benchmarking and Profiling on AMD ROCm

## Benchmarking: Getting Accurate Numbers

### The Rules
1. **Always use the benchmark script's default settings first.**
2. **Never reduce warmup/iterations to "save time."** You get garbage numbers.
3. **Never report first-run latency as baseline.** First run includes one-time compilation costs.

### What a Proper Benchmark Needs

**Warmup (mandatory):** First N runs are always slower due to CUDA/HIP context init, memory allocation, torch.compile JIT (2-15 min on AMD), Triton/AITER kernel JIT, and hipBLASLt autotuning. Minimum 3 warmup runs; script default (usually 10) is better.

**Multiple iterations (mandatory):** GPU timing has ±5-15% variance. Minimum 10 iterations for mean, 20+ for p50/p99. Report mean AND std. If std > 10% of mean, something is wrong.

### GPU Timing (always use this, not wall-clock)

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result = model(input)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

Wall-clock `time.time()` includes Python overhead and scheduling noise — never use for GPU benchmarks.

### Anti-Patterns

| Bad | Why | Good |
|---|---|---|
| `WARMUP=0 ITERATIONS=1` | No warmup = measuring one-time costs | Use script defaults |
| `time.time()` around GPU ops | Includes Python/scheduling overhead | Use CUDA events |
| Reporting first-run latency | Includes compilation time | Mean of post-warmup iterations |
| Disabling torch.compile to "get it working" | Loses 2-5x speedup | Fix compile issues, keep it enabled |
| Reducing denoising steps to 1 | Hides KV cache bugs | Use model's default step count |

### Settings by Purpose

| Purpose | WARMUP | ITERATIONS | Timeout |
|---|---|---|---|
| "Does it crash?" sanity check | 0 | 1 | 120s |
| Quick A/B comparison | 3 | 5 | 300s |
| Proper measurement | default | default | 600s |
| torch.compile first run | default | default | 600s+ |

---

## Profiling: Find Where Time Is Spent

### CRITICAL: Profiling adds 2-5x overhead. Profiled latency ≠ real latency.
Profile tells you WHERE time is spent, not absolute performance. Measure real latency separately.

### torch.profiler (preferred method)

```python
# Warmup OUTSIDE the profiler
for _ in range(5):
    with torch.no_grad():
        output = model(inputs)
torch.cuda.synchronize()

# Profile 3 iterations
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        for _ in range(3):
            output = model(inputs)
            torch.cuda.synchronize()

# Sort by self_cuda_time_total (not cuda_time_total — self_ excludes children)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
prof.export_chrome_trace("trace.json")
```

### Reading the Profile Table

| Kernel Pattern | Category | Action |
|---|---|---|
| `aten::mm` / `aten::linear` / `Cijk_*` | GEMM | If >50%, fuse projections, try TunableOp → gemm-and-linear.md |
| `flash_attn_*` / `mha_fwd*` | Flash Attention | Check if aiter FA is being used |
| `aten::bmm` + `aten::_softmax` | Manual attention | Replace with `F.scaled_dot_product_attention` |
| Many small ops, high call count | Launch overhead | torch.compile + CUDAGraph |
| `aten::copy_` | Dtype/device copies | Fix dtype mismatches at model init |
| CPU total >> CUDA total | CPU bottleneck | Profile preprocessing, data loading |

### Typical Time Distribution (transformer inference)
**GEMM (50-70%) > Attention (15-25%) > Elementwise/Norm (5-10%) > Launch overhead (rest)**

Optimize the largest category first. Don't tune 5% of runtime when 60% is unoptimized.

### Chrome Trace Analysis
```python
import json
with open('trace.json') as f:
    data = json.load(f)
events = [e for e in data['traceEvents'] if e.get('cat') == 'kernel' and 'dur' in e]
events.sort(key=lambda e: -e['dur'])
for e in events[:20]:
    print(f"  {e['dur']/1000:.2f} ms  {e['name'][:80]}")
```

---

## Before Implementing Any Optimization: Calculate First

1. **What is the current bottleneck?** (from profiling — not a guess)
2. **What will this change improve?** (specific kernel, specific operation)
3. **By how much?** (estimate in ms or %)
4. **Is it torch.compile-compatible?** (no graph breaks?)
5. **How will I verify?** (benchmark command)

If you can't answer all 5, profile more or read more code.

### Quick FLOP Estimates (transformer layer, hidden_dim=H, seq_len=S)
- QKV projection: `3 × 2 × S × H²` FLOPs
- Attention scores: `2 × S² × H` FLOPs
- MLP (up + down): `2 × 2 × S × H × 4H` FLOPs
- Total per layer ≈ `24 × S × H²` (GEMM-dominated)

### Common Estimation Mistakes
| Mistake | Reality |
|---|---|
| "Fusing 2 ops saves 50%" | Fusion saves launch overhead (~10μs), not compute time |
| "FP16 is 2x faster than FP32" | Only compute-bound ops. Memory-bound ≈ 1.5x |
| "torch.compile gives 3x" | Only if graph is fully captured. Check for breaks. |
