# Reading `torch.profiler` Output

Use `torch.profiler` in two complementary ways:

- the summary table for fast triage
- the trace viewer for overlap, gaps, and sequencing

## Summary Table

Start with:

```python
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

The load-bearing columns are:

| Column | Meaning | Use |
| --- | --- | --- |
| `Self CPU` | CPU time spent in this op only | host bottlenecks, Python overhead, launch overhead |
| `CPU total` | CPU time in the range including children | full `record_function` region cost |
| `Self CUDA` | GPU time spent in this op only | main ranking column for hot kernels |
| `CUDA total` | GPU time including child kernels | useful for labeled ranges |
| `# of Calls` | number of event firings | derive per-call cost |

Important interpretation rules:

- Actual CUDA kernel rows usually have `Self CPU = 0`; the CPU-side launch cost
  is typically attributed to `cuLaunchKernel`.
- A labeled `record_function(...)` range can appear alongside the underlying
  CUDA kernel row. Both are useful; one is your label, the other is the actual
  kernel name.
- Sort by `self_cuda_time_total` for GPU bottlenecks and by
  `self_cpu_time_total` for host bottlenecks.

## Trace Export

Export a trace with:

```python
prof.export_chrome_trace("trace.json")
```

Open it in:

- `chrome://tracing`
- `https://ui.perfetto.dev/`

Use the trace to inspect:

- gaps between kernels
- CPU work that delays GPU launches
- overlap across streams
- unexpected synchronizations
- module loading or JIT compile artifacts that escaped warmup

## What To Look For

### Launch overhead

Symptoms:

- `cuLaunchKernel` has large `Self CPU`
- many tiny kernels with visible gaps between launches
- per-launch CPU time is close to per-kernel GPU time

Likely fix directions:

- increase work per launch
- fuse adjacent kernels
- use CUDA graphs when appropriate

### CPU-bound sections

Symptoms:

- wide CPU regions with little or no concurrent GPU work
- `Self CPU` dominates the summary table

Likely fix directions:

- optimize host preprocessing
- reduce Python overhead
- move repeated work out of the hot path

### Warmup failure

Symptoms:

- the first profiled iteration is much slower
- compile or module-loading effects appear inside the trace

Likely fix directions:

- add more warmup iterations
- synchronize after warmup and before leaving the profile block

### Good overlap

Symptoms:

- multiple CUDA stream tracks stay busy
- host launches keep the device fed without large idle gaps

This usually means the issue is inside the hot kernel rather than in host
dispatch.

## Grouping Options

Use grouping when shape sensitivity or call-site attribution matters:

```python
prof.key_averages(group_by_input_shape=True)
prof.key_averages(group_by_stack_n=5)
```

Notes:

- `group_by_input_shape=True` requires `record_shapes=True`
- stack grouping increases profiler overhead

## Memory Views

When debugging allocation behavior:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    forward_pass()
```

Then export a memory timeline or use PyTorch's memory snapshot tooling for
deeper allocator analysis.

## Common Pitfalls

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| Forgot `ProfilerActivity.CUDA` | CUDA columns are empty | add CUDA activity |
| Forgot `torch.cuda.synchronize()` | tail kernels missing from the trace | synchronize before the context exits |
| Left `record_shapes=True` on for precise benchmarking | measured wall time is inflated | keep it for attribution, disable for final microbench timing |
| Read only kernel names | misclassifies what the workload is doing | inspect both labels and timing columns |
| Assumed trace proves tensor-core usage | wrong instruction-path conclusion | corroborate with generated source and achieved throughput |
