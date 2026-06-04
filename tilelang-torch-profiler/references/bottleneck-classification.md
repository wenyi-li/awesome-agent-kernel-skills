# Bottleneck Classification With `torch.profiler`

This guide classifies a TileLang kernel as:

- IO-bound
- CUDA-core bound
- Tensor-core bound
- launch-overhead-bound
- underperforming but inconclusive from timing alone

`torch.profiler` is the lightweight first pass. It attributes time well, but it
does not expose low-level counters such as occupancy, bank conflicts, or
register pressure. Use it to narrow the problem before escalating.

## Core Method

Use a roofline-style workflow with source corroboration:

1. measure per-call GPU time
2. convert time into achieved throughput
3. compare that throughput with hardware peaks
4. corroborate with generated source when the instruction path matters

## Step 1: Measure Per-Call GPU Time

For a labeled region:

```python
def gpu_us_per_call(prof, label):
    for evt in prof.key_averages():
        if evt.key == label and evt.self_device_time_total > 0 and evt.count:
            return evt.self_device_time_total / evt.count
    return None
```

Why filter on `self_device_time_total > 0`:

- a `record_function` range can show up as a CPU-side row and a GPU-attributed
  row
- the GPU-attributed row is the one that carries kernel time

## Step 2: Convert Time Into Throughput

### GEMM-like work

```python
flops = 2 * M * N * K
tflops = flops / (us_per_call * 1e-6) / 1e12
```

### Memory-dominated work

```python
bytes_moved = bytes_read + bytes_written
gb_per_s = bytes_moved / (us_per_call * 1e-6) / 1e9
```

Count every distinct global-memory byte once per kernel invocation.

## Step 3: Compare Against Peaks

Use vendor specs or known hardware limits for the target GPU.

Interpretation:

- achieved bandwidth near HBM peak -> IO-bound
- achieved TFLOPS near scalar peak -> CUDA-core bound
- achieved TFLOPS near tensor-core peak -> Tensor-core bound
- far below every peak with one dominant kernel -> timing alone is not enough
- far below every peak with many tiny kernels -> likely launch-overhead-bound

## Step 4: Corroborate With Source

TileLang often emits tensor-core behavior through included helper headers, so
the top-level CUDA source may not literally contain `wgmma` or `mma.sync`.

Look for strong indicators such as:

- TileLang MMA template includes
- `CUtensorMap`
- `mbarrier`
- async bulk copy or related async-path markers

Use a quick inspection pattern:

```python
src = kernel.get_kernel_source()
print("CUtensorMap" in src)
print("mbarrier" in src)
print("mma.h" in src)
```

Interpret with care:

- tensor-core indicators present and achieved TFLOPS is healthy ->
  Tensor-core-bound or Tensor-core-efficient
- tensor-core indicators present but throughput is poor -> likely deeper kernel
  issue; timing alone cannot prove why
- no tensor-core indicators and throughput aligns with scalar peak ->
  CUDA-core bound

If a kernel should be tensor-core based but performs far below expectation,
profiler timing has probably reached its limit.

## Class Signatures

### IO-bound

Typical signs:

- one memory-heavy kernel dominates `Self CUDA`
- achieved GB/s is close to hardware memory bandwidth
- source shows ordinary loads, stores, and light arithmetic

Typical next action:

- reduce memory traffic
- fuse producers and consumers
- improve data reuse rather than chasing arithmetic changes

### CUDA-core bound

Typical signs:

- one compute kernel dominates `Self CUDA`
- achieved TFLOPS is much lower than tensor-core peak but near scalar peak
- source looks like scalar arithmetic rather than MMA-heavy lowering

Typical next action:

- decide whether this kernel should stay scalar
- if not, revisit the TileLang formulation and lowering path

### Tensor-core bound

Typical signs:

- one GEMM-like kernel dominates `Self CUDA`
- achieved TFLOPS is in the expected tensor-core regime for the hardware
- source contains TileLang tensor-core path indicators

Typical next action:

- tune tile sizes and staging only if more headroom matters
- otherwise focus upstream or downstream

### Launch-overhead-bound

Typical signs:

- many small kernels
- `cuLaunchKernel` CPU time is close to total kernel GPU time
- the trace shows repeated small gaps between short kernels

Typical next action:

- increase work per launch
- fuse kernels
- use CUDA graphs when appropriate

### Inconclusive Underperformance

Typical signs:

- one kernel dominates runtime
- throughput is far below any plausible roofline
- source suggests the "right" lowering path, but timing is still poor

This usually means the real cause lives below what `torch.profiler` exposes:

- register pressure or spills
- occupancy limits
- bank conflicts
- pipeline stalls

At that point, say explicitly that the profiler established attribution, not
root-cause proof.

## Escalation Rules

Recommend deeper GPU analysis when:

- tensor-core indicators exist but TFLOPS is still unexpectedly low
- timing is far below every peak and launch overhead does not explain it
- the user asks about occupancy, spills, bank conflicts, or warp stalls

Use this skill to justify the escalation, not to guess beyond the evidence.
