# FlashInfer Performance Reference

Sources:
- [FlashInfer Paper](https://arxiv.org/abs/2501.01005)
- [Official Blog](https://flashinfer.ai/)

## Benchmark Results

### vs Compiler Backends

**MLSys 2025 Paper Results**:
- **Inter-token latency**: 29-69% reduction
- **Long-context inference**: 28-30% latency reduction  
- **Parallel generation**: 13-17% speedup

### vs PageAttention (vLLM)

**Cascade Inference (H100)**:
- Shared prefix (4K tokens), batch size 64: **31x speedup**
- Shared prefix (2K tokens), batch size 32: **18x speedup**
- Standard batching: **26x speedup**

### GPU Performance Characteristics

| GPU | Architecture | Decode TFLOPS | Prefill TFLOPS | Memory BW (GB/s) |
|-----|--------------|---------------|----------------|------------------|
| H100 SXM | Hopper | ~200 | ~800 | 3350 |
| A100 80GB | Ampere | ~80 | ~300 | 2039 |
| RTX 4090 | Ada | ~60 | ~250 | 1008 |
| L40S | Ada | ~55 | ~240 | 864 |
| RTX 6000 Ada | Ada | ~50 | ~220 | 960 |

## Roofline Analysis

### Operational Intensity

**Decode Attention**:
- Intensity: O(1) per token
- Memory-bound on all GPUs
- Batch size doesn't improve intensity
- Limited by memory bandwidth

**Prefill Attention**:
- Intensity: O(l_q) where l_q = query length
- Compute-bound for sequences >512 tokens
- Tensor cores fully utilized
- Limited by FLOPS

### Performance Model

```python
def predict_decode_latency(batch_size, seq_len, hidden_size, memory_bw):
    # Bytes loaded per token
    kv_bytes = 2 * seq_len * hidden_size * 2  # K and V, FP16
    
    # Total bytes for batch
    total_bytes = batch_size * kv_bytes
    
    # Latency (memory-bound)
    latency_ms = (total_bytes / memory_bw) * 1000
    
    return latency_ms

# Example: H100
latency = predict_decode_latency(
    batch_size=64,
    seq_len=2048,
    hidden_size=4096,
    memory_bw=3350e9,  # 3.35 TB/s
)
# ~3.1 ms
```

## Optimization Strategies

### Split-K for Decode

**Purpose**: Parallelize KV-cache processing

**How it works**:
- Divide sequence into chunks
- Process chunks on different SMs
- Reduce partial results

**Speedup**: 2-3x for long sequences (>2K tokens)

```python
# Automatically enabled in FlashInfer
output = flashinfer.single_decode_with_kv_cache(
    q=query,
    kv_data=kv_cache,
    # Split-K applied when beneficial
)
```

### Grouped-Query Attention

**Memory Savings**:
- MHA (Multi-Head Attention): 100% KV-cache
- GQA (num_qo_heads=32, num_kv_heads=8): 25% KV-cache
- MQA (num_kv_heads=1): 3% KV-cache

**Performance**:
- GQA: 2-3x speedup via tensor cores
- MQA: 3-5x speedup

### FP8 Quantization

**Memory Bandwidth**:
- FP16: 100% bandwidth
- FP8: 50% bandwidth → **2x speedup**

**Accuracy Impact**:
- Perplexity increase: <0.1 points
- Minimal quality degradation

```python
# FP8 decode
output = flashinfer.single_decode_with_kv_cache_fp8(
    q=query,
    kv_data=kv_cache_fp8,
    kv_scale=kv_scale,
)
```

### Fused RoPE

**Traditional**:
1. Apply RoPE kernel: 0.5 ms
2. Attention kernel: 3.0 ms
3. Total: 3.5 ms

**FlashInfer Fused**:
1. Fused RoPE + Attention: 3.1 ms
2. Overhead: ~3% vs separate kernels

## Profiling Tools

### Benchmarking API

```python
import flashinfer.bench

# Benchmark decode attention
results = flashinfer.bench.bench_decode_attention(
    batch_sizes=[1, 8, 16, 32, 64],
    seq_lengths=[512, 1024, 2048, 4096],
    num_heads=32,
    head_dim=128,
    num_kv_heads=8,
)

# Print results
for result in results:
    print(f"Batch={result.batch_size}, Seq={result.seq_len}: "
          f"{result.latency_ms:.2f} ms, {result.throughput_tflops:.1f} TFLOPS")
```

### FLOPS Measurement

```python
# Measure achieved FLOPS
flops, time_ms = flashinfer.bench.measure_flops(
    kernel_fn=lambda: flashinfer.single_decode_with_kv_cache(...),
    num_warmup=10,
    num_iterations=100,
)

print(f"Achieved: {flops / 1e12:.1f} TFLOPS")
print(f"Latency: {time_ms:.2f} ms")
```

### Memory Profiling

```python
import torch

# Track memory usage
torch.cuda.reset_peak_memory_stats()

output = flashinfer.batch_decode_with_kv_cache(...)

peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory: {peak_memory_mb:.1f} MB")
```

## Tuning Parameters

### Page Size

**Tradeoff**:
- Smaller pages (8): Less fragmentation, more overhead
- Larger pages (32): More fragmentation, less overhead
- **Recommended**: 16 (good balance)

```python
# Configure page size
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
wrapper.begin_forward(
    page_size=16,  # Optimal for most workloads
    # ...
)
```

### Workspace Size

**Purpose**: Scratch memory for intermediate results

```python
# Allocate workspace
workspace_size = 128 * 1024 * 1024  # 128 MB
workspace = torch.empty(
    workspace_size,
    dtype=torch.uint8,
    device='cuda'
)

# Reuse across layers
for layer in range(num_layers):
    output = wrapper.forward(
        query,
        kv_cache,
        workspace_buffer=workspace,
    )
```

### Backend Selection

**cuDNN**: Best for batch decode on H100/A100
**TensorRT-LLM**: Best for production, FP8
**FlashAttention-2**: Best for prefill, long contexts

```python
# Explicitly select backend
output = flashinfer.cudnn_batch_decode_with_kv_cache(...)  # cuDNN
output = flashinfer.trtllm_batch_decode_with_kv_cache(...)  # TRT-LLM
```

## Performance Patterns

### Continuous Batching

**Problem**: Static batching wastes resources

**Solution**: Dynamic batching with FlashInfer

```python
class ContinuousBatcher:
    def __init__(self):
        self.active_requests = []
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
    
    def add_request(self, request):
        self.active_requests.append(request)
    
    def remove_request(self, request_id):
        self.active_requests = [r for r in self.active_requests 
                               if r.id != request_id]
    
    def step(self):
        if not self.active_requests:
            return
        
        # Prepare batch
        queries = torch.cat([r.get_query() for r in self.active_requests])
        
        # Process batch
        self.wrapper.begin_forward(
            indptr=self.get_indptr(),
            indices=self.get_indices(),
            # ...
        )
        
        outputs = self.wrapper.forward(queries, self.kv_cache)
        self.wrapper.end_forward()
        
        # Distribute outputs
        for request, output in zip(self.active_requests, outputs):
            request.update(output)
```

### Mixed Batching (POD-Attention)

**Overlap prefill and decode**:

```python
# Concurrent streams
prefill_stream = torch.cuda.Stream()
decode_stream = torch.cuda.Stream()

# Prefill (compute-bound)
with torch.cuda.stream(prefill_stream):
    prefill_output = prefill_wrapper.forward(...)

# Decode (memory-bound) - overlaps with prefill
with torch.cuda.stream(decode_stream):
    decode_output = decode_wrapper.forward(...)

# Synchronize
prefill_stream.synchronize()
decode_stream.synchronize()
```

## Bottleneck Analysis

### Identify Bottleneck

```python
import time

# Measure components
start = time.perf_counter()
query = model.get_query(tokens)
query_time = time.perf_counter() - start

start = time.perf_counter()
output = flashinfer.single_decode_with_kv_cache(...)
attention_time = time.perf_counter() - start

start = time.perf_counter()
logits = model.lm_head(output)
proj_time = time.perf_counter() - start

print(f"Query projection: {query_time*1000:.2f} ms")
print(f"Attention: {attention_time*1000:.2f} ms")
print(f"Output projection: {proj_time*1000:.2f} ms")
```

### Common Bottlenecks

**1. Memory Bandwidth (Decode)**:
- Symptom: Low GPU utilization (<30%)
- Solution: Use FP8, larger batch sizes

**2. Compute (Prefill)**:
- Symptom: High GPU utilization (>90%)
- Solution: Reduce sequence length, use chunking

**3. KV-Cache Management**:
- Symptom: High page allocation overhead
- Solution: Increase page size, use page pool

## Best Practices

1. **Profile first**: Identify actual bottlenecks
2. **Use FP8 for decode**: 2x memory bandwidth
3. **Batch aggressively**: Amortize overhead
4. **Reuse wrappers**: Avoid re-planning
5. **Enable cascade inference**: For shared prefixes
6. **Monitor KV-cache memory**: Track fragmentation
7. **Use CUDAGraph**: For latency-critical serving
8. **Choose right page size**: 16 is usually optimal

## Performance Checklist

- [ ] Installed `flashinfer-cubin` for pre-compiled kernels
- [ ] Using appropriate batch sizes (>8 for throughput)
- [ ] Enabled FP8 quantization for KV-cache
- [ ] Configured optimal page size (16)
- [ ] Allocated workspace buffers (reuse across layers)
- [ ] Using wrappers for multi-layer models
- [ ] Profiled to identify bottlenecks
- [ ] Enabled cascade inference if applicable
- [ ] Monitoring KV-cache memory usage
- [ ] Using CUDAGraph for static configurations
