# FlashInfer Advanced Techniques Reference

Sources:
- [Cascade Inference](https://flashinfer.ai/2024/02/02/cascade-inference.html)
- [FlashInfer Paper](https://arxiv.org/abs/2501.01005)

## Cascade Inference

### Problem Statement

Common LLM serving pattern: Multiple requests share a long prompt prefix (e.g., document QA with same document)

**Traditional approach issues**:
- Redundant KV-cache storage
- Wasted memory bandwidth reading shared prefix repeatedly
- Poor cache locality

### Solution: Cascade Attention

**Core Idea**: Decouple attention for shared prefix and unique suffixes

**Two-Stage Attention**:
1. **Multi-query attention** for shared prefix (stored in GPU SMEM)
2. **Batch decode attention** for unique suffixes
3. **Merge operator** combines attention states

### Implementation

```python
def cascade_attention(queries, shared_kv, suffix_kv):
    # Stage 1: Shared prefix attention (in SMEM)
    shared_output, shared_lse = multi_query_attention(
        q=queries,
        kv=shared_kv,  # Loaded into shared memory once
    )
    
    # Stage 2: Suffix attention  
    suffix_output, suffix_lse = batch_decode_attention(
        q=queries,
        kv=suffix_kv,  # Per-request unique KV
    )
    
    # Stage 3: Merge using recursive attention
    final_output = merge_attention_states(
        output1=shared_output,
        lse1=shared_lse,
        output2=suffix_output,
        lse2=suffix_lse,
    )
    
    return final_output

# Merge formula:
# output = (exp(lse1) * output1 + exp(lse2) * output2) / (exp(lse1) + exp(lse2))
```

### Performance Results

**Benchmark (H100 SXM 80GB)**:
- **31x speedup** vs vLLM PageAttention
- **26x speedup** vs FlashInfer batch decode
- Optimal for: Long shared prompts (>1K tokens), large batch sizes (>64)

### Use Cases

1. **Document QA**: Multiple users query same document
2. **Code completion**: Shared codebase context
3. **Multi-turn chat**: Shared system prompt + history
4. **RAG systems**: Shared retrieved documents

## Multi-Head Latent Attention (MLA)

### Overview

Specialized attention for DeepSeek models using latent compression

**Key Features**:
- Compressed KV representation
- Reduced KV-cache memory
- Maintains model quality

### API

```python
# MLA-specific wrapper
wrapper = flashinfer.BatchMLAPagedAttentionWrapper()

wrapper.begin_forward(
    indptr=kv_indptr,
    indices=kv_indices,
    last_page_len=last_page_len,
    num_qo_heads=num_heads,
    num_kv_heads=num_kv_heads,
    latent_dim=latent_dim,  # MLA-specific
    head_dim=head_dim,
    page_size=16,
)

# MLA decode
output = flashinfer.trtllm_batch_decode_with_kv_cache_mla(
    q=latent_query,          # (num_heads, latent_dim)
    kv_data=latent_kv_cache, # Compressed KV
    # ... page table parameters
)

wrapper.end_forward()
```

### Benefits

- 30-40% KV-cache memory reduction
- Comparable performance to standard attention
- Production-ready in DeepSeek-V2/V3

## XQA (Extended Query Attention)

### Purpose

Optimized kernels for specific hardware architectures

**Variants**:
- `xqa()`: Standard extended query attention
- `xqa_mla()`: Extended query with MLA

```python
# XQA kernel
output = flashinfer.xqa(
    q=query,
    kv_data=kv_cache,
    # Hardware-specific optimizations
)
```

## POD-Attention

### Overview

**POD**: Prefill-Overlap-Decode

**Purpose**: Mixed batching of prefill and decode requests

**Challenge**: Different operational intensities
- Prefill: Compute-bound (high FLOPS)
- Decode: Memory-bound (low FLOPS)

### Strategy

```python
# Overlap prefill and decode computation
def pod_attention_step(prefill_reqs, decode_reqs):
    # Launch decode kernels (memory-bound)
    decode_stream = torch.cuda.Stream()
    with torch.cuda.stream(decode_stream):
        decode_outputs = batch_decode(decode_reqs)
    
    # Launch prefill kernels (compute-bound)
    # Overlaps with decode memory transfers
    prefill_stream = torch.cuda.Stream()
    with torch.cuda.stream(prefill_stream):
        prefill_outputs = batch_prefill(prefill_reqs)
    
    # Synchronize
    decode_stream.synchronize()
    prefill_stream.synchronize()
    
    return prefill_outputs, decode_outputs
```

## Just-In-Time Compilation

### Customizable Templates

FlashInfer uses JIT compilation for kernel customization

**Benefits**:
- Adapt to specific head dimensions
- Optimize for deployment settings
- No performance penalty

**Configuration**:
```python
# Kernel compiled for specific parameters
@flashinfer.jit
def custom_attention_kernel(head_dim: int, page_size: int):
    # Template specialized at compile time
    pass

# First call triggers compilation
output = custom_attention_kernel(head_dim=128, page_size=16)

# Subsequent calls use cached kernel
output = custom_attention_kernel(head_dim=128, page_size=16)
```

### Cache Management

```bash
# View compilation cache
flashinfer show-config

# Clear JIT cache
rm -rf ~/.cache/flashinfer/*

# Use pre-compiled cache
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

## Dynamic Scheduling

### Load-Balanced Scheduling

**Problem**: Variable request lengths cause load imbalance

**Solution**: Work-stealing scheduler

```python
# FlashInfer's dynamic scheduler
# Automatically balances work across SMs
output = flashinfer.batch_decode_with_kv_cache(
    q=queries,
    kv_data=kv_cache,
    # Scheduler handles load balancing
)
```

### CUDAGraph Compatibility

**For static batch sizes**:
```python
# Capture CUDAGraph
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    wrapper.begin_forward(...)
    output = wrapper.forward(query, kv_cache)
    wrapper.end_forward()

# Replay graph (low latency)
for step in range(num_steps):
    graph.replay()
```

## Memory Optimization

### Workspace Buffers

Reuse scratch memory across layers:

```python
# Allocate workspace once
workspace = torch.empty(
    workspace_size,
    dtype=torch.uint8,
    device='cuda'
)

# Reuse for all layers
for layer in range(num_layers):
    wrapper.forward(
        query,
        kv_cache,
        workspace_buffer=workspace,  # Shared scratch space
    )
```

### Page Recycling

```python
class PagePool:
    def __init__(self, num_pages, page_size):
        self.free_pages = list(range(num_pages))
        self.allocated_pages = {}
    
    def allocate(self, request_id, num_pages):
        if len(self.free_pages) < num_pages:
            # Evict LRU request
            evict_request = self.find_lru_request()
            self.free(evict_request)
        
        pages = [self.free_pages.pop() for _ in range(num_pages)]
        self.allocated_pages[request_id] = pages
        return pages
    
    def free(self, request_id):
        pages = self.allocated_pages.pop(request_id)
        self.free_pages.extend(pages)
```

## Best Practices

1. **Use cascade inference** for shared prefix scenarios
2. **Enable MLA** for DeepSeek models  
3. **Leverage POD-Attention** for mixed batching
4. **Pre-compile kernels** with flashinfer-jit-cache
5. **Reuse workspace buffers** across layers
6. **Use CUDAGraph** for static configurations
7. **Monitor KV-cache memory** with page recycling
