# FlashInfer Attention Kernels Reference

Sources:
- [Attention API](https://docs.flashinfer.ai/api/attention.html)
- [Accelerating Self-Attentions](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- [KV-Cache Layout Tutorial](https://docs.flashinfer.ai/)

## Three Attention Stages

### Prefill Stage

**Purpose**: Process entire input sequences during initial prompt encoding

**Characteristics**:
- Compute attention between all query-key pairs
- Operational intensity: O(l_q) where l_q is query length
- Compute-bound for long sequences
- High parallelism opportunity

**API**:
```python
import flashinfer

# Single request prefill
output = flashinfer.single_prefill_with_kv_cache(
    q=query,              # (qo_len, num_qo_heads, head_dim)
    k=key,                # (kv_len, num_kv_heads, head_dim)
    v=value,              # (kv_len, num_kv_heads, head_dim)
    causal=True,          # Enable causal masking
    sm_scale=1.0 / math.sqrt(head_dim),
    rope_scale=1.0,
    rope_theta=10000.0,
)

# With return LSE (log-sum-exp) for numerical stability
output, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
    q=query,
    k=key,
    v=value,
    causal=True,
)
```

**Batch Prefill**:
```python
# Using ragged KV-cache (no padding)
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper()

# Plan the computation
wrapper.begin_forward(
    qo_indptr=qo_indptr,  # Query offsets: [0, len(q1), len(q1)+len(q2), ...]
    kv_indptr=kv_indptr,  # KV offsets
    num_qo_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
)

# Execute for each layer
for layer in range(num_layers):
    output = wrapper.forward(query, kv_cache)

wrapper.end_forward()
```

### Decode Stage

**Purpose**: Generate tokens one at a time during autoregressive generation

**Characteristics**:
- Single query attends to full KV-cache
- Operational intensity: O(1) - memory-bound
- Batch size doesn't improve intensity
- Critical for latency

**Single Decode**:
```python
# Decode with paged KV-cache
output = flashinfer.single_decode_with_kv_cache(
    q=query,                    # (num_qo_heads, head_dim)
    kv_data=kv_cache,          # (num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_indices=page_indices,    # (num_pages,)
    kv_indptr=page_indptr,      # (batch_size + 1,)
    kv_last_page_len=last_page_len,  # (batch_size,)
    causal=True,
)
```

**Batch Decode**:
```python
# cuDNN backend
output = flashinfer.cudnn_batch_decode_with_kv_cache(
    q=queries,                  # (total_num_qo_heads, head_dim)
    kv_data=kv_cache,
    qo_indptr=qo_indptr,       # Query head offsets
    paged_kv_indptr=kv_indptr,
    paged_kv_indices=kv_indices,
    paged_kv_last_page_len=last_page_len,
    causal=True,
)

# TensorRT-LLM backend
output = flashinfer.trtllm_batch_decode_with_kv_cache(
    q=queries,
    kv_data=kv_cache,
    # ... same parameters
)

# Using wrapper (recommended for multi-layer)
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()

wrapper.begin_forward(
    indptr=kv_indptr,
    indices=kv_indices,
    last_page_len=last_page_len,
    num_qo_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=16,
)

for layer in range(num_layers):
    output = wrapper.forward(query, kv_cache)

wrapper.end_forward()
```

### Append Stage

**Purpose**: Add new tokens to KV-cache (for speculative decoding)

**Characteristics**:
- Multiple new queries attend to extended KV-cache
- Similar to prefill but appends to existing cache
- Used in speculative decoding verification

**API**:
```python
# Append tokens to existing KV-cache
output = flashinfer.single_prefill_with_kv_cache(
    q=new_queries,        # (num_new_tokens, num_qo_heads, head_dim)
    kv_data=kv_cache,
    causal=True,
    append_mode=True,     # Enable append mode
)
```

## KV-Cache Formats

### 1. Paged KV-Cache

**Concept**: Divide KV-cache into fixed-size pages (inspired by virtual memory)

**Benefits**:
- Reduced memory fragmentation
- Efficient for dynamic batching
- Easy memory sharing across requests

**Layout (NHD)**:
```python
kv_data = torch.empty(
    num_pages,        # Total number of pages
    2,                # K and V
    num_kv_heads,     # Number of KV heads
    page_size,        # Elements per page (typically 16)
    head_dim,         # Head dimension
    dtype=torch.float16,
    device='cuda'
)
```

**Layout (HND)**:
```python
kv_data = torch.empty(
    num_pages,
    2,                # K and V
    page_size,
    num_kv_heads,
    head_dim,
    dtype=torch.float16,
    device='cuda'
)
```

**Page Table**:
```python
# Maps sequences to pages
page_indices = torch.tensor([
    [0, 1, 2],    # Sequence 0 uses pages 0, 1, 2
    [3, 4, 5],    # Sequence 1 uses pages 3, 4, 5
    [6, 7, -1],   # Sequence 2 uses pages 6, 7 (shorter)
], device='cuda')

# Indptr for ragged page table
page_indptr = torch.tensor([0, 3, 6, 8], device='cuda')

# Last page lengths (partial pages)
last_page_len = torch.tensor([10, 16, 5], device='cuda')
```

### 2. Ragged Tensor

**Concept**: Variable-length sequences concatenated without padding

**Benefits**:
- No wasted memory on padding
- Efficient for variable-length batches
- Simple memory layout

**Format**:
```python
# Concatenate all KV sequences
kv_data = torch.cat([
    seq0_kv,  # Shape: (len0, 2, num_kv_heads, head_dim)
    seq1_kv,  # Shape: (len1, 2, num_kv_heads, head_dim)
    seq2_kv,  # Shape: (len2, 2, num_kv_heads, head_dim)
], dim=0)

# Indptr marks sequence boundaries
kv_indptr = torch.tensor([
    0,
    len0,
    len0 + len1,
    len0 + len1 + len2
], device='cuda')
```

### 3. Padded Tensor

**Concept**: Standard dense tensor with padding to max length

**Format**:
```python
kv_data = torch.zeros(
    batch_size,
    max_seq_len,
    2,                # K and V
    num_kv_heads,
    head_dim,
    dtype=torch.float16,
    device='cuda'
)
```

## Backend Selection

FlashInfer automatically selects the best backend based on workload:

### cuDNN Backend

**When to use**:
- Batch decode on H100/A100
- FP16/BF16 data types
- Standard attention patterns

**Features**:
- Highly optimized for NVIDIA GPUs
- Good for medium batch sizes
- CUDAGraph compatible

### TensorRT-LLM Backend

**When to use**:
- Production deployment
- Mixed precision (FP8)
- Large batch sizes

**Features**:
- Optimal for throughput
- FP8 quantization support
- Integration with TensorRT

### FlashAttention-2 Backend

**When to use**:
- Prefill stage
- Long sequences
- Research/prototyping

**Features**:
- Memory-efficient
- High performance on long contexts
- Supports various mask patterns

## Grouped-Query Attention (GQA)

**Concept**: Multiple query heads share KV heads

**Benefits**:
- Reduced KV-cache memory
- 2-3x speedup via tensor core utilization
- Common in modern LLMs (Llama, Mistral)

**Implementation**:
```python
# Automatically handled by FlashInfer
num_qo_heads = 32
num_kv_heads = 8  # GQA with 4 queries per KV head

output = flashinfer.single_decode_with_kv_cache(
    q=query,              # (num_qo_heads, head_dim)
    kv_data=kv_cache,    # (num_pages, 2, num_kv_heads, page_size, head_dim)
    # FlashInfer handles GQA automatically
)
```

## Multi-Query Attention (MQA)

**Concept**: All query heads share single KV head

**Implementation**:
```python
num_qo_heads = 32
num_kv_heads = 1  # MQA

output = flashinfer.single_decode_with_kv_cache(
    q=query,
    kv_data=kv_cache,
    # Works seamlessly
)
```

## Sparse Attention

**Block-Sparse Attention**:
```python
# Define block sparse mask
block_sparse_mask = torch.tensor([
    [1, 0, 0, 1],  # Attend to blocks 0 and 3
    [1, 1, 0, 1],
    [1, 1, 1, 1],
], device='cuda')

# Apply sparse attention
output = flashinfer.block_sparse_attention(
    q=query,
    k=key,
    v=value,
    block_sparse_mask=block_sparse_mask,
    block_size=64,
)
```

## Performance Optimization

### Split-K Technique

**Purpose**: Parallelize KV-cache processing across SMs

**How it works**:
- Divide KV dimension into chunks
- Process chunks in parallel on different SMs
- Reduce results

**Benefit**: 2-3x speedup for long sequences in decode

### Fused RoPE

**Purpose**: Apply rotary embeddings on-the-fly

**Benefit**: Negligible overhead vs separate RoPE kernel

```python
output = flashinfer.single_decode_with_kv_cache(
    q=query,
    kv_data=kv_cache,
    rope_scale=1.0,
    rope_theta=10000.0,
    # RoPE applied automatically
)
```

### FP8 Quantization

**Purpose**: Reduce memory bandwidth

**Benefit**: ~2x speedup with minimal accuracy loss

```python
# Quantize KV-cache to FP8
kv_cache_fp8 = kv_cache.to(torch.float8_e4m3fn)
kv_scale = compute_scale(kv_cache)

output = flashinfer.single_decode_with_kv_cache_fp8(
    q=query,
    kv_data=kv_cache_fp8,
    kv_scale=kv_scale,
)
```

## Common Patterns

### Single-Request Serving

```python
# Simple single-request pattern
def serve_single_request(prompt_tokens, model):
    # Prefill
    kv_cache = model.init_kv_cache()
    output = flashinfer.single_prefill_with_kv_cache(
        q=model.get_query(prompt_tokens),
        kv_data=kv_cache,
        causal=True,
    )
    
    # Decode loop
    for _ in range(max_new_tokens):
        next_token = sample(output)
        output = flashinfer.single_decode_with_kv_cache(
            q=model.get_query(next_token),
            kv_data=kv_cache,
        )
        yield next_token
```

### Continuous Batching

```python
# Continuous batching with paged KV-cache
class ContinuousBatcher:
    def __init__(self):
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()
        self.page_manager = PageManager()
    
    def step(self, active_requests):
        # Separate prefill and decode requests
        prefill_reqs = [r for r in active_requests if r.is_prefill]
        decode_reqs = [r for r in active_requests if not r.is_prefill]
        
        # Process prefill
        if prefill_reqs:
            self.prefill_wrapper.begin_forward(...)
            prefill_outputs = self.prefill_wrapper.forward(...)
            self.prefill_wrapper.end_forward()
        
        # Process decode
        if decode_reqs:
            self.decode_wrapper.begin_forward(...)
            decode_outputs = self.decode_wrapper.forward(...)
            self.decode_wrapper.end_forward()
        
        return prefill_outputs + decode_outputs
```
