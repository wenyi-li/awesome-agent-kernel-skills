---
name: flashinfer
description: "FlashInfer — High-performance kernel library for LLM inference with optimized attention, paged KV-cache, FP8/FP4 quantization"
license: MIT
metadata:
  author: Agent Cluster
  tags: [flashinfer, llm, inference, attention, kv-cache, fp8, quantization, cuda, serving]
---

# FlashInfer Skill

High-performance GPU kernel library for Large Language Model inference delivering state-of-the-art performance across diverse GPU architectures with optimized attention, GEMM, and MoE operations.

**Official Sources:**
- [FlashInfer Documentation](https://docs.flashinfer.ai/)
- [GitHub Repository](https://github.com/flashinfer-ai/flashinfer)
- [Official Blog](https://flashinfer.ai/)
- [MLSys 2025 Paper](https://arxiv.org/abs/2501.01005) (Best Paper Award)

## What is FlashInfer?

**Definition:**
> "A library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, PageAttention and LoRA."

**Key Features:**
- **Unified APIs**: Attention, GEMM, MoE with multiple backend implementations
- **Paged & Ragged KV-Cache**: Efficient memory management for dynamic batching
- **Multi-Backend**: FlashAttention-2/3, cuDNN, CUTLASS, TensorRT-LLM
- **Quantization**: FP8 and FP4 for attention, GEMM, MoE operations
- **Production-Ready**: CUDAGraph and torch.compile compatible
- **Wide GPU Support**: SM75 (Turing) through SM121 (Blackwell)

## Quick Start

### Installation

```bash
# Basic installation
pip install flashinfer-python

# With pre-compiled kernels (recommended)
pip install flashinfer-python flashinfer-cubin

# With JIT cache for specific CUDA version
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

**System Requirements:**
- Linux only
- Python 3.10-3.14
- CUDA 12.6, 12.8, 13.0, or 13.1
- GPU: Turing (T4) through Blackwell

### Verify Installation

```bash
flashinfer show-config
```

### Basic Usage

```python
import torch
import flashinfer

# Single decode with paged KV-cache
output = flashinfer.single_decode_with_kv_cache(
    q=query,                    # (num_qo_heads, head_dim)
    kv_data=kv_cache,          # (num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_indices=kv_page_indices,  # (num_pages,)
    kv_indptr=kv_page_indptr,    # (batch_size + 1,)
    kv_last_page_len=last_page_lengths,  # (batch_size,)
)
```

## Core Capabilities

### Attention Operations

**Decode (Token-by-Token Generation):**
```python
# Single request decode
output = flashinfer.single_decode_with_kv_cache(
    q=query,
    kv_data=kv_cache,
    # ... KV-cache parameters
)

# Batch decode with cuDNN backend
output = flashinfer.cudnn_batch_decode_with_kv_cache(
    q=queries,           # (total_num_qo_heads, head_dim)
    kv_data=kv_cache,
    qo_indptr=qo_indptr,  # Query offsets
    kv_indptr=kv_indptr,  # KV offsets
)

# Using wrapper for multi-layer inference
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
wrapper.begin_forward(...)
for layer in model.layers:
    output = wrapper.forward(query)
wrapper.end_forward()
```

**Prefill (Prompt Processing):**
```python
# Single request prefill
output = flashinfer.single_prefill_with_kv_cache(
    q=query,              # (qo_len, num_qo_heads, head_dim)
    kv_data=kv_cache,
    causal=True,          # Causal masking
)

# Batch prefill with ragged KV-cache
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper()
wrapper.begin_forward(
    qo_indptr=qo_indptr,
    kv_indptr=kv_indptr,
)
output = wrapper.forward(query, kv_cache)
```

**Append (Speculative Decoding):**
```python
# Append new tokens to KV-cache
output = flashinfer.single_prefill_with_kv_cache(
    q=new_queries,        # (num_new_tokens, num_qo_heads, head_dim)
    kv_data=kv_cache,
    causal=True,
    append_mode=True,
)
```

### KV-Cache Formats

**1. Paged KV-Cache:**
```python
# Page-based storage (like virtual memory)
kv_cache = torch.empty(
    num_pages, 2, num_kv_heads, page_size, head_dim,
    dtype=torch.float16, device='cuda'
)

# Page table maps sequences to pages
page_indices = torch.tensor([[0, 1, 2], [3, 4, 5]], device='cuda')
```

**2. Ragged Tensor:**
```python
# Variable-length sequences without padding
kv_data = torch.cat([seq1_kv, seq2_kv, seq3_kv], dim=0)
kv_indptr = torch.tensor([0, len(seq1), len(seq1)+len(seq2), ...])
```

**3. Padded Tensor:**
```python
# Standard dense format
kv_cache = torch.zeros(
    batch_size, max_seq_len, 2, num_kv_heads, head_dim
)
```

### GEMM & MoE Operations

**FP8 Matrix Multiplication:**
```python
# FP8 GEMM with groupwise scaling
output = flashinfer.gemm_fp8_nt_groupwise(
    x=input_fp8,           # FP8 input
    w=weight_fp8,          # FP8 weight
    x_scale=input_scale,   # Per-group scales
    w_scale=weight_scale,
    group_size=128,
)
```

**FP4 Quantized GEMM:**
```python
# FP4 matrix multiplication
output = flashinfer.mm_fp4(
    x=input,
    w_q=weight_fp4,        # Quantized to FP4
    scales=scales,
    group_size=64,
)
```

**Mixture of Experts:**
```python
# Fused MoE with FP8 quantization
output = flashinfer.trtllm_fp8_block_scale_moe(
    x=hidden_states,       # (num_tokens, hidden_size)
    w1=expert_weights_w1,  # FP8 weights
    w2=expert_weights_w2,
    topk_weights=routing_weights,  # (num_tokens, topk)
    topk_ids=expert_ids,           # (num_tokens, topk)
    scales_w1=scales_w1,
    scales_w2=scales_w2,
)
```

### Sampling Operations

**Top-K and Top-P Sampling:**
```python
# Top-K sampling
next_token = flashinfer.top_k_sampling_from_probs(
    probs=probs,          # (batch_size, vocab_size)
    top_k=40,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)

# Top-P (nucleus) sampling
next_token = flashinfer.top_p_sampling_from_probs(
    probs=probs,
    top_p=0.9,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)

# Combined Top-K and Top-P
next_token = flashinfer.top_k_top_p_sampling_from_logits(
    logits=logits,
    top_k=40,
    top_p=0.9,
    uniform_samples=torch.rand(batch_size, device='cuda'),
)
```

**Speculative Decoding:**
```python
# Chain speculative sampling
accepted_tokens = flashinfer.chain_speculative_sampling(
    draft_probs=draft_model_probs,    # (batch_size, num_draft_tokens, vocab)
    draft_tokens=draft_tokens,          # (batch_size, num_draft_tokens)
    target_probs=target_model_probs,   # (batch_size, num_draft_tokens+1, vocab)
    uniform_samples=torch.rand(batch_size, num_draft_tokens+1),
)
```

## Advanced Features

### Cascade Inference

Optimize shared prefix scenarios (e.g., document QA):

```python
# Shared prefix attention (stored in SMEM)
shared_output = flashinfer.single_prefill_with_kv_cache(
    q=queries,
    kv_data=shared_prefix_kv,
    # ... shared KV parameters
)

# Unique suffix attention
suffix_output = flashinfer.batch_decode_with_kv_cache(
    q=queries,
    kv_data=suffix_kv,
    # ... suffix KV parameters
)

# Merge attention states
final_output = merge_attention_states(shared_output, suffix_output)
```

**Performance**: Up to 31x speedup vs baseline PageAttention

### Multi-Head Latent Attention (MLA)

For DeepSeek models:

```python
# MLA-specific wrapper
wrapper = flashinfer.BatchMLAPagedAttentionWrapper()
wrapper.begin_forward(...)

# MLA decode
output = flashinfer.trtllm_batch_decode_with_kv_cache_mla(
    q=latent_query,
    kv_data=latent_kv_cache,
    # ... MLA-specific parameters
)
```

### RoPE (Rotary Position Embeddings)

```python
# Apply RoPE in-place
flashinfer.apply_rope_inplace(
    q=query,              # (seq_len, num_heads, head_dim)
    k=key,
    indptr=indptr,
    offsets=position_offsets,
    rotary_dim=head_dim,
)

# Apply RoPE with position IDs
flashinfer.apply_rope_pos_ids(
    q=query,
    k=key,
    pos_ids=position_ids,
    rotary_dim=head_dim,
)
```

### Normalization

```python
# RMSNorm
output = flashinfer.rmsnorm(
    input=hidden_states,
    weight=norm_weight,
    eps=1e-6,
)

# Fused Add + RMSNorm
output = flashinfer.fused_add_rmsnorm(
    input=hidden_states,
    residual=residual,
    weight=norm_weight,
    eps=1e-6,
)
```

## Performance Characteristics

### Benchmarks (vs Compiler Backends)

- **Inter-token latency**: 29-69% reduction
- **Long-context inference**: 28-30% latency reduction
- **Parallel generation**: 13-17% speedup

### Roofline Analysis

**Decode Attention**: O(1) operational intensity (memory-bound)
**Prefill Attention**: O(l_q) operational intensity (compute-bound for long sequences)

### GPU Performance

| GPU | Decode TFLOPS | Prefill TFLOPS |
|-----|---------------|----------------|
| H100 | ~200 | ~800 |
| A100 | ~80 | ~300 |
| RTX 4090 | ~60 | ~250 |

## Integration Examples

### vLLM Integration

```python
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

class FlashInferAttention:
    def __init__(self, num_heads, head_dim):
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper()
    
    def forward(self, query, kv_cache, kv_indptr, kv_indices):
        return self.wrapper.forward(query)
```

### SGLang Integration

```python
import flashinfer

# Initialize wrappers
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()

# Use in forward pass
if is_prefill:
    output = prefill_wrapper.forward(query, kv_cache)
else:
    output = decode_wrapper.forward(query)
```

## Best Practices

1. **Use pre-compiled kernels**: Install `flashinfer-cubin` for faster startup
2. **Choose appropriate KV-cache format**: Paged for batching, ragged for variable lengths
3. **Enable cascade inference**: For shared prefix scenarios
4. **Use FP8 quantization**: 2x speedup with minimal accuracy loss
5. **Leverage wrappers**: For multi-layer models to amortize planning overhead
6. **Profile workload**: Use `flashinfer.bench` for performance analysis

## References

- **[Attention Kernels](references/attention-kernels.md)** - Prefill, decode, append, KV-cache formats
- **[Advanced Techniques](references/advanced-techniques.md)** - Cascade inference, MLA, optimizations
- **[GEMM & MoE](references/gemm-moe.md)** - Matrix operations, quantization, mixture of experts
- **[Sampling](references/sampling.md)** - Top-k, top-p, speculative decoding
- **[Integration](references/integration.md)** - vLLM, SGLang, deployment patterns
- **[Performance](references/performance.md)** - Benchmarks, optimization strategies
