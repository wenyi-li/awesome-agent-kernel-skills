# FlashInfer GEMM & MoE Reference

Sources:
- [GEMM API](https://docs.flashinfer.ai/api/gemm.html)
- [MoE API](https://docs.flashinfer.ai/api/fused_moe.html)

## GEMM Operations

### BF16 Matrix Multiplication

```python
# Standard matrix multiply (bf16)
output = flashinfer.mm_bf16(
    x=input,              # (M, K)
    w=weight,             # (K, N)
)

# Batch matrix multiply
output = flashinfer.bmm_bf16(
    x=input,              # (batch, M, K)
    w=weight,             # (batch, K, N)
)
```

### FP8 Quantized GEMM

**Per-Tensor Scaling**:
```python
output = flashinfer.bmm_fp8(
    x=input_fp8,          # (batch, M, K) in FP8
    w=weight_fp8,         # (batch, K, N) in FP8
    x_scale=input_scale,   # Scalar scale
    w_scale=weight_scale,  # Scalar scale
)
```

**Groupwise Scaling**:
```python
output = flashinfer.gemm_fp8_nt_groupwise(
    x=input_fp8,          # (M, K) in FP8
    w=weight_fp8,         # (N, K) in FP8 (note: NT layout)
    x_scale=input_scale,   # (M // group_size,)
    w_scale=weight_scale,  # (N // group_size,)
    group_size=128,        # Elements per group
)
```

**Grouped GEMM (for MoE)**:
```python
# Process multiple independent GEMMs
output = flashinfer.group_gemm_fp8_nt_groupwise(
    x=inputs,             # List of input tensors
    w=weights,            # List of weight tensors
    x_scale=input_scales,
    w_scale=weight_scales,
    group_size=128,
)
```

### FP4 Quantized GEMM

**Purpose**: 4-bit weights for extreme compression

```python
output = flashinfer.mm_fp4(
    x=input,              # (M, K) in FP16/BF16
    w_q=weight_fp4,       # (N, K // 2) packed FP4
    scales=scales,        # (N, K // group_size)
    zeros=zeros,          # (N, K // group_size) optional
    group_size=64,
)
```

**Mixed Precision FP8-FP4**:
```python
output = flashinfer.group_gemm_mxfp4_nt_groupwise(
    x=inputs_fp8,         # FP8 activations
    w=weights_fp4,        # FP4 weights
    x_scale=x_scales,
    w_scale=w_scales,
    group_size=64,
)
```

### Segment GEMM

**Purpose**: Variable-length sequence batches

```python
wrapper = flashinfer.SegmentGEMMWrapper()

# Plan segmented computation
wrapper.plan(
    seg_indptr=seg_indptr,    # Segment boundaries
    weight_indices=None,       # Optional weight selection
    batch_size=batch_size,
    d_in=hidden_size,
    d_out=output_size,
)

# Execute across all segments
output = wrapper.run(
    x=inputs,                  # Concatenated inputs
    weights=weights,           # Shared or per-segment weights
    workspace_buffer=workspace,
)
```

## Mixture of Experts (MoE)

### CUTLASS Fused MoE

**Basic Usage**:
```python
output = flashinfer.cutlass_fused_moe(
    x=hidden_states,           # (num_tokens, hidden_size)
    w1=expert_w1,              # (num_experts, inter_size, hidden_size)
    w2=expert_w2,              # (num_experts, hidden_size, inter_size)
    topk_weights=routing_weights,  # (num_tokens, topk)
    topk_ids=expert_ids,           # (num_tokens, topk)
    renormalize=True,              # Normalize routing weights
)
```

**With Activation Function**:
```python
# SwiGLU activation (Llama/Mistral)
output = flashinfer.cutlass_fused_moe(
    x=hidden_states,
    w1=expert_gate_up,         # Combined gate + up projection
    w2=expert_down,
    topk_weights=routing_weights,
    topk_ids=expert_ids,
    activation="swiglu",
)

# GELU activation
output = flashinfer.cutlass_fused_moe(
    x=hidden_states,
    w1=expert_w1,
    w2=expert_w2,
    topk_weights=routing_weights,
    topk_ids=expert_ids,
    activation="gelu",
)
```

### TensorRT-LLM MoE

**FP8 Block-Scaled MoE**:
```python
output = flashinfer.trtllm_fp8_block_scale_moe(
    x=hidden_states,           # (num_tokens, hidden_size)
    w1=expert_w1_fp8,          # FP8 quantized
    w2=expert_w2_fp8,
    topk_weights=routing_weights,
    topk_ids=expert_ids,
    scales_w1=w1_scales,       # Block-wise scales
    scales_w2=w2_scales,
    block_size=128,
)
```

**FP4 Block-Scaled MoE**:
```python
output = flashinfer.trtllm_fp4_block_scale_moe(
    x=hidden_states,
    w1=expert_w1_fp4,          # FP4 quantized (4-bit)
    w2=expert_w2_fp4,
    topk_weights=routing_weights,
    topk_ids=expert_ids,
    scales_w1=w1_scales,
    scales_w2=w2_scales,
    block_size=64,
)
```

### DeepSeek-V3 Routing

**Multiple Routing Strategies**:
```python
# DeepSeek V3 uses different routing per layer
output = flashinfer.trtllm_fp8_block_scale_moe(
    x=hidden_states,
    w1=expert_w1,
    w2=expert_w2,
    topk_weights=routing_weights,
    topk_ids=expert_ids,
    routing_strategy="deepseek_v3",  # Layer-specific routing
    # ... scales
)
```

### Grouped MoE (Tensor Parallelism)

**Expert Parallelism**:
```python
# Each GPU handles subset of experts
local_expert_ids = expert_ids % num_gpus

output = flashinfer.cutlass_fused_moe(
    x=hidden_states,
    w1=local_expert_w1,        # Only local experts
    w2=local_expert_w2,
    topk_weights=routing_weights,
    topk_ids=local_expert_ids,
)

# AllReduce across GPUs
torch.distributed.all_reduce(output)
```

## Quantization Details

### FP8 Formats

**E4M3 (Activations)**:
- 1 sign bit, 4 exponent, 3 mantissa
- Range: ~±448
- Better for activations

**E5M2 (Weights)**:
- 1 sign bit, 5 exponent, 2 mantissa
- Range: ~±57344
- Better for weights

```python
# Convert to FP8
input_fp8 = input.to(torch.float8_e4m3fn)
weight_fp8 = weight.to(torch.float8_e5m2)

# Compute scale factors
input_scale = input.abs().max() / 448
weight_scale = weight.abs().max() / 448
```

### FP4 Packing

**Format**: 2 FP4 values packed per byte

```python
# Quantize to FP4
def quantize_fp4(weight, group_size=64):
    M, N = weight.shape
    num_groups = N // group_size
    
    # Compute per-group scales
    weight_grouped = weight.reshape(M, num_groups, group_size)
    scales = weight_grouped.abs().max(dim=2, keepdim=True)[0]
    
    # Quantize
    weight_normalized = weight_grouped / (scales + 1e-6)
    weight_fp4 = pack_fp4(weight_normalized)  # Pack 2 per byte
    
    return weight_fp4, scales
```

### Block-Wise Scaling

**Purpose**: Finer-grained quantization

```python
# Block-wise scales (better accuracy)
def compute_block_scales(tensor, block_size=128):
    *dims, last_dim = tensor.shape
    num_blocks = last_dim // block_size
    
    tensor_blocked = tensor.reshape(*dims, num_blocks, block_size)
    scales = tensor_blocked.abs().max(dim=-1)[0]
    
    return scales
```

## Performance Optimization

### Grouped GEMM Batching

```python
# Batch multiple GEMMs for efficiency
inputs = [x1, x2, x3, ...]       # Different shapes
weights = [w1, w2, w3, ...]

output = flashinfer.group_gemm_fp8_nt_groupwise(
    x=inputs,
    w=weights,
    x_scale=input_scales,
    w_scale=weight_scales,
    group_size=128,
)

# More efficient than individual GEMMs
```

### Weight Reordering

**For tensor cores**:
```python
# Reorder weights for optimal memory access
def reorder_weight_for_gemm(weight):
    # Transpose for NT layout (GEMM expects N x K)
    weight_t = weight.t()
    
    # Optional: Reorder for better tensor core utilization
    # (implementation-specific)
    return weight_t
```

### Workspace Management

```python
# Allocate workspace once, reuse across calls
workspace_size = flashinfer.get_gemm_workspace_size(...)
workspace = torch.empty(
    workspace_size,
    dtype=torch.uint8,
    device='cuda'
)

# Reuse for all GEMM calls
for layer in range(num_layers):
    output = flashinfer.group_gemm_fp8_nt_groupwise(
        x=inputs,
        w=weights,
        workspace_buffer=workspace,
        # ...
    )
```

## Use Cases

### LoRA Adapters

```python
# Grouped GEMM for multiple LoRA adapters
lora_outputs = flashinfer.group_gemm_fp8_nt_groupwise(
    x=[hidden_states] * num_adapters,
    w=[adapter.lora_down for adapter in adapters],
    # ... scales
)

# Combine with base model output
final_output = base_output + sum(lora_outputs)
```

### Speculative Decoding

```python
# Draft model uses FP4 for speed
draft_logits = flashinfer.mm_fp4(
    x=draft_hidden,
    w_q=draft_lm_head_fp4,
    scales=scales,
    group_size=64,
)

# Target model uses FP8 for accuracy
target_logits = flashinfer.gemm_fp8_nt_groupwise(
    x=target_hidden,
    w=target_lm_head_fp8,
    x_scale=x_scale,
    w_scale=w_scale,
    group_size=128,
)
```

### Multi-Modal Models

```python
# Different modalities use different experts
output = flashinfer.cutlass_fused_moe(
    x=hidden_states,
    w1=expert_w1,
    w2=expert_w2,
    topk_weights=routing_weights,
    topk_ids=expert_ids,  # Routed by modality
)
```

## Best Practices

1. **Use FP8 for activations**: E4M3 format optimal
2. **Use FP4 for weights**: 2-4x memory reduction
3. **Group-wise scaling**: Better accuracy than per-tensor
4. **Batch GEMMs**: Use group_gemm for efficiency
5. **Reuse workspace**: Allocate once, reuse across layers
6. **Profile quantization**: Check accuracy vs performance tradeoff
