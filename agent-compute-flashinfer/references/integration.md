# FlashInfer Integration Reference

Sources:
- [Installation Guide](https://docs.flashinfer.ai/installation.html)
- [GitHub Repository](https://github.com/flashinfer-ai/flashinfer)

## Installation

### Binary Installation

```bash
# Basic installation
pip install flashinfer-python

# Recommended: With pre-compiled kernels
pip install flashinfer-python flashinfer-cubin

# With JIT cache for CUDA 12.9
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129

# For other CUDA versions (12.6, 12.8, 13.0, 13.1)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer

# Build and install
python -m pip install -v .

# Development mode
python -m pip install --no-build-isolation -e . -v
```

### Verify Installation

```bash
# Check version and configuration
flashinfer show-config

# Output:
# FlashInfer version: 0.6.3
# PyTorch version: 2.5.0+cu129
# CUDA version: 12.9
# Kernel compilation status: [OK]
```

## vLLM Integration

### Replace PageAttention

```python
import flashinfer
from vllm.attention import Attention

class FlashInferAttention(Attention):
    def __init__(self, num_heads, head_dim, scale):
        super().__init__(num_heads, head_dim, scale)
        
        # Initialize wrappers
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()
    
    def forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
    ):
        # Check if prefill or decode
        if attn_metadata.is_prompt:
            # Prefill stage
            self.prefill_wrapper.begin_forward(
                qo_indptr=attn_metadata.seq_start_loc,
                kv_indptr=attn_metadata.seq_start_loc,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
            
            output = self.prefill_wrapper.forward(query, kv_cache)
            self.prefill_wrapper.end_forward()
        else:
            # Decode stage
            self.decode_wrapper.begin_forward(
                indptr=attn_metadata.block_tables_ptr,
                indices=attn_metadata.block_tables,
                last_page_len=attn_metadata.last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=16,
            )
            
            output = self.decode_wrapper.forward(query, kv_cache)
            self.decode_wrapper.end_forward()
        
        return output
```

### Configuration

```python
# vLLM engine arguments
from vllm import EngineArgs

engine_args = EngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    # Use FlashInfer backend
    attention_backend="flashinfer",
    # Enable FP8 quantization
    quantization="fp8",
)
```

## SGLang Integration

### Attention Backend

```python
import flashinfer
import sglang as sgl

class SGLangFlashInferBackend:
    def __init__(self, model_config):
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.num_kv_heads = model_config.num_key_value_heads
        
        # Initialize wrappers
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper()
    
    def forward_prefill(self, query, kv_cache, seq_metadata):
        self.prefill_wrapper.begin_forward(
            qo_indptr=seq_metadata.qo_indptr,
            kv_indptr=seq_metadata.kv_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        
        output = self.prefill_wrapper.forward(query, kv_cache)
        self.prefill_wrapper.end_forward()
        return output
    
    def forward_decode(self, query, kv_cache, seq_metadata):
        self.decode_wrapper.begin_forward(
            indptr=seq_metadata.kv_indptr,
            indices=seq_metadata.kv_indices,
            last_page_len=seq_metadata.last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=16,
        )
        
        output = self.decode_wrapper.forward(query, kv_cache)
        self.decode_wrapper.end_forward()
        return output
```

## MLC-Engine Integration

### Custom Attention Module

```python
import tvm
from tvm import relax
import flashinfer

@tvm.script.ir_module
class FlashInferModule:
    @relax.Function
    def attention(
        query: relax.Tensor,
        kv_cache: relax.Tensor,
        kv_indptr: relax.Tensor,
        kv_indices: relax.Tensor,
    ):
        # Call FlashInfer kernel
        output = relax.call_dps_packed(
            "flashinfer.decode",
            (query, kv_cache, kv_indptr, kv_indices),
            relax.TensorStructInfo(query.struct_info.shape, query.struct_info.dtype)
        )
        return output
```

## PyTorch Integration

### Standalone Usage

```python
import torch
import flashinfer

class LlamaAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # Projections
        self.q_proj = torch.nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = torch.nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        
        # KV-cache
        self.kv_cache = None
        
    def forward(self, hidden_states, position_ids, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        if past_key_value is None:
            # Prefill
            output = flashinfer.single_prefill_with_kv_cache(
                q=query[0],  # Assume batch_size=1
                k=key[0],
                v=value[0],
                causal=True,
            )
        else:
            # Decode
            output = flashinfer.single_decode_with_kv_cache(
                q=query[0, 0],  # Single token
                kv_data=past_key_value,
                # ... KV-cache parameters
            )
        
        # Project output
        output = self.o_proj(output.view(batch_size, seq_len, -1))
        return output
```

### CUDAGraph Compatibility

```python
class CUDAGraphAttention:
    def __init__(self):
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper()
        self.graph = None
        self.static_inputs = {}
    
    def warmup(self, query, kv_cache, metadata):
        # Warmup run
        self.wrapper.begin_forward(**metadata)
        output = self.wrapper.forward(query, kv_cache)
        self.wrapper.end_forward()
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.graph):
            self.wrapper.begin_forward(**metadata)
            output = self.wrapper.forward(query, kv_cache)
            self.wrapper.end_forward()
        
        return output
    
    def forward(self, query, kv_cache):
        # Replay captured graph
        self.graph.replay()
        return self.static_inputs['output']
```

## Deployment Configurations

### Docker Container

```dockerfile
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install FlashInfer
RUN pip install flashinfer-python flashinfer-cubin flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu129

# Install serving framework
RUN pip install vllm sglang

# Copy application
COPY app /app
WORKDIR /app

CMD ["python3", "serve.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: llm-server
        image: flashinfer-llm:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: FLASHINFER_JIT_CACHE
          value: "/cache"
        volumeMounts:
        - name: jit-cache
          mountPath: /cache
      volumes:
      - name: jit-cache
        persistentVolumeClaim:
          claimName: flashinfer-cache
```

### Environment Variables

```bash
# Enable logging
export FLASHINFER_LOG_LEVEL=INFO

# Set JIT cache directory
export FLASHINFER_CACHE_DIR=/path/to/cache

# Enable flight recorder
export FLASHINFER_ENABLE_FLIGHT_RECORDER=1

# Set compile workers
export FLASHINFER_NUM_COMPILE_WORKERS=8
```

## Logging and Monitoring

### Flight Recorder

```python
import flashinfer
import flashinfer.logging as logging

# Enable flight recorder
logging.enable_flight_recorder(
    output_dir="/tmp/flashinfer_logs",
    max_events=10000,
)

# Log attention call
output = flashinfer.single_decode_with_kv_cache(
    q=query,
    kv_data=kv_cache,
    # Automatically logged
)

# Flush logs
logging.flush()
```

### Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    output = flashinfer.batch_decode_with_kv_cache(...)

# Export trace
prof.export_chrome_trace("flashinfer_trace.json")
```

## Best Practices

1. **Install pre-compiled kernels**: Use `flashinfer-cubin` for faster startup
2. **Use JIT cache**: Pre-download for offline deployment
3. **Enable CUDAGraph**: For static batch sizes (2-3x lower latency)
4. **Reuse wrappers**: Amortize planning overhead across layers
5. **Profile first**: Identify bottlenecks before optimization
6. **Monitor memory**: Track KV-cache usage with logging
7. **Use environment variables**: Configure without code changes
