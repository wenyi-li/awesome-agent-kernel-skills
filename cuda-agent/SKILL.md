You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance with a minimum requirement of 5% faster than torch.compile baseline.

## 1. CRITICAL RESTRICTIONS

### ⚠️ STRICTLY FORBIDDEN
- **NO torch operators in C++**: NEVER use `torch::*` or `torch::nn::functional::*` in binding.cpp or .cu files
- **NO torch operations in model_new.py**: Only tensor creation and your custom ops allowed
- **NO third-party libraries**: Except cuBLAS (GEMM only) and cuDNN (Conv only)
- **NO modifications to utils/ directory**
- **NO modifications to binding.cpp or binding_registry.h**: These are fixed infrastructure

### ✅ ALLOWED ONLY
- **C++**: Raw CUDA kernels (for custom ops), cuBLAS (for GEMM), cuDNN (MANDATORY for Conv/ConvTranspose)
- **Python**: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
- **Memory**: torch::empty_like for allocation only
- **Focus**: Implement kernels in `kernels/` directory only

## 2. WORKSPACE STRUCTURE

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── utils/                # DO NOT modify - Compilation, verification and profiling tools 
├── model.py              # DO NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops.
```

### File Types and Usage
- **`.cu` files**: CUDA kernels with `__global__` functions (custom implementations)
- **`.cpp` files**: cuDNN/cuBLAS API calls (NO custom kernels)
- **`_binding.cpp` files**: PyTorch tensor handling and Python bindings

## 3. UNIFIED WORKFLOW

### Step 1: Implementation

Create paired files in `kernels/`:

**kernels/my_kernel.cu** (Pure CUDA implementation):
```cuda
#include <cuda_runtime.h>

// Template kernel for performance tuning
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(float* output, const float* input, int size) {
    // Shared memory for tiling
    extern __shared__ float smem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for large data
    for (int i = tid; i < size; i += stride) {
        // Kernel logic with optimizations
        output[i] = /* computation */;
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
) {
    // Dynamic configuration selection
    int blocks = (size + 255) / 256;
    int shared_mem_size = 0;
    
    switch(config) {
        case 0: 
            shared_mem_size = 256 * sizeof(float);
            my_kernel_impl<256, 16><<<blocks, 256, shared_mem_size, stream>>>(
                output, input, size);
            break;
        case 1: 
            shared_mem_size = 128 * sizeof(float);
            my_kernel_impl<128, 32><<<blocks, 128, shared_mem_size, stream>>>(
                output, input, size);
            break;
        default:
            my_kernel_impl<256, 16><<<blocks, 256, 0, stream>>>(
                output, input, size);
    }
}
```

**kernels/my_kernel_binding.cpp** (PyTorch binding):
```cpp
// Use this two headers to replace torch/extension.h for faster compilation
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
);

// PyTorch wrapper with config parameter
torch::Tensor my_kernel_forward(torch::Tensor input, int config = 0) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream (correct way)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher with config
    my_kernel_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        config,
        stream
    );
    
    return output;
}

// Registration function
void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, 
          "My kernel forward",
          py::arg("input"),
          py::arg("config") = 0);
}

// Auto-register
REGISTER_BINDING(my_kernel, register_my_kernel);
```

#### Create model_new.py
```python
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model signature exactly
        super().__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(...))
        self.bias = nn.Parameter(torch.zeros(...))
        
    def forward(self, x):
        # Use custom ops only - NO torch operations
        x = cuda_extension.my_kernel_forward(x, config=0)
        x = cuda_extension.gemm_forward(x, self.weight, self.bias)
        return x
```

### Step 2: Compile and Test
```bash
# Compile with architecture-specific optimizations
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh

# Test in sandbox 
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

### Step 3: Performance Optimization (IF NEEDED)

#### 3.1 Optimization Strategy (Priority Order)

**Priority 1: Algorithmic (>50% impact)**
- Kernel fusion - reduce memory traffic
- Shared memory tiling - improve data reuse
- Memory coalescing - consecutive access patterns

**Priority 2: Hardware Utilization (20-50% impact)**
- Vectorized loads (float2/float4)
- Warp-level primitives (__shfl_sync, __ballot_sync)
- Occupancy tuning (block size, register usage)

**Priority 3: Fine-tuning (<20% impact)**
- Instruction-level parallelism
- Mixed precision (FP16/TF32)
- Prefetching and double buffering

#### 3.2 Parameter Tuning (Last Resort)
Only when within 1.2x of target and algorithmic options exhausted:

```python
# tune_kernel.py - NO recompilation needed
import time, torch, cuda_extension

configs = [
    (0, "256_threads_16_tile"),
    (1, "128_threads_32_tile"),
    (2, "512_threads_8_tile")
]

# Test input
x = torch.randn(batch_size, features).cuda()

# Benchmark each config
best_config, best_time = 0, float('inf')
for config_id, name in configs:
    # Warmup
    for _ in range(10):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"Config {name}: {elapsed:.4f}s")
    if elapsed < best_time:
        best_time, best_config = elapsed, config_id

print(f"Best: config {best_config} ({best_time:.4f}s)")
# Update model_new.py with best_config
```

### Step 4: Iteration Requirements

#### Correctness Failures
**MUST iterate until correctness passes - NO EXCEPTIONS**
1. Debug the specific failing kernel
2. Common issues to check:
   - Boundary conditions (tid < size)
   - Synchronization (__syncthreads placement)
   - Data types and precision
   - Memory alignment
3. Fix in kernels/*.cu and *_binding.cpp ONLY
4. Recompile and test

#### Performance Optimization
**GOAL: Achieve the best possible performance (the faster, the better!)**
**MINIMUM: Must be at least 5% faster than torch.compile baseline**

For each iteration:
1. **Document expectation**: "Fusion will eliminate 3 kernels, expect ~20% speedup"
2. **Apply optimization aggressively**: Don't revert to slow versions
3. **Debug if correctness fails**: Fix the optimized version
4. **Measure and analyze**: Understand why performance changed
5. **Continue optimizing**: Even if you meet the minimum, keep pushing for better performance

**Iteration strategy**:
- First 1-2 iterations: Achieve the minimum 5% improvement
- Next 3-5 iterations: Push for maximum possible speedup
- Continue until no further improvements possible or diminishing returns

**Remember**: The goal is the BEST possible performance, not just meeting the minimum!

### Step 5: Final Cleanup (MANDATORY BEFORE COMPLETION)

Before declaring the task complete, clean up the kernels/ directory to contain ONLY the final optimized version:

**Remove all intermediate attempts**:
```bash
# Remove version files, old attempts, test versions
rm kernels/*_v[0-9].cu kernels/*_old.cu kernels/*_test.cu kernels/*.bak

# Keep only the final optimized implementation
# Example final structure:
# kernels/
#   ├── fused_kernel.cu           # Final implementation
#   └── fused_kernel_binding.cpp  # Final binding
```

## 4. TOOL SCRIPTS REFERENCE

### Verification and Profiling
```bash
# Use sudo to run sandbox utilities
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

### Compilation
```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
```


## 5. OPTIMIZATION CHECKLIST

### Essential Optimizations (Apply First)
- [ ] **Memory Coalescing**: Consecutive threads access consecutive addresses
- [ ] **Kernel Fusion**: Combine operations to reduce memory traffic
- [ ] **Shared Memory**: Cache frequently accessed data
- [ ] **Grid-Stride Loops**: Handle data larger than grid size
- [ ] **Boundary Checks**: Validate all array accesses (tid < size)

### Performance Optimizations (Apply as Needed)
- [ ] **Vectorized Memory**: Use float2/float4 for higher throughput
- [ ] **Warp Primitives**: __shfl_sync for inter-thread communication
- [ ] **Occupancy Tuning**: Balance block size and resource usage
- [ ] **Bank Conflict Avoidance**: Pad shared memory arrays
- [ ] **Loop Unrolling**: Increase instruction-level parallelism

### Advanced Optimizations (For Final Tuning)
- [ ] **Tensor Cores**: Use WMMA/MMA for eligible GEMM operations
- [ ] **Mixed Precision**: FP16/TF32 where appropriate
- [ ] **Persistent Kernels**: Keep data in registers across iterations
- [ ] **CUDA Graphs**: Reduce launch overhead
- [ ] **Double Buffering**: Overlap computation with memory transfers

### Correctness Checklist (Always Verify)
- [ ] **Thread Bounds**: Check tid < N before array access
- [ ] **Synchronization**: __syncthreads() before shared memory reuse
- [ ] **Data Types**: Ensure correct types and conversions
- [ ] **Memory Safety**: No out-of-bounds access
- [ ] **Numerical Stability**: Handle NaN/Inf, use stable algorithms

## 6. COMMON ISSUES AND SOLUTIONS

### Compilation Errors
| Error | Solution |
|-------|----------|
| undefined symbol | Check extern "C" declarations match |
| no kernel image | Verify TORCH_CUDA_ARCH_LIST matches GPU |

### Correctness Failures
| Issue | Debug Steps |
|-------|-------------|
| Wrong output values | 1. Check kernel math<br>2. Verify indexing<br>3. Test with simple inputs |
| NaN/Inf results | 1. Check division by zero<br>2. Verify numerical stability<br>3. Add bounds checking |
| Mismatched shapes | 1. Print tensor shapes<br>2. Check dimension calculations<br>3. Verify reduction logic |

### Performance Issues
| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Slower than baseline | No fusion | Combine kernels |
| Low SM efficiency | Poor occupancy | Tune block size |
| Low memory throughput | Uncoalesced access | Restructure memory pattern |
| High kernel count | Missing fusion | Implement compound operations |

## 7. SUCCESS CRITERIA

**OPTIMIZATION GOALS:**
- 🎯 **MINIMUM REQUIREMENT**: At least 5% faster than torch.compile (≤ 0.95× baseline time)
- 🚀 **TARGET**: Achieve the best possible performance - every microsecond counts!
- ✅ **Correctness**: Test must pass (atol=1e-2, rtol=1e-2)
- 🧹 **Clean Final Code**: kernels/ directory contains ONLY final optimized version (no intermediate attempts)

**Performance metric clarification:**
- If torch.compile baseline = 1.0ms:
  - MINIMUM: Your implementation must be ≤ 0.95ms (5% faster)
  - GOAL: Push for ≤ 0.8ms or better (20%+ faster)
- The faster your implementation, the better the result
- Continue optimizing even after meeting the minimum requirement

## 8. KEY REMINDERS

1. **Keep .cu and _binding.cpp files separate** - Faster compilation
2. **Pass config parameters through bindings** - Enable runtime tuning without recompilation
3. **Focus modifications in kernels/ directory** - Never modify infrastructure files
4. **Be aggressive with optimizations** - Don't revert to slow versions when debugging
5. **Document performance expectations** - Before implementing, state expected gains
6. **Test with descriptive names** - Show which optimizations are applied
7. **Clean up before completion** - Remove ALL intermediate attempts from kernels/, keep ONLY final version

## Your Task

Optimize the PyTorch model in model.py.