# Optimization Checklist

Detailed checklist for optimizing TileLang kernels. Work through each section in order, measuring after every change.

## Table of Contents

1. [Tile Size Selection](#1-tile-size-selection)
2. [Inner Tile (block_K)](#2-inner-tile-block_k)
3. [Pipeline Depth](#3-pipeline-depth)
4. [Thread Count](#4-thread-count)
5. [L2 Swizzle](#5-l2-swizzle)
6. [Memory Access Patterns](#6-memory-access-patterns)
7. [Epilogue Fusion](#7-epilogue-fusion)
8. [Layout and Vectorization](#8-layout-and-vectorization)
9. [Advanced Techniques](#9-advanced-techniques)

---

## 1. Tile Size Selection

Tile sizes (block_M, block_N) determine how much work each thread block handles. Larger tiles amortize the cost of loading shared memory but require more on-chip resources.

### Impact

Going from 64x64 to 128x128 tiles is the single highest-impact optimization for compute-bound GEMM kernels. The exact speedup depends on the GPU and problem shape — always measure.

### How to Choose

**Compute-bound kernels (GEMM):**
- Start with 128x128. This is the sweet spot for most GPUs.
- Try 256x128 or 128x256 for rectangular problem shapes (when M >> N or N >> M).
- Don't go below 64x64 unless shared memory is a hard constraint.

**Memory-bound kernels (elementwise, normalization):**
- 64x64 or even 32x128 is fine. The bottleneck is memory bandwidth, not compute.
- Match block_N to the inner dimension for coalesced access.

### Shared Memory Budget

Every tile configuration has a shared memory cost:

```
shared_bytes = (block_M * block_K + block_K * block_N) * dtype_bytes * num_stages
```

For 128x128x32, fp16, 2 stages:
```
(128*32 + 32*128) * 2 * 2 = 32,768 bytes = 32 KB
```

GPU shared memory limits (default / max with opt-in):
- Blackwell (sm_120) / Hopper (sm_90): 228 KB max
- Ampere (sm_80): 164 KB max
- Check your GPU's limit with ncu

If your config exceeds the default 48 KB, the compiler may automatically request more. If it exceeds the GPU max, the kernel will fail to launch.

## 2. Inner Tile (block_K)

block_K controls the reduction dimension per loop iteration. Larger block_K means fewer iterations but more shared memory per stage.

### Impact

Increasing block_K (e.g. 32→64) can improve throughput on compute-bound kernels, at the cost of doubling shared memory per stage. The gain depends on the GPU and kernel — always measure.

| block_K | Shared bytes (128x128, 2 stages) |
|---------|----------------------------------|
| 32 | 32 KB |
| 64 | 64 KB |

### When to Increase

- The kernel is compute-bound (high arithmetic intensity)
- Shared memory budget allows it
- K dimension is large (many iterations in the pipelined loop)

### When to Keep Small

- Shared memory is already tight (many pipeline stages, large tiles)
- K is small (few iterations anyway)
- Memory-bound kernel (block_K doesn't help much)

## 3. Pipeline Depth

`T.Pipelined(iters, num_stages=N)` overlaps memory transfers with computation. More stages allow deeper overlap.

### How It Works

- `num_stages=0`: No pipelining. Load tile, compute, load next tile. Useful for debugging.
- `num_stages=2`: Double-buffered. While computing on tile N, load tile N+1.
- `num_stages=3`: Triple-buffered. While computing on tile N, tile N+1 is in transit, tile N+2 is being requested.

### Impact

| num_stages | Shared bytes (128x128x32) |
|-----------|--------------------------|
| 2 | 32 KB |
| 3 | 48 KB |

The throughput difference between 2 and 3 stages is typically small. The optimal stage count is hardware-dependent — on some GPUs, 3 stages is slightly worse due to increased shared memory pressure, while on others it helps. Always measure.

### Guidelines

- Start with `num_stages=2`
- Try 3 if latency is dominated by memory access (check with ncu)
- Each additional stage costs `(block_M * block_K + block_K * block_N) * dtype_bytes` more shared memory
- If stages=3 is slower, the shared memory pressure is outweighing the pipelining benefit

## 4. Thread Count

`threads` in `T.Kernel(..., threads=N)` sets the number of threads per block.

### Impact

Increasing from 128 to 256 threads can improve throughput when tiles are large enough (128x128 or bigger). The gain depends on the GPU and kernel — always measure.

### Guidelines

- 128 threads is a safe default
- 256 can improve performance when tiles are large enough (128x128 or bigger)
- Higher thread count increases register pressure, which can reduce occupancy
- Check with ncu: if achieved occupancy is already high, more threads won't help

## 5. L2 Swizzle

```python
T.use_swizzle(panel_size=10, enable=True)
```

Reorders thread block scheduling to improve L2 cache reuse for the B matrix in GEMM.

### When It Helps

- N dimension is much larger than L2 cache can hold
- Multiple rows of the output share the same B tiles
- Problem is large (N >= 8192)

### When It Doesn't Help

- Small to medium problems (N <= 4096): the B matrix may already fit in L2
- Memory-bound kernels: L2 hit rate isn't the bottleneck

### Tuning panel_size

The `panel_size` parameter controls how many rows of blocks share the same B panel. Start with 10, try 4-16. The optimal value depends on L2 cache size and problem dimensions.

## 6. Memory Access Patterns

### Coalesced Global Reads

Threads should access consecutive memory addresses. For row-major tensors (C-contiguous), this means threads should read along the innermost dimension.

TileLang's `T.copy` handles coalescing automatically when source and destination shapes are compatible. Ensure your tensors are contiguous:

```python
assert A.is_contiguous()
```

### Vectorized Loads

For maximum memory throughput, inner dimensions should be multiples of 8 (fp16) or 4 (fp32). This allows the hardware to use 128-bit loads.

### Shared Memory Bank Conflicts

Shared memory is organized in 32 banks. If multiple threads in a warp access the same bank, accesses are serialized (bank conflict). TileLang's layout inference handles most cases, but unusual access patterns can cause conflicts. Check with ncu:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum python script.py
```

## 7. Epilogue Fusion

Every separate kernel launch incurs:
- Kernel launch overhead (~5-10 microseconds)
- Global memory writes + reads between kernels

Fusing operations into a single kernel eliminates both costs.

### Common Fusions

```python
# GEMM + activation (ReLU, sigmoid, GELU, etc.)
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.sigmoid(C_local[i, j])

# GEMM + bias add
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = C_local[i, j] + T.cast(bias[bx * block_N + j], accum_dtype)

# GEMM + scaling
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = C_local[i, j] * T.cast(scale, accum_dtype)

# GEMM + cast (fp32 accumulator -> fp16 output)
# This happens automatically in T.copy if C_local is fp32 and output is fp16
```

### Epilogue Template

```python
# Accumulate in fp32
C_local = T.alloc_fragment((block_M, block_N), T.float32)
T.clear(C_local)

for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# Epilogue: fuse bias + activation + cast
for i, j in T.Parallel(block_M, block_N):
    val = C_local[i, j] + T.cast(bias[bx * block_N + j], T.float32)
    C_local[i, j] = T.max(val, T.cast(0.0, T.float32))  # ReLU

T.copy(C_local, C[by * block_M, bx * block_N])  # fp32 -> fp16 cast happens here
```

## 8. Layout and Vectorization

### Layout Annotations

For kernels using `T.atomic_add` (common in backward passes), annotate the target layout for efficient writes:

```python
from tilelang.utils import make_dq_layout  # or define custom layout
T.annotate_layout({dQ: make_dq_layout(dQ)})
```

Without annotation, atomic writes may be scattered inefficiently across memory.

### When to Use Layout Annotations

- Backward pass dQ accumulation with `T.atomic_add`
- Any kernel with non-standard access patterns
- When ncu shows low store efficiency

## 9. Advanced Techniques

These are specialized optimizations for specific scenarios.

### Split-K for Tall-Skinny GEMM

When M or N is small but K is large, a single tile handles the full M or N dimension, leaving many SMs idle. Split-K divides the reduction dimension across multiple blocks:

```python
# Conceptual: split K across split_k blocks, each does partial reduction
# Final atomic_add combines partial results
# This is typically handled by the autotuner config
```

### Persistent Kernels

For problems where many small GEMMs are launched sequentially, persistent kernels keep threads alive across multiple GEMM tiles, reducing launch overhead. This is an advanced pattern used in flash attention implementations.

### Warp Specialization

Different warps within a block handle different tasks (e.g., some warps do memory loads while others do compute). This is an advanced pattern in TMA-based kernels on Hopper/Blackwell.

## Quick Reference: What to Try First

| Kernel Type | Priority 1 | Priority 2 | Priority 3 |
|-------------|-----------|-----------|-----------|
| GEMM | Tile sizes (128x128) | block_K (64) | threads (256) |
| GEMM + epilogue | Fuse epilogue | Tile sizes | Pipeline depth |
| Elementwise | Vectorized loads | Tile sizes | Shared memory staging |
| Reduction | block_N (wide reduce) | Coalescing | Shared memory |
| Attention fwd | block_M/block_N | num_stages | Causal masking |
| Attention bwd | dQ atomic layout | Split-K | Thread count |
