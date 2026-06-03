# Bottleneck Classification Heuristics

This document describes the rules and thresholds used to classify GPU kernel performance bottlenecks.

## Contents
1. [Bottleneck Types](#bottleneck-types)
2. [Classification Algorithm](#classification-algorithm)
3. [Threshold Values](#threshold-values)
4. [Optimization Strategies](#optimization-strategies)

---

## Bottleneck Types

### 1. Compute-Bound

The kernel spends most time executing compute instructions (VALU, MFMA).

**Indicators:**
- High VALU instruction ratio (>60% of total instructions)
- High GPU busy cycles (>70% utilization)
- Low memory stall cycles
- MFMA units frequently busy (for matrix workloads)

**Typical Causes:**
- Math-intensive algorithms (matrix multiply, convolution)
- Complex per-element operations
- High arithmetic intensity

---

### 2. Memory-Bound

The kernel is limited by memory bandwidth or latency.

**Indicators:**
- High VMEM instruction ratio (>30% of total instructions)
- High memory stall cycles (>40% of busy cycles)
- Low L2 cache hit rate (<50%)
- High memory request traffic

**Sub-categories:**

| Type | Key Indicator |
|------|---------------|
| **Bandwidth-bound** | High HBM traffic, saturated memory channels |
| **Latency-bound** | Low occupancy hiding memory latency |
| **Cache-thrashing** | Very low L2 hit rate, repeated misses |

**Typical Causes:**
- Large data access patterns that exceed cache
- Non-coalesced memory accesses
- Random access patterns
- Working set larger than cache capacity

---

### 3. LDS-Bound

The kernel is limited by Local Data Share (shared memory) access.

**Indicators:**
- High LDS bank conflict rate (>0.1 conflicts per wave)
- High LDS instruction ratio (>20% of instructions)
- Low LDS throughput despite high usage

**Typical Causes:**
- Bank conflicts from strided access patterns
- Too many threads accessing same LDS bank
- Inefficient tiling strategies

**Bank Conflict Example:**
```
// Bad: All threads in warp access same bank
shared[threadIdx.x * 32]  // Stride of 32 = all same bank

// Good: Sequential access across banks
shared[threadIdx.x]       // Stride of 1 = different banks
```

---

### 4. Latency-Bound

The kernel cannot hide instruction latency due to insufficient parallelism.

**Indicators:**
- Low occupancy (<30%)
- High register usage (VGPR/SGPR)
- Low instruction-level parallelism
- Long dependency chains

**Typical Causes:**
- Too many registers per thread (limits waves per CU)
- Too much LDS per workgroup
- Large workgroup sizes with register pressure
- Serial dependency chains in code

---

### 5. Balanced

No single dominant bottleneck; kernel is reasonably optimized.

**Indicators:**
- No metric significantly above thresholds
- Balanced instruction mix
- Reasonable cache hit rates
- Adequate occupancy

---

## Classification Algorithm

```python
def classify_bottleneck(metrics):
    scores = {
        "compute_bound": 0,
        "memory_bound": 0,
        "lds_bound": 0,
        "latency_bound": 0,
    }
    
    # Compute-bound scoring
    if valu_inst_pct > 60:
        scores["compute_bound"] += 2
    if gpu_busy_pct > 70:
        scores["compute_bound"] += 1
    
    # Memory-bound scoring
    if vmem_inst_pct > 30:
        scores["memory_bound"] += 2
    if memory_stall_pct > 40:
        scores["memory_bound"] += 2
    if l2_hit_rate < 50:
        scores["memory_bound"] += 1
    
    # LDS-bound scoring
    if lds_conflict_per_wave > 0.1:
        scores["lds_bound"] += 2
    if lds_inst_pct > 20:
        scores["lds_bound"] += 1
    
    # Latency-bound scoring
    if occupancy_pct < 30:
        scores["latency_bound"] += 2
    if vgpr_count > 128:
        scores["latency_bound"] += 1
    
    # Select highest score
    bottleneck = max(scores, key=scores.get)
    confidence = "high" if scores[bottleneck] >= 3 else \
                 "medium" if scores[bottleneck] >= 2 else "low"
    
    return bottleneck, confidence
```

---

## Threshold Values

| Parameter | Threshold | Classification |
|-----------|-----------|----------------|
| VALU instruction % | > 60% | Compute-bound |
| GPU busy % | > 70% | Compute-bound |
| VMEM instruction % | > 30% | Memory-bound |
| Memory stall % | > 40% | Memory-bound |
| L2 hit rate | < 50% | Memory-bound |
| LDS conflicts/wave | > 0.1 | LDS-bound |
| LDS instruction % | > 20% | LDS-bound |
| Occupancy % | < 30% | Latency-bound |
| VGPR count | > 128 | Latency-bound |

---

## Optimization Strategies

### For Compute-Bound Kernels

1. **Use matrix instructions (MFMA)** for applicable workloads
2. **Optimize ALU operations** - reduce redundant calculations
3. **Loop unrolling** - increase ILP
4. **Vectorization** - use vector types (float4, etc.)
5. **Fast math** - use approximate math functions when precision allows

### For Memory-Bound Kernels

1. **Improve coalescing** - ensure contiguous memory access
2. **Use LDS caching** - stage data in shared memory
3. **Prefetching** - hide memory latency with prefetch
4. **Data layout** - use SoA (Structure of Arrays) over AoS
5. **Reduce precision** - use FP16/BF16 for bandwidth reduction
6. **Tiling** - work on cache-sized blocks

### For LDS-Bound Kernels

1. **Pad shared arrays** - add padding to avoid bank conflicts
   ```cpp
   // Before: bank conflicts
   __shared__ float smem[32][32];
   
   // After: padding eliminates conflicts
   __shared__ float smem[32][33];
   ```
2. **Reorganize access patterns** - transpose or permute indices
3. **Reduce LDS usage** - less data per workgroup
4. **Use shuffle instructions** - when applicable

### For Latency-Bound Kernels

1. **Reduce register pressure** - fewer VGPRs per thread
2. **Smaller workgroups** - more waves can fit per CU
3. **Reduce LDS per workgroup** - allows more concurrent workgroups
4. **Break dependency chains** - enable more parallel execution
5. **Increase occupancy targets** - aim for >50% occupancy

---

## Roofline Model Integration

The roofline model provides complementary analysis:

```
                   Compute Roof
                  _______________
                 /
                /
               / Memory Roof
              /
    GFLOPS   /
            /
           /________________
              Arithmetic Intensity (FLOP/Byte)
```

- **Left of ridge point**: Memory-bound
- **Right of ridge point**: Compute-bound
- **Far below roof**: Latency or efficiency issues

Use `rocprof-compute` (formerly Omniperf) for automated roofline analysis.

---

## Confidence Levels

| Level | Score | Interpretation |
|-------|-------|----------------|
| **High** | ≥3 | Strong evidence; optimize this bottleneck first |
| **Medium** | 2 | Moderate evidence; likely significant |
| **Low** | 1 | Weak evidence; may have multiple bottlenecks |
| **None** | 0 | Balanced or insufficient data |

When confidence is low, the kernel may have multiple bottlenecks or be well-optimized. Consider profiling with additional counters or using `rocprof-compute` for deeper analysis.
