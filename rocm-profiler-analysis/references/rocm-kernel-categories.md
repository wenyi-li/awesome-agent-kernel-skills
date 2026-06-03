# ROCm Kernel Categories for SGLang / vLLM

The upstream SGLang profiler skill is useful, but its category language comes from CUDA-centric
practice. For MI355X / gfx950 work we need buckets that better match ROCm execution reality.

## Recommended Categories

### Communication

- RCCL collectives
- all-reduce / all-gather / reduce-scatter
- peer transfer / synchronization overhead

### Attention

- Triton attention kernels
- AITER attention paths
- SDPA wrappers
- MIOpen attention runtime kernels where applicable

### GEMM / Linear

- hipBLASLt
- rocBLAS
- CK / composable kernel GEMM
- Triton GEMM

### Quantization

- FP8 / INT4 / INT8 kernels
- dequant / requant / scale-apply kernels

### Norm / Elementwise

- RMSNorm
- LayerNorm
- fused add-norm patterns

### MoE / Routing

- dispatch / combine
- gating
- expert-local compute

### Memory / Scheduler

- copies
- cache movement
- host scheduling / launch overhead

## MI355X-Specific Notes

- Always distinguish `gfx950` from `gfx942`
- Treat AITER-native paths as first-class categories
- Keep exact Docker image tags because kernel availability can change between builds on the same
  ROCm minor version

## Reviewer Rule

If a profiling summary says only "GEMM is slow" or "attention dominates", it is not specific
enough. The category scheme should be detailed enough that a reviewer can decide whether the next
step is:

- AITER tuning
- Triton tuning
- RCCL overlap
- CK/native kernel substitution
- scheduler/runtime investigation
