---
name: mi300-hip-vs-nvidia
description: MI300 HIP programming differences vs NVIDIA—wavefront vs warp, memory hierarchy, MFMA usage, occupancy, and profiling pitfalls.
---

# MI300 vs NVIDIA (HIP Porting Guide)

Use when porting/tuning kernels from NVIDIA to MI300 or comparing behaviors.

Key differences and guidance:
- Execution: 64-thread wavefronts vs 32-thread warps—adjust coalescing, tails, and occupancy; watch register/LDS pressure per wavefront.
- Memory: CDNA3 cache hierarchy and vector cache; coalesce for 64 threads; exploit LDS; understand cache line sizes/policies.
- Matrix cores: MFMA set, mixed precision; adapt tiling/layout for Matrix Cores; leverage scalar unit for uniform ops.
- Synchronization/atomics: wave-level sync, memory ordering; extra fences for relaxed coherency; cooperative groups considerations.
- Compiler/feature guards: use HIP macros and arch detection; conditional MFMA paths.
- Optimization: occupancy sizing (multiples of 64), instruction scheduling, mixed precision, load balancing across shader engines, fusion when resources allow.
- Profiling/debug: use ROCProfiler/ROCgdb for wavefront-aware analysis; check launch config impacts on occupancy/memory.

References:
- `references/HIP Kernel Programming Guide for MI300_ Key Differences from NVIDIA AI Chips.md`
