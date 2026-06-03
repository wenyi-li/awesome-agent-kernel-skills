---
name: mi300-cdna3-architecture
description: MI300/CDNA3 architecture guide for HIP/Triton optimization—MFMA variants, dual register files, data formats, sparsity, LDS/GWS, and best practices.
---

# MI300 CDNA3 Architecture (Hardware Tuning Guide)

Use this when optimizing HIP/Triton kernels for MI300/CDNA3 hardware specifics.

Key points to remember (see reference for details):
- MFMA choices: pick variants by matrix size (small -> high block count like 4x4x1_16B; large -> bigger tiles). Use CBSZ/BLGP for broadcast/permutation.
- Dual register files: minimize VGPR↔AccVGPR traffic; keep hot accumulations in AccVGPR.
- Data formats: FP8/BF8 support, conversions/rounding; validate accuracy when using reduced precision.
- Structured sparsity: 4:2 pattern via SMFMAC; ensure encoding matches pattern to get gains.
- Memory: LDS bank-conflict avoidance; coalesced global buffer ops; use cache control bits (GLC/NV) and schedule loads early with `s_waitcnt`.
- Registers/occupancy: respect SGPR single-read limits; align VGPR usage; watch overall pressure to keep occupancy.
- Control/exec: wavefront characteristics, dependency scheduling; hide latency via instruction scheduling.
- Best practices section summarizes matrix, memory, register, and optimization checklists.

References:
- `references/AMD Instinct MI300 CDNA3 Architecture Guide for High-Quality HIP Kernel Development.md`
