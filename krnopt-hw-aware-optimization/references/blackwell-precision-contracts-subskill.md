# Blackwell Precision Contracts Subskill

Use this integrated subskill when the target hardware is SM100-class and the
main question is whether a low-precision or block-scaled path is actually
compatible with the kernel, library, and data layout.

The core rule is:

- on Blackwell, precision choice is not just a dtype choice
- it is often a hardware and layout contract that tensor cores enforce
  directly through `tcgen05.mma`

The Blackwell shift is that microscaling became a hardware contract. On
Ampere and Hopper, "dynamic quantization" meant INT8 or FP8 with
software-managed scales. On Blackwell (SM100, SM120), tensor cores natively
consume microscaled operands: grouping and scaling are part of the data
format that `tcgen05.mma` expects.

| Aspect | Pre-Blackwell (Ampere/Hopper) | Blackwell (SM100+) |
|---|---|---|
| Dominant format | INT8, FP8 E4M3/E5M2 | NVFP4 (E2M1), MXFP8 |
| Scale granularity | Per-tensor, per-token, per-group | Per-block (16 for NVFP4, 32 for MXFP8) |
| Scale format | FP32 | FP8 E4M3 (packed with data) |
| Hardware support | Software dequant before/after tensor cores | Tensor cores consume microscaled operands directly |
| Scale computation | Fully software | Software computes, hardware layout enforced |

## The Main Precision Families

### Plain FP8 Or BF16

Use when:

- the workload wants a simpler path with fewer scale-layout constraints
- the upstream stack already produces compatible operands
- contest or model accuracy pressure disfavors more aggressive microscaling

On A100/SM80 there is no native FP8 tensor-core path; treat FP8 E4M3 as
packed storage and dequantize to BF16/FP16 before the GEMM. On Hopper and
Blackwell, plain FP8 tensor-core MMA is available, but the throughput ceiling
is below the block-scaled paths on the same hardware.

Main trap:

- assuming a plain FP8 path and a block-scaled FP8 path are interchangeable
  at the kernel-interface level; they are not, because the block-scaled
  variant requires a scale tensor in a specific layout and the tensor-core
  instruction is a different `kind::` of `tcgen05.mma`

### Block-Scaled FP8

Use when:

- the intended backend really supports the chosen scale granularity and
  layout
- the workload benefits from the reduced bandwidth and the hardware-native
  `tcgen05.mma.kind::f8f6f4` or `tcgen05.mma.kind::mxf8f6f4.block_scale` path

Contracts to verify:

- scale block size (contest: `128`; many libraries: `32` or `16`)
- scale dtype (FP32 for contest; packed `UE8M0` for SM100 MX paths)
- scale tensor layout or swizzle (GEMM-swizzled vs contiguous)
- operand alignment
- CUTLASS or library backend support for the exact format

Hardware and library support to cite by name:

- CUTLASS on SM100 exposes `tcgen05.mma.kind::f8f6f4` (mixed precision) and
  `tcgen05.mma.kind::mxf8f6f4.block_scale` (block-scaled mxf8/mxf6/mxf4)
- cuBLAS 12.9 ships block-scaled FP4/FP8 matmuls with 2.3x (FP4) and 4.6x
  (FP8) speedups versus standard FP8 on Blackwell and Hopper
- cuDNN Frontend ships block-scaled Grouped GEMM + SwiGLU and Grouped GEMM +
  Quant Unified kernels with per-block scale factors and per-row gating; the
  SwiGLU kernel explicitly uses 128-element block size
- DeepGEMM's `sm100_fp8_gemm_1d1d` is a compact SM100 FP8 GEMM that expects
  packed `UE8M0` scale tensors as `torch.int`
- TensorRT-LLM's `fp8_blockscale_gemm/` ships TMA utilities for block-scaled
  layouts

Contest reality check: MLSys26 uses FP8 E4M3 storage with block-size-128
scales for both activations and weights, BF16 output, SwiGLU-sandwiched MoE.
Several Blackwell library primitives use smaller K-vector scale granularities
(32 or 16), so before locking a backend, verify that its scale-factor tensor
shape matches what the contest spec provides, or plan an explicit scale
repack.

The software pressure behind all of this comes from training: DeepSeek-V3
uses 1x128 activation scales, 128x128 weight scales, and per-group scaling
along the GEMM K dimension, and notes that standard FP8 GEMM does not
directly support that scale contract. This is exactly what SM100 block-scaled
MMA and DeepGEMM's SM100 scale layout address.

Main trap:

- assuming every "block-scaled FP8" backend uses the same layout contract;
  blaming hardware when the real problem is a format or layout mismatch

### NVFP4

Use when:

- the model or workload explicitly targets NVFP4
- the data path can satisfy the two-level scaling contract
- the accuracy tradeoff is acceptable

Format details:

- FP4 E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (non-uniform step sizes)
- block size `16`: 16 elements share one FP8 E4M3 scale factor
- optional global scale: one FP32 per-tensor scale (the amax-derived scale)
- 3.5x memory reduction vs FP16
- Blackwell `tcgen05.mma` delivers 4x throughput over FP8

Two-level scaling contract:

```
global_sf = max(|X|) / max_fp8_value              (per-tensor, from current amax)
per_block_sf[b] = max(|X_block[b]|) / (global_sf * max_fp4_value)
X_fp4[b] = round_to_fp4(X_block[b] / (global_sf * per_block_sf[b]))
```

Dynamic path (no calibration): compute both levels of scale at runtime by
reducing to get `tensor_amax`, deriving `global_sf` as one FP32 value, then
per block of 16 reducing `local_amax`, deriving `per_block_sf` as one FP8
value, and quantizing. TorchAO implements this via
`NVFP4DynamicActivationNVFP4WeightConfig`, which hard-checks SM100+.
FlashInfer provides `nvfp4_quantize(...)` as a lower-level building block.

Known accuracy issues (two sources of error):

1. FP8 block scale factors: limited E4M3 precision introduces rounding error
2. FP4 values: near-maximal values (around 5 when scaled to 6) have no
   representable FP4 value

Sensitivity placement: MLP up/down projections are the most FP4-sensitive
components, while attention projections are substantially less sensitive.
Down projections exhibit extreme outlier behavior (Max/P99.9 ratios 10-100x)
from post-SiLU/SwiGLU activations. This means FP4 is better suited for
attention than for MLP/MoE GEMMs in most cases.

Improved variants worth knowing:

- Four Over Six: adaptively scales blocks to 4 or 6 based on per-block MSE
  comparison, reducing worst-case error for near-maximal values; less than
  15% overhead
- RaZeR: remaps the redundant FP4 zero (+0 = -0 in sign-magnitude) to an
  additional quantization value, 34.6% lower perplexity loss
- MR-GPTQ (QuTLASS): format-optimized PTQ with fused online Hadamard that
  hits near-ideal FP4 throughput on B200; surprising finding that MXFP4 is
  about 15% faster than NVFP4 on B200 but lower accuracy (93.31% vs 96.08%
  recovery after MR-GPTQ)

Contracts to verify:

- block size `16`
- FP8 E4M3 scale format for per-block scales
- optional global FP32 scale path
- output or epilogue compatibility (typically BF16 output)
- whether the library path is NVFP4 or MXFP4; these are not identical

Main trap:

- importing contest-format assumptions (FP8, block 128) into an NVFP4 path;
  they share the microscaling idea but sit at different points in the design
  space

### MXFP8 And Related MX Paths

Use when:

- the stack already speaks MX formats
- Blackwell-native microscaling is a real goal
- the scale layout and vector size match the intended kernel family

Contracts to verify:

- block size `32` for MXFP8, following the OCP MX specification
- UE8M0 packed scale expectations where relevant (power-of-two scales)
- backend-specific swizzled scale layout

Main trap:

- treating MX formats as only a higher-level quantization policy rather than
  a kernel-interface constraint
- forgetting that MXFP4 E8M0 power-of-two scales can introduce quantization
  artifacts that differ from NVFP4 E4M3 behavior

### Mixed-Precision Microscaling

Beyond single-format dynamic quantization, recent Blackwell kernels explore
assigning different MX precisions to different channels or layers based on
sensitivity. Two named systems worth knowing:

- MicroMix: three-way MXFP4/MXFP6/MXFP8 channel partitioning with dequant
  fused into MMA; offline calibration (64 samples), fused reorder-and-quant
  at runtime; CUTLASS-based GEMM kernel with three parallel MMA paths
- QuTLASS: format-optimized PTQ (MR-GPTQ) with high-performance NVFP4/MXFP4
  GEMM kernels on CUTLASS/FlashInfer; near-ideal FP4 throughput on B200

Both use `tcgen05.mma` and absorb dequant into the tensor-core instruction.
Named pattern across all such systems: sort by magnitude, reorder, then
quantize, which is the MX-format evolution of Atom-style channel reordering
and LLM.int8()-style outlier decomposition.

## Software Stack By Layer

| Layer | Tool | Role |
|---|---|---|
| Framework | TorchAO | `NVFP4DynamicActivationNVFP4WeightConfig`, hard-checks SM100+ |
| Kernel templates | CUTLASS | `72b_blackwell_nvfp4_nvfp4_gemm.cu`, block-scaled examples |
| Attention/MoE | FlashInfer | `nvfp4_quantize(...)` utility, fused NVFP4 kernels |
| Training recipes | Transformer Engine | "Current scaling" from current stats, NVFP4/MXFP recipes |
| Serving | vLLM | Dynamic per-token FP8, NVFP4 via FlashInfer on SM100+ |
| Compact GEMM | DeepGEMM | `sm100_fp8_gemm_1d1d` with packed UE8M0 scale layout |

## Precision Decision Pivots

Ask these before recommending a Blackwell precision rewrite:

1. Does the intended kernel backend accept the exact value format?
2. Does it accept the exact scale format and block size?
3. Does it require a GEMM-swizzled or otherwise non-naive scale layout?
4. Is the contest or application format actually the same as the library
   path, or is an explicit scale repack required?
5. Is the fallback plain FP8 or BF16 path still available if the contract
   does not line up?
6. Is the precision question actually about throughput headroom, or is it
   really about accuracy sensitivity of specific layers (MLP vs attention)?

## What This Should Prevent

This subskill should stop the user from:

- treating block-scaled FP8, NVFP4, MXFP8, and MXFP4 as one generic
  "Blackwell low-precision" bucket
- assuming a CUTLASS example and a cuDNN or Transformer Engine path have the
  same scale layout
- blaming hardware when the real problem is a format or layout mismatch
- reaching for NVFP4 on MLP/MoE GEMMs when the accuracy evidence says those
  are the most sensitive layers
- confusing the NVFP4 two-level scale contract (global FP32 + per-block FP8)
  with the contest's single-level FP8 + block-128 scheme

## Where To Escalate

- Use `references/b200-sm100-subskill.md` when the precision choice also
  depends on TMEM, 1SM vs 2SM scheduling, or cluster-aware kernel structure.
- Use `references/cutlass-hw-source-map-subskill.md` when the user needs to
  know where the backend compatibility or schedule rules live in source.
