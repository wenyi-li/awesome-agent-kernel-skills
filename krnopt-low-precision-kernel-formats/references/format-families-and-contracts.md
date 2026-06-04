# Format Families And Contracts

Use this reference when the task is to understand what low-precision contract a
kernel actually obeys.

## Format Families

### 1. Plain FP8

- value format is FP8, commonly E4M3 or E5M2
- scale may be absent or managed outside the kernel contract
- useful when the user is asking about dtype path, Tensor Core path, or plain
  FP8 storage

### 2. Block-Scaled FP8

- FP8 values plus explicit scale factors shared by blocks of values
- scale granularity is part of the ABI
- the contest path belongs here, not under NVFP4
- kernel implications:
  scale layout, block granularity, and dequant placement all matter

### 3. NVFP4

- Blackwell-era FP4 family with FP8 scales and hardware-native microscaling
- not the same contract as contest FP8 block scaling
- kernel implications:
  hardware-native path, specific block contract, and tighter hardware/library
  coupling

### 4. MXFP4 / MXFP6 / MXFP8

- Blackwell microscaling family with mixed-precision variants
- often appears in hardware-aware or library-specific discussions
- kernel implications:
  format choice is often architecture-specific and should often hand off to
  `krnopt-hw-aware-optimization`

### 5. Dynamic Or Transparent Quantization

- runtime computes scales or chooses lower-precision execution internally
- may preserve a high-level model interface while changing kernel-side format
  handling
- kernel implications:
  quant or dequant may be fused into prologues, epilogues, or library-managed
  paths

### 6. Low-Precision KV-Cache Formats

- separate domain from GEMM or MoE weight-activation formats
- common question is whether decode kernels read compressed cache directly or
  dequant into a higher-precision path
- route decode-specific bottleneck or implementation questions to the relevant
  profiling or coding skill after the format is clear

## Questions To Answer Before Routing Elsewhere

Resolve these first:

1. what are the stored values
2. what are the stored scales
3. what is the scale granularity
4. what layout or packing contract does the consumer kernel expect
5. where is dequant or requant fused

## Routing Table

| Question type | Owning skill |
| --- | --- |
| "What format is this?" | `krnopt-low-precision-kernel-formats` |
| "What scale layout does the kernel ABI require?" | `krnopt-low-precision-kernel-formats` |
| "Why is this quantized path slow?" | `krnopt-cuda-profiling` |
| "What experiment should we try next?" | `krnopt-cuda-generic-optimization` |
| "How do I implement the kernel for this format?" | `krnopt-cuda-coding` |
| "Should I choose NVFP4, MXFP4, or another Blackwell-native path?" | `krnopt-hw-aware-optimization` |

## Contest Guardrail

The contest's low-precision path is not a generic "Blackwell low-precision"
question by default.

Treat:

- contest FP8 block-scaled formats as contest ABI and kernel-contract questions
- NVFP4 or MXFP formats as Blackwell-specific format families that may inspire
  design choices but are not automatically valid substitutes
