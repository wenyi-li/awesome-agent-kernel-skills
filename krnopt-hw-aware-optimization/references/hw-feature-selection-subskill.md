# Hardware Feature Selection Subskill

Use this integrated subskill when the first question is "which hardware
features should matter for this kernel on this GPU?"

## Core Rules

- Start from workload shape, not from feature hype.
- A hardware feature matters only if the code can satisfy its contracts.
- Feature choice should change the kernel structure, not just the vocabulary.
- Keep a fallback path in mind in case the hardware-native route is a bad fit.
- Separate instruction choice from scheduler choice and from
  precision-contract choice.

## Ordered Questions

1. What is the target GPU and compute capability?
2. What kind of kernel is this: GEMM, attention, reduction, dispatch, sparse
   routing, epilogue-heavy, or something else?
3. Which of these hardware surfaces plausibly matter?
   - instruction family
   - accumulator location
   - staging path
   - scheduler or launch path
   - precision and scale-layout contract
4. Which generation branch is actually relevant: pre-SM90, SM90, or SM100?
5. What code or launch structure is required to use each candidate well?
6. What is the likely payoff, and what could go wrong?

## Generation-First Decision Tree

Once the target hardware is resolved, work through these branches rather
than debating every feature in the abstract.

### Branch A: pre-SM90 (Ampere SM80 or earlier)

The central features are:

- `cp.async` staging for global-to-shared copy without register hops,
  typically with `cuda::pipeline` and `cuda::memcpy_async` for N-stage
  prefetch
- Tensor Core MMA appropriate to the generation (`mma.m16n8k16` family on
  SM80, etc.)
- No TMA, no WGMMA, no TMEM; accumulators live in registers
- No native FP8 tensor-core path on A100/SM80; treat FP8 as packed storage
  and dequantize to BF16/FP16 before the GEMM

Typical feature-to-workload mapping:

- regular GEMM or attention: `cp.async` multi-stage pipeline + register-accum
  MMA; CUTLASS SM80 templates, or TensorRT-LLM SM80 launchers such as
  `fused_moe_gemm_launcher_sm80`
- MoE grouped with FP8 storage: dequantize in the GEMM prologue, then plain
  tensor-core GEMM

### Branch B: SM90 Hopper (H100 / H20 / H800)

Treat H100, H20, H800, and other SM90-family devices as Hopper for feature
selection, but resolve local device limits before copying H100 capacities or
throughput assumptions. Blackwell-only surfaces such as TMEM, CLC, 2-CTA MMA,
and `tcgen05.mma` do not apply here.

Candidate features, in rough order of relevance:

1. Instruction family: WGMMA for warpgroup async MMA
2. Staging path: TMA when tiles are regular; `cp.async` for irregular
   residuals
3. Accumulator location: registers (Hopper has no TMEM)
4. Scheduler: static, DeepGEMM-style static persistent, or Stream-K for
   long-K imbalance; no CLC on SM90
5. Warp roles: producer/consumer warp specialization is the usual high-end
   shape
6. Precision: plain FP8, block-scaled FP8 via Hopper-compatible CUTLASS
   paths, or a Transformer Engine recipe; no native microscaling with
   FP8-E4M3 per-block scales

Workload-to-feature triggers:

- attention with long K and good TMA fit: FlashAttention-3 shape (producer
  warpgroup doing TMA, consumer warpgroups doing WGMMA + softmax,
  interleaved GEMM/softmax pipelining)
- grouped GEMM for routed experts: CUTLASS example 68-style pointer-array
  grouped blockwise FP8 with on-device TMA descriptor modification, or
  TMA-adaptive pattern for variable residuals; use DeepGEMM SM90 1D2D/1D1D
  and scheduler sources as compact H20-oriented references
- back-to-back kernel pipelines: PDL with `overlap_ratio` and
  `prefetch_ratio` knobs around CUTLASS weight prefetch

### Branch C: SM100 data-center Blackwell (B200)

Candidate features, in rough order of impact:

1. Instruction family: `tcgen05.mma` variants (`f16`, `bf16`, `f8f6f4`,
   `mxf8f6f4.block_scale`), 2-4x faster than Hopper WGMMA
2. Accumulator location: TMEM, addressed as `lane = addr[31:16]`,
   `col = addr[15:0]`; 256 KB per SM, column allocation, power-of-two
   minimum 32 columns
3. CTA cooperation: 1SM vs 2SM (`cta_group::2`) is a real execution-contract
   branch
4. Staging path: TMA with SM100 extensions (2SM TMA, `tma_gather4`,
   tensormap replacement); `cp.async` still available for irregular parts
5. Scheduler: CLC for persistent dynamic load balancing at cluster
   granularity; DeepGEMM-style static persistent also legitimate
6. Cluster features: thread-block clusters up to 8 CTAs portable, 16 CTAs
   nonportable; DSMEM for cross-CTA SMEM sharing inside a cluster
7. Precision contracts: block-scaled FP8 / NVFP4 / MXFP8 / MXFP4 with
   hardware-enforced scale layout; often `UE8M0` packed scales and
   GEMM-swizzled scale tensors
8. Cross-kernel: PDL + CUDA Graphs for MoE launch-overhead elimination
9. Scalar-side: software exp emulation when MUFU at 16 ops/clock/SM becomes
   a softmax co-bottleneck versus MMA at ~8192 ops/clock/SM

Workload-to-feature triggers:

- throughput-first dense GEMM with large uniform tiles: 2SM `tcgen05.mma`
  with TMEM accumulator and TMA staging, CUTLASS builder schedule from
  example 71
- grouped GEMM with many small experts: 1SM `tcgen05.mma` pointer-array
  grouped pattern (CUTLASS example 75), optionally with CLC for dynamic
  load balancing and tensormap replacement for per-expert descriptors
- attention with long K and softmax-bound: FlashAttention-4 shape (TMEM P/S,
  separate correction warpgroup, software exp emulation, optional 2-CTA)
- low-latency decode: cluster shape `1x1xMAX_SPLITS`, flash-decoding-style
  work partitioning over KV length, DSMEM-based cluster reduction (CUTLASS
  example 93)
- FP8 + block-scale-128 (contest shape): verify library backend
  (`mxf8f6f4.block_scale` in CUTLASS, DeepGEMM's `sm100_fp8_gemm_1d1d`,
  cuDNN Frontend grouped-blockwise, cuBLAS 12.9) accepts the contest block
  size before committing

### Branch D: Unknown or mismatched hardware

Stop. Do not recommend TMEM-, CLC-, or block-scaled-specific rewrites until
the compute capability is resolved. Consumer Blackwell (CC 12.x, RTX
50-series) does not have TMEM; datacenter Blackwell (CC 10.0) does. Mixing
them produces wrong advice. If a code example comes from RTX 50-series
work, do not assume its tile and storage choices transfer upward unchanged
to B200.

## Surface-By-Surface Quick Map

For a given workload, these are the surfaces to check in order:

### Instruction family

- pre-SM90: generation-appropriate `mma.*`
- SM90: WGMMA
- SM100: `tcgen05.mma` (`f16`, `bf16`, `tf32`, `f8f6f4`,
  `mxf8f6f4.block_scale`, `i8`, `u8`; dense and `.sp` sparse variants across
  `cta_group::1` and `cta_group::2`)

### Accumulator location

- pre-SM90 and SM90: registers (mainloop designed around register pressure)
- SM100 datacenter: TMEM (mainloop can be nearly register-free; epilogue
  becomes the first-class tuning problem; TMEM readback is warp-scoped, so
  full-tile epilogues often need warpgroup coordination)
- SM100 consumer (CC 12.x): no TMEM

### Staging path

- pre-SM90: `cp.async` with `cuda::pipeline`
- SM90: TMA for regular tiles; `cp.async` for irregular residuals
- SM100: TMA plus 2SM TMA, `tma_gather4`, tensormap replacement, L2 cache
  hints via `cache_hint`; `cp.async` still available

### Scheduler / launch path

- static tile ownership: default when tiles are uniform
- static persistent: launch-overhead-sensitive or expert-count-variable
  grouped problems (DeepGEMM pattern works on SM90 and SM100)
- Stream-K: long-K imbalance on SM90 / SM100
- CLC: SM100-only, cluster-granularity dynamic persistent scheduling
- PDL: cross-kernel overlap, typically with CUDA Graphs

### Precision / scale-layout contract

- plain FP8 or BF16: simplest, fewest contracts, fallback for mismatches
- block-scaled FP8: contest format uses block 128 FP32 scales; libraries
  often use 32 or 16, UE8M0 packed, possibly GEMM-swizzled; CUTLASS
  `tcgen05.mma.kind::f8f6f4` and `mxf8f6f4.block_scale`, cuBLAS 12.9,
  DeepGEMM `sm100_fp8_gemm_1d1d`, cuDNN Frontend, TensorRT-LLM
  `fp8_blockscale_gemm/`
- NVFP4: FP4 E2M1 with block-16 + FP8 E4M3 scales + optional FP32 global;
  TorchAO, FlashInfer, CUTLASS
  `72b_blackwell_nvfp4_nvfp4_gemm.cu`
- MXFP8 / MXFP4: OCP MX spec, block 32, UE8M0 power-of-two scales; QuTLASS
  and MicroMix for format-optimized PTQ; MXFP4 is ~15% faster than NVFP4 on
  B200 but lower accuracy

## Pipeline And Warp-Specialization Are Coupled

A repeated lesson from modern tensor-core kernels: do not design software
pipelining and warp specialization separately. Tile shape, stage count, copy
path, warp role split, accumulator home, and epilogue structure all
interact. Blackwell intensifies the coupling because tiles and throughput
targets are larger, TMEM shifts where accumulator pressure lives, non-MMA
work often needs explicit movement between TMEM and registers, and
scheduler structure matters more for keeping the math path busy.

Practical design loop:

1. choose a plausible tile and memory path
2. decide where accumulators live and how the epilogue will read them back
3. search stage count and warp-role split together
4. profile for stalls, register pressure, and underfed tensor cores
5. only then refine low-level details

## What This Should Prevent

This should stop the user from:

- forcing an architecture feature onto the wrong workload
- assuming a feature is free once a library exposes it
- mixing Hopper and Blackwell guidance as if they had the same instruction
  and accumulator model
- mixing datacenter SM100 and consumer SM120 guidance as if they had the
  same storage and feature surface
- overlooking descriptor, layout, cluster, or synchronization contracts
- overlooking scale-layout and dtype-contract mismatches on Blackwell
- treating architecture-specific tricks as replacements for correctness or
  profiling
- porting Hopper schedules to Blackwell by only enlarging tiles; the whole
  schedule question should be reopened at the architecture break

## Escalation

- Use `krnopt-cuda-coding` when the question is still mostly generic source
  structure.
- Use `krnopt-cuda-profiling` when runtime dominance or stall cause still
  needs measurement.
- Use the matching SM90 or SM100 reference when the generation branch is
  clear and the question now depends on that hardware's specific constraints.
- Use `references/blackwell-precision-contracts-subskill.md` whenever the
  decision involves FP8 block scaling, NVFP4, or MX-format layout contracts.
- Use `references/scheduler-and-launch-control-subskill.md` when the
  question has narrowed to scheduling model.
