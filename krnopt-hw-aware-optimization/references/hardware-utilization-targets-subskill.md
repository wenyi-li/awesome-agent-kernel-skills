# Hardware Utilization Targets Subskill

Use this reference when the user asks how to enhance GPU utilization on a specific NVIDIA GPU, or when the hardware-aware question is about low SM utilization, low Tensor Core utilization, underfed WGMMA/`tcgen05`, memory-pipe saturation, scheduler underfill, or launch gaps.

## Source Basis

- Official CUDA methodology:
  [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/),
  [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),
  and [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).
- Architecture references:
  [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
  and [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-tuning-guide/contents.html).
- Blackwell programming references:
  [CUTLASS](https://github.com/NVIDIA/cutlass),
  [Colfax Blackwell TMEM GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/),
  and [Colfax Blackwell block-scaling tutorial](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/).
- Microarchitecture context:
  [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789)
  and [Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures](https://arxiv.org/abs/2605.04178).
- Distilled idea: utilization work must name the limiting path, useful-work
  waste, hardware surface, and confirmation metric. Occupancy alone is not a
  target, and SM100/B200 features must not be assumed on consumer Blackwell
  devices.

## Core Framing

Hardware-aware utilization means choosing the architecture surface that increases useful work on the limiting hardware path. It is not the same as maximizing occupancy.

For each plan, name:

- `utilization_target`: Tensor Core / MMA, CUDA core / SFU, HBM, L2, SMEM / MIO, registers or TMEM, scheduler occupancy, or launch/runtime.
- `waste_source`: padding, masked lanes, low tile reuse, descriptor churn, scale/dequant traffic, spills, tail waves, fragmented launches, or non-MMA epilogue work.
- `hardware_surface`: instruction family, accumulator location, staging path, scheduler/launch control, precision/scale contract, or cluster feature.
- `confirmation_metric`: the measured counter or timing change that would prove the hardware feature is actually helping.

Prefer the feature that changes the limiting path. Do not recommend TMA, WGMMA, TMEM, CLC, or low-precision MMA only because the GPU exposes it.

## Surface Map

| Utilization target | Hardware-aware question | Likely surfaces | Main risk |
| --- | --- | --- | --- |
| Tensor Core / MMA | Are tensor cores idle because operands, tiles, or instruction family are wrong? | SM80 `mma`, SM90 WGMMA, SM100 `tcgen05.mma`, 1SM/2SM MMA mode | Register/TMEM pressure, poor staging, epilogue bottleneck |
| HBM / L2 bytes | Is each useful FLOP moving too many bytes? | TMA/cp.async staging, L2-aware scheduling, lower precision, fusion boundary | Layout conversion or scale traffic erases byte savings |
| SMEM / MIO | Is shared-memory movement starving compute? | SMEM swizzle/padding, wider shared ops, fewer SMEM round trips, TMA layout | Bank conflicts, MIO throttle, over-large stages |
| Register / TMEM | Are accumulators or live state limiting residency? | Smaller tiles, split epilogues, SM100 TMEM, epilogue warpgroup planning | Spills on SM90, TMEM readback/column pressure on SM100 |
| Scheduler occupancy | Are SMs idle because work is ragged or tail-heavy? | Persistent kernels, Stream-K, SM100 CLC, grouped scheduling, cluster shape | More scheduling overhead than useful work |
| Launch/runtime | Is the GPU idle between many short kernels? | CUDA Graphs, PDL, plan/run separation, cached descriptors, persistent execution | Graph capture misses, too many shape buckets |
| CUDA core / SFU | Does scalar work cap a tensor-core kernel? | Epilogue simplification, softmax scheduling, software exp path, split/fuse tradeoff | Moving work off SFU can increase memory or register pressure |

## Architecture Branches

### Pre-SM90

Use generation-appropriate Tensor Core MMA and `cp.async` staging. Utilization wins usually come from coalescing, multi-stage `cp.async`, tile shape, avoiding spills, and making enough CTAs to cover latency. Do not use TMA, WGMMA, TMEM, or CLC vocabulary.

Useful when:

- regular GEMM or attention has enough tile reuse for `cp.async` staging
- HBM traffic dominates and shared-memory tiling can reduce bytes per useful output
- FP8 is only a storage format and must be dequantized into a supported MMA path

Avoid when:

- the workload is too small or jagged to amortize staging
- register pressure from larger tiles lowers eligible warps enough to lose

### SM90 / Hopper

SM90 utilization is usually about keeping WGMMA fed without letting register accumulators, SMEM pressure, or TMA descriptor work dominate.

Primary surfaces:

- WGMMA for warpgroup asynchronous tensor-core work
- TMA for regular multidimensional global-to-shared tile movement
- warp-specialized producer/consumer pipelines to overlap TMA, WGMMA, and scalar side work
- static persistent or Stream-K scheduling for ragged grouped work or long-K imbalance
- SM90-compatible FP8 and block-scaled layouts with explicit scale metadata

Use WGMMA/TMA when:

- the work is GEMM-like and has enough M/N/K tile size to amortize setup
- operands can be staged in layouts that feed WGMMA
- producer/consumer roles can overlap copy, compute, and epilogue
- register-resident accumulators do not cause spills or collapse occupancy

Avoid or downshift when:

- tiny decode tiles, jagged residuals, or descriptor churn dominate
- scalar softmax/epilogue work leaves WGMMA waiting
- larger WGMMA tiles spill or reduce resident warpgroups too much
- cluster/DSMEM usage reduces occupancy more than it improves reuse

Confirmation should include tensor-core issue or active cycles, TMA/SMEM stalls, register spills, achieved occupancy or eligible warps, and wall time on the target workload.

### SM100 / B200 Data-Center Blackwell

SM100 utilization is usually about exploiting `tcgen05.mma` and TMEM without moving the bottleneck to epilogue readback, scale layout, CLC overhead, or non-MMA scalar work.

Primary surfaces:

- `tcgen05.mma` instruction variants, including block-scaled low-precision modes
- TMEM accumulator storage and warp-scoped readback
- 1SM versus 2SM MMA execution contracts
- TMA extensions such as 2SM TMA, `tma_gather4`, tensormap replacement, and cache hints
- CLC for cluster-level dynamic persistent scheduling
- DSMEM and cluster shapes for cross-CTA reuse or decode reductions
- native low-precision scale-layout contracts for FP8/FP6/FP4/MX/NVFP4 paths

Use SM100-specific paths when:

- dense or grouped GEMM is large enough to benefit from wider `tcgen05` throughput
- accumulator register pressure is a real limiter and TMEM can move the pressure out of registers
- 2SM mode improves throughput for large uniform tiles
- 1SM mode fits many smaller grouped experts better than 2SM
- CLC or cluster scheduling addresses measured ragged work or tail-wave underfill
- native block-scale layout removes standalone dequant/scale kernels

Avoid or downshift when:

- the epilogue cannot efficiently read TMEM or needs too much register state
- 2SM mode lowers scheduling flexibility for small or irregular groups
- CLC overhead exceeds the imbalance it removes
- scale conversion, packing, or layout swizzling consumes the low-precision win
- the target is consumer Blackwell / CC 12.x, where SM100 TMEM and CLC assumptions may not apply

Confirmation should include `tcgen05`/tensor pipe throughput, TMEM allocation/readback pressure where available, eligible warps, tail-wave or cluster occupancy, memory bytes per useful output, and end-to-end time.

## Workload-Specific Utilization Pivots

- Dense GEMM: choose the architecture-native MMA path first, then tune tile shape, staging, accumulator home, and epilogue pressure.
- Attention prefill: prioritize avoiding materialized QK, staging Q/K/V tiles efficiently, and overlapping matmul with softmax or rescale work.
- Decode attention: prioritize split over KV length, cluster or block scheduling, graph-compatible planning, and KV-cache locality over raw occupancy.
- Grouped GEMM / MoE: prioritize persistent or grouped scheduling, padding reduction, scale layout, and one-kernel execution before hand-tuning individual expert tiles.
- Quantized linear: prioritize native MMA scale contracts or fused dequant/scale; byte savings do not count if scalar conversion becomes the bottleneck.
- Epilogue-heavy kernels: check whether the architecture-native MMA path is already fast enough that scalar/SFU/register pressure is now the limiter.

## Output Pattern

For hardware-specific utilization plans, include:

- Target GPU and exact compute capability branch.
- Utilization target and waste source.
- Hardware surface selected and the code shape it requires.
- Why that surface should increase useful work per byte, per launch, or per active hardware cycle.
- Architecture-specialized build target, such as `sm_90a` or `sm_100a`, when required.
- The metrics that would confirm or reject the utilization hypothesis.
- Fallback if the hardware-specific feature is a poor fit.
