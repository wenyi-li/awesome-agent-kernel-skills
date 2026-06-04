# GPU Utilization Enhancement Subskill

Use this subskill when the user asks to improve GPU utilization, low SM utilization, low Tensor Core utilization, poor useful work per byte, underfilled launches, padding waste, decode launch overhead, or low-precision kernels that do not speed up.

## Core Framing

Do not treat high occupancy as the goal. The goal is higher useful work per byte moved and per launch, with the limiting resource made explicit.

Classify which resource is underused or overused before proposing edits:

- Tensor Core / MMA path: GEMM-like work exists, but tensor pipe issue or active cycles are low.
- CUDA core / SFU path: scalar math, dequantization, reductions, exponentials, predicates, or epilogues dominate.
- HBM / L2 path: global bytes or sectors dominate useful FLOPs.
- SMEM / MIO path: shared-memory transactions, bank conflicts, or MIO throttle dominate.
- Register / local-memory path: spills or excessive live state block occupancy or add hidden memory traffic.
- Launch / runtime path: CPU gaps, many tiny kernels, graph capture misses, or descriptor planning dominate.
- Scheduler / shape path: ragged groups, tail waves, too few CTAs, or uneven work assignment leave SMs idle.

Count only useful work. Treat padding, masked lanes, repeated layout conversions, standalone dequant/requant, intermediate tensor round trips, redundant metadata loads, and descriptor setup as waste until proven otherwise.

## Intake Additions

When utilization is the stated goal, add these fields to the normal bottleneck intake:

- `utilization_target`: the resource or pipeline the experiment intends to feed better.
- `waste_source`: padding, fragmented launches, tail waves, low reuse, scale/dequant traffic, page metadata, spills, non-MMA epilogue work, or conversion-only kernels.
- `architecture_gate`: features required, such as `cp.async`, `wgmma`, `tma`, `tcgen05`, TMEM, CUDA Graphs, or specific low-precision MMA support.
- `confirmation_metric`: the counter or timing signature that should improve if the experiment works.

Avoid recommending occupancy tuning by itself unless evidence shows too few eligible warps, too few resident CTAs, or tail-wave underfill. Occupancy that increases memory pressure, spills, or SMEM conflicts can make utilization worse.

## Symptom To Experiment Map

| Symptom | Primary experiment | Why it can improve utilization | Confirmation |
| --- | --- | --- | --- |
| GEMM-shaped code has low Tensor Core utilization | First verify a vendor/library MMA path is not available; then retile CTA/warp/MMA shapes, increase K-stage pipelining, and select the correct dtype/intrinsic/build arch | Tensor Cores stall when operands are not staged and issued at the right granularity | Tensor pipe throughput and MMA issue rate rise; wall time falls |
| HBM throughput is high while SM/Tensor Core throughput is low | Reduce bytes per useful FLOP through fusion, tiling reuse, vectorized/coalesced loads, compacted padding, or valid lower precision | Moving the same data fewer times raises arithmetic intensity | DRAM bytes/sectors per output fall; useful FLOPs unchanged |
| L2 traffic dominates repeated operand reads | Reorder work for reuse, stage hot operands through SMEM/registers, improve page/block locality, or batch shapes with similar metadata | L2 helps only when reuse distance and access pattern allow hits | L2 hit rate or bytes per useful output improve |
| SMEM/MIO throttle, bank conflicts, or shared wavefronts dominate | Change SMEM layout with padding/swizzle, widen shared-memory ops, reduce SMEM round trips, or keep more values in registers | Tensor Cores can starve behind shared-memory issue or bank conflict pressure | MIO throttle and shared conflict metrics fall |
| Few CTAs, ragged groups, or tail waves leave SMs idle | Use persistent scheduling, grouped work queues, split-K/Stream-K, cooperative launch control when available, or merge compatible small jobs | Work distribution, not raw math, is limiting residency and tail utilization | Active SMs, eligible warps, and tail-wave behavior improve |
| Decode or serving path shows CPU launch gaps | Use CUDA Graphs with shape buckets, plan/run separation, cached descriptors, or fuse tiny kernels | GPU can sit idle between short kernels even when each kernel is locally efficient | Kernel count and CPU launch gaps fall; graph replay succeeds |
| Low precision reduces bytes but not time | Fuse unpack/dequant/scale into the MMA path or epilogue, use native block-scale layouts where supported, and remove conversion-only kernels | Quantization helps only if conversion overhead does not become the bottleneck | HBM bytes fall without scalar/SFU or instruction overhead taking over |
| Epilogue, reductions, softmax, or transcendental math dominates | Simplify epilogue expressions, use stable fused reductions, reduce SFU calls, or split/fuse only after checking register pressure | Non-MMA work can cap end-to-end utilization even when GEMM is fast | SFU/scalar instruction share falls and total time improves |

## Technique Notes

Prefer standard primitives before custom rewrites for GEMM-like kernels. cuBLASLt, CUTLASS/CuTe, cuDNN attention paths, and runtime-specific fused kernels often already implement architecture-specific tiling, epilogues, split-K, grouped scheduling, and low-precision MMA paths. A custom kernel needs a clear reason, such as unsupported shape, fusion boundary, metadata layout, or launch/runtime constraint.

For attention and decode-like kernels, utilization usually comes from avoiding materialization and improving KV-cache locality rather than increasing raw occupancy. Online softmax, tiled QK/V processing, split over sequence length, page-aware scheduling, and graph-compatible planning are utilization techniques because they reduce HBM traffic and expose enough independent work.

For MoE and grouped work, utilization usually comes from converting many small irregular expert jobs into fewer high-occupancy grouped or persistent kernels. The key waste sources are launch count, padding to uniform expert shapes, unbalanced expert loads, gather/scatter traffic, and scale-factor layout for low precision.

For low-precision kernels, smaller operands are not sufficient. Check where unpacking, scale loads, dequantization, requantization, and accumulator conversion execute. If these run as standalone kernels or dominate scalar instructions, the byte savings may not reach useful Tensor Core work.

## Output Pattern

For a utilization-focused answer, include:

- The diagnosed utilization target.
- The likely waste source.
- One primary experiment and the next fallback experiment.
- Why the experiment should improve useful work per byte, per launch, or per active SM cycle.
- The exact confirmation and rejection metrics.
- The architecture gate and build/runtime assumptions.
- A routing note if the next step belongs in hardware-aware, structural, domain-specific, low-precision, profiling, coding, or local integration work.
