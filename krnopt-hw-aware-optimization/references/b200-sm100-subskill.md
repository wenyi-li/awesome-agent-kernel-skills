# B200 SM100 Subskill

Use this integrated subskill when the target hardware is NVIDIA B200 / SM100
Blackwell and the question is which hardware features should change how the
kernel is written or restructured.

This subskill is deliberately opinionated: B200 optimization is usually not
"pick the biggest tensor core path and pray." The hardware rewards kernels that
budget TMEM, shared memory, epilogue work, and scheduler shape as carefully as
they budget MMA throughput.

## B200 Facts That Change Kernel Decisions

Treat these as first-order constraints, not trivia:

- `148` SMs on B200
- `228 KB` SMEM per SM, but only `227 KB` per CTA because CUDA reserves `1 KB`
  per thread block
- `256 KB` TMEM per SM, organized as `512 columns x 128 lanes` of 32-bit cells
- `64K` 32-bit registers per SM, `255` max registers per thread
- `64` warps per SM, up to `32` thread blocks per SM
- portable thread-block cluster size `8 CTAs`; nonportable up to `16 CTAs`
  via `cudaFuncAttributeNonPortableClusterSizeAllowed`
- BF16 MMA throughput around `8192 ops/clock/SM`
- MUFU throughput stays at `16 ops/clock/SM`, same as Hopper
- SMEM read bandwidth stays at `128 B/clock/SM`, same as Hopper
- HBM3/3e up to `180 GB`; L2 cache `126 MB` on GB200 with persistence control
  retained from Ampere
- SMEM carveout set: `0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB`, configured
  via `cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout)`
- static SMEM cap per kernel is still `48 KB`; dynamic SMEM above `48 KB`
  requires explicit opt-in

The key asymmetry is the whole game:

- MMA throughput improved much faster than SMEM bandwidth, exp throughput, and
  general scalar side work

So on B200, optimization often means:

- reducing non-MMA work
- overlapping non-MMA work aggressively
- moving accumulator lifetime out of registers
- making epilogues, correction logic, and scheduler policy explicit design
  choices

## B200 Is CC 10.0, Not Generic "Blackwell"

Do not blur B200 / SM100 with consumer Blackwell.

- B200 is the data-center CC `10.0` path this repo cares about
- consumer Blackwell CC `12.x` (RTX 50-series, SM120) has a different resource
  surface: max `48` warps per SM, SMEM per SM only `128 KB`, and no TMEM
- the Colfax UMMA/TMEM GEMM tutorial explicitly targets SM100, SM101, SM103

That changes recommendations immediately:

- TMEM-heavy UMMA guidance applies to `SM100`, `SM101`, `SM103` datacenter
  parts only
- cluster and shared-memory budgets should be reasoned about with CC `10.0`
  limits, not generic "Blackwell" assumptions
- if a code example comes from RTX 50-series work, do not assume its tile and
  storage choices transfer upward unchanged to B200

When someone says "the Blackwell paper says X," the first question should be:
which Blackwell paper, and on which class of device? The consumer microbench
work (RTX 5080 / GB203) and the datacenter microbench work (B200 / H200)
answer different kernel-authoring questions. Use the datacenter paper for
TMEM-heavy, UMMA-heavy, and SM100 scheduling questions; use the consumer
paper only when the question is what survives outside SM100.

## Compile Targeting Is Part Of The Plan

CUDA 12.9 introduces three practical target modes that matter for Blackwell:

| Target style | Example | Use when | Portability |
|---|---|---|---|
| generic | `compute_100` | kernel does not require family-specific features | normal PTX/cubin compatibility |
| family-specific | `compute_100f` | kernel uses Blackwell family features but should still run on compatible later minor variants | same major family, equal or higher minor CC |
| architecture-specific | `compute_100a` | kernel depends on exact architecture-specific behavior; portability intentionally sacrificed | exact architecture only |

Default rule for this skill: when the kernel relies on Blackwell-specific
instructions or features, make the specialized target explicit in the build.
Use `sm_100a`, `compute_100a`, or the build-system equivalent for exact
architecture-specific paths, and do not use a generic `sm_100` build as
evidence that the specialized path is enabled. `100f` can be relevant when a
deployment deliberately wants family-specific forward compatibility, but the
kernel handoff must still state that choice and why it is sufficient for the
selected feature surface. The practical pattern is to keep a generic
implementation path, build the specialized fast path with explicit targeting,
and benchmark and validate them separately.

## Most Useful B200-Specific Features

Treat these as the first B200 feature set to consider:

- TMA for async multidimensional tile movement, extended with 2SM TMA and
  `tma_gather4`
- TMEM for accumulator placement separate from registers
- 2-CTA MMA (`cta_group::2`) for larger cooperative tensor-core tiles
- thread-block clusters and DSMEM for cross-CTA cooperation
- Cluster Launch Control for persistent scheduling
- `tcgen05.mma` low-precision and block-scaled paths
- Blackwell-native block-scaled FP8 / MXFP8 / NVFP4 layout contracts
- Software-exp emulation when MUFU bandwidth becomes a co-bottleneck

## Feature To Code Mapping

### TMA On B200

Use when:

- tiles are multidimensional and regular enough for descriptor-driven movement
- global-to-shared traffic is a major structural concern
- the kernel can afford explicit producer/barrier choreography

SM100 extends TMA with several primitives worth knowing by name:

- 2SM TMA (`cp.async.bulk.tensor.Nd.cta_group::2`): both peer CTAs execute the
  TMA op, hardware routes data, barrier update targets only CTA0 via
  `Sm100MmaPeerBitMask = 0xFEFFFFFF`
- `tma_gather4` (`cp.async.bulk.tensor.2d.tile::gather4`): fetches 4
  non-contiguous rows in a single TMA op; critical for sparse attention and
  any gather-heavy staging
- Tensormap replacement (`tensormap.replace.tile.global_address`): dynamically
  updates TMA descriptors in SMEM, used in per-expert routing in DeepGEMM
- L2 cache hints via a `uint64_t cache_hint` parameter

Implications for code:

- explicit async copy pipeline
- explicit barrier choreography
- descriptor lifecycle and update strategy matter
- multicast / cluster behavior may become part of the kernel contract

Likely wins:

- reduced GMEM-to-SMEM round trips and clean multi-stage pipelines
- 2SM TMA can feed 2-CTA MMA with cooperative operand staging
- tensormap replacement lets one kernel serve many experts without per-expert
  host-side descriptor work

Likely traps:

- using TMA when descriptors churn so often that management overhead dominates
- ignoring the possibility that a more direct copy path is better for small,
  irregular operands; CUTLASS MoE examples explicitly call out cases where
  `cp.async` for an irregular decode-side operand beats a pure TMA design
- forgetting that the public microbench repos separate `ldgsts_*` and
  `tma2d_*` for a reason: benchmark the simple copy path and the
  descriptor-driven tile path and pick the one that matches the movement
  pattern

Fallback path:

- `cp.async`-style pointer-driven loads for irregular residuals or
  decode-side operands; mixed TMA + `cp.async` kernels are legitimate on SM100

### TMEM

Use when:

- tensor-core accumulators are large enough that register writeback becomes a
  structural problem
- the kernel needs accumulator or correction state to persist across a long
  mainloop without crushing registers

TMEM is a 256 KB on-chip memory per SM introduced on SM100, specifically
designed for storing MMA accumulator results. It is fundamentally different
from SMEM.

Critical B200 details:

- TMEM is addressed as `lane = addr[31:16]`, `col = addr[15:0]`
- TMEM is organized as `512 columns x 128 lanes` of 32-bit cells
- allocation is column-based, power-of-two, minimum `32` columns (16 KB)
  granules via `TMEM::Allocator1Sm` or `Allocator2Sm`
- warp 0 typically allocates TMEM and broadcasts the base pointer through
  SMEM; `tcgen05.alloc` writes the allocated base address into shared memory
- each warp only accesses `32` of the `128` TMEM lanes; full-tile readback
  often needs warpgroup coordination
- fencing is mandatory: `tcgen05_before_thread_sync()` and
  `tcgen05_after_thread_sync()` around every `__syncthreads()` that crosses
  TMEM access
- movement primitives are `tcgen05.ld`, `tcgen05.st`, and `tcgen05.cp`;
  loading to registers uses `tmem_ld_32dp32bNx<N>` and storing back uses
  `tmem_st_32dp32bNx<N>`
- deallocation requires `release_allocation_lock()` before `free()` to
  unblock the next CTA
- TMEM is scarce and SM-level: a full 512-column allocation limits CTA
  occupancy to 1

Implications for code:

- accumulator lifetime is now a TMEM design concern
- occupancy can collapse if TMEM allocation is too large
- epilogue organization becomes a first-class design problem because TMEM
  readback is warp-scoped
- shared-memory communication may be needed just to distribute TMEM base
  addresses
- `tcgen05.ld` shape and `.num` choices trade instruction count against
  register pressure

Likely wins:

- mainloops can become almost register-free because MMA accumulation stays in
  TMEM
- register pressure can fall in the main MMA loop, freeing occupancy for
  staging or role-specialized warps
- decoupled rescaling patterns become possible: FlashAttention-4 stores P and
  S matrices in TMEM and uses a separate correction warpgroup

Likely traps:

- treating TMEM as "free accumulator space" instead of scarce SM-level state
- moving accumulators into TMEM but forgetting that the epilogue can now
  become the bottleneck
- ignoring the fact that larger MMA atoms consume real TMEM column budget
- underestimating that TMEM is not just faster accumulators; it shifts
  pressure from registers to epilogues and staging

Fallback path:

- smaller tile shapes that accumulate in registers, or hybrid kernels where
  only the inner accumulator is TMEM-resident and the epilogue reads out in
  smaller slabs

### `tcgen05.mma`

Use when:

- the kernel really targets SM100 tensor cores
- dtype, scale layout, and operand placement match a Blackwell path

Blackwell introduces seven new `tcgen05.mma` variants, all 2-4x faster than
Hopper WGMMA:

- `kind::tf32` (2x Hopper TF32)
- `kind::f16` (2x Hopper FP16)
- `kind::bf16`
- `kind::f8f6f4` (mixed precision, 2x Hopper FP8 throughput, FP4/FP6/FP8 operands)
- `kind::mxf8f6f4.block_scale` (block-scaled mixed precision)
- legacy `i8`, `u8`

All variants support TN/NT/TT/NN layout combinations and exist in both dense
and sparse (`.sp`) forms across `cta_group::1` and `cta_group::2`.

Launch and operand contract (from the Colfax UMMA/TMEM GEMM tutorial):

```
A -> SMEM or TMEM
B -> SMEM
D -> TMEM
launcher -> one elected thread
```

That differs from the mental model Hopper readers bring over from WGMMA. Even
in CTA-pair mode, UMMA launch itself is a single-thread action. The rest of
the kernel exists to keep descriptors, SMEM staging, and TMEM accumulator
state valid around that launch. That is why Blackwell code often looks
"under-threaded" around the MMA issue point: one elected thread does the
launch while the rest of the CTA prepares descriptors, stages memory, or
handles epilogue work.

Useful dense-shape reminders:

- `64 x N x 16`, `N` multiple of `8`, `N <= 256`
- `128 x N x 16`, `N` multiple of `16`, `N <= 256`
- K is treated as `32 bytes`
- the large `128 x 256 x 16` atom consumes `256` TMEM columns, half of the
  full `512`-column TMEM space

The instruction descriptor (`idesc`) carries mode bits such as transpose,
negate, and `enable-input-d`:

```
enable_input_d = true   // D = A * B + D (accumulate into existing TMEM state)
enable_input_d = false  // D = A * B     (overwrite)
```

A kernel that expects accumulation across K-stages must preserve TMEM state;
a kernel using overwrite mode can simplify lifetime and initialization.

In CUTLASS / CuTe, the hardware surface shows up through atoms and traits:

```cpp
using Atom   = cute::MMA_Atom<cute::SM100_MMA_F16BF16_SS>;
using Traits = cute::MMA_Traits<cute::SM100_MMA_F16BF16_SS>;
```

guarded by `CUTE_ARCH_MMA_SM100_ENABLED`. An atom name such as
`SM100_MMA_F16BF16_SS` is not just a type alias; it encodes the operand
contract, descriptor form, and memory placement assumptions the rest of the
kernel has to satisfy.

Likely wins:

- 2-4x tensor-core throughput over Hopper WGMMA on matching dtypes
- mainloop register pressure can drop substantially once D lives in TMEM
- direct hardware endpoint for the contest's FP8 + block-scale-128 workload
  via `mxf8f6f4.block_scale`

Likely traps:

- importing a Hopper WGMMA mental model directly into B200 code
- assuming "UMMA enabled" means the surrounding memory / epilogue design is
  solved
- forgetting that `enable-input-d` changes whether the op overwrites or
  accumulates into existing TMEM state
- treating bigger tensor-core atoms as free: the large `128 x 256 x 16` atom
  consumes half of TMEM's column budget

Fallback path:

- smaller UMMA atoms, or a Hopper-style WGMMA path on compatible generations,
  or a CUTLASS builder schedule that accepts the existing layout

### 2-CTA MMA

Use when:

- one CTA cannot saturate the Blackwell MMA path
- a larger cooperative tile is worth paired-CTA structure and synchronization

2-CTA MMA is an SM100 tensor core mode (`cta_group::2`) where a CTA pair
within the same cluster cooperatively executes a single `tcgen05.mma`
instruction:

- partitions A in M and B across N between the two CTAs
- each CTA stages only half of B in its own SMEM; hardware consumes the
  combined B tile during the multiply
- supports M up to `256` (vs `128` for single-CTA mode); valid shapes include
  `128 x {64..256} x 64` and `256 x {64..256} x 64` for f16
- requires `cluster_shape_M >= 2`
- both CTAs must be launched and remain active while the operation is in
  flight; cannot be dynamically disabled within a kernel (unlike SM90
  multicast, which could be conditionally skipped)
- TMA loads use `cp.async.bulk.tensor.Nd.cta_group::2`, and barrier update
  targets only CTA0 via `Sm100MmaPeerBitMask = 0xFEFFFFFF`

Real kernels: FlashAttention-4 uses 2-CTA mode for its forward pipeline,
processing 8 Q tiles across 4 CTA pairs on 4 SMs; FlashMLA defines
`SM100_MMA_F16BF16_2x1SM_TS_NOELECT` for prefill with M=128 or 256.

Implications for code:

- CTA ownership must be designed in pairs
- cluster participation is part of correctness, not just optimization
- peer and leader behavior must stay consistent through the kernel
- scheduler and tile mapping often need to become cluster-aware

Likely wins:

- larger effective tile shapes that saturate the SM100 math path
- reduced per-CTA operand-staging bandwidth because each CTA stages half of B

Likely traps:

- bolting 2-CTA mode onto a kernel designed as isolated CTAs
- ignoring the way paired ownership changes epilogue and synchronization shape
- choosing 2SM for portability or aesthetic reasons when 1SM is a better fit

Practical schedule note:

- 1SM vs 2SM is not just a tile-size choice; it is a different execution
  contract with cluster-shape constraints and launch-shape implications
- CUTLASS Blackwell examples (example 71, example 75) treat schedule
  compatibility between mainloop and epilogue as a real contract that must
  stay aligned with the 1SM or 2SM path
- CUTLASS's Python generator logic rejects invalid 2SM cluster shapes
- prefer 1SM when portability, divisibility, or occupancy quantization make
  2SM awkward; prefer 2SM when the cluster shape and workload really feed the
  larger math path

### Clusters And DSMEM

Use when:

- one CTA's local SMEM is too small
- cross-CTA cooperation is cheaper than bouncing through global memory
- paired or clustered scheduling is already part of the kernel design

Distributed Shared Memory (DSMEM) is the thread-block-cluster feature where a
thread block can read, write, and atomically update the shared memory of
other thread blocks in the same cluster. The Blackwell Tuning Guide frames it
as the middle ground between local SMEM and global memory when a working set
no longer fits inside a single CTA's SMEM allocation.

DSMEM access rules (mirror global-memory guidance):

1. Coalesce DSMEM accesses whenever possible
2. Align accesses to 32-byte segments when possible
3. Avoid non-unit stride when possible
4. Use local SMEM to reshape or stage irregular access patterns before
   touching DSMEM

DSMEM can be used concurrently with L2 cache accesses, so some kernels
benefit by combining DSMEM bandwidth with L2 bandwidth rather than bouncing
everything through global memory.

Implications for code:

- cluster occupancy must be reasoned about explicitly
- DSMEM access still wants coalesced, aligned 32-byte patterns
- cluster size is a real tradeoff against residency

Likely traps:

- treating DSMEM like magical shared state while ignoring access shape
- optimizing for cluster-local sharing while forgetting the residency hit
- assuming "always choose 16" for nonportable cluster size; larger clusters
  reduce scheduling freedom and may cut active blocks

Important occupancy note:

- on B200, cluster-based kernels should be reasoned about with
  `cudaOccupancyMaxActiveClusters`, not ordinary single-CTA occupancy
  intuition
- use a preferred dynamic cluster for the fast path plus a fallback cluster
  that evenly divides it to recover idle SMs when the device SM count is not
  a neat multiple of the preferred cluster size

### Cluster Launch Control

Use when:

- the kernel is persistent or tile load balancing genuinely matters
- irregular tile ownership or queueing is part of the problem

Useful concrete details:

- CLC work assignment happens at cluster granularity; a 2x2 cluster consumes
  4 ClcIDs at once
- the first tile comes from `blockIdx`, then later work can come from
  `clusterlaunchcontrol.try_cancel`, which returns either success with
  coordinates or decline (all tiles done)
- CUTLASS examples commonly use pipeline depth `3` for latency hiding
- transaction bytes for the CLC response are 16; one elected thread from the
  scheduler warp issues the query
- the scheduler warp is a first-class role in the CTA

In the CLC persistent pattern, the 8 warps per block are typically assigned:

| Warp | Role |
|------|------|
| 0 | MMA |
| 1 | Scheduler (CLC producer/consumer) |
| 2 | Mainloop Load |
| 3 | Epilogue Load |
| 4-7 | Epilogue |

Likely traps:

- adopting persistence or CLC without enough irregularity to justify it
- assuming CLC is a generic speedup rather than a scheduling tool
- recommending CLC on Hopper, where it does not exist

### Programmatic Dependent Launch

Use when:

- the performance question is really overlap between two dependent kernels
- a downstream kernel can start while the previous kernel tail is draining
- CUDA Graphs or a fixed pipeline already exist around the kernel chain

PDL lets a downstream kernel begin executing while the upstream kernel's tail
is still draining, as long as the dependency graph permits. Combined with
CUDA Graphs, it cut DeepSeek-R1 from 67 to 253 tokens/s on 8xB200 in
TensorRT-LLM. CUDA Graphs are especially valuable for MoE decoder blocks that
fire many small kernels (router, top-k, permute, grouped GEMM1, SwiGLU,
grouped GEMM2, unpermute, reduce) where per-kernel launch latency dominates.

CUTLASS exposes runtime knobs `overlap_ratio` and `prefetch_ratio` for the
PDL-based weight-prefetch pattern (example 63 and `dependent_kernel_launch`
docs). The relevant architecture helpers live in
`include/cutlass/arch/grid_dependency_control.h` with
`launch_dependent_grids()` and `wait_on_dependent_grids()`.

Likely traps:

- recommending PDL when the real issue is still grouped work distribution
  within one kernel
- talking about PDL as if it replaced persistence or CLC

### Block-Scaled And Microscaled Paths

Use when:

- the workload truly wants FP8 / MXFP8 / NVFP4 and shared scale factors
- the backing library or handwritten kernel really matches the scale layout
  and dtype contract

Implications for code:

- scale-factor layout and granularity are part of the kernel interface
- repacking or swizzling may be required before dispatch
- dtype / alignment / layout choices determine whether the hardware path is
  real

Concrete B200 cautions:

- the contest uses block size `128`, but Blackwell libraries often expose
  smaller or differently packed scale layouts (e.g. `32` or `16`)
- SM100 commonly uses packed `UE8M0` scale factors
- some Blackwell kernels require GEMM-swizzled scale layout rather than a
  naive contiguous scale tensor
- DeepGEMM's `sm100_fp8_gemm_1d1d` accepts scale-factor K granularities of
  32 or 128 and requires scale tensors transformed into packed `UE8M0`
  `torch.int` layout before dispatch

Likely traps:

- assuming every "block-scaled FP8" backend matches the same layout
- treating scale format as metadata instead of part of the operand contract

See the precision-contracts subskill for the full list of per-format rules.

### Software Exp Emulation

Use when:

- the softmax `exp` path on B200 has become a co-bottleneck with MMA
- the kernel already runs FlashAttention-4-style per-tile softmax

On B200, MUFU provides only 16 ops/clock/SM for exponential, while MMA
delivers around 8192 ops/clock/SM. For a 128^3 attention tile, the exp unit
needs about 1024 cycles, equal to MMA compute time, which makes it a real
co-bottleneck rather than a rounding-error concern.

FlashAttention-4's software exp emulation decomposes `2^x` into integer and
fractional parts:

- integer part `2^floor(x)`: computed by bit manipulation of the IEEE 754
  float exponent field (shift and add), running on the integer ALU, not MUFU
- fractional part `2^frac(x)`, `x_frac in [0,1)`: a degree-n polynomial
  `sum p_i * x_frac^i` with `p_0 = 1.0`, running on FMA units in parallel
  with MUFU

The `ex2_emu_freq` knob controls how often the software path is used vs
hardware MUFU. On SM103 (B300/B300A) with native higher-throughput `exp2`,
`freq=0` is correct (use hardware only). Do not apply this technique
reflexively; it is a B200 / SM100 answer to a specific bandwidth imbalance.

## B200 Decision Pivots

Use these as fast decision questions before recommending a rewrite:

1. Is the kernel actually limited by non-MMA work such as exp, correction,
   epilogue, or descriptor management?
2. Is the data movement regular enough for TMA to pay off?
3. Is accumulator pressure high enough that TMEM changes the structure, not
   just the wording?
4. Does the kernel truly benefit from CTA pairs or clusters?
5. Is the schedule irregular enough to justify persistence or CLC?
6. Do the scale-factor layout and dtype contracts match the intended SM100
   backend?
7. Is the question really in-kernel scheduling, or would PDL only help across
   a kernel boundary?
8. What is the clean fallback if the B200-native route is too constrained?

## Best Public Code Corpora To Cite Or Inspect

- FlashAttention-4 for TMEM-first attention design, 2-CTA, software-exp, and
  role-specialized register budgets (reaches ~1613 TFLOPs/s in FP16 at 71%
  utilization on B200 using fully asynchronous MMA with TMEM accumulation)
- DeepGEMM for compact SM100 FP8/BF16 GEMM and scheduler/TMA helpers
- CUTLASS SM100 examples 71, 75, 77, 81, 82, 93, 95 for grouped GEMM, FMHA,
  CLC, low-latency GQA, distributed GEMM, and block-scaled kernels
- ThunderKittens for public H100/B200/B300 kernels and a reusable LCF
  producer/consumer skeleton
- Microbench-Blackwell source repo for `ldgsts_*`, `tma2d_*`, `compare_mem_*`,
  and `sm_l2_distance/` experiments to separate memory-hierarchy from
  math-pipeline questions

## B200 Coding Checklist

Ask these before choosing a B200-specific path:

1. Is non-MMA work the thing that needs to be reduced or overlapped?
2. Is the tile and layout regular enough for TMA?
3. Is accumulator pressure large enough to justify TMEM?
4. Does the kernel truly benefit from paired CTAs or clusters?
5. Are exp / correction / epilogue paths now the real limiter?
6. Do scale-factor layout and dtype contracts match the chosen backend?
7. What is the clean fallback if the Blackwell-native route is too
   constrained?

## When B200 Knowledge Is Not Enough

If a feature detail is still unclear after this subskill:

- do targeted docs or source research, then follow the source order it reveals
- do not improvise feature semantics from memory
