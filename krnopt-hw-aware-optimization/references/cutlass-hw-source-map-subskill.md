# CUTLASS Hardware Source Map Subskill

Use this integrated subskill when the user already knows the relevant hardware
feature, but now needs to inspect the CUTLASS or CuTe layer that actually
implements the contract.

The goal is to stop answers like "look somewhere in CUTLASS." Point to the
layer that owns the behavior.

## Core Mental Model For CUTLASS 3.x

The right mental model for CUTLASS 3.x is "a staged GEMM construction kit
with explicit tuning surfaces," not "a bag of opaque iterators." The two big
architectural shifts from CUTLASS 2.x are decoupling and CuTe adoption.
Hopper's warpgroup-wide WGMMA was the canonical driver: it did not fit
cleanly into the old warp/thread decomposition, so CUTLASS 3.x made the
interface more algorithmic than architectural. A kernel author should think
in terms of *problem shape*, *mainloop*, *epilogue*, *schedule*, and *layout
contracts* first, then map onto a specific SM90 or SM100 implementation.

Three design promises the docs emphasize:

- correctness by default, with static asserts when contracts are violated
- fewer named types; policies replace many configuration aliases
- clear, local performance knobs instead of hidden iterator behavior

The GEMM hierarchy is still the performance backbone. The blocked triple
loop still matters: CTA tile -> warp tile -> instruction tile. CTA tiles
balance occupancy, cache reuse, and boundary waste; warp tiles balance reuse
and bank-conflict-free SMEM access; thread or instruction tiles balance
register reuse and math issue.

The canonical 3.x launch stack is:

```cpp
CollectiveEpilogue
CollectiveMainloop
GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>
GemmUniversalAdapter<GemmKernel>
```

Epilogue often comes first in code even though it runs last at runtime.
This is load-bearing on SM100 because the mainloop builder needs to know
the epilogue SMEM carveout to choose stage count correctly. The SM100
quickstart instantiates the epilogue before the mainloop and feeds its
shared-storage size into `StageCountAutoCarveout`.

When reading a new CUTLASS kernel, the highest-yield checklist is:

1. What are the operand element types, layouts, and alignments?
2. What `MmaTileShape` and `ClusterShape` are being chosen?
3. Which `KernelSchedule` and `EpilogueSchedule` are selected?
4. Is the kernel using auto policies, or is the schedule pinned explicitly?
5. Which part of the code owns performance policy: the builder, a runtime
   scheduler, or an external autotuner?

## CuTe Is The Layout Vocabulary

CuTe provides the vocabulary that makes the tuning seams legible. Layouts
are not just bookkeeping; they are the mechanism by which thread/data
mapping, tiling, and instruction contracts become explicit.

The recurring CuTe recipe:

- `make_shape(...)` defines problem or tile shape
- `make_stride(...)` defines layout semantics
- `make_tensor(...)` combines pointer or iterator, shape, and layout
- `local_tile(...)` projects a full tensor into the CTA's view

Reading trick: when a kernel is hard to parse, look for which mode is
stride-1 and reason from that instead of mentally translating BLAS
transpose flags. The tutorial reframes "row-major vs column-major" as
`M-major`, `N-major`, `K-major`.

Layout algebra is the real "iterator logic." Operations like `coalesce`,
`composition`, `divide`, and `product` let CUTLASS derive partitioning
behavior from a single layout vocabulary:

- `divide` partitions data tiles across CTAs or threads
- `composition` combines nested layout transformations
- `coalesce` simplifies ugly multi-mode layouts without changing the
  underlying mapping

TMA tensors explain the weird coordinate code. TMA instructions do not
consume raw GMEM pointers; they consume a descriptor plus coordinates
within that descriptor's view. CuTe builds implicit tensors of coordinates
rather than ordinary pointer tensors. `ArithmeticTuple`, basis elements
like `E<0>{}`, and TMA-coordinate layouts exist so CUTLASS can tile,
slice, and offset descriptor coordinates the same way it would tile
ordinary memory.

MMA atoms bridge PTX and generic code. CuTe wraps each MMA instruction as:

1. an Operation struct matching the PTX instruction's physical interface
2. an `MMA_Traits` specialization describing logical types, shape, thread
   mapping, and operand layouts
3. an `MMA_Atom` combining the two
4. optionally a `TiledMMA` that repeats or interleaves atoms into larger
   tiles

An atom name like `SM100_MMA_F16BF16_SS` encodes the operand contract,
descriptor form, and memory placement assumptions the rest of the kernel
has to satisfy. `CUTE_ARCH_MMA_SM100_ENABLED` is the Blackwell gate.

## Which Layer Owns What

### Instruction Wrappers, Descriptor Rules, TMEM Helpers

Open `include/cute/arch/` when the question is about:

- Hopper TMA wrapper behavior
- WGMMA or UMMA (`tcgen05`) instruction wrappers
- descriptor formats
- TMEM allocation helpers

Typical files:

- `include/cute/arch/copy_sm90_tma.hpp` (Hopper TMA load/store wrappers and
  multicast variants at the PTX-interface level)
- `include/cute/arch/mma_sm90.hpp` (Hopper GMMA surface)
- `include/cute/arch/mma_sm100.hpp` (Blackwell UMMA / `tcgen05` surface)
- `include/cute/arch/tmem_allocator_sm100.hpp` (Blackwell TMEM lifetime
  and allocation behavior)

This layer is for understanding what the hardware wrapper actually exposes,
not for understanding whole-kernel scheduling.

### Mainloop Or Epilogue Construction

Open the collective layer when the question is about:

- which schedule family a kernel uses
- how staging, barriers, and MMA are composed
- how the epilogue schedule interacts with the mainloop

Typical files:

- `include/cutlass/gemm/collective/collective_builder.hpp` (generic entry
  point for mainloop construction)
- `include/cutlass/epilogue/collective/collective_builder.hpp` (matching
  entry point for epilogues)
- `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`
  (Hopper TMA + GMMA warp-specialized collective internals)
- `include/cutlass/gemm/collective/sm100_mma_array_tma_gmma_ss_warpspecialized.hpp`
  (Blackwell collective for UMMA-array / TMA patterns)

This is where schedule policy names stop being symbolic and start turning
into real barrier, load, and MMA choreography.

### Warp Roles, Persistence, Tile Ownership, CLC

Open the kernel layer when the question is about:

- which warp is the scheduler
- how tile ownership advances
- persistent-kernel wrappers
- Stream-K or SM100 tile schedulers

Typical files:

- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` (main
  Hopper warp-specialized kernel wrapper)
- `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` (main
  Blackwell persistent wrapper)
- `include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp` (Hopper
  Stream-K behavior)
- `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp` (Blackwell
  persistent and CLC tile assignment behavior)

This layer is where you learn which warp is the scheduler, which warp does
mainloop load, which warps run epilogue, and how work advances between
output tiles.

### Barriers, Async Stages, PDL, CLC Fetch Machinery

Open the pipeline and architecture helpers when the question is about:

- async stages
- barrier flow
- multicast or peer masks
- PDL helpers
- CLC fetch behavior

Typical files:

- `include/cutlass/pipeline/sm90_pipeline.hpp` (Hopper async pipeline
  machinery used by TMA producer/consumer kernels)
- `include/cutlass/pipeline/sm100_pipeline.hpp` (Blackwell UMMA-aware
  pipelines, multicast logic, peer masks, and `PipelineCLCFetchAsync` for
  Cluster Launch Control)
- `include/cutlass/arch/grid_dependency_control.h` (PDL surface:
  `launch_dependent_grids()` and `wait_on_dependent_grids()`)

When an example README talks about overlapping stages, CLC queries, or
dependent launch, this is usually the layer that makes the behavior
concrete.

### Generation Breadth, Pruning, And Schedule Compatibility

Open `python/cutlass_library/` when the question is about:

- why a kernel exists or does not exist
- which shapes or schedules were generated
- analytical pruning or heuristics

Typical files:

- `python/cutlass_library/manifest.py` (library manifest bookkeeping)
- `python/cutlass_library/generator.py` (stamps operations)
- `python/cutlass_library/sm90_shapes.py`, `sm100_shapes.py` (enumerate
  available instruction and cluster shapes)
- `python/cutlass_library/sm90_utils.py`, `sm100_utils.py` (pruning and
  schedule-compatibility rules)
- `python/cutlass_library/heuristics.py`,
  `python/cutlass_library/heuristics_provider.py` (analytical narrowing
  via NVIDIA Matmul Heuristics)
- `python/cutlass_library/emit_kernel_listing.py` (selective profiler
  regression/testlist kernel emission)

Reading trick: shape files define what instruction or cluster shapes
exist; utils files define pruning rules and schedule compatibility;
generator / manifest files stamp those into library operations. So if a
kernel did not get generated, the answer is often in the utils pruning
logic, not in the example code.

Hopper and Blackwell use a four-digit global instantiation level to drive
kernel generation. The docs explicitly warn that exhaustive generation
can reach millions of candidate kernels, so "just instantiate everything"
is not a realistic default.

### Profiler Runtime Behavior

Open `tools/profiler/src/` when the question is about:

- how a CUTLASS profiler flag is interpreted
- grouped or blockwise profiler behavior
- reporting shape

Typical files:

- `tools/profiler/src/cutlass_profiler.cu` (top-level driver)
- `tools/profiler/src/gemm_operation_profiler.cu`,
  `grouped_gemm_operation_profiler.cu`,
  `blockwise_gemm_operation_profiler.cu` (parse problem-space arguments,
  cluster preferences and fallbacks, raster order, PDL flags, runtime
  dtypes, and per-operation workspace rules; grouped profiler encodes
  CUTLASS's own assumptions about grouped and MoE-like problem shapes,
  including file-based problem-size input and cluster fallback handling)
- `tools/profiler/src/performance_report.cpp` (what CSV metadata is
  preserved)

Heuristics prune first, profiler validates second. Heuristics are a
search-space reducer; they still expect runtime measurement to pick the
winner.

### CuTe DSL Autotuning And Debugging

The CuTe DSL autotuning guide is a parallel control surface separate from
the profiler. Recommended cache key includes dtype plus configuration
knobs such as `use_2cta_instrs`, `mma_tiler`, `cluster_shape`, and
`use_tma_store`. For input-level caching, approximate keys such as
rounded powers of two are explicitly endorsed when exact-shape tuning is
too expensive.

DSL debugging support covers line info for PTX/SASS correlation, IR/PTX/
CUBIN dumping, file or console logging via environment variables, access
to dumped `__mlir__`, `__ptx__`, `__cubin__` programmatically, and runtime
`cute.printf()` vs compile-time Python `print()`.

### Multi-GPU Expansion

Open `include/cutlass/experimental/distributed/`, especially
`gemm_collective_builder.hpp`, when the question is about distributed
GEMM. This is where the builder pattern begins to extend into
tensor-parallel territory. For single-GPU work it is rarely the first
thing to read.

## How Hopper And Blackwell Examples Teach The Layers

Public CUTLASS examples form a cross-generation pattern library.

Hopper examples (builder schedules, warp specialization, overlap):

- Example 49 (core Hopper builder tutorial): `CollectiveBuilder` is where
  schedule, stage count, and epilogue style become explicit; walks through
  automatic schedules, warp-specialized TMA schedules, ping-pong variants,
  Stream-K cooperative schedules, and custom EVT epilogues
- Example 50: explicit WGMMA + TMA configuration showing cluster shape,
  multicast, and epilogue-schedule interaction
- Examples 67 and 68: blockwise/grouped FP8 story using schedules like
  `KernelTmaWarpSpecializedCooperativeFP8Blockwise` or pointer-array
  grouped schedules; example 68 demonstrates on-device TMA descriptor
  modification for grouped problems
- Example 63 and `dependent_kernel_launch.md`: PDL cross-kernel overlap
  with `overlap_ratio` and `prefetch_ratio` runtime knobs

Blackwell examples (instruction contracts, schedule compatibility, dynamic
persistence):

- Example 71 (builder analogue to Hopper 49): mainloop and epilogue
  schedules must be compatible across 1SM and 2SM families
- Example 75 (canonical grouped GEMM on SM100): pointer-array grouped
  problems, runtime cluster-shape handling, device-side TMA descriptor
  updates, separate 1SM/2SM schedule families
- Example 81: how to profile and interpret blockwise/groupwise kernels,
  including kernel naming and the practical trick of swapping M/N when one
  dimension is small
- Example 77: Blackwell FMHA, where forward, backward, and generation paths
  mix TMA and `cp.async` and fusion depends on whether extra memory
  traffic can be orchestrated through the mainloop
- Example 93: low-latency GQA with cluster shape `1x1xMAX_SPLITS`,
  flash-decoding-style work partitioning over KV length, seven warps
  across DMA/MMA/epilogue/reduction, DSMEM-based cluster reduction
- Example 95: CLC-backed scheduling under Green Context partitioning;
  adapts to partial-SM availability better than older static persistent
- Example 82: experimental distributed GEMM path for Blackwell

`media/docs/cpp/blackwell_functionality.md` is the compatibility table for
supported data types, alignments, layout combinations, tile shapes,
epilogue schedules, and auto dispatch policies.

Cross-generation reading rule: use Hopper examples to learn the *shape* of
CUTLASS 3.x schedules; use Blackwell examples to learn the *constraints*
imposed by the new instruction and memory model.

## Fast Mapping Table

- descriptor or instruction question -> `include/cute/arch/`
- mainloop or epilogue schedule question -> collectives
- persistent or tile-scheduler question -> kernel layer
- async pipeline or PDL / CLC helper question -> pipeline and arch helpers
- generation support or omitted kernel question -> `python/cutlass_library/`
- profiler behavior question -> `tools/profiler/src/`
- distributed-GEMM question -> `include/cutlass/experimental/distributed/`

## Where To Escalate

- Do targeted docs or source research if the relevant source layer is still
  unclear after following this map.
- Use the generation-specific SM90 or SM100 subskill when the question is
  still "which hardware path should I choose?" rather than "where does
  CUTLASS encode it?"
