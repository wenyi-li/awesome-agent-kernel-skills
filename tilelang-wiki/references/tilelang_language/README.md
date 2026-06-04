# TileLang Language API Guide V2

This directory is organized by `tilelang.language` API area. Each topic has two
files:

- `basic.md`: common APIs and patterns used across the examples.
- `advanced.md`: explicit scheduling, target-specific, lower-level, or rarely
  used APIs.

Start with the basic page for the topic you need. Open the advanced page only
when the basic page points there or when the kernel uses an explicit hardware
path such as TMA, WGMMA, TCGEN05, cluster launch control, raw pointers, or
manual synchronization.

## Instruction Families

Use this map when you know the API or instruction family but not the document:

- `loop/basic.md`: everyday loop helpers, including `T.Parallel`,
  `T.Pipelined`, `T.serial`, and `T.Serial`.
- `loop/advanced.md`: parallel loop hints, manual pipeline scheduling,
  `T.unroll`, `T.vectorized`, persistent scheduling, and lower-level loop
  controls.
- `allocate/basic.md`: `T.Tensor`, `T.alloc_shared`, `T.alloc_local`,
  `T.alloc_fragment`, `T.alloc_var`, and eager outputs.
- `allocate/advanced.md`: global workspace allocation, barriers, tensor memory,
  descriptor buffers, reducers, and specialized allocation forms.
- `copy_op/basic.md`: `T.copy`, common movement patterns, and `T.im2col`.
- `copy_op/advanced.md`: copy annotations, async copy, TMA copy, cluster copy,
  gather/scatter TMA, transpose, multicast, bulk copy, fence/wait controls, and
  tensor-map helpers.
- `gemm_op/basic.md`: dense `T.gemm`, sparse GEMM, common transpose flags, and
  warp-policy options.
- `gemm_op/advanced.md`: WGMMA, TCGEN05, block-scaled GEMM, warpgroup-level
  GEMM controls, and specialized MMA paths.
- `basic_operations/basic.md`: tensor arguments, `T.fill`, `T.clear`, and
  elementwise tile assignments.
- `basic_operations/advanced.md`: buffer proxy helpers, pointer views,
  fast-math intrinsics, IEEE rounded intrinsics, and packed x2 intrinsics.
- `reduce_op/basic.md`: `T.reduce_sum`, `T.reduce_max`, `T.reduce_min`,
  `T.reduce_abssum`, cumulative sum, and common reduction patterns.
- `reduce_op/advanced.md`: generic reduction controls, batched reductions,
  NaN-propagating reductions, and warp reductions.
- `kernel_warpgroup_cluster_builtins/basic.md`: `T.Kernel`, cluster launch
  basics, barrier and mbarrier basics, and warpgroup usage.
- `kernel_warpgroup_cluster_builtins/advanced.md`: launch-frame details,
  external CUDA source kernels, warp specialization, WGMMA ordering, cluster
  synchronization and launch control, and Blackwell/TCGEN05 helpers.
- `annotations/basic.md`: threadblock swizzle, layout maps, buffer aliasing,
  and launch bounds.
- `annotations/advanced.md`: safe values, L2 hit-ratio hints, compile flags,
  pass configs, and specialized target/compiler controls.
- `misc/basic.md`: atomics, debug helpers, dynamic symbols, boolean buffer
  reductions, and small utilities.
- `misc/advanced.md`: access pointers, TMA store and proxy helpers, lane/warp
  shuffle and vote helpers, explicit global memory access, grid/global
  synchronization, random numbers, PDL, assumptions, branch hints, DP4A, and
  raw TIR exports.
