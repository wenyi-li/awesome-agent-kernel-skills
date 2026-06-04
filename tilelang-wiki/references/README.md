# TileLang Reference Guide

This is the main reference entry point for the local TileLang skill. Use it to
choose the right document quickly; use the linked pages for the actual kernel
syntax, API details, examples, and debugging workflows.

TileLang is a Python-first DSL for writing high-performance kernels for targets
such as CUDA and HIP. The files in this skill are the source of truth for local
answers. Use `../examples/README.md` for runnable example selection and
`../FAQs.md` for known failure modes.

## Start Here

Use these pages early in development:

- `language_basics.md`: compact kernel-writing cheatsheet. Covers the basic
  kernel shape, launch regions, memory scopes, data movement, compute,
  reductions, and a commented FlashAttention 2 forward kernel.
- `jit_autotune.md`: compact compile/tuning cheatsheet. Covers basic
  `tilelang.jit`, `.compile(...)`, direct calls, `tilelang.autotune`, config
  scans, and cache-debug reminders.
- `tilelang/README.md`: top-level `import tilelang` API index, including JIT,
  autotune, environment controls, AutoDD, and debug tools.
- `tilelang_language/README.md`: `import tilelang.language as T` API index,
  organized by instruction family.
- `python_compat.md`: supported Python syntax, symbolic dimensions, tensor
  annotations, dtype forms, casts, and host-vs-kernel boundaries.

The previous narrative programming guides are archived under
`old_programming_guiles/`. Use them only for historical walkthrough context
after checking the current cheatsheets and API references above.

## Reading Order

1. Start with `language_basics.md` for the kernel skeleton and core TileLang
   language.
2. Read `jit_autotune.md` when you need to compile, call, profile, tune, or
   reason about cache behavior.
3. Read `python_compat.md` when the question is about supported Python syntax,
   shapes, tensors, dtypes, or casts.
4. Jump into `tilelang_language/<topic>/basic.md` for exact `T.*` API usage.
5. Jump into `tilelang/<topic>/basic.md` for exact top-level `tilelang.*` API
   usage.
6. Open `advanced.md` pages only when a basic page points there or the kernel
   uses an explicit advanced path such as TMA, WGMMA, TCGEN05, cluster launch
   control, manual synchronization, AutoDD, grouped autotune compile, or
   low-level debug tooling.

## API Detail Map

Top-level `tilelang.*` APIs:

- `tilelang/jit/basic.md`: `tilelang.jit`, eager calls, lazy factory style,
  `.compile(...)`, `out_idx`, source inspection, and profiling helpers.
- `tilelang/jit/advanced.md`: `tilelang.compile(...)`,
  `tilelang.par_compile(...)`, mode inference, cache keys, pass configs,
  target/backend defaults, and debug output.
- `tilelang/autotune/basic.md`: `tilelang.autotune`,
  `AutoTuner.from_kernel(...)`, compile/profile arguments, captured inputs, and
  common tuning pitfalls.
- `tilelang/autotune/advanced.md`: config binding, explicit caller values,
  result objects, validation paths, tuning caches, worker controls, timeouts,
  grouped compile, and multi-GPU benchmarking.
- `tilelang/env/basic.md`: cache helpers, environment defaults,
  target/backend settings, and import-time behavior.
- `tilelang/env/advanced.md`: cache directories, environment-variable
  precedence, target/backend resolution, and runtime setup debugging.
- `tilelang/autodd_tools/basic.md`: AutoDD and practical debug tooling exposed
  through `tilelang.tools`.
- `tilelang/autodd_tools/advanced.md`: layout plotting, generated-source
  callbacks, and lower-level instrumentation.

`tilelang.language as T` APIs:

- `tilelang_language/loop/`: serial, unrolled, parallel, pipelined, vectorized,
  and persistent loop forms.
- `tilelang_language/allocate/`: tensor annotations, eager outputs, shared,
  local, fragment, scalar, barrier, tensor-memory, descriptor, and reducer
  allocations.
- `tilelang_language/copy_op/`: `T.copy`, `T.async_copy`, TMA, cluster copy,
  gather/scatter, transpose, multicast, bulk copy, and tensor-map helpers.
- `tilelang_language/gemm_op/`: dense GEMM, sparse GEMM, WGMMA, TCGEN05,
  block-scaled GEMM, warp policies, and specialized MMA paths.
- `tilelang_language/basic_operations/`: `T.fill`, `T.clear`, tensor
  annotations, elementwise assignments, proxy buffers, pointer views, and math
  intrinsics.
- `tilelang_language/kernel_warpgroup_cluster_builtins/`: `T.Kernel`, thread
  and block builtins, barriers, mbarriers, warpgroup helpers, cluster helpers,
  external CUDA source kernels, and low-level MMA builtins.
- `tilelang_language/annotations/`: swizzle, layout maps, restrict-buffer,
  launch bounds, safe values, L2 hints, compile flags, and pass configs.
- `tilelang_language/reduce_op/`: tile reductions, cumulative sum, generic
  reductions, batched reductions, NaN-propagating reductions, and warp
  reductions.
- `tilelang_language/misc/`: atomics, debug helpers, dynamic symbols, boolean
  buffer reductions, explicit loads/stores, random numbers, PDL, branch hints,
  DP4A, and raw TIR exports.

Each API topic uses `basic.md` for common usage and `advanced.md` for
target-specific, lower-level, or rarely used behavior.

## Other Reference Areas

- distilled tutorials:
  - `auto_tuning.md`
  - `debug_tools_for_tilelang.md`
- operator walkthroughs:
  - `deeplearning_operators/elementwise.md`
  - `deeplearning_operators/gemv.md`
  - `deeplearning_operators/matmul.md`
  - `deeplearning_operators/matmul_sparse.md`
  - `deeplearning_operators/deepseek_mla.md`
- practical details:
  - `god_blessing/magic_enums.md`
  - `god_blessing/pass_config.md`
- archived setup, narrative guides, compiler internals, and runtime internals:
  - `old_programming_guiles/get_started/Installation.md`
  - `old_programming_guiles/get_started/overview.md`
  - `old_programming_guiles/get_started/targets.md`
  - `old_programming_guiles/compiler_internals/letstmt_inline.md`
  - `old_programming_guiles/compiler_internals/inject_fence_proxy.md`
  - `old_programming_guiles/compiler_internals/tensor_checks.md`
  - `old_programming_guiles/runtime_internals/stubs.md`

## Practical Routing

- For a first kernel, open `language_basics.md`, then choose the closest
  runnable example from `../examples/README.md`.
- For an API lookup, open `tilelang/README.md` or
  `tilelang_language/README.md`, then the relevant `basic.md`.
- For tuning, start with `jit_autotune.md`, then use
  `tilelang/autotune/basic.md` only when you need the detailed API.
- For debugging, start with `../FAQs.md`; use
  `debug_tools_for_tilelang.md` when you need AutoDD, source inspection, layout
  plotting, or generated-code callbacks.
- For advanced hardware paths, start from the basic page, then open the
  matching `advanced.md` page only after the basic kernel is correct.
