# Programming Guides Overview

This section maps the current TileLang programming model onto the working
examples under `examples`. Treat those examples as the source of truth when an
older document disagrees with current behavior.

TileLang is a Python DSL for tiled kernel development across targets such as
CUDA, HIP, Metal, LLVM/C, WebGPU, and CuTeDSL. In practice, most kernels follow
the same high-level shape:

1. Define a kernel with `@tilelang.jit` or a nested `@T.prim_func`.
2. Bind symbolic dimensions with `T.const(...)` or `T.dynamic(...)`.
3. Declare tensors with `T.Tensor(...)` and outputs with `T.empty(...)`.
4. Enter a launch region with `T.Kernel(...)`.
5. Allocate explicit memory scopes such as shared, local, or fragment storage.
6. Move tiles with `T.copy` and compute with tile operators such as `T.gemm`.
7. Compile, call, inspect, and profile the resulting kernel from Python.

For the canonical minimal GEMM skeleton that demonstrates that flow, start with
`language_basics.md`.

## How To Read This Section

These guides are organized by responsibility rather than by operator family:

- `language_basics.md` owns the core kernel shape: JIT structure, tensors,
  launch regions, memory scopes, copies, and host-side invocation.
- `control_flow.md` owns branching, loop forms, guards, thread bindings, and
  boundary handling.
- `instructions.md` owns the semantics of tile-level operations and lower-level
  primitives such as `T.copy`, `T.gemm`, reductions, atomics, and sync helpers.
- `software_pipeline.md` owns `T.Pipelined`, manual `stage` and `order`
  annotations, and async/TMA pipeline rules.
- `python_compatibility.md` owns the "what Python syntax means inside TileLang"
  support matrix.
- `autotuning.md` owns config spaces, decorator/programmatic tuning flows,
  validation, caching, and advanced run options.
- `type_system.md` owns dtype normalization, tensor annotations, casts, and
  mixed-precision rules.

## Recommended Reading Order

1. Start with `examples/quickstart.py`.
2. Read `language_basics.md` to learn the canonical TileLang kernel skeleton.
3. Read `control_flow.md` and `python_compatibility.md` to understand what loop
   and branch syntax means inside captured kernels.
4. Read `instructions.md` when you need precise behavior for copy, GEMM,
   reductions, atomics, or synchronization.
5. Read `software_pipeline.md` when tuning or manually annotating pipelined
   loops.
6. Read `type_system.md` when dtype normalization, casts, or low-precision
   behavior matter.
7. Read `autotuning.md` after the kernel is already correct.

## Common Entry Points

- Start from `examples/quickstart.py` for a minimal eager JIT kernel.
- Compare against `examples/gemm/example_gemm.py` for the common tiled GEMM
  structure.
- Compare against `examples/elementwise` when the kernel is mostly
  element-parallel.
- Use `examples/gemm/example_gemm_autotune.py` together with `autotuning.md`
  once the untuned kernel is stable.

## Guide Boundaries

These guides intentionally cross-reference one another instead of repeating the
same full kernel walkthrough in every page:

- If you need the surrounding kernel structure, go back to `language_basics.md`.
- If you need loop semantics, go to `control_flow.md`.
- If you need pipeline semantics, go to `software_pipeline.md`.
- If you need instruction semantics, go to `instructions.md`.

That split keeps the recurring GEMM teaching pattern in one place while still
keeping each page standalone.
