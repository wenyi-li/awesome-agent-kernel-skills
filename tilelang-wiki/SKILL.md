---
name: "tilelang-wiki"
description: "A local, code-grounded TileLang reference skill for DSL semantics, kernel authoring, API lookup, tuning, debugging, examples, and compiler/runtime behavior. Use whenever you need to write, modify, explain, optimize, validate, or troubleshoot TileLang programming work."
---

# TileLang Wiki

Use this skill when the user needs grounded help with TileLang concepts, kernel
authoring, optimization, debugging, operator selection, or compiler/runtime
internals.

Treat the bundled local files as the source of truth:

- `references/README.md`: main TileLang reference entry point
- `examples/README.md`: example catalogue and operator inventory
- `FAQs.md`: common failure modes, debugging shortcuts, and issue-driven fixes

Do not rely on external TileLang docs unless the user explicitly asks for
outside information.

## When To Invoke

Invoke this skill when the user asks:

- how to install TileLang or choose a target/backend
- how to write or understand a TileLang kernel
- what a TileLang construct means, such as `@tilelang.jit`, `T.Kernel`, `T.copy`, `T.gemm`, `T.Pipelined`, or memory scopes
- for a quick API lookup, such as which allocation, loop, math, profiling, or debugging primitive to use
- whether a Python pattern is supported inside TileLang kernels
- how to tune, profile, or debug a TileLang kernel
- which local example best matches an operator, datatype, hardware target, or optimization goal
- how TileLang compiler or runtime internals behave

Do not invoke this skill for unrelated CUDA, Triton, or generic TVM questions
unless the user is clearly working through TileLang material.

## Primary Sources

Start with:

- `references/README.md` for the compact language overview, quick API reference, kernel templates, and routing to deeper guides
- `examples/README.md` for the example catalogue and operator-family lookup

Open `references/README.md`, `references/language_basics.md`, and
`references/jit_autotune.md` early in development. They are the short path for
the current guide structure, basic kernel syntax, and compile/tuning workflow.
Use `references/tilelang/README.md` for top-level `tilelang.*` APIs and
`references/tilelang_language/README.md` for exact `tilelang.language as T`
APIs. Use `references/python_compat.md` for supported Python syntax, shapes,
tensors, dtypes, and casts.

Open `FAQs.md` early when the user reports a concrete compiler error, profiler
surprise, autotuning failure, or an example that compiles but does not validate
or perform as expected.

Also open `FAQs.md` early when the user says they changed kernel source but a
profile or autotune rerun did not appear to recompile. Cache behavior and cache
keys can be the real issue even when the kernel body changed.

Then open deeper files only as needed.

## Language Basics

Always keep this mental model available when answering:

1. Define a kernel with `@tilelang.jit` or a nested `@T.prim_func`.
2. Declare shapes with `T.const(...)` or `T.dynamic(...)`.
3. Annotate buffers with `T.Tensor(...)` and outputs with `T.empty(...)`.
4. Launch work with `T.Kernel(...)`.
5. Allocate storage with `T.alloc_shared(...)`, `T.alloc_fragment(...)`, `T.alloc_local(...)`, or `T.alloc_var(...)`.
6. Move tiles with `T.copy(...)`.
7. Compute with `T.gemm(...)`, reductions, or elementwise loops.
8. Optimize with `T.Pipelined(...)`, swizzle, autotuning, or target-specific primitives.
9. Validate against a reference and inspect generated code or profiling output if needed.

Important distinctions to explain clearly:

- `T.const(...)` is for dimensions inferred from concrete input tensors
- `T.dynamic(...)` keeps dimensions symbolic in the compiled kernel
- `T.Parallel(...)` is the usual high-level elementwise loop
- `T.Pipelined(...)` is the usual staged copy/compute loop
- `T.if_then_else(...)` is for value-producing conditionals
- `T.copy(...)` is the default movement primitive; `T.async_copy(...)` is for deliberately managed overlap

If the user is new to TileLang, start from `references/README.md` and the local
`examples/quickstart.py` pattern.

## Working Style

Route by user intent, not by file tree.

1. Identify the goal:
   - setup
   - first kernel
   - DSL semantics
   - operator/example selection
   - optimization
   - debugging
   - internals
2. Open `references/README.md` first for the compact overview, quick API table, and template shape.
3. Open `references/language_basics.md` next for fast DSL basics, then route to `references/tilelang_language/README.md` for exact `T.*` APIs.
4. Open `examples/README.md` when the user needs a matching runnable example.
5. Open `FAQs.md` when the task sounds like a known failure mode or a practical
   "why did this happen?" question.
6. Use deeper pages under `references/` only for exact semantics, caveats, or internals.
7. Prefer the simplest correct example before advanced architecture-specific variants.

## Answering Rules

- Prefer practical guidance over file dumps.
- Use `references/README.md` for conceptual explanation and `examples/README.md` for locating the right example family.
- Use `references/README.md` for fast routing between the cheatsheets and the `tilelang/` or `tilelang_language/` API detail pages.
- If docs and examples overlap, use docs for semantics and examples for implementation patterns.
- Surface target assumptions early, especially CUDA vs HIP/AMD and Hopper/Blackwell-specific behavior.
- Separate semantic correctness from performance tuning.
- For debugging, use a stepwise workflow: reproduce, inspect generated artifacts, compare against a reference, then minimize.
- For autotune or profiler surprises, explicitly consider cache reuse before
  assuming the source edit was ignored.
- When recommending an example, explain why it matches the user's operator, datatype, and hardware.

## Important Caveats

- The current guide defaults are `references/README.md`, `references/language_basics.md`, `references/jit_autotune.md`, `references/python_compat.md`, `references/tilelang/README.md`, and `references/tilelang_language/README.md`.
- Older narrative guides, setup pages, compiler-internal notes, and
  runtime-internal notes live under `references/old_programming_guiles/`; use
  them as archived background only after checking the newer cheatsheets and API
  references.
- The local example tree is broad and evolves faster than prose docs, so prefer `examples/README.md` to choose an operator family and then open the specific example directory or README.
- `FAQs.md` is intentionally issue-driven and may mention pitfalls not yet
  surfaced in the higher-level guides; use it for surgical debugging advice.
- Autotune and JIT cache behavior can explain why a rerun does not visibly
  recompile after a source edit. Do not assume the kernel body change alone
  invalidated the active cache key; check the cache rules in the docs and FAQs.
