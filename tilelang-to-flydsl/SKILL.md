---
name: tilelang-to-flydsl
description: >
  Port a kernel written in TileLang (the `@T.prim_func` / `with T.Kernel(...)` DSL
  used by TileKernels and other tile-ai projects) into an equivalent FlyDSL
  kernel (`@flyc.kernel` / `@flyc.jit` with explicit layout algebra,
  copy atoms, MMA atoms, and SmemAllocator). Use whenever the task is to
  rewrite, translate, or migrate a TileLang `@T.prim_func` body into FlyDSL,
  including converting individual operators in the TileKernels test suite
  while preserving the Python wrapper signatures so the existing pytest
  cases keep validating the new implementation.
allowed-tools: Read Edit Write Bash Grep Glob Agent
---

# TileLang → FlyDSL Conversion Skill

## What this skill is for

You are translating a kernel written in **TileLang** (Apache TVM TIR-based,
`@T.prim_func` decorated, used in `tile-ai/tilelang` and the `TileKernels`
operator collection from DeepSeek) into **FlyDSL** (Python DSL on top of an
MLIR-native `fly` dialect that targets ROCm/HIP via ROCDL). FlyDSL is a
*lower-level* kernel authoring surface: tile and thread mappings are explicit,
copies are atom-call based, and GEMMs are emitted through MFMA atoms instead of
a dispatched `T.gemm`.

Treat the TileLang source as a **specification** — match its observable
behaviour (input/output shapes, dtypes, math) — and re-derive an equivalent
FlyDSL kernel. Mechanical line-by-line translation rarely works because the two
DSLs sit at different abstraction levels.

## When to invoke

Invoke (or load) this skill when any of the following is true:

- The user asks to "port", "convert", "translate", "rewrite" a kernel from
  TileLang to FlyDSL, or vice versa references both projects.
- The user references TileKernels (`tile_kernels/...`), `@T.prim_func`,
  `T.Kernel`, `T.alloc_shared`, etc. and wants the result to land in a FlyDSL
  codebase.
- A test in `TileKernels/tests/...` must keep passing while the Python wrapper
  is re-implemented on top of FlyDSL.

Do **not** invoke for changes that stay within a single DSL.

## Operating constraint: no execution

This skill assumes that you cannot build or run either project from this
environment (no GPU, neither `tilelang` nor `flydsl` is necessarily installed).
That means:

- Validation is **review-based**, not test-based. Use the workflow in
  `references/workflow.md` and the gotcha list in `references/gotchas.md` to
  audit each converted kernel before declaring it complete.
- Never claim "tests pass" or "kernel verified" unless the user has actually
  run them and shared the output.
- When a behavioural ambiguity blocks progress, surface it to the user as a
  question rather than guessing.

## Mental model — the four hard problems

Every TileLang→FlyDSL conversion reduces to four sub-problems. Solve them in
order:

1. **Launch shape.** Translate `with T.Kernel(gx, gy, gz, threads=N) as
   (pid_x, pid_y, pid_z):` into a `@flyc.jit` host function that calls
   `kernel(...).launch(grid=(gx, gy, gz), block=(N, 1, 1), stream=stream)`,
   plus `bid = fx.block_idx.{x,y,z}` and `tid = fx.thread_idx.x` reads inside
   the `@flyc.kernel` body. Multi-dim block (`threads=(tx, ty)`) is rare in
   TileKernels but does occur — preserve it.
2. **Memory map.** Map every `T.alloc_*` to its FlyDSL counterpart:
   `alloc_shared` → `SmemAllocator.allocate_array` (with `finalize()` in the
   jit body), `alloc_local`/`alloc_fragment` → `fx.memref_alloca` with a
   register-space memref type, scalar `alloc_var(init=v)` → a Python value
   threaded through `range(..., init=[...])` if it must be loop-carried,
   otherwise just a normal Float/Int variable.
3. **Loop nest.** Map `T.Parallel`, `T.serial`, `T.unroll`, `T.vectorized`,
   `T.Pipelined` (see `references/api_mapping.md` row "Loops"). The trickiest
   one is `T.Parallel` — it has *no direct* FlyDSL counterpart and you must
   thread-distribute by hand using `tid` arithmetic or a tiled copy.
4. **Data movement.** `T.copy(global, shared)` and `T.copy(shared, global)`
   call sites become tiled copies in FlyDSL: `fx.rocdl.make_buffer_tensor` →
   `zipped_divide`/`logical_divide` → `make_tiled_copy(...)` →
   `thr_copy.partition_S/D` → `fx.copy(copy_atom, ...)`. A literal
   `arr[i, j] = ...` write becomes a single `fx.copy_atom_call(...)` with a
   one-element register memref, or — when many threads write contiguously —
   the same tiled copy machinery.

The compute body in between (arithmetic, reductions, control flow) usually
maps cleanly once 1–4 are in place.

## Workflow

Follow `references/workflow.md` step-by-step on each kernel. The high-level
loop is:

1. **Read the TileLang source end-to-end.** Note the kernel signature, every
   `T.alloc_*`, every `T.copy` / `T.gemm`, every loop kind, every reduction.
   Identify what is *constexpr* vs *runtime dynamic*.
2. **Read the test that calls it.** The Python wrapper signature
   (`transpose(x: torch.Tensor)`, `topk_gate(scores, num_topk)`, ...) is the
   contract you must preserve. The test imports
   `tile_kernels.<module>.<func>` and compares against a reference. Keep the
   wrapper, replace the JIT factory body.
3. **Sketch the FlyDSL skeleton** using `references/idioms.md` for the closest
   matching pattern (elementwise / reduction / shared-mem rearrange / GEMM).
4. **Fill in compute, copy, and reduction blocks.** Use `references/api_mapping.md`
   as a translation dictionary, but do not blindly inline — re-derive the
   thread-to-data mapping each time.
5. **Audit against `references/gotchas.md`.** Run through every entry; each
   one is something the agent has gotten wrong before.
6. **Report status.** Tell the user exactly which review checks you completed
   and which require GPU execution to verify.

## Reference files

Read these on demand — they are organised as quick lookups, not narrative:

- `references/api_mapping.md` — comprehensive symbol table:
  every `T.*` primitive used in TileKernels paired with its FlyDSL spelling.
- `references/idioms.md` — side-by-side patterns for the five kernel
  archetypes: elementwise/cast, reduction (norm / softmax / topk),
  shared-memory rearrange (transpose), GEMM, fused
  copy+compute+reduce.
- `references/gotchas.md` — pitfalls and review checklist. Read **before**
  declaring a conversion done.
- `references/workflow.md` — the procedure to follow per kernel, including
  how to handle the "I cannot run it" constraint with a static review pass.
- `references/worked_examples/normalize_weight.md` — full conversion of a
  small reduction kernel (`tile_kernels/moe/normalize_weight_kernel.py`).
- `references/worked_examples/batched_transpose.md` — full conversion of a
  shared-memory rearrange kernel (`tile_kernels/transpose/...`).
- `references/worked_examples/gemm_skeleton.md` — annotated FlyDSL GEMM
  skeleton you can adapt when a TileLang kernel uses `T.gemm`.

## Project structure assumptions

The skill is portable. When applying to a new project, locate these things in
the target codebase rather than assuming the paths from the worked examples:

- TileLang sources usually live at `<repo>/tilelang/tilelang/language/...`
  if vendored, otherwise `tilelang` is just a pip dependency. Public surface
  is `from tilelang import language as T`.
- FlyDSL Python expression API is at
  `<flydsl-checkout>/python/flydsl/expr/{primitive,gpu,buffer_ops,rocdl,vector,math,numeric,typing}.py`.
- FlyDSL pre-built kernels at `<flydsl-checkout>/kernels/*.py` are the most
  reliable references for production patterns (softmax_kernel.py and
  rmsnorm_kernel.py are the closest to TileKernels-style elementwise+reduction
  kernels; preshuffle_gemm.py is the closest to GEMM).

When asked to convert an unfamiliar TileKernels kernel, default to:

1. `Read` the TileLang source.
2. `Read` the matching test under `TileKernels/tests/<module>/test_<name>.py`.
3. `Grep` `<flydsl>/kernels/` for an analogous existing FlyDSL kernel and
   `Read` it as a structural template.
4. Then proceed with the workflow above.
