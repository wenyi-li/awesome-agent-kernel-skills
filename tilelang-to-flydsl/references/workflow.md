# Per-kernel conversion workflow

The recipe to apply to every TileLang kernel handed to you. Steps 1–3 are
research; 4–6 are the actual port; 7–8 are review. Do not skip any.

---

## Step 1 — Read the TileLang kernel end-to-end

Open the source file (e.g. `tile_kernels/<group>/<name>_kernel.py`) and read
**every line**, not just the `@T.prim_func` body. Build a mental table:

| Item | What to capture |
|---|---|
| JIT factory | Name, signature (`compile-time int / str / dtype` args), `pass_configs`. |
| Symbolic dims | Every `T.dynamic('...')`. |
| Constants | Every Python `int` / `str` defined above the `@T.prim_func`. |
| Kernel signature | Name, every `T.Tensor[...]` / `T.StridedTensor[...]`. |
| Launch | `with T.Kernel(...) as (...):` — grid, threads, axis names. |
| Allocations | One row per `T.alloc_*`: name, shape, dtype, scope. |
| Loops | One row per `for ... in T.{Parallel,serial,unroll,vectorized,Pipelined}(...)`: kind, bounds, body summary. |
| Copies/GEMMs | Every `T.copy`, `T.gemm`, `T.async_copy`. |
| Reductions | `T.reduce_*`, `T.alloc_reducer`, `T.finalize_reducer`. |
| Atomics | `T.atomic_*`. |
| Control flow | Every `if`, every `T.assume`, every special-case branch. |
| Math | Non-trivial math intrinsics (`T.exp`, `T.rsqrt`, ...) and casts. |

This table is the *spec* you are porting. Nothing in the FlyDSL output should
do something the table doesn't account for.

## Step 2 — Read the test

Open `TileKernels/tests/<group>/test_<name>.py`. Note:

- The Python public function the test imports
  (`tile_kernels.<group>.<func>(...)`).
- The signature, the assertions, the tolerances passed to `assert_equal` /
  `assert_close`.
- Any tensor that comes from a stride-disturbing helper (`twice_stride`,
  `unsqueeze`, etc.) — these need `mark_layout_dynamic` in the FlyDSL
  wrapper.
- The parametrise grid (`generate_test_params_*`) — you must support every
  combination it covers.
- Empty-input handling (`if num_tokens == 0:` early return).

This is the contract. The wrapper signature is the boundary you must
preserve.

## Step 3 — Find the closest existing FlyDSL kernel

Run a `grep` / file search across `<flydsl>/kernels/` for the closest
analogue:

| Your kernel does | Use as template |
|---|---|
| Per-token softmax/normalisation | `softmax_kernel.py`, `rmsnorm_kernel.py` |
| Per-token quantisation/cast | `silu_and_mul_fq.py`, `silu_and_mul_fq` patterns |
| Tiled transpose / shuffle | `mfma_preshuffle_pipeline.py` (helper), or write fresh |
| GEMM with prefetch | `preshuffle_gemm.py`, `hgemm_splitk.py` |
| MoE token routing | `moe_gemm_2stage.py` (two-stage skeleton, not the GEMM) |
| Attention | `pa_decode_fp8.py`, `mla_fwd_decode.py`, `flash_attn_func.py` |
| LayerNorm | `layernorm_kernel.py` |

Read the template. Extract: the `SmemAllocator` ceremony, the
`@flyc.kernel` body shape, the `wave_reduce`/`block_reduce` helpers if any,
the `tiled_copy` pattern for the dominant data movement, the launch shape.

## Step 4 — Sketch the FlyDSL skeleton

Write a stub that has:

- The `SmemAllocator` block (LDS allocations matching the Step 1 table).
- The `@flyc.kernel` signature with `fx.Tensor` / `fx.Int32` /
  `fx.Constexpr[...]` per the TileLang signature, including any `dtype_str`
  Constexpr if the original kernel was generic over dtype.
- The `bid = fx.block_idx.x` / `tid = fx.thread_idx.x` reads matching the
  `T.Kernel` axes.
- An empty body.
- A matching `@flyc.jit` launcher that wires the launch shape and the
  allocator's `finalize()` ceremony.
- An updated Python wrapper at the top of the original module that calls
  this new launcher.

Compile the **mental shape** at this point — does the launcher signature
match the test's call site? If not, fix now, before filling in the body.

## Step 5 — Fill in the body

Translate one `T.*` block at a time. Use `references/api_mapping.md` as a
dictionary and `references/idioms.md` for the larger patterns:

1. Outer loop / `T.Pipelined` first — establish the iteration structure.
2. Inside-loop loads (`T.copy`, indexed reads).
3. Compute (math, casts, reductions).
4. Inside-loop stores.
5. Post-loop epilogue.

After each block: re-read it. Does it match the corresponding TileLang block
in semantics (same data flow, same compute)? If anything is "I'll come back
to this", note it as a TODO and continue — but do not declare done.

## Step 6 — Audit against the gotcha list

Open `references/gotchas.md`. Walk through every section linearly.
For each item, do one of:

- **Verify**: read the corresponding code, confirm the rule is followed.
- **Justify**: the code violates the rule but the violation is intentional —
  write down why.
- **Fix**: the code violates the rule and that's a bug — fix it now.

Do not skip the silent items (B, C-3, D-1, E-3, G-2, H-1). These are the
ones that produce kernels that compile and run but give wrong outputs. They
will not be caught by review-only validation.

## Step 7 — Diff against the reference

Re-open the closest FlyDSL reference kernel from Step 3. Walk down both files
in parallel. List every structural deviation:

```
LDS layout:    softmax uses (RED_SLOTS,) f32; mine uses (NUM_TOKENS,) f32.   - JUSTIFIED: per-token reduction.
prefetch loop: softmax does not prefetch; mine pipelines K=2.                - DEVIATION: TileLang source had T.Pipelined, kept it.
wave_reduce:   softmax uses sum + max; mine uses sum only.                   - JUSTIFIED: kernel is sum-only.
```

If any line of the deviation list reads "no idea why", treat that as a bug.

## Step 8 — Report

State explicitly to the user:

- Which kernel you converted (path).
- What the wrapper signature now is (should be unchanged).
- A summary of structural choices: dtype variant chosen (FNUZ vs OCP), copy
  atom widths used, LDS layout, pipeline depth.
- The list of gotcha checks completed.
- The list of items you could NOT verify because they require running the
  test (typically: numerical correctness, shape edge cases the test
  parametrises). Phrase as "I cannot verify X, Y, Z without GPU execution
  — please run `pytest tests/<...>/test_<name>.py -n 4` to confirm."
- Any open questions that blocked progress.

Do not say "tests pass" or "kernel verified" unless the user has run the
test and shared output.

---

## Deliberate non-goals

- **Performance tuning.** A faithful port is the goal of this skill. Match
  the TileLang shape; do not over-engineer. If the FlyDSL kernel is slow,
  hand off to `gemm-optimization` / `lds-optimization` /
  `prefetch-data-load` / `kernel-trace-analysis` after correctness is
  verified.
- **Refactoring the wrapper.** Keep the Python public API byte-identical.
  Renames, signature tweaks, return-type changes are out of scope.
- **Touching unrelated kernels.** One TileKernels module ↔ one converted
  FlyDSL module. Do not opportunistically clean up neighbours.
