---
name: flydsl-kernel-authoring
description: >
  FlyDSL is a Python DSL with MLIR-native backend for authoring custom AMD
  GPU kernels with explicit layout algebra (pre-installed at /opt/FlyDSL on
  images tagged *-flydsl:*). Use this skill when profiling identifies a hot
  per-row reduction (RMSNorm / LayerNorm / softmax), a fused elementwise
  chain (norm + residual add, activation + multiplier), or an unusual-shape
  grouped GEMM that the standard AMD backends (Triton / aiter / CK /
  hipBLASLt / TransformerEngine) don't serve well. Essential for any workload
  where Python/config/Triton-tuning gains have plateaued and the profile
  shows a custom kernel opportunity. Covers the `/opt/FlyDSL` availability
  check, the integration playbook (dispatcher + direct site-packages edit +
  autograd-safe output handling), kernel authoring patterns (elementwise via
  layout API, block reductions via wave_reduce_add, fused dx+dw designs, MFMA
  GEMM preshuffle), torchrun gotchas, and the critical rule that custom
  kernels typically only win end-to-end when stacked with
  `torch.compile(mode="default")`.
---

# FlyDSL Kernel Authoring (AMD MI300X / MI350)

FlyDSL (Flexible Layout Python DSL) is a Python-native way to write
high-performance AMD GPU kernels with explicit layout algebra, tiled
copies/MMAs, LDS management, MFMA intrinsics, and autotune — compiled through
an MLIR stack to ROCDL. On images where it is pre-installed, it lives at
`/opt/FlyDSL` and is ready to `import` without any build step.

## When to use FlyDSL (decision table)

FlyDSL is a **Level 5** optimization — only reach for it AFTER:
1. You have a verified post-warmup baseline.
2. You have profiled with kernel-level breakdown (see your `gpu-profiling` skill).
3. `torch.compile(mode="default")` already works and its graph breaks are fixed.
4. You have tried the standard AMD backends for the hotspot kernel class.

| Hotspot profile | FlyDSL good fit? |
|---|---|
| Standard matmul/GEMM in batch-heavy range | NO — hipBLASLt / aiter tuned GEMM / CK are well tuned here |
| SDPA / flash attention in common shapes | NO — aiter FMHA or TE attention usually wins |
| Elementwise chain that torch.compile already fuses | NO — compile already handles this |
| Custom per-row reduction (RMSNorm / softmax / layernorm variant) at unusual shape or dtype | YES — layout API + block_reduce_add pattern is a natural fit |
| Fused row op + normalization (e.g. add-then-norm) where the chain is NOT already fused | YES — fuse 2+ kernels into 1 pass |
| Grouped GEMM variant where upstream libs lack a good preshuffle config | YES if you can verify a library-side config is missing |
| Any hotspot where the standard library lacks coverage for your exact (M,N,K)/dtype | YES if measured, not assumed |

If none of the YES rows apply, do NOT spend trials on FlyDSL. A standard-library
config or fusion change is more productive.

## Critical rule: stack with `torch.compile`

Custom kernels usually do NOT move end-to-end metrics on their own, even when
the isolated kernel is faster. The wins almost always require stacking a custom
kernel with `torch.compile(mode="default")`. Before claiming a FlyDSL win e2e:

1. Verify `torch.compile(mode="default")` is active somewhere in the hot path.
2. Report the e2e metric with AND without your FlyDSL kernel, with compile on.
3. If isolation shows a speedup but e2e is flat, consider Python dispatch
   overhead — a C-extension replacement may have been fast enough already.

## Availability check

Before touching FlyDSL code, confirm the image has it installed:

```bash
# Is FlyDSL available on this image?
ls /opt/FlyDSL/build-fly/python_packages/flydsl 2>/dev/null || echo "FlyDSL NOT installed on this image"
echo "FLYDSL_ROOT=${FLYDSL_ROOT:-unset}"
python3 -c "from kernels.rmsnorm_kernel import build_rmsnorm_module; print('imports OK')"
```

If any of the above fail, FlyDSL is not available on this image. Do not attempt
FlyDSL work — focus on the other optimization levels instead.

When the image supports FlyDSL, `FLYDSL_ROOT=/opt/FlyDSL`, `PYTHONPATH` already
contains the FlyDSL package paths, and `LD_LIBRARY_PATH` contains the MLIR libs.
Do NOT re-export these yourself in the main shell — they are set image-wide.

## Find a Python that can actually USE FlyDSL — discovery first

FlyDSL ships as compiled C++ extensions tagged for a specific CPython
ABI (e.g. `_mlir.cpython-310-x86_64-linux-gnu.so`). Images that
"have FlyDSL" usually expose it in two ways:

1. **A pip-installed FlyDSL**, sitting in some venv's
   `site-packages/flydsl/` directory with a `.dist-info`. This is the
   *intended* way to use FlyDSL — its symlinks resolve, its dependencies
   are pinned, and `flydsl.compile(module)` round-trips cleanly.
2. **A raw build directory** at `/opt/FlyDSL/build-fly/python_packages/`
   left over from the image build process. This directory has dangling
   symlinks (e.g., `_mlir/ir.py` -> `/workspace/llvm-project/.../ir.py`
   that no longer exists) and importing from it bypasses the proper
   pip dependency resolution. **Do not put this on PYTHONPATH** — it
   shadows the working pip-installed flydsl with a broken view.

The image's pre-set `PYTHONPATH=/opt/FlyDSL/build-fly/python_packages:/opt/FlyDSL`
is **not** a recommendation; it's a holdover from the build environment
that actively breaks downstream imports. Step 0 of any FlyDSL work is
to find the right Python and DROP the bad PYTHONPATH.

### Step 0a — discovery: which Python actually has FlyDSL?

```bash
# Always start with PYTHONPATH cleared. Globally-set PYTHONPATH is the
# #1 source of "import flydsl crashes nanobind" symptoms.
unset PYTHONPATH

# A. Check the workload's Python (the one your benchmark already uses).
#    Use `-L` so symlinks (most managed venvs are symlinks) are followed.
WORKLOAD_PY=$(find -L /workspace -maxdepth 5 -path '*/.venv/bin/python' \
              -executable 2>/dev/null | head -1)
WORKLOAD_PY="${WORKLOAD_PY:-$(which python3)}"
"$WORKLOAD_PY" -c 'import flydsl, sys; print("WORKLOAD_PY OK:", flydsl.__file__, sys.version_info[:2])' 2>&1 | head -3

# B. Check well-known image Pythons that may host FlyDSL pip-installed.
#    Common locations: /opt/venv (rocm/sgl-dev images), /opt/conda (anaconda),
#    /opt/python*. Do not stop at the first match — you want one whose
#    `flydsl.compile(...)` actually works, see Step 0b.
for cand in /opt/venv/bin/python /opt/conda/bin/python /usr/bin/python3 \
           $(ls /opt/uv-pythons/cpython-*/bin/python 2>/dev/null); do
    [ -x "$cand" ] || continue
    if "$cand" -c 'import flydsl' 2>/dev/null; then
        echo "candidate FLYDSL_PY: $cand ($("$cand" --version))"
    fi
done
```

If `WORKLOAD_PY` itself imports flydsl cleanly, you're in the easy case
— skip ahead to "Once you have a working FLYDSL_PY".

If `WORKLOAD_PY` fails but a `/opt/venv` (or similar) Python succeeds,
record both: `FLYDSL_PY=/opt/venv/bin/python` for kernel authoring,
`WORKLOAD_PY=...` for the end-to-end benchmark. **You will need a
cross-Python integration plan** — see Tier 2 below.

### Step 0b — verify: does kernel COMPILATION + EXECUTION work?

`import flydsl` succeeding is a necessary but not sufficient signal.
The C++ type registry only fully populates after `flydsl.compile(...)`
runs, and the `.so` runtime only proves itself when a kernel actually
executes on the GPU. Smoke-test with the simplest known-good kernel
shipped by the FlyDSL distribution (`elementwise_add` from
`flydsl_tests`):

```bash
"$FLYDSL_PY" - <<'PY'
import sys, importlib.util, importlib
import torch, flydsl

# flydsl_tests is shipped alongside flydsl in pip-installed images and
# contains the canonical kernel test harness. Its tests `from tests.*`
# but the package itself is installed as `flydsl_tests`, so alias.
import flydsl_tests
sys.modules['tests'] = flydsl_tests

# Locate flydsl_tests.kernels.test_eltwise_add and run a tiny shape.
import os
fp = os.path.join(os.path.dirname(flydsl_tests.__file__), 'kernels', 'test_eltwise_add.py')
spec = importlib.util.spec_from_file_location('_smoke', fp)
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.test_compile_and_run(M=129, N=255)  # asserts internally on max-err
print('FlyDSL kernel compiled + ran on GPU OK')
PY
```

Expected output ends with `[FLIR INFO] Verification: Max error: 0.00e+00`
followed by `FlyDSL kernel compiled + ran on GPU OK`. Anything else is
a real failure (do not paper over it).

### Step 0c — workload-shape isolation correctness check (mandatory)

Step 0b proves the toolchain works on a tiny shape; Step 0c proves it
works at the shapes your real workload uses. **Do this before any
profiling or source reading.** Use one of the kernels empirically
verified to build cleanly under pip-flydsl 0.0.1.dev (most images
ship this pinned version):

| kernel | dtype | M, N | status (validated 2026-04-28) |
| --- | --- | --- | --- |
| `flydsl.kernels.softmax_kernel` | bf16, f16, f32 | any (M, N), N is the reduction axis | works for arbitrary N, fast path for N % BLOCK_THREADS*VEC_WIDTH == 0 |
| `flydsl_tests.kernels.test_eltwise_add` | f32 | any (M, N) | works for arbitrary shape |
| `flydsl.kernels.rmsnorm_kernel` | — | — | **broken on pip-flydsl 0.0.1.dev**: ValueError at build time on bf16 / non-power-of-2 N. Skip on this version. |
| `flydsl.kernels.layernorm_kernel` | — | — | **broken on pip-flydsl 0.0.1.dev**: ValueError on memref.store at build time. Skip on this version. |

Recipe (substitute your workload's real M, N, dtype):

```bash
"$FLYDSL_PY" - <<'PY'
import torch, flydsl, torch.nn.functional as F
from flydsl.kernels.softmax_kernel import build_softmax_module

M, N, dtype_str = 256, 2048, 'bf16'   # <- replace with your shape
torch_dtype = {'f16': torch.float16, 'bf16': torch.bfloat16, 'f32': torch.float32}[dtype_str]

mod = build_softmax_module(M, N, dtype_str)
exe = flydsl.compile(mod)

x   = torch.randn(M, N, dtype=torch_dtype, device='cuda').contiguous()
out = torch.empty_like(x)
exe(x, out, M)
torch.cuda.synchronize()

ref = F.softmax(x.float(), dim=-1).to(torch_dtype)
err = (out - ref).abs().max().item()
print(f'softmax M={M} N={N} {dtype_str}: max_err={err:.4e}',
      'PASS' if err < 1e-2 else 'FAIL')
PY
```

Expected `max_err < 1e-5` for bf16, `< 1e-4` for f16. If the chosen
kernel hits a `ValueError` / `TypeError` at compile time (the table
above lists known-bad ones for pip-flydsl 0.0.1.dev), record the
exact error in `optimization_state.json`, then **do not debug
upstream pip-flydsl internals** — pick the next kernel from the
"works" rows of the table and try again. Do NOT waste trial time
trying to fix `rmsnorm_kernel.py` or `layernorm_kernel.py` by hand;
those are upstream bugs.

A passing Step 0c IS a valid FlyDSL trial attempt by itself — it
demonstrates the toolchain works on workload shapes, which is the
foundation for any later integration work. Counts as one of the two
required attempts under the cross-Python case.

### If Step 0b crashes — read the error *before* trying workarounds:

- `nanobind: type 'IntTupleType' ... base type "mlir::python::mlir::PyType" not known to nanobind!`
  → You're loading the broken raw-build `flydsl` package via a
  PYTHONPATH that shadows the pip-installed one. `unset PYTHONPATH` and
  re-try with this Python — do not retry the same path with a different
  Python interpreter, the path is the bug.
- `ModuleNotFoundError: No module named 'flydsl.expr'`
  → You're using an older pip-installed flydsl (v0.0.1.dev) that does
  not have the `flydsl.expr` API. The kernel files in this version use
  `build_<name>_module(...)` + `flydsl.compile(...)` instead of
  `@flyc.kernel`. Use the older API directly; do not search for `flyc`.
- `cpython-3XX-x86_64-linux-gnu.so: ELF class mismatch / undefined symbol`
  → Genuine ABI mismatch. Move to "Tier 1: parallel Python via uv" below.

### Once you have a working FLYDSL_PY

- Always launch `"$FLYDSL_PY" -c '...'` *with* `PYTHONPATH` cleared.
- If your `FLYDSL_PY` differs from `WORKLOAD_PY` (cross-Python case),
  remember that you cannot just `from flydsl import ...` in
  `WORKLOAD_PY`. The integration paths are:
  - **Same Python case** (FLYDSL_PY == WORKLOAD_PY): import flydsl
    directly in the workload's torch path; this is the easy case.
  - **Cross-Python case** (FLYDSL_PY != WORKLOAD_PY): use Tier 2 below
    (kernel artifact handoff via `.hsaco`/`hipModuleLoad`).

### Tier 1 (rare last-resort): install a parallel matching Python via uv

Only fall through to this when:

- `WORKLOAD_PY` cannot import flydsl (ABI mismatch or missing), AND
- No image-bundled Python (Step 0a B) has flydsl pip-installed, AND
- `/opt/FlyDSL` exists with `.so` files for a specific cp3XX ABI you
  could match.

```bash
unset PYTHONPATH

# Detect FlyDSL's required CPython ABI from its .so files.
ABI_DIGITS=$(ls /opt/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs/*.cpython-*.so \
             2>/dev/null | head -1 | grep -oP 'cpython-\K[0-9]+' | head -1)
FLYDSL_PYVER="${ABI_DIGITS:0:1}.${ABI_DIGITS:1}"
uv python install "$FLYDSL_PYVER"
FLYDSL_PY=$(uv python find "$FLYDSL_PYVER")

# Verify with PYTHONPATH set ONLY for this call.
PYTHONPATH=/opt/FlyDSL/build-fly/python_packages \
    "$FLYDSL_PY" -c 'import flydsl; print("flydsl ok in parallel python")'
```

**Tier 1 caveat: even when import succeeds, `flydsl.compile(...)` may
abort with `nanobind: ... IntTupleType ... base type "mlir::python::mlir::PyType"
not known to nanobind!`** The uv-installed CPython
(`python-build-standalone`) uses a different `libstdc++` runtime than
the one FlyDSL's `.so` files were linked against, so MLIR's nanobind
type registry never finishes populating. When this happens you have
exhausted Tier 1; move to Tier 2 (cross-Python kernel artifact handoff)
or Tier 3 (skip FlyDSL).

### Tier 2 (advanced): cross-Python kernel artifact handoff

When your kernel wins under `$FLYDSL_PY` in isolation, integrate it into the
workload's `$WORKLOAD_PY` torch path WITHOUT importing FlyDSL there:

1. Compile the FlyDSL kernel for the target shape using `$FLYDSL_PY`.
   Dump the final ROCDL ASM via `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps`
   (already documented in the gotchas). The dump produces a `final_isa.s`
   plus per-pass MLIR.
2. Convert the ASM to a `.hsaco` (HSA Code Object) using `clang-offload-bundler`
   or `llvm-mc` from ROCm's toolchain. The `.hsaco` is the runtime artifact
   that `hipModuleLoad` consumes.
3. From `$WORKLOAD_PY`, write a small `torch.utils.cpp_extension.load_inline`
   wrapper that does `hipModuleLoad` on the `.hsaco`, looks up the kernel
   symbol, and exposes a `(torch.Tensor, ...) -> torch.Tensor` callable.
   The kernel call path is then pure C++/HIP at runtime — no Python-version
   compatibility issue.

This is advanced. Only attempt after the isolation bench under `$FLYDSL_PY`
already shows a meaningful win for your target shape. If your kernel works
in isolation but the cross-Python integration is too heavy for the trial
budget, document the isolation result in `optimization_state.json` and
fall through to Tier 3.

### Tier 3 (last resort): document and pivot to non-FlyDSL kernel work

After Tiers 1 and 2 fail (e.g. uv install offline-blocked, ABI tag too
exotic, no compatible CPython available), document the mismatch in
`optimization_state.json` and pivot to **non-FlyDSL** kernel-fusion work
in this stage: kernel fusion via `torch.compile`, attention backend
selection, Inductor config tuning, hipBLASLt autotune, custom Triton
kernels — these don't depend on the FlyDSL Python frontend and often
produce comparable wins on memory-bound or fusion-bound hotspots.

Do **not** spend the trial fighting the FlyDSL build itself: the
`/opt/FlyDSL` distribution is per-image, and rebuilding from source
(MLIR + nanobind + ROCm) is hours of work that won't fit in a trial.

## Existing kernels are HEADSTART code — inventory `/opt/FlyDSL/kernels/` FIRST

Before authoring any new kernel, list `/opt/FlyDSL/kernels/`. The directory
ships ready-made kernels covering most common ops: RMSNorm, fused add+RMSNorm,
LayerNorm, softmax, preshuffle GEMM, attention. Each is a complete, tested
implementation that builds correctly.

```bash
ls /opt/FlyDSL/kernels/*.py
```

If a kernel matching your hotspot already exists, your job is to:

1. Read that kernel — it solves the FlyDSL API issues for that operation
   class (vector types, `reduce`, `copy_atom_call`, layout divides).
2. Use it as the starting point for your dispatcher / integration.
3. Only modify it for shape-specific tile sizes.

Authoring the same kernel from scratch when a working sample exists is the
single most common way to burn the FlyDSL trial. The MLIR-style errors you
will hit ("`reduce` on ArithValue", "memref type mismatch", "vector
broadcast required") are all already-solved problems in the sample kernels.

## Integration playbook (the only pattern that consistently works)

The safe way to deploy a FlyDSL kernel into a live PyTorch / training run:

1. **Pick one hot function.** Do NOT globally replace an op. You want a single
   Python entry point whose inputs/outputs you can dispatch on.
2. **Author the kernel builder** (see patterns below) that takes shape/dtype
   constants and returns a callable.
3. **Write a dispatcher** that:
   - Caches the compiled kernel per `(M, N, dtype)`.
   - Falls through to the original library for any shape/dtype NOT matching.
   - Is callable with the exact signature the host expects.
4. **Patch the call site**. Prefer a direct module-source edit inside the
   container's site-packages (this is robust under torchrun), over
   `usercustomize.py` tricks which often do NOT reach torchrun child processes.
5. **Verify correctness** against the reference op with a BF16-appropriate
   tolerance (e.g. `max abs diff < 2e-2`).
6. **Benchmark** with the task's standard harness, not a micro-bench.
7. **Verify stacking with torch.compile**. Turn compile on if it was off.

## Kernel authoring patterns

A minimal FlyDSL kernel has this structure (see `/opt/FlyDSL/examples/` and
`/opt/FlyDSL/kernels/` for full examples):

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.numeric import Float32
from flydsl._mlir import ir

@flyc.kernel
def my_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    # compile-time constants
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    # body: layout divides, copy atoms, reductions, stores

@flyc.jit
def launch(A, B, C, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, C).launch(grid=(M,), block=(256,), stream=stream)
```

### Pattern 1: element-wise / vectorized copy (the layout API)

Best for element-wise ops, activations, and the "load + compute + store" core
of norms. Use `logical_divide` + `slice` + `copy_atom_call`; NOT raw
`buffer_ops.buffer_load` (the raw path can crash for compute kernels, whereas
the layout API is proven for elementwise).

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
vec_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(VEC, 1),
                               fx.AddressSpace.Register)
r = fx.memref_alloca(vec_reg_ty, fx.make_layout(VEC, 1))
fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, tid)), r)
v = fx.memref_load_vec(r).to(Float32)
# ... compute ...
fx.memref_store_vec(out_bits, r)
fx.copy_atom_call(copy_atom, r, fx.slice(div_out, (None, tid)))
```

### Pattern 2: per-row block reduction (for RMSNorm / softmax / etc.)

Use the `wave_reduce_add` + `block_reduce_add2` idiom from
[references/reduction-and-norms.md](references/reduction-and-norms.md).
`block_reduce_add2` is strictly faster than two separate reductions because it
shares the LDS barrier.

### Pattern 3: fused dx + dw (shared reads)

A two-kernel "reduce first, then finalize" design doubles memory bandwidth
because both kernels re-read the same input tensors. If profiling shows
memory bandwidth is the ceiling for a reduction kernel, FUSE the main pass
with a per-block partial write and add a small finalize kernel that only
reads the partials. See
[references/reduction-and-norms.md](references/reduction-and-norms.md)
for the TILE_M-block design.

### Pattern 4: GEMM with MFMA + LDS ping-pong (preshuffle)

For GEMM variants only when hipBLASLt / aiter / CK genuinely do not cover the
shape/dtype. Start from `/opt/FlyDSL/kernels/preshuffle_gemm.py` or
`/opt/FlyDSL/examples/04-preshuffle_gemm.py`. Weight preshuffle (`shuffle_weight`)
is a ONE-TIME cost at model init; cache the shuffled weight, but see the
"data_ptr cache invalidation" gotcha below.

### Pattern 5: autotune

Wrap the `@flyc.jit` with `@autotune(configs=[...], key=[...])` from
`flydsl.autotune` to sweep tile sizes. Use this only after the kernel is
correct.

## Gotchas (these WILL cost you trials if you miss them)

1. **torchrun + `kernels/` name collision.** Under torchrun, sometimes another
   package named `kernels` is cached in `sys.modules` and shadows FlyDSL's.
   Use this robust loader for FlyDSL kernels inside any patch:

   ```python
   import importlib.util, sys, os

   def _load_file(name, path):
       spec = importlib.util.spec_from_file_location(name, path)
       mod = importlib.util.module_from_spec(spec)
       sys.modules[name] = mod
       spec.loader.exec_module(mod)
       return mod

   # Before importing FlyDSL kernels, purge any bad `kernels` in sys.modules
   if "kernels" in sys.modules:
       _kp = getattr(sys.modules["kernels"], "__path__", None)
       if not (_kp and any(str(p).startswith("/opt/FlyDSL") for p in _kp)):
           for _n in list(sys.modules):
               if _n == "kernels" or _n.startswith("kernels."):
                   del sys.modules[_n]
   _load_file("kernels", "/opt/FlyDSL/kernels/__init__.py")
   _load_file("kernels.kernels_common", "/opt/FlyDSL/kernels/kernels_common.py")
   _fwd_mod = _load_file("kernels.my_fwd_kernel",
                         "/opt/FlyDSL/kernels/my_fwd_kernel.py")
   ```

2. **`data_ptr()` weight cache invalidates under in-place optimizer updates.**
   If you cache preshuffled weights keyed on `tensor.data_ptr()`, the optimizer
   can update the weight in place without changing the pointer, leaving you
   with a stale shuffled copy. Either re-shuffle every call (cheap if the
   weight is small) or key on a version counter.

3. **Fresh output buffers are required by autograd.** Do NOT cache an output
   tensor and `.zero_()` it between calls — autograd will detect the inplace
   modification and fail with "expected version N, got N+1". Allocate
   `torch.zeros(...)` fresh each call; it is cheap.

4. **`buffer_ops.buffer_load` auto-promotes bf16 to f32.** If you want the
   native bf16 vector, pass `dtype=T.bf16` explicitly. For store-back,
   convert with `.to(BFloat16)` on gfx950 (`USE_HW_CVT_PK_BF16_F32`) or use
   the manual round-to-nearest-even pack on gfx942.

5. **Python `id()` shadowing.** If you use `id` as a local variable (very
   common in minified patches), a later `id(kf)` crashes. Rename to `idim`
   or `import builtins as _bi; _bi.id(kf)`.

6. **FlyDSL MoE grouped GEMM is not a universal speedup.** Careful measured
   work shows it can win or lose vs. the library grouped GEMM depending on
   shapes and end-to-end overheads (routing construction, weight reshuffle,
   output zeroing). It is NOT a default "always try this" — only take this
   on when profiling specifically implicates a shape the library does not
   cover well, and measure end-to-end before locking it in.

7. **First-call compile latency.** Each unique shape/dtype triggers a JIT
   compile (~1-15 s on AMD). Warm up kernels at model init or accept the
   first-iteration overhead; do NOT include the first iteration in timing.

8. **IR inspection.** `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python3 my.py`
   emits per-pass MLIR dumps and a `final_isa.s`. Use this to diagnose
   compile errors.

9. **Cache invalidation.** FlyDSL's on-disk cache at `~/.flydsl/cache/` auto-
   invalidates on source/closure changes, so you rarely need to clear it.
   Only set `FLYDSL_RUNTIME_ENABLE_CACHE=0` when modifying C++ passes or
   non-closure helper functions in the FlyDSL source itself.

10. **LDS (shared memory) budget is a HARDWARE limit — check before scaling
    tiles.** The error ``out of resource: shared memory, Required: X, Hardware
    limit: Y`` means the kernel requested more LDS than the target CU has.
    Per-CU LDS budgets are fixed:

    | GPU arch | GPU chip | LDS per CU |
    |---|---|---|
    | gfx942 | MI300X / MI308X |  64 KB |
    | gfx950 | MI350 / MI355X | 160 KB (163 840 bytes) |
    | gfx1250 | MI450 | 160 KB |

    For a standard double-buffered GEMM kernel, LDS usage scales roughly as:

    ```
    lds_bytes  ≈  (BLOCK_M * BLOCK_K  +  BLOCK_K * BLOCK_N)
                    * dtype_bytes
                    * num_stages
    ```

    Concrete guard (the missing check that caused the real-world failure we
    observed: `BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, num_stages=3, bf16`
    wants 192 KB on gfx950's 160 KB limit):

    ```python
    def _lds_ok(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, dtype_bytes, arch="gfx950"):
        lds_limit = {"gfx942": 65536, "gfx950": 163840, "gfx1250": 163840}[arch]
        needed = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * dtype_bytes * num_stages
        return needed <= lds_limit

    # When accepting autotune / origami / hand-picked configs, always check:
    if not _lds_ok(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, 2, "gfx950"):
        # fall back to the safe default, e.g. (128, 128, 64, 2)
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = 128, 128, 64, 2
    ```

    If the kernel is not in your own code (e.g. you're relaxing a
    library-side guard so an autotune suggestion with a larger
    `BLOCK_K` is accepted), the LDS check MUST be added to that
    selection path — otherwise isolation-benchmark-passing configs
    will crash at verification time on different shapes.

    Separately: epilogue storage, `cache_a` / `cache_b` preshuffle buffers,
    and register spills also count against the CU's LDS budget. When
    adjacent kernels share a stream, their LDS is not additive (each
    kernel has exclusive access during its launch) — only one kernel's
    budget needs to fit. But within one kernel, everything must fit.

## Recovery patterns — when you hit a FlyDSL API error

These are the most common stumbles authors hit on a first FlyDSL kernel.
Each has a worked recovery rather than "give up and abandon the hotspot".

### "`reduce` on ArithValue" / "vector type required for `reduce`"

`reduce()` and `wave_reduce_add()` operate on **vector types**, not on the
scalar `ArithValue` you get from a single load or arithmetic op. The fix
is to keep the data in vector form through the full reduction chain:

```python
# WRONG — `x_safe` becomes a scalar after the select, reduce() fails:
x_vec   = fx.memref_load_vec(r).to(Float32)
x_safe  = is_valid.select(x_vec, Float32(0.0))   # ArithValue
sq      = (x_safe * x_safe).reduce(...)           # crash

# RIGHT — keep the operand a vector, mask via vector select:
x_vec    = fx.memref_load_vec(r).to(Float32)
zero_vec = fx.vec_broadcast(Float32(0.0), shape=x_vec.shape)
x_safe   = fx.vec_select(is_valid_vec, x_vec, zero_vec)  # still a vector
sq       = (x_safe * x_safe).reduce(...)                  # OK
```

The existing `kernels/layernorm_kernel.py` and `kernels/rmsnorm_kernel.py`
both do exactly this — read them once, then mimic the pattern.

### Shape-divisibility — your tile must divide the shape

`out of resource: shape not divisible` (and silent "wrong-output" bugs) come
from picking a tile that does not divide the workload dimension. The fix is
simple: pick a tile that does. Worked example for the Qwen3-30B-A3B
hidden_size = **2560** (a non-power-of-2 that breaks naive 2048 / 1024 tiles):

```text
hidden_size = 2560 = 2^9 · 5

Valid tile choices (divide 2560 evenly):
  256  → 10 columns of work per row
  320  →  8 columns of work per row
  512  →  5 columns of work per row
  640  →  4 columns of work per row

Invalid tile choices for this hidden_size:
  1024 → 2.5 columns — leaves a remainder
  2048 → 1.25 columns — leaves a remainder
```

In the kernel, set `BLOCK_N` (the per-row column tile) to the chosen size:

```python
BLOCK_N = 256          # divides hidden_size=2560
NCOLS   = N // BLOCK_N # exact, no remainder
for j in range(NCOLS):
    ...
```

If the workload's shape genuinely cannot be divided by any reasonable tile
(e.g. a prime hidden_size), use a masked tail loop pattern from
`kernels/rmsnorm_kernel.py` — but that is the exception, not the default.

### "Custom kernel works in isolation but doesn't move e2e"

The skill's golden rule says: stack with `torch.compile(mode="default")`.
Some authors skip this step because they observed `torch.compile` regress
on the same model in an earlier trial without the custom kernel. **That
observation is not transferable to the compile + custom-kernel pair.**
The Python dispatch overhead on a freshly-allocated output tensor — which
your custom kernel introduces — is what `torch.compile` is best at
removing. Always measure (a) baseline, (b) custom kernel without compile,
(c) custom kernel WITH compile. (c) is the one expected to win.

### "Custom kernel claims a win in isolation but verification disagrees"

The most common cause is dispatcher routing: your hot path is calling the
ORIGINAL library op, not your FlyDSL kernel. Sanity check:

```python
# Add a one-line print at the top of the FlyDSL dispatcher:
def _flydsl_dispatch(x, weight, eps):
    print(f"[flydsl_path] M={x.shape[0]} N={x.shape[1]} dtype={x.dtype}", flush=True)
    ...
```

Run one iteration. If the line never fires, your patch did not reach the
hot module (common with torchrun's process tree caching `sys.modules`); use
the loader pattern from gotcha #1 above. If it fires but the count is
suspiciously low compared to the expected number of layers, your dispatcher
is falling through to the reference op for shapes it should handle —
inspect the shape filter.

## Reference files

These reference docs (trimmed from the FlyDSL repository) cover the patterns in
depth. Read them when you need the exact API:

- [references/layout-algebra.md](references/layout-algebra.md) — Shape, Stride,
  Layout, make_layout / make_tile / logical_divide / raked_product, copy atoms
- [references/kernel-authoring.md](references/kernel-authoring.md) — `@flyc.kernel`
  and `@flyc.jit` full reference, scf.for loop-carried values, frontend
  restrictions (no mutated closures, no inter-branch variables).
- [references/reduction-and-norms.md](references/reduction-and-norms.md) — wave
  and block reductions, dual `block_reduce_add2`, 2-kernel vs fused
  dx+dw patterns.
- [references/gemm-optimization.md](references/gemm-optimization.md) — MFMA
  tiles, LDS ping-pong, hot_loop_scheduler, CShuffle epilogue, preshuffle.

## Integration checklist

When you ship a FlyDSL kernel into the live workload, verify in order:

1. `/opt/FlyDSL` exists on this image? If no — stop, use another level.
2. Direct Python import test passes (see "Availability check" above).
3. Correctness: your kernel matches the reference within BF16 tolerance on a
   representative input.
4. Isolation bench: your kernel is faster than the library call it replaces,
   on the task's shape, as a standalone Python benchmark.
5. E2E bench WITHOUT your kernel: recorded as the comparison baseline.
6. E2E bench WITH your kernel: must improve vs. (5).
7. E2E bench WITH your kernel + `torch.compile(mode="default")`: must improve
   vs. both (5) and (6). If (7) is the only one that improves meaningfully,
   the combination is the optimization — document BOTH as required.

If (4) wins but (6) does NOT, your kernel's Python dispatch overhead is
probably eating the isolated kernel gain. Either move the dispatcher into the
hot path (no extra Python), or accept that this specific kernel is not the
right target.
