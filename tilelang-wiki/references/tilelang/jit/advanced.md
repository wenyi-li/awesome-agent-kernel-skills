# JIT Advanced

This page covers lower-level JIT behavior: explicit compile paths, cache layers,
pass configs, backend resolution, parallel compile, and generated artifacts.

## Top-Level Exports

Normal `import tilelang` exports:

- `tilelang.jit`
- `tilelang.JITKernel`
- `tilelang.compile`
- `tilelang.par_compile`

`tilelang.jit` is the decorator path. `tilelang.compile(...)` compiles an
already-built `T.prim_func`. `tilelang.par_compile(...)` compiles many
`T.prim_func` objects concurrently.

## Explicit `tilelang.compile(...)`

Use top-level compile when you already have a `T.prim_func`:

```python
kernel = tilelang.compile(
    func,
    out_idx=[-1],
    target="cuda",
    execution_backend="auto",
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
```

Important behavior:

- `func` must be a TileLang `PrimFunc`.
- `out_idx` may be an int, a list of ints, or `None`.
- If the function carries `tilelang_out_idx`, passing `out_idx` again is an
  error.
- Function-level `tilelang_pass_configs` are merged with explicit
  `pass_configs`; explicit values override matching function-level values.
- Function-level `tilelang_compile_flags` are extended by explicit
  `compile_flags`.

`tilelang.compile(...)` returns `JITKernel`, the same runtime wrapper produced
by JIT compile.

## Parallel Compile

`tilelang.par_compile(...)` accepts an iterable of `PrimFunc` objects and
returns kernels in the same order:

```python
kernels = tilelang.par_compile(
    funcs,
    out_idx=[-1],
    target="cuda",
    num_workers=8,
    ignore_error=False,
)
```

Use it for offline preparation of multiple specializations. If
`ignore_error=True`, failed entries are logged and returned as `None`; otherwise
the first failed compile raises.

The JIT wrapper also exposes `.par_compile(configs, ...)`. Each config is
elaborated into a `PrimFunc` first, then passed through the same parallel
compile path:

```python
kernels = matmul.par_compile(
    [
        {"M": 1024, "N": 1024, "K": 1024, "block_M": 128},
        {"M": 2048, "N": 2048, "K": 2048, "block_M": 128},
    ],
    num_workers=4,
)
```

Config entries may be dictionaries of keyword arguments or tuples of positional
arguments. The wrapper uses the decorator's compile options, including
`out_idx`, target/backend choices, pass configs, and compile flags.

## JIT Mode Inference

`@tilelang.jit` starts in `mode="auto"`.

- If the wrapped function returns a `PrimFunc`, the wrapper uses lazy mode and
  returns a compiled kernel object.
- If the wrapped function uses the eager builder pattern, the wrapper uses eager
  mode and direct calls execute the compiled kernel immediately.

Once inferred, the wrapper stores the mode. `out_idx` is only legal in lazy
mode.

## In-Process Kernel Cache

Each JIT wrapper has its own in-process cache from specialization key to
compiled kernel. The key is built by the wrapped TileLang function after
argument parsing, so tensor dimensions, ordinary specialization arguments, and
hidden autotune parameters can all affect the selected entry.

Practical implications:

- Reusing the same JIT wrapper and same specialization reuses the compiled
  kernel.
- Changing specialization arguments such as tile sizes creates a different
  entry.
- Hidden `__tune_params` are folded into the specialization path for autotune.
- This wrapper cache is separate from the on-disk kernel cache controlled by
  TileLang environment/cache settings.

## On-Disk Kernel Cache

The compile path goes through TileLang's cache layer unless caching is disabled.
The cache key includes the lowered function script, output indices, target,
execution backend, pass configs, compile flags, TileLang version, platform, and
native-library stamp when available.

This means changing the actual kernel, target, backend, pass config, or compile
flag should invalidate the kernel cache. Editing unrelated Python scaffolding
around the same lowered function may not.

Use `tilelang.disable_cache()` or `TILELANG_DISABLE_CACHE=1` when debugging
cache behavior. Autotune has a separate result cache described in the autotune
guide.

## Pass Configs

Pass configs can be passed through the decorator or compile call:

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def kernel(...):
    ...
```

Examples frequently use pass configs for:

- fast math
- disabling warp specialization
- disabling TMA lowering
- enabling PTXAS verbose output
- enabling layout visualization

Prefer `tilelang.PassConfigKey` constants when available. Existing examples also
show raw string keys; constants are easier to audit.

## Backend And Target Defaults

If `target`, `execution_backend`, or `verbose` is `None`, compile reads the
environment defaults:

- `TILELANG_TARGET`, default `auto`
- `TILELANG_EXECUTION_BACKEND`, default `auto`
- `TILELANG_VERBOSE`, default `0`

The `"auto"` execution backend is resolved against the target. Keep explicit
target/backend settings in examples that must be reproducible across machines.

## Debug Output

`debug_root_path` is a JIT wrapper option, not a `JITKernel` method:

```python
@tilelang.jit(debug_root_path="debug_tilelang")
def kernel(...):
    ...
```

When a wrapper compiles, TileLang writes generated kernel source and the
TileLang program text under that directory. Relative paths are resolved by the
JIT implementation before writing. For user-controlled output paths, compile
the kernel and use `JITKernel.export_sources(...)`.

The wrapper also has a convenience source helper:

```python
src = matmul.get_kernel_source(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
```

This compiles the requested specialization and returns the generated kernel
source. Use the compiled `JITKernel` object when you also need profiling,
launching, host source, or source export.

## Autotune Interaction

`@tilelang.autotune(...)` expects to wrap a `@tilelang.jit` result. During
tuning, hidden parameters and compile arguments are passed through the JIT
wrapper. Keep tunable config keys aligned with the kernel factory signature;
otherwise the autotuner rejects unused keys before benchmarking.
