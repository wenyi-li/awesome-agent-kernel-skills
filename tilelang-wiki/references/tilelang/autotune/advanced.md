# Autotune Advanced

This page covers cache behavior, config binding, validation paths, CPU worker
settings, timeout behavior, and advanced `run(...)` options.

## Config Binding

Each config is a dictionary. The autotuner filters a config against the kernel
factory signature and rejects unused keys. This prevents a common bug where a
misspelled tuning key silently has no effect.

Example:

```python
configs = [
    {"block_M": 128, "block_N": 128, "threads": 256},
]
```

Every key must be a parameter accepted by the JIT function or kernel factory.
Shape arguments such as `M`, `N`, and `K` may be supplied by the call site,
while tile sizes and scheduling choices usually come from the config.

In decorator form, `configs` may also be a callable. It is evaluated with the
kernel call arguments for the current specialization, which is useful when the
candidate list depends on shapes.

## Explicit Caller Values

If the caller explicitly supplies a tunable parameter, that value wins for the
current call. This is useful for forcing a known-good configuration during
debugging. It also means the autotuner may have fewer free parameters to search
than the config list suggests.

If all tunable keys are already supplied by the caller, the decorator path logs
a warning, skips tuning, and compiles the fixed configuration directly.

Use `do_not_specialize=(...)` on the decorator when a call argument should not
participate in the decorator's per-specialization tuner cache key. This is only
for arguments that do not change the valid config choice or generated program.

## Result Object

The programmatic API returns an autotune result with:

- `latency`: best candidate latency.
- `config`: winning configuration dictionary.
- `ref_latency`: reference-program latency when measured.
- `libcode`: generated source from the best kernel.
- `func`: tuned TileLang `PrimFunc`.
- `kernel`: compiled `JITKernel`.

Prefer returning or storing `result.kernel` when the caller needs to execute the
tuned kernel repeatedly.

## Validation Paths

The profiler can validate candidates in three ways:

- Default validation: run candidate and `ref_prog`, then compare outputs using
  tolerances.
- `manual_check_prog`: custom candidate/reference validation.
- `skip_check=True`: benchmark without correctness checks.

Use `skip_check=True` only when another path already guarantees correctness.
For kernels with non-random-valid metadata, capture real inputs with
`set_autotune_inputs(...)` or provide `supply_prog`.

## Input Supply Precedence

Input supply order is:

1. Active `set_autotune_inputs(...)` context.
2. `supply_prog`, when provided.
3. Auto-generated tensors from `supply_type`.

Captured inputs are frozen into a `supply_prog` when `set_profile_args(...)`
runs. That detail matters for programmatic tuning: enter
`set_autotune_inputs(...)` before calling `set_profile_args(...)`. Captured
inputs are the safest option for grouped, sparse, masked, paged, and varlen
kernels.

## Cache Behavior

The autotuner has an in-memory result cache and an on-disk result cache. The
disk cache is separate from the regular compiled-kernel cache.

The autotune cache key includes normalized specialization arguments, config
space, selected compile/profile settings, TileLang version, simple closure
values, and source for the callable passed into the tuner. It does not hash the
full contents of reference functions, supply functions, or captured tensors.

Practical cache gotcha: if the callable passed to autotune is a small wrapper
that calls a helper, editing only the helper may not change the autotune cache
key. In that case the tuner can reuse an older result and skip visible
re-tuning.

For a fresh debug run:

```bash
TILELANG_AUTO_TUNING_DISABLE_CACHE=1 TILELANG_DISABLE_CACHE=1 python tune.py
```

`TILELANG_AUTO_TUNING_DISABLE_CACHE=1` disables autotune result cache lookup and
disk save for the run. `TILELANG_DISABLE_CACHE=1` disables TileLang cache use
more broadly, including the regular compiled-kernel cache underneath tuning.

## CPU Worker Controls

Autotune uses CPU workers for compiling and benchmarking candidates. Environment
variables control worker selection:

| Variable | Meaning |
| --- | --- |
| `TILELANG_AUTO_TUNING_CPU_UTILITIES` | Fraction of available CPUs to use. Default `0.9`. |
| `TILELANG_AUTO_TUNING_CPU_COUNTS` | Explicit worker count. `-1` means auto. |
| `TILELANG_AUTO_TUNING_MAX_CPU_COUNT` | Upper bound for auto worker count. `-1` means no limit. |

Use explicit worker counts when tuning inside shared CI or a busy workstation.
These variables affect compilation parallelism; benchmark workers are controlled
by the `run(...)` arguments described below.

## Timeout Behavior

`run(..., timeout=...)` applies a per-candidate benchmark deadline. On POSIX
main-thread runs, the implementation uses signal alarms. In other contexts it
uses a watchdog thread that injects a timeout exception at Python boundaries.

Timeouts can stop a Python-level benchmark path, but a kernel or driver call
stuck inside C/CUDA code may only respond after control returns to Python.
Choose conservative configs when testing a new search space.

## Grouped Compile And Multi-GPU Notes

`run(...)` has optional controls for larger search spaces:

```python
result = autotuner.run(
    warmup=3,
    rep=20,
    timeout=30,
    use_pipeline=True,
    enable_grouped_compile=True,
    group_compile_size=2,
    benchmark_multi_gpu=True,
    benchmark_devices=[0, 1],
)
```

`use_pipeline=True` starts benchmarking as compile results become available.
Grouped compilation is currently active only for CUDA with the `tvm_ffi`
execution backend; otherwise TileLang falls back to per-config compilation.
Keep config spaces small until a single fixed config validates.

For multi-GPU benchmarking, ensure the process has the intended visible CUDA
devices and captures device-local inputs. The autotuner benchmarks concrete
kernels; it does not infer distributed placement from the config.

## Logging

Autotuner logs are written to `autotuner.log` in the current working directory
and also to stdout through the autotuner logger. Use this file to distinguish:

- candidates rejected by compile failure
- candidates rejected by validation
- candidates that timed out
- candidates that benchmarked successfully

## Minimal Debug Checklist

1. Validate one fixed config without autotune.
2. Add autotune with a tiny config list.
3. Capture real inputs if metadata values matter.
4. Disable both autotune and kernel caches when debugging source changes.
5. Increase search space only after compile, validation, and timing are stable.
