# Autotune Basics

`tilelang.autotune(...)` searches candidate kernel configurations, compiles
candidates, validates them when a reference is supplied, benchmarks valid
candidates, and returns the fastest compiled kernel for the current
specialization.

There are two public styles:

- Decorator form for kernels already written as `@tilelang.jit`.
- `AutoTuner.from_kernel(...)` for a plain factory that returns `T.prim_func`.

Most examples use the decorator form. Use `AutoTuner.from_kernel(...)` when you
need explicit control over compile and profiling arguments or want the full
autotune result object.

## Decorator Form

Place `@tilelang.autotune(...)` above `@tilelang.jit(...)`:

```python
import itertools
import tilelang
import tilelang.language as T


def get_configs():
    params = {
        "block_M": [64, 128],
        "block_N": [64, 128],
        "block_K": [32, 64],
        "num_stages": [2, 3],
        "threads": [128, 256],
    }
    return [
        {k: v for k, v in zip(params, values)}
        for values in itertools.product(*params.values())
    ]


@tilelang.autotune(configs=get_configs(), warmup=3, rep=20)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=2, threads=128):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

The keys in each config dictionary must match parameters of the decorated
function. In the example above, `block_M`, `block_N`, `block_K`, `num_stages`,
and `threads` are tunable because they are kernel factory parameters.

Calling the decorated function triggers tuning for that specialization:

```python
kernel = matmul.compile(4096, 4096, 4096)
```

`.compile(...)` returns the best `JITKernel`. A direct call also tunes on first
use; in lazy mode it returns the best `JITKernel`, while in eager mode it runs
the best kernel immediately with the call inputs.

## Programmatic `AutoTuner`

Use `AutoTuner.from_kernel(...)` when you want explicit control over compile
and profile arguments:

```python
import tilelang as tl
from tilelang.autotuner import AutoTuner


def ref_program(A, B):
    return A @ B.T


def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
    # Return a T.prim_func that uses these tuning parameters.
    ...


result = (
    AutoTuner.from_kernel(kernel=kernel, configs=configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        backend="event",
    )
    .run(warmup=3, rep=20)
)

best_kernel = result.kernel
best_config = result.config
best_latency = result.latency
```

The usual call chain is:

1. `set_compile_args(...)`
2. `set_profile_args(...)`
3. `run(...)`

The returned result contains the best latency, best config, reference latency
when measured, generated source, tuned `PrimFunc`, and compiled `JITKernel`.

## Compile Arguments

`set_compile_args(...)` controls candidate compilation:

```python
autotuner.set_compile_args(
    out_idx=[-1],
    target="auto",
    execution_backend="auto",
    pass_configs=None,
)
```

- `out_idx` marks output tensor parameters for lazy/factory kernels.
- `target=None` reads `TILELANG_TARGET`.
- `execution_backend=None` reads `TILELANG_EXECUTION_BACKEND`.
- `verbose=None` reads `TILELANG_VERBOSE`.
- `pass_configs` is forwarded to the compiler.

Use explicit `out_idx` when the kernel writes output buffers instead of
returning `T.empty(...)`. Negative indices are allowed, so `out_idx=[-1]`
marks the last `T.prim_func` parameter as output.

## Profile Arguments

`set_profile_args(...)` controls validation and timing:

```python
autotuner.set_profile_args(
    warmup=25,
    rep=100,
    timeout=30,
    supply_type=tilelang.TensorSupplyType.Auto,
    ref_prog=None,
    supply_prog=None,
    rtol=1e-2,
    atol=1e-2,
    skip_check=False,
    manual_check_prog=None,
    cache_input_tensors=False,
    backend="event",
)
```

- `ref_prog` validates candidates when `skip_check=False`.
- `supply_type` controls generated inputs when `supply_prog` is absent.
- `supply_prog` returns concrete inputs and overrides `supply_type`.
- `manual_check_prog` replaces default allclose-style validation.
- `cache_input_tensors=True` reuses generated inputs across compatible
  candidates.
- `backend` selects profiler timing: `"event"`, `"cupti"`, or `"cudagraph"`.
- `timeout` is the per-candidate benchmark timeout.
- `max_mismatched_ratio` can relax validation for a small fraction of
  mismatched elements.

Arguments passed to `run(warmup=..., rep=..., timeout=...)` override the stored
profile defaults for that tuning run.

## Capturing Real Inputs

Generated tensors work for simple dense kernels. Use real captured inputs when
correctness depends on metadata values such as offsets, lengths, masks, grouped
GEMM tables, or variable-length sequence arrays.

```python
from tilelang.autotuner import AutoTuner, set_autotune_inputs

with set_autotune_inputs(packed_lhs, packed_rhs, group_sizes):
    result = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(ref_prog=reference, skip_check=False)
        .run(warmup=3, rep=20)
    )
```

`set_autotune_inputs(...)` accepts positional inputs or a single list/tuple of
inputs. Captured inputs are frozen when `set_profile_args(...)` runs and become
the active `supply_prog` for that tuner. If you also pass `supply_prog`, the
captured inputs take precedence.

## Common Pitfalls

- Bare `@tilelang.autotune` without arguments is not supported; call it as
  `@tilelang.autotune(configs=...)`.
- Put `@tilelang.autotune(...)` above `@tilelang.jit(...)`.
- Config keys must match kernel factory parameters.
- If every tunable parameter is supplied explicitly by the caller, the decorator
  skips the search and compiles that fixed configuration.
- Use real captured inputs for sparse, grouped, masked, or varlen kernels.
- Autotune result caching is separate from the regular JIT kernel cache; disable
  both caches when debugging source changes.
