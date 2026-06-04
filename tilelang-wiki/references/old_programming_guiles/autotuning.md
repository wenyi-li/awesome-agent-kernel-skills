# Autotuning

TileLang's autotuner searches a list of configuration dictionaries, compiles
each candidate, optionally validates the result, benchmarks candidates, and
returns the fastest compiled kernel.

There are two supported workflows:
- Decorator workflow: `@tilelang.autotune(...)` stacked above `@tilelang.jit(...)`
- Programmatic workflow: `AutoTuner.from_kernel(...).set_*().run(...)`

This guide assumes you already know the surrounding kernel structure from
`language_basics.md`. It focuses on how tunable parameters, config lists,
validation, and benchmarking wrap that kernel shape.

Implementation anchors:
- `import tilelang.autotuner.tuner`
- `import tilelang.autotuner.param`
- `import tilelang.autotuner.capture`
- `import tilelang.env`

Example anchors:
- `examples/gemm/example_gemm_autotune.py`
- `examples/gemm/example_gemm_advanced_autotune.py`
- `examples/gdn/example_chunk_delta_h.py`
- `examples/flash_attention/example_mha_fwd_bshd.py`

## What a Config Is

A config is a dictionary whose keys match tunable arguments in the kernel
factory:

```python
configs = [
    {
        "block_M": 128,
        "block_N": 256,
        "block_K": 32,
        "num_stages": 2,
        "thread_num": 128,
        "enable_rasteration": True,
    },
]
```

During tuning, TileLang calls the factory once per config by passing those keys
as keyword arguments. If a config contains a key that is not a parameter of the
factory, `AutoTuner.run()` raises `ValueError("Unused keys in config: ...")`.

Tune compile-time scheduling parameters, not runtime data:
- tile sizes: `block_M`, `block_N`, `block_K`
- pipeline depth: `num_stages`
- block threads: `threads` or `thread_num`
- memory/layout knobs: swizzle enable flags, GEMM transpose choices, split-K
  factors, backend-specific staging choices

## Decorator Workflow

The decorator must be written above `@tilelang.jit` because Python applies
decorators from bottom to top, and `tilelang.autotune` expects to receive a
`JITImpl`.

```python
import itertools
import tilelang


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[32, 64],
        num_stages=[0, 1, 2, 3],
        thread_num=[128, 256],
        enable_rasterization=[False, True],
    )
    return [
        dict(zip(iter_params.keys(), values))
        for values in itertools.product(*iter_params.values())
    ]


@tilelang.autotune(configs=get_configs(), warmup=3, rep=20)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K,
           block_M=128,
           block_N=128,
           block_K=32,
           num_stages=2,
           thread_num=128,
           enable_rasterization=False):
    # Build the usual TileLang kernel here.
    # The kernel body is the same minimal structure shown in
    # `language_basics.md`, except the tuning parameters now drive
    # tile sizes, thread count, swizzle choice, and pipeline depth.
    ...
```

Call the function with problem sizes. The first call for a cache key tunes and
returns the best compiled kernel; later calls reuse the cached best kernel.

```python
kernel = matmul(1024, 1024, 1024)
c = kernel(a, b)
```

Use concrete tuning inputs when automatic input generation cannot infer them,
especially with symbolic dimensions or scalar kernel inputs:

```python
from tilelang.autotuner import set_autotune_inputs

with set_autotune_inputs(a, b):
    kernel = matmul(M, N, K)
```

`set_autotune_inputs` accepts either varargs or a single list/tuple:

```python
with set_autotune_inputs(a, b):
    ...

with set_autotune_inputs([a, b]):
    ...
```

This is especially important for metadata-driven kernels, not just symbolic
shapes. If correctness depends on offsets, masks, lengths, grouped-GEMM size
tables, or similar structured tensors, capture a real consistent input set so
autotune validates each config against the actual contract:

```python
group_sizes = torch.tensor(..., device="cuda", dtype=torch.int32)

with set_autotune_inputs(packed_lhs, packed_rhs, group_sizes):
    result = autotuner.run(warmup=3, rep=20)
```

If the kernel takes metadata tensors, make sure the reference program accepts
the same non-output input signature as the kernel. A mismatched reference
signature or auto-generated metadata tensor is a common reason every config
appears to fail validation even when the kernel implementation is otherwise
correct.

### Decorator Arguments

```python
@tilelang.autotune(
    configs=...,
    warmup=25,
    rep=100,
    timeout=100,
    supply_type=tilelang.TensorSupplyType.Auto,
    ref_prog=None,
    supply_prog=None,
    rtol=1e-2,
    atol=1e-2,
    max_mismatched_ratio=0.01,
    skip_check=False,
    manual_check_prog=None,
    cache_input_tensors=False,
    do_not_specialize=None,
)
```

Notes from the implementation:
- Bare `@tilelang.autotune` without arguments is not supported.
- `configs` can be a list or a callable. In decorator mode, callable configs are
  called with the kernel factory's non-tuning arguments.
- `do_not_specialize` excludes selected call arguments from the decorator's
  in-process tune-cache key. This is useful when changing a value should not
  trigger retuning.
- If the caller explicitly provides all tunable parameters, the autotuner logs a
  warning and uses direct JIT compilation instead of tuning.

## Programmatic Workflow

The programmatic API gives explicit control over compile arguments, profile
arguments, and advanced `run()` options. The GEMM autotune example uses this
form.

```python
import tilelang as tl
from tilelang.autotuner import AutoTuner


def ref_program(A, B):
    return A @ B.T


def get_best_config(M, N, K, profile_backend="event"):
    def kernel(block_M=None,
               block_N=None,
               block_K=None,
               num_stages=None,
               thread_num=None,
               enable_rasteration=None):
        # Return the usual TileLang kernel factory here.
        ...

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K))
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
            backend=profile_backend,
        )
    )

    return autotuner.run(warmup=3, rep=20)
```

`AutoTuner.run()` returns an `AutotuneResult`:

```python
result = get_best_config(M, N, K)
print(result.config)
print(result.latency)
kernel = result.kernel
```

For the underlying kernel ingredients referenced here, see:

- `language_basics.md` for the canonical tiled kernel skeleton.
- `software_pipeline.md` for `num_stages` and `T.Pipelined`.
- `instructions.md` for `T.copy`, `T.gemm`, and `T.use_swizzle`.

### Compile Arguments

```python
autotuner.set_compile_args(
    out_idx=[-1],
    target="auto",
    execution_backend="auto",
    target_host=None,
    verbose=False,
    pass_configs=None,
)
```

If `target`, `execution_backend`, or `verbose` are omitted, TileLang reads the
environment-backed defaults:
- `TILELANG_TARGET`
- `TILELANG_EXECUTION_BACKEND`
- `TILELANG_VERBOSE`

`target` is normalized through `determine_target(...)` and then wrapped as a TVM
`Target`. The execution backend is resolved against that target.

Supported execution backend strings in the compile args are:
- `"auto"`
- `"tvm_ffi"`
- `"cython"`
- `"nvrtc"`
- `"torch"`

Some backend-specific code also handles CuTeDSL artifacts, but the public
compile-argument type currently exposes the strings above.

### Profile Arguments

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
    max_mismatched_ratio=0.01,
    skip_check=False,
    manual_check_prog=None,
    cache_input_tensors=False,
    backend="event",
)
```

For the programmatic API, pass timing controls to `autotuner.run(...)`. The
`warmup`, `rep`, and `timeout` values stored in `ProfileArgs` participate in the
profile-argument object and cache hashing, but the active benchmark loop receives
the `warmup`, `rep`, and `timeout` arguments from `run()`.

Profiler backends are:
- `"event"`
- `"cupti"`
- `"cudagraph"`

Validation behavior:
- If `skip_check=False` and `ref_prog` is provided, the profiler validates the
  candidate against `ref_prog`.
- If `manual_check_prog` is also provided, TileLang calls
  `manual_assert_close(...)`.
- Otherwise it calls `assert_allclose(...)` with `rtol`, `atol`, and
  `max_mismatched_ratio`.
- If `ref_prog` is `None`, the autotuner benchmarks candidates without a
  correctness reference.
- `ref_prog` should return reference output tensors. Do not return a boolean
  from `ref_prog`; use `manual_check_prog` for custom boolean/assertion logic.

## Input Supply

The autotuner needs concrete tensors to benchmark each candidate. Use the
following priority order.

### 1. `set_autotune_inputs`

This is the most predictable path for real kernels and is required when a
non-output PrimFunc parameter is scalar rather than a buffer.

```python
with set_autotune_inputs(a, b):
    kernel = matmul(M, N, K)
```

In `set_profile_args`, TileLang freezes captured inputs immediately and builds a
device-aware `supply_prog`. If benchmarking uses multiple CUDA devices, captured
tensors are cloned to the worker device.

### 2. Custom `supply_prog`

`supply_prog` receives profiler parameters and returns input tensors:

```python
def supply_prog(params):
    # params is a list of KernelParam-like descriptors for the current candidate.
    return [a, b]

autotuner.set_profile_args(supply_prog=supply_prog)
```

If `set_autotune_inputs(...)` is active, TileLang warns that `supply_prog` is
ignored. If `supply_prog` is provided, `supply_type` is ignored.

### 3. Built-in Tensor Suppliers

Without captured inputs or a custom supplier, TileLang uses the kernel profiler's
input generation:

```python
autotuner.set_profile_args(supply_type=tl.TensorSupplyType.Integer)
```

Common values used by examples:
- `tl.TensorSupplyType.Auto`
- `tl.TensorSupplyType.Integer`
- `tl.TensorSupplyType.Normal`
- `tl.TensorSupplyType.Randn`

Automatic input generation is convenient for static tensor-only kernels. For
symbolic shapes, dynamic values, scalar inputs, or unusual dtype requirements,
provide explicit inputs.

Keep the kernel factory closure serializable. `AutoTuner.run()` inspects the
factory closure when it hashes and schedules tuning work, so avoid capturing
live `torch.Tensor` objects in the closure. Hoist tensor-derived metadata such
as shape integers, TileLang dtypes, booleans, and scalar constants first, then
capture only those serializable values:

```python
rope_dtype = torch_dtype_to_tilelang_dtype(q.dtype)
freq_dtype = torch_dtype_to_tilelang_dtype(freqs.dtype)
batch, seq_len, heads, head_dim = q.shape

def kernel(block_M=None, threads=None):
    return make_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        block_M=block_M,
        threads=threads,
        dtype=rope_dtype,
        freq_dtype=freq_dtype,
    )
```

If you instead capture a tensor directly, the autotuner can fail early with a
message that the closure cell contents are not serializable.

## Caching

TileLang caches autotune results in memory and on disk when caching is enabled.
The cache key includes:
- TileLang version
- function source
- default parameter values
- selected closure values
- kernel call parameters
- config list
- compile-argument hash
- profile-argument hash

Disk cache directory:

```text
<TileLang kernel cache namespace>/autotuner/<cache-key>/
```

The namespace root is derived from the kernel cache, which is controlled by
`TILELANG_CACHE_DIR`.

Files written by `AutotuneResult.save_to_disk(...)` include:
- `best_config.json`
- `latency.json`
- `function.pkl`
- `out_idx.json`
- `params.pkl`
- `device_kernel.cu`
- `host_kernel.cu`
- one backend artifact such as `kernel_lib.so`, `kernel.cubin`, `kernel.py`, or
  `executable.so`

Cache controls:
- `TILELANG_CACHE_DIR`: default `~/.tilelang/cache`
- `TILELANG_TMP_DIR`: default `$TILELANG_CACHE_DIR/tmp`
- `TILELANG_DISABLE_CACHE=1`: disable TileLang caches globally
- `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`: disable autotune disk cache

The `"torch"` execution backend logs that disk cache saving is not supported for
that path. In that case, rely on in-process reuse or use a backend that persists
artifacts.

## Parallelism and Advanced Run Options

Basic run:

```python
result = autotuner.run(warmup=3, rep=20, timeout=100)
```

Advanced run, as used by `example_gemm_advanced_autotune.py`:

```python
result = autotuner.run(
    warmup=warmup,
    rep=rep,
    timeout=timeout,
    use_pipeline=use_pipeline,
    enable_grouped_compile=enable_grouped_compile,
    group_compile_size=group_compile_size,
    benchmark_multi_gpu=benchmark_multi_gpu,
    benchmark_devices=benchmark_devices,
)
```

Compile workers use a `ThreadPoolExecutor`. Worker count is controlled by:
- `TILELANG_AUTO_TUNING_CPU_UTILITIES`: CPU fraction, default `0.9`
- `TILELANG_AUTO_TUNING_CPU_COUNTS`: explicit worker count, `-1` means auto
- `TILELANG_AUTO_TUNING_MAX_CPU_COUNT`: cap, `-1` means no cap

Grouped compile:
- Enabled by `enable_grouped_compile=True` and `group_compile_size > 1`.
- Currently active only for CUDA plus `tvm_ffi`.
- Other target/backend combinations fall back to per-config compilation with a
  warning.

Pipelined benchmark:
- `use_pipeline=True` lets benchmark workers start as soon as compiled
  candidates become available.
- `use_pipeline=False` waits until compile progress reaches the benchmark phase.

Multi-GPU benchmark:
- `benchmark_multi_gpu=True` distributes benchmark work across CUDA device
  ordinals.
- `benchmark_devices=[0, 1, ...]` restricts the device list.
- Non-CUDA targets or invalid devices fall back to a single current device with
  warnings.

Timeouts:
- `timeout > 0` applies per benchmark call.
- In the active autotuner benchmark path, worker threads run each benchmark call
  in a daemon sub-thread and join with the configured timeout.
- A POSIX `SIGALRM` helper exists in the module, but the threaded benchmark path
  is the path used by `AutoTuner.run()`.

Logs are written to `autotuner.log` in the current working directory.

## Saving and Loading Results

`AutotuneResult` can be saved manually:

```python
from pathlib import Path

result = autotuner.run(warmup=3, rep=20)
result.save_to_disk(Path("out/best/matmul_1024"), verbose=True)
```

Reload with matching compile arguments:

```python
from tilelang.autotuner.param import AutotuneResult, CompileArgs

restored = AutotuneResult.load_from_disk(
    "out/best/matmul_1024",
    CompileArgs(out_idx=[-1], target="auto", execution_backend="auto"),
)
kernel = restored.kernel
```

For backend paths that do not persist executable artifacts, reloading may return
no kernel and require recompilation.

## Config Space Guidance

Keep config spaces legal and small at first:

```python
def get_configs(M, N, K):
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]

    configs = []
    for BM in block_M:
        for BN in block_N:
            for BK in block_K:
                for stages in num_stages:
                    for threads in thread_num:
                        for swizzle in enable_rasterization:
                            configs.append({
                                "block_M": BM,
                                "block_N": BN,
                                "block_K": BK,
                                "num_stages": stages,
                                "thread_num": threads,
                                "enable_rasterization": swizzle,
                            })
    return configs
```

Filter impossible configs before tuning:
- shared-memory use must fit the target
- threads must fit the target block limit
- tile shapes must be compatible with the data layout and GEMM transpose flags
- pipeline stages should match the copy/computation structure
- output shape and `out_idx` must match the PrimFunc parameters

For GEMM, the examples also show a Roller-backed path through
`MatmulTemplate(...).recommend_hints(topk=...)`; use that when you want
device-aware candidate generation instead of a raw Cartesian product.

## Troubleshooting

- `Use tilelang.autotune to decorate func without arguments is not supported yet`:
  call it as `@tilelang.autotune(configs=...)`.
- `The @autotune decorator can only be applied to @tilelang.jit decorated instances`:
  put `@tilelang.autotune(...)` above `@tilelang.jit(...)`.
- `Unused keys in config`: remove keys that are not parameters of the kernel
  factory or rename the factory parameter.
- `No configurations to tune`: return a non-empty config list.
- Scalar input error mentioning `set_autotune_inputs`: provide concrete inputs
  with `with set_autotune_inputs(...)`.
- Validation is slow or failing: verify `ref_prog` receives the same logical
  inputs as the kernel and returns outputs in the shape/dtype expected by the
  profiler.
- Cached inputs have incompatible shape or dtype across configs: set
  `cache_input_tensors=False` or provide a `supply_prog` that regenerates inputs
  per candidate.
- `Cell contents ... is not serializable`: remove captured `torch.Tensor`
  objects from the kernel-factory closure and capture only shapes, dtypes,
  flags, and other scalar metadata.
- Disk cache does not appear: check `TILELANG_DISABLE_CACHE`,
  `TILELANG_AUTO_TUNING_DISABLE_CACHE`, and the selected execution backend.
