# JIT And Autotune Cheatsheet

Use this page as the short path for compiling a TileLang program with
`tilelang.jit` and scanning a small set of kernel configurations with
`tilelang.autotune`. For full option lists and lower-level cache behavior, see
`tilelang/jit/basic.md`, `tilelang/jit/advanced.md`,
`tilelang/autotune/basic.md`, and `tilelang/autotune/advanced.md`.

## Compile With `tilelang.jit`

The usual flow is:

1. Write a Python function decorated with `@tilelang.jit`.
2. Define the TileLang program using `tilelang.language as T`.
3. Call `.compile(...)` with shape and specialization arguments.
4. Launch the returned `JITKernel` with runtime tensors.

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B, block_M: int, block_N: int, threads: int):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), T.float32)
    B: T.Tensor((M, N), T.float32)
    C = T.empty((M, N), T.float32)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            row = by * block_M + i
            col = bx * block_N + j
            if row < M and col < N:
                C[row, col] = A[row, col] + B[row, col]

    return C


kernel = add.compile(M=1024, N=1024, block_M=32, block_N=32, threads=128)
C = kernel(A, B)
```

`T.const(...)` dimensions are fixed for the compiled specialization. Values
such as `block_M`, `block_N`, and `threads` are Python specialization
arguments, so changing them creates a different compiled kernel entry.

For quick smoke tests, an eager-style JIT function can also be called directly:

```python
C = add(A, B, block_M=32, block_N=32, threads=128)
```

Compile first when you want to inspect source, profile, export generated files,
or reuse the same specialization explicitly:

```python
print(kernel.get_kernel_source())
latency_ms = kernel.get_profiler().do_bench()
```

## Tune With `tilelang.autotune`

Use `tilelang.autotune(...)` to try a finite list of configuration dictionaries.
Place it above `@tilelang.jit(...)`. Each config key must match a parameter of
the decorated kernel factory.

```python
configs = [
    {"block_M": 32, "block_N": 32, "threads": 128},
    {"block_M": 64, "block_N": 32, "threads": 128},
    {"block_M": 64, "block_N": 64, "threads": 256},
]


@tilelang.autotune(configs=configs, warmup=3, rep=20)
@tilelang.jit
def tuned_add(A, B, block_M=32, block_N=32, threads=128):
    ...


best_kernel = tuned_add.compile(M=1024, N=1024)
C = best_kernel(A, B)
```

During tuning, TileLang compiles candidates, optionally validates them, times
valid candidates, and returns the fastest `JITKernel` for the current
specialization. Keep the first search space small until one fixed config
compiles and validates.

For kernels that write explicit output buffers, use lazy/factory style with
`out_idx`:

```python
@tilelang.autotune(configs=configs, warmup=3, rep=20)
@tilelang.jit(out_idx=[-1])
def matmul_factory(M, N, K, block_M=128, block_N=128, block_K=32, threads=256):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        ...

    return main


best_kernel = matmul_factory.compile(4096, 4096, 4096)
best_kernel(A, B, C)
```

## Practical Rules

- Start with `@tilelang.jit` plus `.compile(...)` for ordinary examples and
  benchmarks.
- Use direct calls only for short correctness checks.
- Use `@tilelang.autotune(configs=...)`; bare `@tilelang.autotune` is not the
  supported form.
- Put `@tilelang.autotune(...)` above `@tilelang.jit(...)`.
- Make every config key match a kernel parameter, such as `block_M`, `block_N`,
  `block_K`, `num_stages`, or `threads`.
- If correctness depends on real metadata tensors, capture real inputs with
  `set_autotune_inputs(...)` or use the programmatic `AutoTuner` API.
- Pass basic compiler switches through `pass_configs` on `@tilelang.jit(...)`
  or `.compile(...)`; prefer `tilelang.PassConfigKey` constants when available.
- When debugging stale tuning results, disable both cache layers:
  `TILELANG_AUTO_TUNING_DISABLE_CACHE=1 TILELANG_DISABLE_CACHE=1`.

## Where To Go Next

- `tilelang/jit/basic.md`: eager JIT, lazy factory style, direct calls,
  `.compile(...)`, `out_idx`, source inspection, and profiling helpers.
- `tilelang/jit/advanced.md`: `tilelang.compile(...)`, parallel compile, JIT
  mode inference, cache keys, pass configs, target/backend defaults, and debug
  output.
- `tilelang/autotune/basic.md`: decorator autotune, `AutoTuner.from_kernel(...)`,
  compile/profile arguments, captured inputs, and common pitfalls.
- `tilelang/autotune/advanced.md`: config binding, explicit caller values,
  result objects, validation paths, tuning caches, worker controls, timeouts,
  and grouped or multi-GPU benchmarking.
