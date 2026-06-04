# JIT Basics

`@tilelang.jit` is the normal Python entry point for a TileLang kernel. It wraps
a kernel definition and returns a JIT wrapper. The wrapper can compile lazily
through `.compile(...)`, or it can run directly when called with tensors.

```python
import tilelang
import tilelang.language as T
```

## Eager-Style Kernels

Most examples use eager-style JIT. Tensor arguments are ordinary Python
parameters, tensor shapes are bound with `T.const(...)`, and outputs are
created with `T.empty(...)`. Direct calls compile and run the kernel.

```python
@tilelang.jit
def elementwise_add(A, B, block_M, block_N, threads):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), T.float32)
    B: T.Tensor((M, N), T.float32)
    C = T.empty((M, N), T.float32)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        for i, j in T.Parallel(block_M, block_N):
            row = by * block_M + i
            col = bx * block_N + j
            C[row, col] = A[row, col] + B[row, col]

    return C
```

`T.const("M, N")` names dimensions that are fixed for the compiled
specialization. In a direct call they are inferred from the tensor arguments; in
an explicit `.compile(...)` call they can be supplied by name. Parameters such
as `block_M`, `block_N`, and `threads` are Python specialization arguments and
also become part of the JIT specialization. The compact example assumes `M` and
`N` are divisible by the tile sizes; add bounds checks for ragged edge tiles.

## Compile Then Call

Use `.compile(...)` when you want a reusable compiled object:

```python
kernel = elementwise_add.compile(M=1024, N=1024, block_M=32, block_N=32, threads=128)
C = kernel(A, B)
```

The returned object is a `tilelang.JITKernel`. It is callable and exposes
source/profiling helpers:

```python
print(kernel.get_kernel_source())

profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency_ms = profiler.do_bench()
```

Explicit compile is the clearest form for examples, benchmarks, generated
source inspection, and repeated launches with the same specialization.

## Direct Calls

You can call an eager-style JIT function directly:

```python
C = elementwise_add(A, B, block_M=32, block_N=32, threads=128)
```

The wrapper infers eager mode, compiles the first matching specialization,
stores it in the wrapper's in-process cache, executes it, and returns the
output tensor or tensors. Direct calls are concise for smoke tests. Compile
first when you need inspection, profiling, exporting, or explicit control over
launch reuse.

## Lazy Factory Style

Lazy style returns a nested `@T.prim_func` from the JIT-decorated function.
Calling the wrapper returns a compiled `JITKernel`; the returned kernel is then
called with runtime tensors. This style is common when output buffers are
explicit parameters.

```python
@tilelang.jit(out_idx=[2])
def add_factory(M: int, N: int, block_M: int, block_N: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]

    return main
```

`out_idx` marks which argument positions are outputs returned after launch.
Negative indices are supported, for example `out_idx=[-1]` for the final
argument. If more than one output is produced, pass a list such as
`out_idx=[-3, -2, -1]`.

Do not use `out_idx` with eager-style `T.empty(...)` outputs. The JIT wrapper
raises an error if eager mode is selected and `out_idx` is set.

## Common Options

`@tilelang.jit(...)` and `.compile(...)` share compile-oriented options:

```python
@tilelang.jit(
    target="cuda",
    execution_backend="auto",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def kernel(...):
    ...
```

Common options:

- `target`: compilation target, defaulting to the environment target or
  `"auto"`.
- `execution_backend`: runtime adapter, defaulting to the environment backend
  or `"auto"`.
- `target_host`: optional host target for cross-compilation.
- `pass_configs`: TileLang/TVM lowering switches.
- `compile_flags`: extra compiler flags.
- `verbose`: lower-level compile and backend-resolution logging.
- `debug_root_path`: save generated kernel source and TileLang program text
  when compiling through the JIT wrapper.

## Inspecting A Kernel

After compiling, use `JITKernel` helpers:

```python
print(kernel.get_kernel_source())
print(kernel.get_host_source())

kernel.show_source("kernel")
kernel.show_source("host")
kernel.show_source("both")

kernel.export_sources(kernel_path="kernel.cu", host_path="host.cc")
```

`show_source(...)` prints generated source. `export_sources(...)` writes source
files. `print_source_code(...)` still exists as a compatibility alias, but new
docs should use `show_source(...)`.

## Decision Rule

- Ordinary kernel or benchmark: `@tilelang.jit` plus `.compile(...)`.
- Quick correctness test: direct JIT call.
- Explicit output buffers: lazy factory style with `out_idx`.
- Need generated source, profiling, exported source files, or repeated launches:
  compile first and use the returned `JITKernel`.
