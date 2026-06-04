# TileLang Overview

TileLang is a Python-embedded DSL for writing tiled accelerator kernels. A TileLang
program is ordinary Python code that builds TVM TIR with TileLang-specific tile
operations such as `T.copy`, `T.gemm`, `T.Parallel`, and `T.Pipelined`. The compiler
then runs TileLang passes that infer layouts, lower tile operations, split host and
device code, and emit target-specific source for CUDA, HIP, Metal, or C-style
backends.

The best way to understand the current programming model is to read the working
examples in `examples`. The minimal GEMM examples use the same
core pieces that appear in larger attention, convolution, FP8, and sparse kernels:

- Declare runtime tensors and symbolic dimensions with `T.Tensor`, `T.const`, and
  `T.dynamic`.
- Create a tiled launch frame with `T.Kernel`.
- Allocate explicit memory scopes with `T.alloc_shared`, `T.alloc_local`, and
  `T.alloc_fragment`.
- Move tiles with `T.copy`.
- Express tiled computation with tile ops such as `T.gemm`, reductions, or explicit
  `T.Parallel` loops.
- Compile, run, inspect, and profile the generated kernel through `tilelang.jit` and
  `JITKernel`.

## A Small GEMM Kernel

The following shape mirrors `examples/quickstart.py` and
`examples/gemm/example_gemm.py`:

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

This is not a Python loop that launches GPU work immediately. The decorated function
builds a TileLang/TIR program. `T.Kernel` records a grid and thread-block frame;
`T.Pipelined` records a loop with software-pipeline annotations; `T.copy` and
`T.gemm` emit TileLang tile operations that are lowered later.

## Running A Kernel

There are two common working flows.

Use explicit compilation when you want a reusable kernel object:

```python
kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

c = kernel(a, b)
print(kernel.get_kernel_source())

profiler = kernel.get_profiler()
latency_ms = profiler.do_bench()
```

Use direct JIT-call style when runtime tensors should drive specialization and the
result can be returned immediately:

```python
c = matmul(a, b, block_M=128, block_N=128, block_K=32)
```

The direct-call style is used by examples such as `elementwise/example_elementwise_add.py`
and `dynamic_shape/example_dynamic.py`. Internally, `tilelang.jit` caches compiled
kernels by argument/configuration key, so repeated calls with the same specialization
reuse the compiled kernel.

For correctness checks, the examples compare against PyTorch references with
`torch.testing.assert_close`. For benchmarking, use `kernel.get_profiler().do_bench()`
after explicit compilation, or `tilelang.profiler.do_bench(lambda: kernel_call(...))`
for direct-call examples.

## Buffers And Memory Scopes

TileLang exposes memory placement directly. These APIs map to buffer scopes that the
compiler later lowers for the selected backend.

| API | Scope | Typical use |
| --- | --- | --- |
| `T.Tensor(shape, dtype)` | Global parameter buffer | Input and output tensors passed from Python. |
| `T.empty(shape, dtype)` | Returned output buffer | Output allocation for eager-style JIT functions. |
| `T.alloc_shared(shape, dtype)` | `shared.dyn` | Thread-block shared staging memory. |
| `T.alloc_local(shape, dtype)` | `local` | Per-thread local storage. |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` | Register/fragment storage whose layout is inferred by TileLang. |

Prefer `T.Tensor` in new code. `T.Buffer` still exists for compatibility, but the
frontend marks it as deprecated.

## Loops And Tile Operations

`T.Parallel` expresses logical element parallelism. It creates annotated parallel
TIR loops; later passes validate the loop and lower it into thread-partitioned code
using inferred or explicit layouts. It should be read as a TileLang parallel loop,
not as Python threads.

`T.Pipelined` expresses a serial loop with pipeline metadata such as `num_stages`.
The software pipeline is materialized by compiler passes such as pipeline planning
and software-pipeline injection. The annotation is what enables asynchronous copy
and staged execution opportunities during lowering.

`T.copy` is the default tile movement API. Depending on the source/destination
regions, target, and annotations, TileLang can lower it to ordinary SIMT copies,
CUDA `cp.async`, TMA bulk load/store, LDSM/STSM, or TMEM-related paths. Start with
`T.copy`; use lower-level copy APIs only when a specific example or optimization
requires them.

`T.gemm` validates tile regions, shapes, strides, transposition flags, and optional
barrier metadata, then emits a `tl.tileop.gemm` operation. Backend-specific lowering
chooses implementations for NVIDIA, AMD, or other supported paths. Higher-level
operators such as convolution often reduce to tiled data movement plus `T.gemm`
or related tile operations.

Reductions such as `T.reduce_sum`, `T.reduce_max`, `T.cumsum`, and
`T.finalize_reducer` are available for fragment/shared-memory workflows. They are
part of the tile programming model, but most getting-started kernels are easier to
understand through copy, GEMM, and explicit `T.Parallel` loops first.

## Compilation Path

The implemented lowering path is:

```text
@tilelang.jit / @T.prim_func
  -> Python frontend builds TIR with tl.tileop calls
  -> pre-lower semantic checks
  -> frontend legalization and simplification
  -> pipeline planning and software-pipeline injection
  -> layout reducer and layout inference
  -> LowerTileOp for copy/gemm/reduce/parallel lowering
  -> target-specific optimization
  -> host/device split
  -> CUDA, HIP, Metal, C, or other codegen path
  -> JIT adapter for execution from Python
```

This means TileLang is neither a pure scheduling language nor a black-box operator
library. You write the tiled memory movement and computation structure directly, and
the compiler fills in layouts, thread mappings, software-pipeline mechanics, and
backend-specific code generation where it has enough information to do so.
