# Python Compatibility And Dtypes

TileLang kernels are written in Python syntax, but the kernel body is lowered
to TileLang/TIR rather than executed by the Python interpreter. Use Python
freely for host-side configuration, shapes, autotune spaces, and factory code;
inside `@tilelang.jit` or `@T.prim_func` bodies, stay within the syntax that
TileLang lowers.

There are two frontend styles:

- `@tilelang.jit` eager-style kernels use a Python AST frontend around
  `with T.Kernel(...)`.
- `@T.prim_func` kernels use the TileLang TVM-script parser.

Both share the same core TileLang language: `T.Tensor`, `T.empty`, `T.Kernel`,
`T.Parallel`, scoped allocations, `T.copy`, scalar expressions, and typed
buffers.

## Quick Rules

- Host Python builds kernels; TileLang syntax describes device execution.
- Use `range(...)`, `T.serial(...)`, `T.Parallel(...)`, `T.Pipelined(...)`, and
  TileLang buffer indexing for kernel loops and data access.
- Use Python containers only for compile-time configuration, not as device-side
  storage.
- Use TileLang dtypes, strings, NumPy dtypes, Torch dtypes, or Python scalar
  types where a dtype is expected; TileLang normalizes them to TVM dtypes.
- Keep accumulation dtype explicit for mixed-precision kernels.

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B, block: int, dtype=T.float32):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), dtype)
    B: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block), T.ceildiv(M, block), threads=128) as (bx, by):
        for i, j in T.Parallel(block, block):
            row = by * block + i
            col = bx * block + j
            C[row, col] = A[row, col] + B[row, col]

    return C
```

## Supported Kernel Python

| Pattern | Use it for | Notes |
| --- | --- | --- |
| `for i in range(n)` | Serial kernel loops | Lowered like `T.serial(n)`. |
| `for i in range(a, b, s)` | Serial loops with bounds/step | Constant positive and negative steps are supported in current tests. |
| `for i in T.serial(...)` | Explicit serial loops | Prefer when clarity matters. |
| `for i, j in T.Parallel(...)` | Parallel tile loops | Common for elementwise tile work. |
| `for k in T.Pipelined(...)` | Software-pipelined loops | Used for copy/compute staging. |
| `while condition` | Device loops | Keep loop state in TileLang variables. |
| `if` / `elif` / `else` | Branches | Symbolic conditions lower to TIR; compile-time booleans are resolved by the frontend. |
| `x if cond else y` | Value conditionals | Supported in eager/JIT; use `T.if_then_else` when parser behavior is uncertain. |
| `A[i]`, `A[i, j]` | Buffer load/store | Multidimensional indexing is supported. |
| `A[i:j]` | Buffer regions | Use mainly for `T.copy` and other TileLang operations. |
| `A[-1]` | Negative indexing | Legalized when the extent and sign are provable. |
| `buffer.shape[d]` | Extents | Preferred over `len(buffer)`. |
| `x = expr`, `x: T.int32 = expr` | Local bindings | Creates a TileLang expression or compile-time Python value depending on `expr`. |
| `A[i] = expr`, `+=`, `-=`, `*=` | Stores and updates | Works for local variables and buffer elements. |
| `a, b = b, a` | Tuple assignment | Swap/unpack patterns are tested. |
| `assert cond, msg` | Assertions | `@T.prim_func` parses to `T.Assert`; eager/JIT symbolic conditions become device asserts. |
| `T.print(...)` | Device print | Use ordinary `print(...)` only for host-side setup/debug output. |
| `with T.Kernel(...)` | Launch region | Arbitrary Python context managers do not have device semantics. |

Avoid these inside device code:

- Iterating over Python lists, tuples, dicts, `zip(...)`, or `enumerate(...)`
  as kernel loops.
- Consecutive assignment such as `a = b = c`.
- Stepped buffer slices such as `A[i:j:step]`; use explicit loops.
- `is`, `in`, `not in`, `type(...)`, or `isinstance(...)` as device-side
  predicates.
- Ordinary Python functions or classes for device execution; use `@T.macro`
  for reusable TileLang code blocks.

`break` and `continue` are supported by the eager/JIT AST frontend, but are not
portable across both frontend styles. Avoid them in `@T.prim_func` kernels.

## Dtype Forms

Most dtype-taking APIs accept dtype-like values that TileLang can normalize:

```python
dtype = T.float16
accum_dtype = "float32"

@T.prim_func
def matmul(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), torch.float16),
):
    C_frag = T.alloc_fragment((BM, BN), accum_dtype)
```

Accepted forms include:

- TileLang/TVM dtype objects: `T.float32`, `T.int8`, `T.bfloat16`.
- Strings: `"float32"`, `"int8"`, `"bfloat16"`.
- Python scalar types: `bool`, `int`, `float`.
- NumPy dtypes: `np.float32`, `np.int32`, `np.dtype("float16")`.
- Torch dtypes: `torch.float32`, `torch.int8`, `torch.bfloat16`, plus
  supported FP8/FP4 dtypes when present in the installed Torch version.

Common aliases normalize as follows:

```text
T.float  -> float32
T.half   -> float16
T.double -> float64
T.int    -> int32
T.uint   -> uint32
T.long   -> int64
T.short  -> int16
```

Use `T.dtype(dtype).as_torch()` when test or integration code needs the Torch
equivalent of a TileLang dtype. FP8 Torch mapping can be backend-sensitive:
CUDA and HIP may prefer different `fn`/`fnuz` variants.

## Tensor And Allocation Types

Use `T.Tensor` for kernel parameters and `T.empty` for eager-style outputs:

```python
A: T.Tensor((M, K), T.float16)
B: T.Tensor[[K, N], "float16"]
C = T.empty((M, N), T.float16)
```

`T.Tensor` defaults to global memory and row-major contiguous layout. A scalar
shape is treated as one-dimensional. `T.Buffer` still exists for compatibility,
but new TileLang code should prefer `T.Tensor`.

Use typed allocation helpers for device storage:

```python
A_shared = T.alloc_shared((BM, BK), dtype)
B_shared = T.alloc_shared((BK, BN), dtype)
C_frag = T.alloc_fragment((BM, BN), T.float32)
tmp = T.alloc_local((128,), "float32")
flag = T.alloc_var(T.int32, init=0)
```

`T.alloc_shared`, `T.alloc_fragment`, `T.alloc_local`, `T.alloc_global`,
`T.alloc_var`, and related helpers follow the same dtype normalization rules.
`T.alloc_var(dtype, init=...)` casts numeric initializers to the requested
dtype.

## Casts And Mixed Precision

Dtype objects are callable for scalar construction or conversion:

```python
x = T.float32(1)
y = T.int32(i)
v = T.float16x2(a, b)
```

Use `T.cast(value, dtype)` for numeric conversion and
`T.reinterpret(value, dtype)` for bit reinterpretation.

For mixed precision, choose input, output, and accumulation dtypes separately:

```python
dtype = T.float16
accum_dtype = T.float32

A_shared = T.alloc_shared((BM, BK), dtype)
B_shared = T.alloc_shared((BK, BN), dtype)
C_frag = T.alloc_fragment((BM, BN), accum_dtype)

T.gemm(A_shared, B_shared, C_frag)
```

FP8 GEMM examples commonly use FP8 inputs, `T.float32` scale tensors,
`T.float32` accumulators, and `T.bfloat16` or `T.float32` outputs. Treat
low-precision and vector dtype names as frontend availability first, then check
backend and architecture support.

## Host Python Versus Kernel Python

This is host-side Python and may use ordinary containers freely:

```python
configs = [(64, 64), (128, 64)]
for block_m, block_n in configs:
    kernel = make_kernel(block_m, block_n)
```

Inside the kernel, use TileLang loops and buffers:

```python
for i, j in T.Parallel(block_m, block_n):
    out[i, j] = 0
```

When in doubt, ask whether a value exists while building the kernel or while
executing on the device. Build-time values can use normal Python. Device-time
values should use TileLang expressions, buffers, loops, casts, assertions, and
macros.
