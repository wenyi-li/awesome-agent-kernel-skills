# Python Compatibility

TileLang is embedded in Python, but TileLang kernels are not executed by the
Python interpreter. Python syntax is parsed or rewritten into TileLang/TIR.
Only the supported subset has device semantics.

There are two frontend paths to keep in mind:

- `@T.prim_func` uses TileLang's TVM-script parser.
- `@tilelang.jit` eager-style kernels use a Python AST rewriter around
  `with T.Kernel(...)`.

Most day-to-day examples in `examples` use ordinary Python
on the host to build shapes/configs, then use TileLang loop and memory
constructs inside the kernel body.

```python
import tilelang
import tilelang.language as T
```

For kernel structure and memory scopes, start with `language_basics.md`. For
loop and branch semantics, see `control_flow.md`. This page focuses on frontend
compatibility rather than re-teaching the full kernel walkthrough.

## Rule of Thumb

Inside a kernel body, use Python syntax only when TileLang lowers it to TIR or
when the value is purely compile-time Python. If the value is data-dependent on
device execution, use TileLang operations.

For example:

```python
for k in range(T.ceildiv(K, BK)):
    ...
```

is a TileLang serial loop because the frontend maps `range` to `T.serial`.

```python
for i, j in T.Parallel(BM, BN):
    C[i, j] = 0
```

is a parallel TileLang loop. It is not Python iteration over a container.

## Control Flow And Loops

| Syntax | Status | Notes |
| --- | --- | --- |
| `for i in range(n)` | Supported | Lowered as `T.serial(n)`. |
| `for i in range(a, b)` | Supported | Lowered as `T.serial(a, b)`. |
| `for i in range(a, b, s)` | Supported | Constant positive and negative steps are supported in current tests. |
| `for i in T.serial(...)` | Supported | Explicit serial loop. |
| `for i, j in T.Parallel(...)` | Supported | Parallel loop over one or more dimensions. |
| `for k in T.Pipelined(...)` | Supported | Software-pipelined loop. |
| `for x in list` | Not kernel iteration | Use `range` plus indexing, or unroll with compile-time Python outside the kernel. |
| `enumerate(...)`, `zip(...)` | Not kernel iteration | No TileLang lowering was found; examples use these on the host side. |
| `while condition` | Supported | Use TileLang variables for device-side loop state. |
| `if` / `elif` / `else` | Supported | Symbolic conditions lower to TIR; compile-time booleans are resolved by Python/frontend. |
| `x if cond else y` | Supported in eager/JIT | Used in examples; the eager AST has explicit ternary lowering. Prefer `T.if_then_else` if parser behavior is uncertain. |
| `break`, `continue` | Eager/JIT only | The eager AST rewriter supports them. The local `@T.prim_func` parser does not show explicit handlers, so avoid them in `@T.prim_func` kernels. |

Examples in this repo mostly use:

```python
for i in range(loop_extent):
    ...

for i, j in T.Parallel(block_M, block_N):
    acc[i, j] += ...
```

## Data Access

| Syntax | Status | Notes |
| --- | --- | --- |
| `A[i]` | Supported | Scalar load/store. |
| `A[i, j, k]` | Supported | Multidimensional buffer indexing. |
| `A[i:j]` | Supported | Produces a buffer region for operations such as `T.copy`. |
| `A[i:j:step]` | Not generally supported | A regression test expects stepped buffer slices to fail. Use explicit loops. |
| `A[-1]` | Supported when provable | A legalization pass rewrites provably negative indices by adding the buffer extent. Unknown symbolic signs may be left unchanged. |
| `buffer.shape[d]` | Supported | Preferred way to read static or symbolic buffer extents. |

Use slices primarily as regions for TileLang operations:

```python
T.copy(A[by * BM, k * BK], A_shared)
T.copy(acc, C[by * BM, bx * BN])
```

For strided regions, write an explicit loop:

```python
for i in T.Parallel(N):
    B[i] = A[i * 2]
```

## Assignment And Expressions

| Syntax | Status | Notes |
| --- | --- | --- |
| `x = expr` | Supported | Creates/binds a TileLang expression or compile-time value depending on `expr`. |
| `x: T.int32 = expr` | Supported | Annotated local variable style is supported by the parser/frontend. |
| `A[i] = expr` | Supported | Lowers to buffer store. |
| `+=`, `-=`, `*=`, etc. | Supported | Works for local variables and buffer elements. |
| `a, b = b, a` | Supported | Tuple assignment/swap is tested. |
| `a = b = c` | Not supported | Consecutive assignment is rejected by the parser. |
| Arithmetic/comparison ops | Supported | `+`, `-`, `*`, `/`, `//`, `%`, comparisons, boolean `and`/`or` are lowered for TileLang expressions. |
| `is`, `in`, `not in` | Not device ops | Use explicit comparisons or TileLang boolean expressions. |

## Containers

Python containers can be useful at compile time, but they are not device-side
containers.

| Feature | Status | Notes |
| --- | --- | --- |
| Tuple/list unpacking | Supported for binding | Used for loop variables, `T.Kernel(...) as (bx, by)`, and swaps. |
| Python lists/tuples of constants | Compile-time only | Fine for shapes/configs/macros if fully known while building the kernel. |
| Dicts | Host-side only | No special kernel lowering was found. |
| List/dict/set comprehensions | Host-side only | No dedicated kernel lowering was found. |

If you need device storage, allocate TileLang buffers:

```python
tmp = T.alloc_fragment((BM, BN), "float32")
shared = T.alloc_shared((BM, BK), "float16")
```

## Functions, Classes, And Macros

Do not define ordinary Python functions or classes for device-side execution
inside a kernel. Use `@T.macro` for reusable TileLang code blocks.

```python
@T.macro
def clear_tile(tile):
    for i, j in T.Parallel(tile.shape[0], tile.shape[1]):
        tile[i, j] = 0

@T.prim_func
def kernel(C: T.Tensor((M, N), "float32")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (by, bx):
        acc = T.alloc_fragment((BM, BN), "float32")
        clear_tile(acc)
```

Macros are expanded at compile time. Current tests cover macros returning
constants, expressions, frames, and applying callable arguments. Returning from
inside macro control flow is more restricted; keep macros simple and
straight-line when possible.

## Builtins, Print, And Assert

| Feature | Status | Notes |
| --- | --- | --- |
| `assert cond, msg` | Supported with caveats | In `@T.prim_func`, parses to `T.Assert`; in eager/JIT, symbolic conditions become device asserts and false compile-time booleans raise Python `AssertionError`. |
| `T.device_assert(cond, msg)` | Supported | Explicit device-side assertion helper. |
| `T.print(...)` | Supported | Device-side print for expressions/buffers. |
| `print(...)` | Host-side only | Use for Python setup/debug output, not device printing. |
| `len(...)` | Compile-time only | Works on ordinary Python containers; there is no general device-side `len`. Use `buffer.shape[d]` for buffer extents. |
| `type(...)`, `isinstance(...)` | Host-side only | Useful while constructing kernels, not as device-side predicates. |

## With Statements

`with` is supported for TileLang/TVM frames such as:

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

and other TileLang frame-style constructs. Arbitrary Python context managers do
not have device semantics inside a kernel.

## Host Python vs Kernel Python

This is valid host Python:

```python
configs = [(64, 64), (128, 64)]
for block_m, block_n in configs:
    kernel = make_kernel(block_m, block_n)
```

Inside the kernel, write TileLang loops:

```python
for i, j in T.Parallel(block_m, block_n):
    out[i, j] = 0
```

This separation is the safest way to read existing examples: use Python freely
to choose shapes, dtypes, and launch parameters; use TileLang syntax for
device-side control flow, memory, and math.

## Implementation Pointers

- `@T.prim_func` uses the TileLang TVM-script parsing path.
- Eager `@tilelang.jit` kernels use a Python AST frontend.
- Loop constructors provide the lowering for serial, parallel, and pipelined loops.
- Macro expansion happens in the frontend before lowering.
- Device print and assert are explicit TileLang helpers.
- Negative indices are legalized when the frontend can prove the target extent.
