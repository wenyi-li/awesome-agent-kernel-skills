# Control Flow

TileLang kernels are written as Python, but their bodies are captured and
lowered to TIR. This guide documents the control-flow forms that appear in the
working examples and the implementation: Python conditionals, expression
selection, loop helpers, thread bindings, `while`, `break`, and `continue`.

The examples assume:

```python
import tilelang.language as T
```

Kernel structure, memory scopes, and the canonical minimal GEMM skeleton are
introduced in `language_basics.md`. This page focuses on control constructs and
guarding patterns.

## Conditionals

Use ordinary Python `if`/`elif`/`else` for statement-level branching. Conditions
can be compile-time Python values or TileLang/TIR expressions.

```python
if trans_A:
    T.copy(A[k * block_K, by * block_M], A_shared)
else:
    T.copy(A[by * block_M, k * block_K], A_shared)
```

Compile-time flags are common in JIT functions. TileLang specializes the kernel
for the values used at compile time, so branches such as `if trans_A:` or
`if window_size is not None:` are good ways to build variants.

Runtime predicates are also common:

```python
if bx * block_M < M and by * block_N < N:
    T.copy(C_local, C[by * block_M, bx * block_N])
```

For expression-level selection, use `T.if_then_else`:

```python
acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
```

`T.if_then_else` is preferred for value-producing runtime selection. It is also
the right tool for guarded loads because the untaken branch is not evaluated.

Python ternary expressions can be used when the condition is a compile-time
Python value:

```python
start = T.max(0, (bx * block_M - window_size) // block_N) if window_size is not None else 0
```

For runtime value selection inside buffers or scalar expressions, prefer
`T.if_then_else`.

## Predicate Helpers

Python `and` and `or` are used in many examples:

```python
if i_s <= i_t and i_s >= 0:
    ...
```

For boolean buffers, TileLang also exposes reduction helpers:

```python
if T.any_of(mask):
    ...

if T.all_of(mask_region):
    ...
```

`T.any_of` and `T.all_of` take a boolean buffer or buffer region. They are not
called as `T.all_of(a < b, c < d)`.

## Serial Loops

`T.serial` is the default explicit for-loop. It supports one-argument and
start/stop/step forms:

```python
for k in T.serial(K):
    ...

for k in T.serial(0, K, 2):
    ...
```

Use it for ordered scalar work, simple reductions, explicit load loops, or
control-heavy code that should not be mapped as a TileLang parallel loop.

In eager JIT code, Python `range(...)` is supported and maps to a serial
TileLang loop. Use it for ordinary Python-looking loops when no TileLang-specific
loop annotation is needed.

## Grid Loops

`T.grid(...)` creates a compact Cartesian product loop nest:

```python
for i, j, k in T.grid(M, N, K):
    C[i, j] += A[i, k] * B[k, j]
```

It is useful for CPU-style scalar kernels and simple nested iteration. GPU tile
kernels more often use `T.Parallel` for per-tile element work.

## Unrolled And Vectorized Loops

`T.unroll` requests unrolling:

```python
for k in T.unroll(4):
    acc += a[k] * b[k]
```

`T.vectorized` marks a loop for vectorized memory access or computation:

```python
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
```

Use these for small static loops or explicit data movement patterns where the
lowering intent matters.

## Parallel Loops

`T.Parallel(ext0, ext1, ...)` creates a nested parallel loop. The loop header
returns one index per extent:

```python
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```

This is the normal form for per-element tile work. The compiler maps the
parallel loop nest onto available threads and inferred layouts. Optional hints
include:

```python
for i, j in T.Parallel(block_M, block_N, coalesced_width=4):
    ...

for i, j in T.Parallel(block_M, block_N, loop_layout=layout):
    ...
```

Only use explicit layouts once the default layout inference is not enough.

Keep `T.Parallel` nests structurally simple. Continuous nested parallel loops
can be fused by the checker, but non-continuous nested parallel loops are
rejected. Tile operations inside `T.Parallel` and `T.Pipelined` inside
`T.Parallel` are also rejected; put tile-level operations outside the parallel
element loop.

## Pipelined Loops

`T.Pipelined` describes a loop whose body has producer and consumer stages. It
is the common loop around tiled GEMM and attention:

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

The one-argument form starts at zero. The two-argument form is also supported:

```python
for k in T.Pipelined(loop_start, loop_end, num_stages=2):
    ...
```

For most kernels, set `num_stages` and let TileLang infer the software
pipeline. Use manual `order=...` and `stage=...` annotations only when the
automatic pipeline is not sufficient. `software_pipeline.md` owns those
semantics and restrictions.

## Thread Bindings

`T.Kernel(..., threads=...)` creates thread dimensions. Most tile code can ignore
the raw thread indices, but low-level kernels can read them:

```python
with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
    tn = T.get_thread_binding(0)
    tk = T.get_thread_binding(1)
    ...
```

Use thread bindings when each thread owns a specific scalar/vector lane, when
writing atomics, or when implementing kernels below the tile-operator level.

## While, Break, And Continue

The eager frontend supports `while`, `break`, and `continue` inside captured
kernels. The older `@T.prim_func` parser supports `while`; use `break` and
`continue` primarily in eager `@tilelang.jit` code. Keep these constructs simple
and bounded.

```python
i = T.alloc_var("int32", init=0)
while i[0] < limit:
    if done:
        break
    i[0] += 1
```

The eager builder detects obvious constant-true Python `while` loops and warns
when statements after `break` or `continue` cannot run. Prefer `T.serial` or
`T.Pipelined` when the trip count is known; use `while` for genuinely
data-dependent loops.

`continue` is valid in captured Python loops and host-side helper loops. Inside
performance-critical kernel loops, a predicate plus `T.if_then_else` is often
easier for the compiler to optimize.

## Boundary Handling

Tile kernels often operate on full tiles even when the problem size is not a
multiple of the tile size. You can write explicit guards:

```python
for i, j in T.Parallel(block_M, block_N):
    row = by * block_M + i
    col = bx * block_N + j
    if row < M and col < N:
        C[row, col] = A[row, col] + B[row, col]
```

Many examples rely on TileLang copy and memory legalization passes for simple
global-memory tile-edge safety. Do not treat that as a replacement for all
guards: shared/local accesses are not generally made safe the same way. Write
explicit guards when the edge path has custom behavior, when a value should be
masked before compute, when shared/local indexing could go out of range, or when
clarity matters.

For value masking, use `T.if_then_else`:

```python
S[i, j] = T.if_then_else(col < valid_cols, S[i, j], -T.infinity(S.dtype))
```

## Practical Pattern

This is a compact control-flow pattern that highlights branching and masking
without re-teaching the full GEMM lifecycle:

```python
for i, j in T.Parallel(block_M, block_N):
    row = by * block_M + i
    col = bx * block_N + j
    in_bounds = row < M and col < N
    value = T.if_then_else(in_bounds, A[row, col], 0)
    if in_bounds:
        C[row, col] = value
```

Use compile-time Python branches to select kernel variants, TileLang loops to
describe lowering intent, and `T.if_then_else` for runtime value selection.

For the surrounding kernel shape, see `language_basics.md`. For how pipelined
loops are lowered, see `software_pipeline.md`.
