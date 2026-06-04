# Loop Basics

This page covers the loop forms most TileLang kernels use first:
`T.Parallel`, `T.Pipelined`, and `T.serial`.

## `T.Parallel`

Use `T.Parallel(...)` for elementwise work over a tile.

```python
for i, j in T.Parallel(block_M, block_N):
    C_frag[i, j] = A_shared[i, j] + B_shared[i, j]
```

`T.Parallel(block_M, block_N)` creates a parallel loop nest and returns one
loop variable per extent. One-dimensional forms are common for row-wise state:

```python
for i in T.Parallel(block_M):
    scores_sum[i] = scores_sum[i] * scores_scale[i]
```

Keep `T.Parallel` bodies scalar or elementwise. Tile-level operations such as
`T.gemm`, `T.copy`, and software-pipelined copy/compute loops normally sit
outside the element loop.

## `T.Pipelined`

Use `T.Pipelined(...)` for repeated copy/compute stages, especially tiled GEMM,
attention, convolution, and dynamic-shape GEMM loops.

```python
T.clear(C_frag)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_frag)
```

The one-argument form starts at zero and stops at the value passed:

```python
for k in T.Pipelined(num_tiles, num_stages=2):
    ...
```

The two-argument form uses an explicit start and stop:

```python
for k in T.Pipelined(start_tile, stop_tile, num_stages=2):
    ...
```

For normal kernels, start with `num_stages` only. `num_stages=0` is still a
valid loop, but it does not request compiler-inferred software pipelining.

## `T.serial` And `T.Serial`

Use `T.serial(...)` when a loop should remain serial in the generated TIR.
`T.Serial` is an alias.

```python
for i in T.serial(N):
    acc[0] += x[i]

for i in T.serial(0, N, 2):
    tmp[i // 2] = x[i]
```

The one-argument form starts at zero. The three-argument form supports an
explicit step. In eager JIT code, ordinary Python `range(...)` is also accepted
for serial loops, but `T.serial(...)` makes the intended lowering explicit.
