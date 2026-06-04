# Basic Operations: Basic

This page covers the construction and fill operations used by most kernels.
Allocation primitives such as `T.alloc_shared`, `T.alloc_fragment`, and
`T.alloc_local` create per-kernel storage; the tensor proxy APIs below describe
kernel arguments and explicit views.

## Tensor Arguments

Use `T.Tensor` for contiguous tensor arguments:

```python
A: T.Tensor((M, K), T.float16)
B: T.Tensor((K, N), "float16")
C: T.Tensor((M, N), T.float32)
```

`T.Tensor(shape, dtype)` creates a global-scope buffer annotation with
contiguous row-major strides. A scalar integer or symbolic shape is treated as a
one-dimensional shape.

For non-contiguous global-memory arguments, use `T.StridedTensor`:

```python
A: T.StridedTensor((M, K), (stride_m, stride_k), T.float16)
```

The rank of `shape` and `strides` must match.

## Fill And Clear

```python
T.fill(buffer, value)
T.clear(buffer)
```

`T.fill` writes a value over a buffer, buffer region, or recoverable buffer-load
region. `T.clear(buffer)` is the zero-fill form and is the common way to
initialize accumulators:

```python
C_local = T.alloc_fragment((block_M, block_N), T.float32)
T.clear(C_local)
```

Use `T.fill` when the initial value is not zero. Flash-attention style kernels
often initialize softmax state this way:

```python
T.fill(acc_o, 0)
T.fill(logsum, 0)
T.fill(scores_max, -T.infinity(accum_dtype))
```

## Elementwise Tiles

For ordinary elementwise work, combine local buffers with `T.Parallel`:

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```

For scalar math in those loops, use the usual public math operators and
functions such as `+`, `*`, `T.max`, `T.min`, `T.exp2`, `T.sqrt`, and
`T.clamp`. Reserve the lower-level intrinsic names for cases where you need the
specific lowering behavior described in the advanced page.
