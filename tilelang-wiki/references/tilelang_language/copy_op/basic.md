# Copy Basics

`T.copy` is the default data movement primitive in TileLang kernels. Use it for
ordinary movement among global, shared, fragment, and local buffers.

## `T.copy`

```python
T.copy(A[by * block_M, k * block_K], A_shared)
T.copy(B[k * block_K, bx * block_N], B_shared)
T.copy(C_frag, C[by * block_M, bx * block_N])
```

The first two statements load global tiles into shared memory. The last
statement stores a computed fragment tile to the output tensor.

`T.copy` accepts buffers, buffer regions, and buffer loads. Full buffers use
their whole shape. Sliced or indexed regions use the region extents. This lets
common tiled code pass the head of a tile instead of spelling every extent:

```python
T.copy(A[row_start, k_start], A_shared)
T.copy(C_frag, C[row_start, col_start])
```

Treat this as tile-head convenience, not general broadcasting. If both sides
are full buffers, their shapes must match structurally. If both sides are
scalar buffer loads with no region extent, TileLang lowers the operation to a
direct scalar store.

## Common Patterns

GEMM-like kernels usually copy inputs into shared memory inside a pipelined
loop, compute into a fragment, then copy the fragment out:

```python
T.clear(C_frag)
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_frag)

T.copy(C_frag, C[by * block_M, bx * block_N])
```

Elementwise kernels can also use `T.copy` to move tiles through shared or
fragment storage before and after a `T.Parallel` loop:

```python
T.copy(A[by * block_M, bx * block_N], A_shared)
for i, j in T.Parallel(block_M, block_N):
    C_frag[i, j] = A_shared[i, j] + 1
T.copy(C_frag, C[by * block_M, bx * block_N])
```

## Convolution Tiles With `T.im2col`

Convolution kernels often use `T.im2col` to form the activation tile consumed
by GEMM:

```python
T.im2col(data, data_shared, by, k_iter, KH, stride, dilation, pad)
T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
T.gemm(data_shared, kernel_shared, out_frag)
```

`T.im2col(img, col, nhw_step, c_step, kernel, stride, dilation, pad,
eviction_policy=None)` emits an image-to-column tile operation. The optional
eviction policy accepts the same policy names as `T.copy`.
