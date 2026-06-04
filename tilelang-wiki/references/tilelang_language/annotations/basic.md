# Annotations: Basic

Annotations attach scheduling, layout, aliasing, or compile metadata to the
current TileLang function or block. Put them close to the code they describe.

## Threadblock Swizzle

`T.use_swizzle(panel_size, order="row", enable=True)` annotates the kernel with
a 2-D rasterization swizzle:

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    T.use_swizzle(panel_size=10)
```

`order="row"` selects row rasterization. Any other order value currently
selects column rasterization. Passing `enable=False` returns `None`, which is
useful when an autotuned boolean controls the swizzle.

This is common in GEMM-like examples because the launch tile order affects L2
locality.

## Layout Maps

`T.annotate_layout(layout_map)` tells later passes how buffers are laid out:

```python
layout = T.Layout(A_shared.shape, lambda i, j: (i, j))
T.annotate_layout({A_shared: layout})
```

Values may be:

- a `T.Layout`;
- a `T.Fragment` layout for fragment buffers;
- a callable that TileLang wraps as a `T.Layout` over the buffer shape.

Invalid layout values raise immediately. For fragment buffers, the layout must
be a `T.Fragment`.

Use layout annotations for swizzled shared memory or explicit fragment layout
control. Keep the layout function local and easy to audit; incorrect layout
maps can silently make loads and stores disagree.

## Buffer Aliasing

`T.annotate_restrict_buffers(*buffers)` marks kernel buffer parameters as
non-restrict so generated code omits `__restrict__` for those parameters:

```python
T.annotate_restrict_buffers(A, B)
```

Use it when arguments may alias, such as overlapping views of the same
allocation. Passing no buffers is a no-op. Non-buffer arguments raise
`TypeError`.

## Launch Bounds

`T.annotate_min_blocks_per_sm(n)` sets the generated CUDA launch-bounds minimum
blocks per multiprocessor:

```python
T.annotate_min_blocks_per_sm(2)
```

`n` must be a positive Python integer. This can improve occupancy by limiting
register use, but may introduce spilling. Treat it as a performance knob and
validate the generated kernel.

