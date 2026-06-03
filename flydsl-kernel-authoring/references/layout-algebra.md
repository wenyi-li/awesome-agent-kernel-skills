# FlyDSL Layout Algebra Reference

The layout system is the core abstraction that separates FlyDSL from
Triton/Gluon. It gives you precise control over how data is arranged in
registers, shared memory, and global memory, and how threads map to data.

## Core types

| Type | Description | Example |
|------|-------------|---------|
| `!fly.int_tuple` | Integer tuple (can be nested) | `(8, 16)`, `(8, (4, 2))` |
| `!fly.layout` | (Shape, Stride) pair | `(8, 16):(1, 8)` (col-major) |
| `!fly.memref` | Memory reference with layout | Typed pointer + layout info |

A **Layout** maps a logical coordinate to a physical linear memory index.

**Formula**: `Index = sum(coord_i * stride_i)`

## Construction

```python
import flydsl.expr as fx

shape  = fx.make_shape(8, 16)            # IntTuple (8, 16)
stride = fx.make_stride(1, 8)            # IntTuple (1, 8)
layout = fx.make_layout(shape, stride)   # Layout (8,16):(1,8)

# Shorthand with Python tuples
layout = fx.make_layout((8, 16), (1, 8))

# Coordinates
coord = fx.make_coord(i, j)

# Identity layout
identity = fx.make_identity_layout((M, N))
```

## Mapping operations

```python
idx   = fx.crd2idx(coord, layout)    # coord -> linear index
coord = fx.idx2crd(idx, layout)      # linear index -> coord
fx.size(layout)                       # total element count
fx.get_shape(layout)
fx.get_stride(layout)
fx.rank(int_tuple)                    # number of top-level modes
```

**Example**. For `layout = (8, 16):(1, 8)` (column-major):
- `crd2idx((3, 5), layout)` = `3*1 + 5*8` = 43
- `idx2crd(43, layout)` = `(43 % 8, 43 / 8)` = `(3, 5)`

## Products (combine layouts)

```python
fx.logical_product(layout, tiler)   # basic mode-wise concatenation
fx.raked_product(thr, val)          # interleaved access (common for TiledCopy)
fx.block_product(layout, tiler)
fx.zipped_product(layout, tiler)
fx.tiled_product(layout, tiler)
fx.flat_product(layout, tiler)
```

## Divides (partition layouts)

```python
fx.logical_divide(layout, divisor)  # basic partition -> (tile, rest)
fx.zipped_divide(layout, divisor)
fx.tiled_divide(layout, divisor)
fx.flat_divide(layout, divisor)
```

## Copy atoms

Pre-built copy atoms for different widths:

| Atom | Bits | Usage |
|------|------|-------|
| `fx.UniversalCopy32b()` | 32 | 1 x f32 element |
| `fx.UniversalCopy(64)` | 64 | 2 x f32 elements |
| `fx.UniversalCopy(128)` | 128 | 4 x f32 elements |
| `fx.rocdl.BufferCopy128b()` | 128 | AMD buffer load 4 x f32 (fast path) |
| `fx.rocdl.BufferCopy32b()` | 32 | AMD buffer load 1 x f32 |

## The standard elementwise pattern (proven, always use)

```python
@flyc.kernel
def my_kernel(In: fx.Tensor, Out: fx.Tensor,
              BLOCK: fx.Constexpr[int], VEC: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile = BLOCK * VEC

    # Divide tensor by tile, select this block
    tIn  = fx.slice(fx.logical_divide(In,  fx.make_layout(tile, 1)), (None, bid))
    tOut = fx.slice(fx.logical_divide(Out, fx.make_layout(tile, 1)), (None, bid))

    # Per-thread sub-divide into VEC wide chunks
    tIn  = fx.logical_divide(tIn,  fx.make_layout(VEC, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC, 1))

    copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    MemTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(VEC, 1),
                              fx.AddressSpace.Register)
    rIn  = fx.memref_alloca(MemTy, fx.make_layout(VEC, 1))
    rOut = fx.memref_alloca(MemTy, fx.make_layout(VEC, 1))

    fx.copy_atom_call(copy, fx.slice(tIn, (None, tid)), rIn)
    v = fx.memref_load_vec(rIn)
    v = fx.arith.mulf(v, v)   # example compute
    fx.memref_store_vec(v, rOut)
    fx.copy_atom_call(copy, rOut, fx.slice(tOut, (None, tid)))
```

## Notes on AMD specifics

- **Wavefront size is 64** on all gfx9xx (not 32 like NVIDIA). All XOR-shuffle
  reductions use shifts `[32, 16, 8, 4, 2, 1]`.
- **LDS size per CU**: 64 KB on gfx942, 160 KB on gfx950.
- **LDS banks**: 32 banks (gfx942) or 64 banks (gfx950). Bank conflict rule
  and swizzle patterns differ between them — see
  [FlyDSL/.claude/skills/lds-optimization/SKILL.md](https://github.com/ROCm/FlyDSL) for
  detail.
- **`tile_k * elem_bytes` must be divisible by 64** for the K64-byte micro-step
  used in MFMA preshuffle paths.
