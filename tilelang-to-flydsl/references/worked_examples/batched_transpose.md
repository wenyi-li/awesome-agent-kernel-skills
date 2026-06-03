# Worked example: batched_transpose

A medium-complexity kernel from
`tile_kernels/transpose/batched_transpose_kernel.py`. Demonstrates:

- Multi-axis grid (`shape_y // bly`, `shape_x // blx`, `num_batches`).
- Strided input (`T.StridedTensor`).
- Shared memory with bank-conflict padding (`block_x + block_k`).
- A `T.Fragment(forward_fn=...)` driving a `T.Parallel` write-out.
- A swizzled inner loop on the LDS write side.

The point of this example is to show **how to read a `T.Fragment.forward_fn`
and turn it into a FlyDSL `tiled_copy`**.

---

## Original (TileLang, abridged)

```python
def create_loop_layout_fn(block_x: int, num_threads: int = 256):
    def loop_layout_fn(i, j):
        elems = i * block_x + j
        forward_thread = (elems // 4) % num_threads
        forward_local  = elems % 4 + elems // (num_threads * 4) * 4
        return forward_thread, forward_local
    return loop_layout_fn


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_batched_transpose_kernel(shape_x_mod_128: int, shape_y_mod_128: int, dtype: T.dtype):
    num_batches = T.dynamic('num_batches')
    shape_x = T.dynamic('shape_x')
    shape_y = T.dynamic('shape_y')
    stride_x = T.dynamic('stride_x')

    num_threads = 256
    block_x = 128 if shape_x_mod_128 == 0 else 64
    block_y = 128 if shape_y_mod_128 == 0 else 64
    block_k = 4
    num_threads_per_row = block_y // block_k

    loop_layout = T.Fragment((block_y, block_x), forward_fn=create_loop_layout_fn(block_x, num_threads))

    @T.prim_func
    def batched_transpose_kernel(
        x:   T.StridedTensor[(num_batches, shape_x, shape_y), (shape_x * stride_x, stride_x, 1), dtype],
        out: T.Tensor[(num_batches, shape_y, shape_x), dtype],
    ):
        with T.Kernel(shape_y // block_y, shape_x // block_x, num_batches, threads=num_threads) as (pid_y, pid_x, pid_batch):
            out_shared = T.alloc_shared((block_y, block_x + block_k), dtype)
            tid = T.get_thread_binding()
            row, col = tid // num_threads_per_row, tid % num_threads_per_row

            tmp     = T.alloc_local((block_k, block_k), dtype)
            tmp_row = T.alloc_local((block_k,), dtype)
            for i_ in T.unroll(block_x // block_k // (num_threads // num_threads_per_row)):
                i = i_ * (num_threads // num_threads_per_row) + row
                for j in T.unroll(block_k):
                    for k in T.vectorized(block_k):
                        tmp_row[k] = x[pid_batch, pid_x*block_x + i*block_k + j, pid_y*block_y + col*block_k + k]
                    for k in T.unroll(block_k):
                        tmp[k, j] = tmp_row[k]

                for j in T.unroll(block_k):
                    swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k
                    for k in T.vectorized(block_k):
                        out_shared[col*block_k + swizzle_j, i*block_k + k] = tmp[swizzle_j, k]

            T.sync_threads()
            for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
                out[pid_batch, pid_y*block_y + i, pid_x*block_x + j] = out_shared[i, j]

    return batched_transpose_kernel
```

## Decoding `loop_layout`

The `forward_fn(i, j)` says: given a logical `(i, j)` coordinate, the
**linear element id** is `elems = i * block_x + j` (row-major over the LDS
shape `(block_y, block_x)`); the **owning thread** is `elems // 4 mod 256`
and the **local index** in that thread is `elems % 4 + (elems // 1024) * 4`.

Read it as a sentence: "Pack 4 consecutive elements per thread, walk
threads contiguously across `j`, then `i`. After 1024 elements (256 threads
× 4) wrap into the next chunk of 4 per thread." That is a standard
contiguous-along-J value layout of length 4, with thread layout
`(num_threads,)` strided by 4. In FlyDSL:

```python
thr_layout = fx.make_layout(num_threads, 4)         # 256 threads, stride 4
val_layout = fx.make_layout(4, 1)                   # 4 contiguous values
copy_atom  = fx.make_copy_atom(fx.rocdl.BufferCopy(elem_b * 4)(), elem_dt)
tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy  = fx.make_tiled_copy(copy_atom, tv, tile_mn)
```

This is the only non-mechanical step. Once `tiled_copy` is built, the LDS →
global write-out is a stock `partition_S/D + fx.copy(...)`.

## Step-1 spec table (abridged)

| Item | Value |
|---|---|
| Symbolic dims | `num_batches`, `shape_x`, `shape_y`, `stride_x` |
| Constants | `num_threads=256`, `block_x ∈ {64,128}`, `block_y ∈ {64,128}`, `block_k=4` |
| Alloc | `out_shared (block_y, block_x+4) dtype`; `tmp (4,4) dtype`; `tmp_row (4,) dtype` |
| Loops | unroll over `block_x // 16 // (256//(block_y//4))` = small constant |
| Strides | input is strided along axis 1 (stride_x); axis 2 contiguous |
| Test | `tests/transpose/test_transpose.py` calls `tile_kernels.transpose.batched_transpose(x)` for various dtypes (`bf16`, `fp8_e4m3fn`, `fp32`) and shapes |

## Conversion design choices

- LDS allocator: one `allocate_array` of size
  `block_y * (block_x + block_k)` with `elem_ty`. Pad preserved.
- Register memrefs: `tmp_row` as `(block_k, 1)` contiguous; `tmp` as
  `(block_k, block_k)` row-major (use `make_layout((block_k, block_k), (block_k, 1))`).
  `tmp` is reused per outer iteration — allocate once outside the
  range_constexpr.
- Strided input: in the Python wrapper, do
  `x_dyn = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=4)`
  before passing to the launcher (the test asserts `stride_x % 4 == 0`).
- Swizzle: `swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k` is
  computed at trace time as Python arithmetic on `tid` (which is an `Int32`
  value) — the result is a `range_constexpr`-compatible expression for the
  `j` outer but uses runtime `tid`. Keep it inside an `scf.for`-friendly
  scope or pre-compute `tid // (8 // elem_bytes)` once.

## Ported (FlyDSL skeleton)

This is a *skeleton* — the inner read with a strided source needs the
`buffer_load` pattern with explicit byte offsets instead of a tiled copy
(because the input has a runtime stride that BufferCopy atoms cannot bake
in). Adapt as needed.

```python
import os, math, torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.ir import InsertionPoint
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr, const_expr
from flydsl.expr import buffer_ops
from flydsl.expr.typing import T as flyT
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch


def get_batched_transpose_kernel(shape_x_mod_128: int, shape_y_mod_128: int,
                                 dtype_str: str):
    arch = get_rocm_arch()
    NUM_THREADS = 256
    BLOCK_X = 128 if shape_x_mod_128 == 0 else 64
    BLOCK_Y = 128 if shape_y_mod_128 == 0 else 64
    BLOCK_K = 4
    THR_PER_ROW = BLOCK_Y // BLOCK_K
    ROWS_PER_PASS = NUM_THREADS // THR_PER_ROW
    OUTER = BLOCK_X // BLOCK_K // ROWS_PER_PASS

    if dtype_str == "bf16":
        elem_dt = fx.BFloat16
    elif dtype_str == "fp8_e4m3":
        elem_dt = fx.Float8E4M3FNUZ if arch.startswith("gfx94") else fx.Float8E4M3FN
    elif dtype_str == "fp32":
        elem_dt = fx.Float32
    else:
        raise ValueError(dtype_str)
    elem_ty = elem_dt.ir_type
    elem_bytes = elem_dt.width // 8

    # 1. LDS allocator — one buffer of (BLOCK_Y, BLOCK_X + BLOCK_K).
    allocator = SmemAllocator(None, arch=arch, global_sym_name="smem_btr")
    LDS_PAD = BLOCK_X + BLOCK_K
    lds_handle = allocator.allocate_array(elem_ty, BLOCK_Y * LDS_PAD)

    @flyc.kernel
    def kernel(
        x:   fx.Tensor,
        out: fx.Tensor,
        num_batches: fx.Int32,
        shape_x:     fx.Int32,
        shape_y:     fx.Int32,
        stride_x:    fx.Int32,
    ):
        bid_y = fx.block_idx.x
        bid_x = fx.block_idx.y
        bid_b = fx.block_idx.z
        tid   = fx.thread_idx.x

        row = tid // THR_PER_ROW
        col = tid %  THR_PER_ROW

        # 2. LDS view (BLOCK_Y, LDS_PAD), row-major.
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, 0, elem_ty, shape=(BLOCK_Y, LDS_PAD))

        # 3. Build a buffer-resource for the strided input. We use raw
        #    buffer_ops because the leading dim has a runtime stride.
        x_rsrc = buffer_ops.create_buffer_resource(x)        # base addr + size
        out_buf = fx.rocdl.make_buffer_tensor(out)

        # 4. Per-thread register memref of length BLOCK_K (one row of `tmp_row`).
        reg_ty  = fx.MemRefType.get(elem_ty,
                                    fx.LayoutType.get(BLOCK_K, 1),
                                    fx.AddressSpace.Register)
        reg_lay = fx.make_layout(BLOCK_K, 1)

        # 5. Read + transpose loop.
        for i_ in range_constexpr(OUTER):
            i = i_ * ROWS_PER_PASS + row

            # Per-row j loop: load BLOCK_K elems along the contiguous axis,
            # store column-major into `tmp` (held implicitly via LDS).
            for j in range_constexpr(BLOCK_K):
                # Compute byte offset for x[bid_b, bid_x*BLOCK_X + i*BLOCK_K + j,
                #                          bid_y*BLOCK_Y + col*BLOCK_K + 0].
                row_idx_x = bid_x * BLOCK_X + i * BLOCK_K + j
                col_idx_y = bid_y * BLOCK_Y + col * BLOCK_K
                # Strided index in elements:
                #   bid_b * shape_x * stride_x + row_idx_x * stride_x + col_idx_y
                elem_off  = (bid_b * shape_x * stride_x
                             + row_idx_x * stride_x
                             + col_idx_y)
                byte_off  = elem_off * elem_bytes

                # buffer_load returns a vector of vec_width lanes.
                vec = buffer_ops.buffer_load(x_rsrc, byte_off, vec_width=BLOCK_K)

                # Compute swizzle along j to dodge LDS bank conflicts.
                swizzle_j = (j + tid // (8 // elem_bytes)) % BLOCK_K

                # Store vec[k] -> lds[col*BLOCK_K + swizzle_j, i*BLOCK_K + k].
                for k in range_constexpr(BLOCK_K):
                    SmemPtr.store(lds, vec[k],
                                  [col * BLOCK_K + swizzle_j,
                                   i   * BLOCK_K + k])

        gpu.barrier()

        # 6. Write LDS back to `out` using a tiled copy whose layout matches
        #    the original `loop_layout`.
        out_slab = fx.slice(out_buf, (bid_b, None, None))    # (shape_y, shape_x)
        out_blk  = fx.zipped_divide(out_slab, (BLOCK_Y, BLOCK_X))
        out_blk  = fx.slice(out_blk, (None, (bid_y, bid_x)))

        # thr_layout: 256 threads, stride 4; val_layout: 4 contiguous.
        thr_layout = fx.make_layout(NUM_THREADS, BLOCK_K)
        val_layout = fx.make_layout(BLOCK_K, 1)
        copy_op    = fx.rocdl.BufferCopy(BLOCK_K * elem_bytes * 8)()  # 4*elem_bytes bytes
        copy_atom  = fx.make_copy_atom(copy_op, elem_dt)
        tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
        tiled_copy  = fx.make_tiled_copy(copy_atom, tv, tile_mn)
        thr_copy    = tiled_copy.get_slice(tid)

        # Wrap LDS as a 2-D memref view for partitioning.
        lds_view = lds.as_memref(fx.make_layout((BLOCK_Y, BLOCK_X), (LDS_PAD, 1)))
        src_part = thr_copy.partition_S(lds_view)
        dst_part = thr_copy.partition_D(out_blk)
        fx.copy(copy_atom, src_part, dst_part)

    @flyc.jit
    def launch(x: fx.Tensor, out: fx.Tensor,
               num_batches: fx.Int32, shape_x: fx.Int32,
               shape_y: fx.Int32, stride_x: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        # Finalise LDS allocator inside the gpu module body.
        ctx = CompilationContext.get_current()
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        kernel(x, out, num_batches, shape_x, shape_y, stride_x).launch(
            grid=(shape_y // BLOCK_Y, shape_x // BLOCK_X, num_batches),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Wrapper kept identical to TileKernels."""
    x = x.unsqueeze(0)
    out = batched_transpose(x)
    return out.squeeze(0)


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3
    num_batches, shape_x, shape_y = x.shape
    assert shape_x % 64 == 0 and shape_y % 64 == 0
    assert x.stride(-2) % 4 == 0 and x.stride(-1) == 1

    if x.dtype == torch.bfloat16:
        dtype_str = "bf16"
    elif x.dtype == torch.float8_e4m3fn:
        dtype_str = "fp8_e4m3"
    elif x.dtype == torch.float32:
        dtype_str = "fp32"
    else:
        raise ValueError(x.dtype)

    kernel = get_batched_transpose_kernel(shape_x % 128, shape_y % 128, dtype_str)

    out = torch.empty((num_batches, shape_y, shape_x), dtype=x.dtype, device='cuda')
    if num_batches > 0 and shape_x > 0 and shape_y > 0:
        # Mark the input's leading-dim stride as runtime-dynamic with the
        # divisibility guaranteed by the asserts.
        x_dyn = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=4)
        kernel(x_dyn, out, num_batches, shape_x, shape_y, x.stride(-2))

    return out
```

## Audit notes

- **A** Launch shape: TileLang `(pid_y, pid_x, pid_batch)` ↔ FlyDSL
  `(block_idx.x, block_idx.y, block_idx.z)`. ✓ The order is preserved.
- **B** Dtype: bf16 / fp8 / fp32. The arch-conditional FNUZ choice is
  applied for fp8. ✓
- **C** LDS allocator: `BLOCK_Y * LDS_PAD` (with the `+BLOCK_K` pad).
  `finalize()` called in the `@flyc.jit` body. ✓
- **D** `T.unroll` ↔ `range_constexpr`; the inner `T.vectorized(BLOCK_K)`
  read is mapped to a single `buffer_load(..., vec_width=BLOCK_K)` plus a
  `range_constexpr(BLOCK_K)` LDS-store loop. ✓
- **E** `buffer_ops.create_buffer_resource(x)` instead of
  `make_buffer_tensor(x)` because the input is strided — the layout API
  cannot encode a runtime stride. The output is contiguous so it gets
  `make_buffer_tensor`. ✓
- **F** No reductions in this kernel. N/A.
- **H** `mark_layout_dynamic(leading_dim=1, divisibility=4)` set at the
  call site to match `T.StridedTensor` semantics. ✓

## Items I cannot verify without GPU

- The exact byte offset arithmetic in `buffer_ops.buffer_load` against the
  3-D strided input — needs a small running test with a known stride to
  confirm.
- The decoded `thr_layout`/`val_layout` mapping vs the original
  `loop_layout`. Specifically, whether `fx.make_layout(NUM_THREADS,
  BLOCK_K)` (i.e. `(256,):(4,)`) plus `val_layout = (4,):(1,)` reproduces
  the *same* 1-D-into-2-D coordinate that
  `forward_fn(i, j) = ((elems // 4) % 256, ...)` produces. If the test
  fails on a corner pattern, this is the first thing to check by reading
  the FlyDSL `LayoutAlgebra/*.mlir` lit tests for `make_layout_tv`.
- The swizzle constant `(j + tid // (8 // elem_bytes)) % BLOCK_K`. For fp8
  it becomes `(j + tid // 8) % 4`. Verify the bank count assumption (32
  banks of 4 bytes each on AMD CDNA — same as NVIDIA).
- That `make_layout` accepts the `LDS_PAD` stride for the LDS view; if it
  rejects strided LDS layouts, fall back to a direct 1-D `SmemPtr` index
  arithmetic in the partition.
