# Side-by-side idioms

Each section shows the **TileLang shape** of the pattern and the **FlyDSL
shape** that produces equivalent semantics. Use these as templates when
sketching the FlyDSL skeleton in workflow step 3.

---

## 1. Vectorised elementwise / cast

This is the bread and butter of TileKernels: load a tile, apply per-element
math, store. TileLang expresses it with `T.vectorized` and direct buffer
indexing; FlyDSL expresses it with a tiled buffer copy + a register memref +
`Vector` arithmetic.

### TileLang

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_elem_kernel(hidden: int):
    num_tokens = T.dynamic('num_tokens')
    NUM_THREADS = 256
    TILE_M, TILE_K = 128, 128
    VEC_K = TILE_K // 32                        # 4 elems per thread
    VEC_M = TILE_M * 32 // NUM_THREADS          # rows per thread

    @T.prim_func
    def k(x:   T.Tensor[(num_tokens, hidden), T.bfloat16],
          out: T.Tensor[(num_tokens, hidden), T.float8_e4m3]):
        with T.Kernel(T.ceildiv(num_tokens, TILE_M), T.ceildiv(hidden, TILE_K),
                      threads=NUM_THREADS) as (pid_m, pid_k):
            in_local  = T.alloc_local((VEC_K,), T.bfloat16)
            out_local = T.alloc_local((VEC_K,), T.float8_e4m3)
            tid = T.get_thread_binding()
            m_id, k_id = tid // 32, tid % 32

            for i in T.serial(VEC_M):
                row = pid_m * TILE_M + m_id * VEC_M + i
                col0 = pid_k * TILE_K + k_id * VEC_K
                for j in T.vectorized(VEC_K):
                    in_local[j] = x[row, col0 + j]
                for j in T.vectorized(VEC_K):
                    out_local[j] = T.cast(in_local[j] * 2.0, T.float8_e4m3)
                for j in T.vectorized(VEC_K):
                    out[row, col0 + j] = out_local[j]
    return k
```

### FlyDSL

```python
def build_elem_kernel(*, hidden: int, in_dtype_str="bf16", out_dtype_str="fp8_e4m3"):
    arch = get_rocm_arch()
    NUM_THREADS = 256
    TILE_M, TILE_K = 128, 128
    VEC_K = TILE_K // 32                        # bytes per copy: 4 * 2B = 8B  -> BufferCopy64b
    VEC_M = TILE_M * 32 // NUM_THREADS

    in_dtype  = fx.BFloat16   if in_dtype_str  == "bf16"      else fx.Float16
    out_dtype = fx.Float8E4M3FNUZ if out_dtype_str == "fp8_e4m3" else fx.Float8E5M2

    @flyc.kernel
    def k(x: fx.Tensor, out: fx.Tensor, num_tokens: fx.Int32):
        bid_m = fx.block_idx.x
        bid_k = fx.block_idx.y
        tid   = fx.thread_idx.x

        # Buffer-backed views so we can use BufferCopy atoms.
        x_buf   = fx.rocdl.make_buffer_tensor(x)
        out_buf = fx.rocdl.make_buffer_tensor(out)

        # Decompose tid -> (m_id, k_id). 32 threads per row, 8 rows per block.
        m_id = tid // 32
        k_id = tid % 32

        # Per-thread register memref of length VEC_K.
        reg_ty  = fx.MemRefType.get(in_dtype.ir_type,
                                    fx.LayoutType.get(VEC_K, 1),
                                    fx.AddressSpace.Register)
        out_reg_ty = fx.MemRefType.get(out_dtype.ir_type,
                                       fx.LayoutType.get(VEC_K, 1),
                                       fx.AddressSpace.Register)
        reg_layout = fx.make_layout(VEC_K, 1)

        # 64-bit BufferCopy = 8 bytes per copy = 4 bf16 elements. Pick the atom
        # whose width matches `VEC_K * elem_bits / 8`.
        copy_atom_in  = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), in_dtype)
        copy_atom_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), out_dtype)

        for i in range_constexpr(VEC_M):
            row  = bid_m * TILE_M + m_id * VEC_M + i
            col0 = bid_k * TILE_K + k_id * VEC_K

            # Per-row global view: row r, all cols. Then divide along K by VEC_K
            # so each tid grabs one VEC_K-wide chunk.
            row_x   = fx.slice(x_buf,   (row, None))
            row_out = fx.slice(out_buf, (row, None))
            x_div   = fx.logical_divide(row_x,   fx.make_layout(VEC_K, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_K, 1))

            r_in  = fx.memref_alloca(reg_ty,     reg_layout)
            r_out = fx.memref_alloca(out_reg_ty, reg_layout)

            # Load: BufferCopy from x[row, col0:col0+VEC_K] -> r_in.
            fx.copy_atom_call(copy_atom_in, fx.slice(x_div, (None, col0 // VEC_K)), r_in)

            # Compute on the whole vector at once.
            v_in  = fx.memref_load_vec(r_in)
            v_f32 = v_in.to(fx.Float32) * fx.Float32(2.0)
            v_out = v_f32.to(out_dtype)
            fx.memref_store_vec(v_out, r_out)

            # Store back.
            fx.copy_atom_call(copy_atom_out, r_out, fx.slice(out_div, (None, col0 // VEC_K)))

    @flyc.jit
    def launch(x: fx.Tensor, out: fx.Tensor, num_tokens: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        gx = (num_tokens + TILE_M - 1) // TILE_M
        gy = (hidden + TILE_K - 1) // TILE_K
        k(x, out, num_tokens).launch(grid=(gx, gy, 1),
                                     block=(NUM_THREADS, 1, 1),
                                     stream=stream)

    return launch
```

**Notes**

- `T.vectorized(VEC_K)` does not appear in the FlyDSL output. The
  vectorisation is instead expressed by choosing `BufferCopy64b()` (one 8-byte
  copy = 4 bf16 elements) and a register layout of `(VEC_K, 1)`.
- The inner `T.serial(VEC_M)` becomes a `range_constexpr` because `VEC_M` is a
  compile-time int.
- All math runs on a `Vector` value (`v_in`, `v_f32`, `v_out`) — the casts and
  the `* 2.0` are vector ops, not per-lane scalars.
- The `T.cast(x, T.float8_e4m3)` becomes `v.to(out_dtype)` on the `Vector`.

---

## 2. Per-thread distributed loop (`T.Parallel`)

`T.Parallel` is the compiler's signal: "distribute these iterations across the
workgroup". FlyDSL has no such construct — you must distribute by hand or by
constructing a tiled copy whose thread layout matches.

### TileLang

```python
for i in T.Parallel(num_aligned_experts):
    if i < num_experts:
        scores_fragment[i] = scores[pid, i]
    else:
        scores_fragment[i] = -T.infinity(T.float32)
```

### FlyDSL — option A: explicit thread distribution

Works when the per-iteration body is small and parallel-friendly.

```python
ITEMS_PER_THREAD = (num_aligned_experts + NUM_THREADS - 1) // NUM_THREADS
for j in range_constexpr(ITEMS_PER_THREAD):
    i = tid + j * NUM_THREADS
    in_range = i < num_aligned_experts
    in_experts = i < num_experts
    val = in_experts.select(scores[bid, i], fx.Float32(float("-inf")))
    if in_range:                          # plain Python `if` — lowers to scf.if
        scores_fragment_lds.store(val, [i])
```

### FlyDSL — option B: tiled copy with thread layout

Preferred when the iteration shape and access pattern can be described as a
layout. For `T.Parallel(M, N, loop_layout=fragment)`, build the copy from the
`fragment.forward_fn`:

```python
# loop_layout from TileLang said: thread_id = (i*block_x + j) // 4 % NUM_THREADS
#                                  local_id  = (i*block_x + j) % 4
#                                              + (i*block_x + j) // (NUM_THREADS*4) * 4
# → thr_layout has shape (NUM_THREADS,) walking *J-then-I* in steps of 4
# → val_layout has shape (4,) per thread, contiguous along the same walk
thr_layout = fx.make_layout((NUM_THREADS,), (4,))
val_layout = fx.make_layout((4,), (1,))
copy_atom  = fx.make_copy_atom(fx.UniversalCopy(VEC_BITS), elem_dtype)
tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
thr        = tiled_copy.get_slice(tid)
src_part   = thr.partition_S(out_shared_view)
dst_part   = thr.partition_D(out_global_view)
fx.copy(copy_atom, src_part, dst_part)
```

When in doubt prefer option A — it is uglier but easier to reason about and
audit.

---

## 3. Shared-memory rearrange (transpose / pre-shuffle)

Pattern: load a (M, N) tile from global, swap a couple of axes through LDS,
write a (N, M) tile back. TileKernels' `batched_transpose_kernel.py` is the
canonical example. The FlyDSL version is split between an LDS allocator
(`SmemAllocator`) outside the kernel and an explicit `gpu.barrier()` between
write and read.

### TileLang skeleton

```python
@T.prim_func
def kernel(x: ..., out: ...):
    with T.Kernel(...) as (pid_y, pid_x, pid_b):
        out_shared = T.alloc_shared((block_y, block_x + block_k), dtype)  # +pad to dodge bank conflicts
        tmp     = T.alloc_local((block_k, block_k), dtype)
        tmp_row = T.alloc_local((block_k,),         dtype)
        tid = T.get_thread_binding()
        row, col = tid // T_PER_ROW, tid % T_PER_ROW

        for i_ in T.unroll(...):
            i = i_ * (NUM_THREADS // T_PER_ROW) + row
            # Load 4 rows × 4 cols into `tmp`, vectorising along k.
            for j in T.unroll(block_k):
                for k in T.vectorized(block_k):
                    tmp_row[k] = x[pid_b, ..., ...]
                for k in T.unroll(block_k):
                    tmp[k, j] = tmp_row[k]
            # Write transposed into LDS with swizzle.
            for j in T.unroll(block_k):
                swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k
                for k in T.vectorized(block_k):
                    out_shared[col*block_k + swizzle_j, i*block_k + k] = tmp[swizzle_j, k]

        T.sync_threads()
        for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
            out[pid_b, pid_y*block_y + i, pid_x*block_x + j] = out_shared[i, j]
```

### FlyDSL skeleton

```python
def build_transpose(*, block_x, block_y, block_k, dtype_str, num_threads=256):
    arch     = get_rocm_arch()
    elem_dt  = dtype_to_elem_type(dtype_str)
    elem_ty  = elem_dt.ir_type
    elem_b   = elem_dt.width                              # 8 for fp8, 16 for bf16

    # 1. LDS allocation outside the kernel — finalised in the JIT body.
    allocator = SmemAllocator(None, arch=arch, global_sym_name="smem_transpose")
    lds_pad   = block_x + block_k
    lds_shape = (block_y, lds_pad)
    lds_buf   = allocator.allocate_array(elem_ty, block_y * lds_pad)

    @flyc.kernel
    def kernel(x: fx.Tensor, out: fx.Tensor,
               num_batches: fx.Int32, shape_x: fx.Int32, shape_y: fx.Int32,
               stride_x: fx.Int32):
        bid_y = fx.block_idx.x
        bid_x = fx.block_idx.y
        bid_b = fx.block_idx.z
        tid   = fx.thread_idx.x

        # 2. LDS view through SmemPtr (2-D).
        base_ptr = allocator.get_base()
        lds      = SmemPtr(base_ptr, 0, elem_ty, shape=lds_shape)

        # 3. Buffer-backed input view sliced down to (block_x, block_y) for this block.
        x_buf  = fx.rocdl.make_buffer_tensor(x)
        x_slab = fx.slice(x_buf, (bid_b, None, None))                      # (shape_x, shape_y)
        x_blk  = fx.zipped_divide(x_slab, (block_x, block_y))               # ((block_x, block_y), tiles)
        x_blk  = fx.slice(x_blk, (None, (bid_x, bid_y)))                    # this block's tile

        # 4. Per-thread distribution of the load. T_PER_ROW threads cover one row.
        T_PER_ROW = block_y // block_k
        row, col  = tid // T_PER_ROW, tid % T_PER_ROW
        ROWS_PER_PASS = num_threads // T_PER_ROW
        copy_atom_g = fx.make_copy_atom(fx.rocdl.BufferCopy(8 * block_k * elem_b // 8)(), elem_dt)

        # 5. Read+transpose loop. Use the same explicit pattern as TileLang —
        #    range_constexpr for the unrolled outer; range_constexpr for the
        #    inner k-vectorised packs; a 1xblock_k register memref per pass.
        reg_ty  = fx.MemRefType.get(elem_ty, fx.LayoutType.get(block_k, 1),
                                    fx.AddressSpace.Register)
        reg_lay = fx.make_layout(block_k, 1)
        for i_ in range_constexpr(block_x // block_k // ROWS_PER_PASS):
            i = i_ * ROWS_PER_PASS + row
            for j in range_constexpr(block_k):
                r = fx.memref_alloca(reg_ty, reg_lay)
                fx.copy_atom_call(copy_atom_g,
                                  fx.slice(x_blk, (i*block_k + j, col*block_k)),  # base addr
                                  r)
                vec = fx.memref_load_vec(r)

                # Swizzle along j to dodge bank conflicts on the LDS write.
                swizzle_j = (j + tid // (8 // (elem_b // 8))) % block_k
                # Write each lane k of `vec` into LDS at (col*block_k + swizzle_j, i*block_k + k).
                for k in range_constexpr(block_k):
                    SmemPtr.store(lds, vec[k], [col*block_k + swizzle_j,
                                                i*block_k + k])

        gpu.barrier()

        # 6. Write LDS back to global. Use a tiled copy keyed off the same
        #    loop_layout TileLang built with `T.Fragment`. See §2 option B
        #    for how to derive thr/val layouts from the forward_fn.
        out_buf = fx.rocdl.make_buffer_tensor(out)
        out_slab = fx.slice(out_buf, (bid_b, None, None))
        out_blk  = fx.zipped_divide(out_slab, (block_y, block_x))
        out_blk  = fx.slice(out_blk, (None, (bid_y, bid_x)))

        thr_layout = fx.make_layout((num_threads,), (4,))
        val_layout = fx.make_layout((4,), (1,))
        tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
        tiled_copy  = fx.make_tiled_copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dt), tv, tile_mn)

        # LDS is the source, global is the destination.
        thr_copy   = tiled_copy.get_slice(tid)
        # Wrap the LDS pointer as a memref-like tensor for `partition_S`.
        lds_view   = lds.as_memref(fx.make_layout(lds_shape, (lds_pad, 1)))
        src_part   = thr_copy.partition_S(lds_view)
        dst_part   = thr_copy.partition_D(out_blk)
        fx.copy(tiled_copy.copy_atom, src_part, dst_part)

    @flyc.jit
    def launch(x: fx.Tensor, out: fx.Tensor,
               num_batches: fx.Int32, shape_x: fx.Int32, shape_y: fx.Int32,
               stride_x: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        ctx = CompilationContext.get_current()
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kernel(x, out, num_batches, shape_x, shape_y, stride_x).launch(
            grid=(shape_y // block_y, shape_x // block_x, num_batches),
            block=(num_threads, 1, 1),
            stream=stream,
        )

    return launch
```

The mapping is: every `T.alloc_shared` becomes one `allocator.allocate_array`
call (you can have several — each gets its own `_offset`). `T.sync_threads`
becomes `gpu.barrier()`. The `T.Fragment` with a `forward_fn` is replaced by a
hand-derived `thr_layout`/`val_layout` pair fed into `make_tiled_copy`.

---

## 4. Reduction (norm / softmax / topk)

The canonical TileLang reduction uses `T.reduce_max` / `T.reduce_sum` /
`T.alloc_reducer`. None of these exist in FlyDSL — re-implement with a
two-stage block reduction (wave-butterfly + LDS round-trip).

### Helpers worth lifting from `kernels/softmax_kernel.py`

```python
WARP_SIZE = get_warp_size()                 # 64 on CDNA, 32 on RDNA / gfx1250
RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)

def wave_reduce(x, mode):                    # mode in ("max","sum","min","and","or")
    w = x
    for _e in range_constexpr(int(math.log2(WARP_SIZE))):
        off  = WARP_SIZE // (2 << _e)
        peer = w.shuffle_xor(off, WARP_SIZE)
        w = w.maximumf(peer) if mode == "max" else \
            w.addf(peer, fastmath=arith.FastMathFlags.fast)
    return w

def block_reduce(val, mode, s_red):
    if const_expr(RED_SLOTS == 1):
        return wave_reduce(val, mode)
    lane = tid % WARP_SIZE
    wave = tid // WARP_SIZE
    w = wave_reduce(val, mode)
    if lane == 0:
        SmemPtr.store(s_red, w, [wave])
    gpu.barrier()
    if wave == 0:
        in_range = lane < RED_SLOTS
        v = SmemPtr.load(s_red, [in_range.select(lane, 0)])
        v = in_range.select(v, NEUTRAL[mode])
        v = wave_reduce(v, mode)
        if lane == 0:
            SmemPtr.store(s_red, v, [0])
    gpu.barrier()
    return SmemPtr.load(s_red, [0])
```

`s_red` is an `SmemPtr` of length `RED_SLOTS` allocated up-front in the
allocator. For multi-output reductions (softmax: max then sum, rmsnorm: sum
of squares only) preallocate one slot-array per quantity.

### Top-K via repeated max

`T.reduce_max` + `T.alloc_reducer('min')` is the classic top-k. Translate
literally:

```python
for k in range_constexpr(num_topk):
    # Replace `T.reduce_max(scores_fragment, amax_fragment)`:
    scores_vec = Vec(...)                  # build the per-thread Vector view
    thread_max = scores_vec.reduce(ReductionOp.MAX)
    block_max  = block_reduce(thread_max, "max", s_red)
    # Replace `T.alloc_reducer((1,), int32, 'min')` and the if/min update:
    cand_idx   = (scores_vec == block_max).select(idx_vec, fx.Int32(0x7fffffff))
    thread_min = cand_idx.reduce(ReductionOp.MIN)
    block_min  = block_reduce(thread_min, "min", s_red_int)
    # Mask the chosen index out for the next iteration.
    scores_vec = (idx_vec == block_min).select(fx.Float32(float("-inf")), scores_vec)
    # Store the chosen index.
    if tid == 0:
        SmemPtr.store(topk_lds, block_min, [k])
```

`Vec.reduce` operates on the per-thread vector; `block_reduce` then gathers
across threads. Two LDS slot arrays — one float for max, one int for min —
both live in `SmemAllocator`.

---

## 5. GEMM via MFMA

`T.gemm(A_shared, B_shared, C_local)` does a *lot* under the hood. The minimal
FlyDSL skeleton:

```python
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))  # M=N=K=16, fp16 → fp32 accum
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
thr_mma   = tiled_mma.thr_slice(tid)

frag_A    = thr_mma.make_fragment_A(A_blk_view)   # A_blk_view = block-level (M, K) view
frag_B    = thr_mma.make_fragment_B(B_blk_view)
frag_C    = thr_mma.make_fragment_C(C_blk_view)

# Stage 1: load A,B from global to shared, then to fragment (or directly to fragment).
# This is the part TileLang's `T.copy(...A_shared)` hides — write it explicitly.
# See `kernels/preshuffle_gemm.py` for the production pattern.

frag_C.fill(0)
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)   # D = A·B + C

# Stage 2: write frag_C back through a copy atom.
copy_atom_C = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
tiled_copy_C = fx.make_tiled_copy_C(copy_atom_C, tiled_mma)
thr_copy_C   = tiled_copy_C.get_slice(tid)
fx.copy(copy_atom_C, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(C_blk_view))
```

Pick an MFMA shape that matches the dtype: `mfma_f32_16x16x4f32` for f32,
`mfma_f32_16x16x16f16` for f16, `mfma_f32_16x16x32_fp8_fp8` for fp8 on gfx94x,
`mfma_scale_f32_16x16x128_f8f6f4` for blockscale on gfx950. The list lives in
`flydsl/expr/rocdl.py`.

For production-quality GEMM with the prefetch pipeline, do **not** open-code
it in the conversion — it is a complete rewrite, not a port. Either point
the user at `kernels/preshuffle_gemm.py` or invoke the `prefetch-data-load`
skill from this repo.

---

## 6. Strided / non-contiguous tensor inputs

TileLang has `T.StridedTensor[(B, M, N), (M*S, S, 1), dtype]` baked into the
kernel signature. FlyDSL has no such per-arg layout annotation — the host
expresses non-contiguity at the launch site:

```python
# Inside the Python wrapper that calls @flyc.jit:
x_dyn = flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1, divisibility=4)
launch(x_dyn, ...)
```

The kernel reads `x` as `fx.Tensor` with the dynamic stride threaded through.
`mark_layout_dynamic` *should* be set whenever the test uses
`twice_stride(x)`-like helpers (i.e., the leading stride is not equal to the
trailing dim). Otherwise the JIT will assume a tighter contiguous layout and
the kernel will read garbage.
