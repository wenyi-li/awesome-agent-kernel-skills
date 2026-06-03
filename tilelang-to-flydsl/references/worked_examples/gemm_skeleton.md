# Worked example: GEMM with `T.gemm` → MFMA atoms

This is *less* of a worked example and more of an **annotated skeleton**.
Production GEMM in FlyDSL is large (preshuffle layout, software prefetch
pipeline, hot-loop scheduling) and reproducing it line-by-line for a port
is wrong — defer to `kernels/preshuffle_gemm.py` for the production version.
Use this skeleton only when the TileLang kernel uses `T.gemm` in a
straightforward "load → gemm → store" pattern with no pipelining.

If the TileLang source has `for ko in T.Pipelined(...): T.copy(...);
T.gemm(...);` that **is** pipelining, and the FlyDSL conversion will need
the `prefetch-data-load` skill (separate, in this same `.claude/skills/`).

---

## Source (TileLang, the README example)

```python
@tilelang.jit
def matmul_relu(A, B,
                block_M: int = 64, block_N: int = 64, block_K: int = 64,
                dtype: T.dtype = T.float16, accum_dtype: T.dtype = T.float32):
    M, N, K = T.const('M, N, K')
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])
    return C
```

## What `T.gemm(A_shared, B_shared, C_local)` actually does

It (a) figures out an MFMA shape that matches the dtypes, (b) loads
fragments from `A_shared` / `B_shared` into per-thread registers in the
correct MFMA-operand layout, (c) emits enough MFMA instructions to cover
`(block_M, block_N, block_K)`, (d) accumulates into `C_local`. Steps (b–d)
are explicit in FlyDSL and do not happen unless you write them out.

## FlyDSL skeleton (no pipeline, no preshuffle)

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T as flyT
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir.ir import InsertionPoint
from flydsl.compiler.kernel_function import CompilationContext


def build_matmul_relu(*, M, N, K, block_M=64, block_N=64, block_K=64,
                     in_dtype_str="f16", accum_dtype_str="f32"):
    arch = get_rocm_arch()
    in_dt    = fx.Float16  if in_dtype_str    == "f16" else fx.BFloat16
    accum_dt = fx.Float32

    # MFMA shape: pick the variant matching the dtype.
    # f16 → mfma_f32_16x16x16f16. The K_INST = 16 here.
    M_INST, N_INST, K_INST = 16, 16, 16

    # Atom layout: how many MFMA atoms per workgroup along (M, N, K).
    #   block_M / M_INST atoms along M, block_N / N_INST along N, 1 along K.
    #   For block_M = block_N = 64 and a 4-warp workgroup, the atom layout is
    #   (block_M / M_INST, block_N / N_INST, 1) = (4, 4, 1).
    atom_layout = fx.make_layout((block_M // M_INST,
                                  block_N // N_INST,
                                  1),
                                 (1, block_M // M_INST, 0))   # warp ordering

    # LDS allocations.
    allocator = SmemAllocator(None, arch=arch, global_sym_name="smem_gemm")
    lds_a = allocator.allocate_array(in_dt.ir_type, block_M * block_K)
    lds_b = allocator.allocate_array(in_dt.ir_type, block_K * block_N)

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        bid_x = fx.block_idx.x        # along N
        bid_y = fx.block_idx.y        # along M
        tid   = fx.thread_idx.x

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        # Block-level views.
        A_blk = fx.zipped_divide(A_buf, (block_M, block_K))
        B_blk = fx.zipped_divide(B_buf, (block_K, block_N))
        C_blk = fx.zipped_divide(C_buf, (block_M, block_N))
        A_blk = fx.slice(A_blk, (None, (bid_y, 0)))
        B_blk = fx.slice(B_blk, (None, (0, bid_x)))
        C_blk = fx.slice(C_blk, (None, (bid_y, bid_x)))

        # MMA atom + tiled mma + per-thread fragment.
        mma_op    = fx.rocdl.MFMA(M_INST, N_INST, K_INST, in_dt)
        mma_atom  = fx.make_mma_atom(mma_op)
        tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout)
        thr_mma   = tiled_mma.thr_slice(tid)

        # Fragments for A, B, C (the per-thread register layouts the MMA expects).
        # Pass the block-level views; FlyDSL derives the operand layouts.
        frag_A = thr_mma.make_fragment_A(A_blk)
        frag_B = thr_mma.make_fragment_B(B_blk)
        frag_C = thr_mma.make_fragment_C(C_blk)
        frag_C.fill(0.0)

        # Copy atoms for the global → fragment loads. Use a tiled copy keyed
        # off the MMA's expected operand layout — `make_tiled_copy_A/B`
        # builds it for you.
        copy_atom_load = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_dt)
        tiled_copy_A   = fx.make_tiled_copy_A(copy_atom_load, tiled_mma)
        tiled_copy_B   = fx.make_tiled_copy_B(copy_atom_load, tiled_mma)
        thr_copy_A     = tiled_copy_A.get_slice(tid)
        thr_copy_B     = tiled_copy_B.get_slice(tid)

        # Outer K loop.
        K_TILES = K // block_K
        for ko in range(K_TILES):
            # Re-slice along K. (Equivalent to T.copy(A[by*BM, ko*BK], A_shared);
            # except we load directly into the fragment rather than via shared.
            # For a true LDS staging pipeline see preshuffle_gemm.py.)
            A_blk_ko = fx.slice(A_blk, (None, (None, ko)))
            B_blk_ko = fx.slice(B_blk, (None, (ko, None)))

            src_A = thr_copy_A.partition_S(A_blk_ko)
            src_B = thr_copy_B.partition_S(B_blk_ko)
            fx.copy(copy_atom_load, src_A, thr_copy_A.retile(frag_A), pred=None)
            fx.copy(copy_atom_load, src_B, thr_copy_B.retile(frag_B), pred=None)

            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        # Epilogue: ReLU on frag_C, store to C.
        # Use the carrier vector from frag_C and apply ReLU element-wise.
        # (Replace this with the appropriate fragment-level vector op.)
        # Then write back via tiled_copy_C.
        copy_atom_store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), accum_dt)
        tiled_copy_C    = fx.make_tiled_copy_C(copy_atom_store, tiled_mma)
        thr_copy_C      = tiled_copy_C.get_slice(tid)
        # ReLU on the in-register C fragment:
        v_c = fx.memref_load_vec(frag_C)
        v_c = v_c.maximumf(fx.Float32(0.0))
        fx.memref_store_vec(v_c, frag_C)
        fx.copy(copy_atom_store,
                thr_copy_C.retile(frag_C),
                thr_copy_C.partition_S(C_blk),
                pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        ctx = CompilationContext.get_current()
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gemm_kernel(A, B, C).launch(
            grid=((N + block_N - 1) // block_N,
                  (M + block_M - 1) // block_M, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    return launch
```

## Why this is a *skeleton*, not a final port

- **No SW pipeline.** TileLang's `T.Pipelined(num_stages=3)` provides
  multi-buffer prefetch for free; this skeleton fuses load and MFMA in
  the same iteration, exposing the global-load latency. Performance will
  be 30–60% of the TileLang baseline.
- **No preshuffle.** Production AMD GEMM uses a B-preshuffle layout to
  avoid LDS swizzle. Without it the LDS bank conflict cost shows up in
  ATT traces.
- **Atom layout (4, 4, 1) is correct only for `block_M = block_N = 64`.**
  Re-derive `atom_layout` per problem shape.
- **Thread layout** (`(1, block_M // M_INST, 0)` ordering) here picks a
  warp tiling. Different tilings produce different register pressure /
  LDS access patterns; refer to `preshuffle_gemm.py` for the production
  choice.

When converting a TileKernels GEMM, do these in order:

1. Get this skeleton to compile and pass the unit test for one
   (M, N, K, dtype) configuration.
2. Move the `fx.copy(... -> frag_A/B)` step through an LDS staging buffer
   (allocate `lds_a`, `lds_b` with appropriate swizzle, copy global → LDS,
   barrier, copy LDS → fragment).
3. Add the prefetch pipeline with `range(K_TILES, init=[lds_a_view,
   lds_b_view, ...])` and double-buffer the LDS allocations. Use the
   `prefetch-data-load` skill at this point.
4. Add hot-loop scheduling (`hot_loop_scheduler` from
   `mfma_preshuffle_pipeline.py`) once everything is correct.

Each step has its own correctness gate. Don't combine them.
