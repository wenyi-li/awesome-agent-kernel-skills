# Worked example: normalize_weight

A small kernel from `tile_kernels/moe/normalize_weight_kernel.py`. One
thread per token, no shared memory, no GEMM, no reductions across threads.
Useful as a baseline for the conversion of "row-local" kernels.

---

## Original (TileLang)

```python
import os
import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_normalize_weight_kernel(num_topk: int):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_blocks = T.ceildiv(num_tokens, 128)

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        denominator: T.Tensor[(num_tokens,), T.float32],
        normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            tid = T.get_thread_binding()
            weights_local = T.alloc_local((num_topk,), T.float32)
            row = pid * num_threads + tid

            if row < num_tokens:
                sum = T.alloc_var(T.float32, init=1e-20)
                for i in T.vectorized(num_topk):
                    weights_local[i] = topk_weights[row, i]

                for i in T.unroll(num_topk):
                    sum += weights_local[i]

                denominator[row] = sum
                for i in T.vectorized(num_topk):
                    normalized_weights[row, i] = weights_local[i] / sum

    return normalize_weight_kernel


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    kernel = get_normalize_weight_kernel(num_topk)

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    denominator = torch.empty((num_tokens,), dtype=torch.float32, device='cuda')
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    if num_tokens > 0:
        kernel(topk_weights, denominator, normalized_weights)

    return (denominator, normalized_weights)
```

## Step-1 spec table

| Item | Value |
|---|---|
| JIT factory | `get_normalize_weight_kernel(num_topk: int)` |
| Symbolic dims | `num_tokens` (runtime) |
| Constants | `num_threads=128`, `num_topk` (factory arg → constexpr) |
| Kernel signature | `topk_weights[N, K] f32` in, `denominator[N] f32` out, `normalized_weights[N, K] f32` out |
| Launch | `T.Kernel(ceildiv(N, 128), threads=128) as (pid,)` |
| Allocations | `weights_local: (num_topk,) f32, register`; `sum: scalar f32 init=1e-20` |
| Loops | `T.vectorized(num_topk)` ×2; `T.unroll(num_topk)` ×1 |
| Copies | scalar reads/writes (no `T.copy`) |
| Reductions | per-thread sum, no cross-thread reduction |
| Math | division |
| Edge case | `if row < num_tokens` mask |
| Test | `tests/moe/test_normalize_weight.py`, calls `tile_kernels.moe.normalize_weight(topk_weights)` |
| Strides | input is contiguous (asserted), no `mark_layout_dynamic` needed |

## Conversion design choices

- One thread per token row. `pid * 128 + tid` → row index, masked to
  `num_tokens`.
- `weights_local: (num_topk,) f32` becomes a register memref of shape
  `(num_topk, 1)`.
- `T.vectorized(num_topk)` becomes a single `BufferCopy*` whose width is
  `num_topk * 32` bits (fp32). For `num_topk == 4` that's `BufferCopy128b`,
  for `num_topk == 8` that's `BufferCopy256b` (two 128b copies). Pick the
  atom by inspecting `num_topk` at trace time.
- `T.unroll(num_topk)` over the local sum is the same as
  `range_constexpr(num_topk)` summing register lanes via `Vector` slicing.
  In practice, just use `Vec.reduce(ReductionOp.ADD)`.
- `T.alloc_var(T.float32, init=1e-20)` is *not* a loop-carried var (the
  loop is unrolled at trace time, so the assignments collapse into a single
  reduction). It can be a normal Python `fx.Float32(1e-20)`.

## Ported (FlyDSL)

```python
import os
import math
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.ir import InsertionPoint
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr
from flydsl.expr.vector import ReductionOp


def get_normalize_weight_kernel(num_topk: int):
    NUM_THREADS = 128

    # Pick the largest BufferCopy atom that fits num_topk fp32s in one shot.
    BITS = num_topk * 32
    if BITS == 32:
        copy_op = fx.rocdl.BufferCopy32b()
    elif BITS == 64:
        copy_op = fx.rocdl.BufferCopy64b()
    elif BITS == 128:
        copy_op = fx.rocdl.BufferCopy128b()
    else:
        # Fall back to a per-element loop if num_topk doesn't fit a single atom.
        copy_op = None

    @flyc.kernel
    def normalize_weight_kernel(
        topk_weights:       fx.Tensor,
        denominator:        fx.Tensor,
        normalized_weights: fx.Tensor,
        num_tokens:         fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        row = bid * NUM_THREADS + tid

        # Wrap inputs/outputs as buffer-backed tensors for BufferCopy*.
        x_buf = fx.rocdl.make_buffer_tensor(topk_weights)
        d_buf = fx.rocdl.make_buffer_tensor(denominator)
        n_buf = fx.rocdl.make_buffer_tensor(normalized_weights)

        # Per-thread register memref for the K-vector.
        reg_ty = fx.MemRefType.get(fx.Float32.ir_type,
                                   fx.LayoutType.get(num_topk, 1),
                                   fx.AddressSpace.Register)
        reg_lay = fx.make_layout(num_topk, 1)

        # Mask: only rows < num_tokens do real work.
        in_range = row < num_tokens

        if in_range:
            # 1. Load row[K] in one BufferCopy.
            r_in = fx.memref_alloca(reg_ty, reg_lay)
            row_view = fx.slice(x_buf, (row, None))                 # (num_topk,)
            row_div  = fx.logical_divide(row_view,
                                         fx.make_layout(num_topk, 1))
            copy_atom = fx.make_copy_atom(copy_op, fx.Float32)
            fx.copy_atom_call(copy_atom, fx.slice(row_div, (None, 0)), r_in)

            # 2. Reduce K to a scalar sum (with an additive epsilon to match TileLang).
            v_in = fx.memref_load_vec(r_in)
            sum_v = v_in.reduce(ReductionOp.ADD,
                                fastmath=fx.expr.arith.FastMathFlags.fast)
            sum_with_eps = sum_v + fx.Float32(1e-20)

            # 3. Write denominator[row] = sum_with_eps.
            d_view = fx.slice(d_buf, (row,))
            d_reg_ty = fx.MemRefType.get(fx.Float32.ir_type,
                                         fx.LayoutType.get(1, 1),
                                         fx.AddressSpace.Register)
            d_reg = fx.memref_alloca(d_reg_ty, fx.make_layout(1, 1))
            fx.memref_store_vec(
                fx.expr.vector.full(1, sum_with_eps, fx.Float32),
                d_reg,
            )
            d_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            fx.copy_atom_call(d_copy_atom, d_reg, d_view)

            # 4. Normalised output = v_in / sum_with_eps.
            v_out = v_in / sum_with_eps                              # vector / scalar
            r_out = fx.memref_alloca(reg_ty, reg_lay)
            fx.memref_store_vec(v_out, r_out)
            n_view = fx.slice(n_buf, (row, None))
            n_div  = fx.logical_divide(n_view, fx.make_layout(num_topk, 1))
            fx.copy_atom_call(copy_atom, r_out, fx.slice(n_div, (None, 0)))

    @flyc.jit
    def launch(
        topk_weights:       fx.Tensor,
        denominator:        fx.Tensor,
        normalized_weights: fx.Tensor,
        num_tokens:         fx.Int32,
        stream:             fx.Stream = fx.Stream(None),
    ):
        gx = (num_tokens + NUM_THREADS - 1) // NUM_THREADS
        normalize_weight_kernel(topk_weights, denominator,
                                normalized_weights, num_tokens).launch(
            grid=(gx, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def normalize_weight(topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same public API as the TileKernels original."""
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32

    num_tokens, num_topk = topk_weights.shape
    kernel = get_normalize_weight_kernel(num_topk)

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        # Best-effort: FlyDSL doesn't expose `get_kernel_source()` directly; the
        # closest equivalent is FLYDSL_DUMP_IR=1.
        pass

    denominator        = torch.empty((num_tokens,),          dtype=torch.float32, device='cuda')
    normalized_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    if num_tokens > 0:
        kernel(topk_weights, denominator, normalized_weights, num_tokens)

    return denominator, normalized_weights
```

## Auditing against the gotcha list (excerpt)

- **A** Launch shape: `T.Kernel(num_blocks, threads=128) as (pid,)` is 1-D
  → `block_idx.x`, `block=(128, 1, 1)`. ✓
- **B** Dtype: f32 only, no FNUZ ambiguity. ✓
- **C** `T.alloc_var(init=1e-20)` is reduction-fold-only, mapped to
  `+ fx.Float32(1e-20)` after the vector reduce. No loop carry needed. ✓
- **D** `T.vectorized(num_topk)` mapped to `BufferCopy{N}b()` per
  `num_topk`. The `T.unroll(num_topk)` mapped to `Vec.reduce(ADD)`. ✓
- **E** `make_buffer_tensor` wrapping done. ✓
- **G** No `T.exp`/`T.log` — N/A.
- **H** Test asserts `topk_weights.is_contiguous()` so no
  `mark_layout_dynamic` needed.
- **I** `num_topk` is a Python int → constexpr. ✓
- **K** Closest reference is `silu_and_mul_fq.py` for the per-row
  load/scalar-write pattern. Diff: this kernel does no FP8 packing; just
  plain fp32 in-out.

## Items I cannot verify without GPU

- Numerical correctness of the division (overflow at very small denominators
  — TileLang uses `init=1e-20` to bias; we do the same).
- BufferCopy alignment when `num_topk * 4` is not 4/8/16. The fallback
  branch (`copy_op = None`) is not implemented — the user must ensure
  `num_topk ∈ {1, 2, 4}` or extend the kernel.
- That `fx.expr.vector.full(1, sum_with_eps, fx.Float32)` is the right
  spelling for a length-1 vector store. (It is what `softmax_kernel.py`
  uses; I copied the pattern.)
