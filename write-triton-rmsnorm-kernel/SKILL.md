# Skill: Write a Triton RMSNorm Kernel

## Purpose

Guide the agent through implementing a correct, numerically stable RMSNorm kernel in Triton: `y = x * rsqrt(mean(x², axis=-1) + eps) * weight`. RMSNorm is the dominant normalization in modern decoder-only LLMs (LLaMA, Mistral, Qwen, Gemma, DeepSeek). This skill covers one-pass sum-of-squares with fp32 accumulation, the persistent kernel pattern when the hidden dim fits in a single tile, masking for non-divisible tails, the affine weight broadcast (no bias), and the backward pass. RMSNorm is structurally simpler than LayerNorm — no mean subtraction, no Welford — but the failure modes around fp16 squaring and weight pointer arithmetic still bite.

---

## Use this when

- You are writing the normalization layer for an LLM inference engine (vLLM-style, TensorRT-LLM-style, or custom) and want to fuse the residual add or a downstream epilogue with the norm.
- You need RMSNorm forward + backward for training a LLaMA-family model and `apex.normalization.FusedRMSNorm` is not available on your target hardware (e.g., AMD CDNA).
- You need a fused `residual + RMSNorm` — the pre-norm pattern that dominates LLM blocks — and want to avoid materializing the residual sum to HBM. The kernel may also need to write the post-residual sum back as the next block's residual stream.
- You are debugging numerical drift between a PyTorch reference and a vendor kernel and need a clean Triton baseline to bisect against.

---

## Do not use this when

- You are on PyTorch 2.4+ and `torch.nn.functional.rms_norm` (or a `torch.compile`'d `nn.RMSNorm`) is sufficient. The compiler fuses the read, square, reduce, scale, and weight broadcast.
- You are using a HuggingFace LLaMA / Mistral / Qwen model and the stock `LlamaRMSNorm` with `torch.compile` meets your perf bar. Only write a custom kernel if you need fusion or you are inside an inference engine that controls launch.
- The hidden dim is very small (< 256). Vendor and warp-level CUDA reductions outperform a Triton tile-based approach at this size.
- You need normalization over a non-trailing dimension. RMSNorm by convention normalizes the last dim; this skill assumes that.
- The model uses LayerNorm, not RMSNorm. Use the Triton LayerNorm skill — re-adding mean subtraction into an "RMSNorm" kernel changes semantics.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Input shape** — usually (B, T, H) or (N, H). Confirm H is the trailing dimension and the tensor is contiguous along H.
2. **Hidden dimension H** — fixed (e.g., 4096 for LLaMA-7B, 8192 for LLaMA-70B, 6144 for Qwen) or dynamic? Multiple of 128? Does it fit in a reasonable BLOCK_SIZE (≤ 8192)?
3. **Input dtype** — fp16, bf16, or fp32. fp16 squaring overflows around `|x| ≈ 256`; bf16 squaring underflows easily for small activations. Accumulation must be fp32 regardless.
4. **Weight presence and dtype** — RMSNorm has a per-feature `weight` of shape (H,). Confirm dtype (often the activation dtype in LLaMA, sometimes fp32). There is **no bias**.
5. **Epsilon value** — typically 1e-5 (LLaMA, Mistral) or 1e-6 (Gemma). Placement is `rsqrt(mean_sq + eps)` (inside the sqrt). `rsqrt(mean_sq) + eps` is wrong and not a real convention.
6. **Forward-only or forward + backward** — if backward is needed, save `rrms = rsqrt(mean_sq + eps)` per row.
7. **Residual fusion** — does the kernel read a residual and add it before squaring? If yes, must the post-add value also be written back as the next block's residual stream?
8. **Target architecture** — NVIDIA SM75/80/90/100 or AMD CDNA2/3. BLOCK_SIZE and `num_warps` choices differ; on CDNA prefer `num_warps=4` or `8` and avoid tensor-core-specific assumptions.

---

## Required reasoning process

1. **Assign one program per row.** `row_idx = tl.program_id(0)`. For (B, T, H), flatten to (B*T, H) and launch grid `(B*T,)`. Row base pointer is `x_ptr + row_idx * row_stride`.

2. **Decide one-pass vs two-pass.** RMSNorm only needs `Σ x²`, no mean-then-variance dance. **One pass is always sufficient when H ≤ BLOCK_SIZE.** No Welford needed. For H > BLOCK_SIZE, loop the row in chunks accumulating `Σ x²` into a fp32 scalar, then loop a second time to apply the scale.

3. **Persistent (single-block) pattern when H fits.** For LLM-class hidden sizes (4096, 5120, 6144, 8192), set `BLOCK_SIZE = next_power_of_2(H)` and load the full row in one tile. The row stays in registers between the reduction and the scale step — no second HBM read.

4. **Load the row in fp32 with masking.**
   ```python
   col_offsets = tl.arange(0, BLOCK_SIZE)
   mask = col_offsets < H
   x = tl.load(x_ptr + row_idx * stride + col_offsets, mask=mask, other=0.0)
   x_f32 = x.to(tl.float32)
   ```
   `other=0.0` is correct: out-of-bounds positions contribute 0 to the sum-of-squares.

5. **Compute mean of squares in fp32.** `mean_sq = tl.sum(x_f32 * x_f32, axis=0) / H`. The denominator is H (the full row length), not the count of valid lanes — masked-as-zero lanes correctly contribute 0.

6. **Compute reciprocal RMS.** `rrms = 1.0 / tl.sqrt(mean_sq + eps)` or equivalently `tl.rsqrt(mean_sq + eps)`. `rsqrt` typically lowers to a single hardware instruction (`rsqrt.approx.f32` on NVIDIA). Eps must be inside the sqrt.

7. **Load the per-feature weight (1D, shape (H,)).** `w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)`. The pointer is offset only by `col_offsets` — no row index. The weight broadcasts across all rows.

8. **Apply normalization and weight.** `y = x_f32 * rrms; y = y * w.to(tl.float32)`. Algebraic refactorings (`(x * w) * rrms`, etc.) are valid in exact arithmetic but reorder rounding — pick one and stay consistent with the saved-stat convention.

9. **Store the output with the same mask.** `tl.store(y_ptr + row_idx * out_stride + col_offsets, y.to(output_dtype), mask=mask)`.

10. **If training, save `rrms`.** Write the scalar `rrms` to a (N_rows,) buffer at index `row_idx`. The backward kernel reads this. Do **not** save `mean_sq` or `rms` — that forces an extra op in the backward.

11. **For H > BLOCK_SIZE, two-pass loop.** Pass 1: accumulate `acc += tl.sum(x_chunk * x_chunk)` in fp32 across chunks, then `rrms = rsqrt(acc / H + eps)`. Pass 2: re-load each chunk, multiply by `rrms` and the corresponding weight chunk, store. Costs one extra HBM read per row; prefer the persistent pattern whenever it fits.

12. **Backward pass formulation.** Given upstream `dy`, with `x_normed = x * rrms`:
    - `dx = rrms * (dy * w) − (x_normed * rrms / H) * Σ_h (dy * w * x_normed)`. This form keeps everything in fp32 and only multiplies by `rrms` once per element. Equivalent to `(1/rms) * (dy*w − (x / (H*rms²)) * Σ(dy*w*x))`.
    - `dweight = Σ_batch (dy * x_normed)` — reduce across batch and time, leaving a (H,) gradient. This is a separate kernel (or a separate launch over feature columns) because it is a cross-row reduction.
    - There is **no `dbias`**.

---

## Kernel design rules

- BLOCK_SIZE must be a power of 2 and declared `tl.constexpr`. For the persistent pattern, use the smallest power of 2 ≥ H (4096→4096, 5120→8192, 6144→8192).
- Cast input to fp32 immediately after loading, before any squaring. fp16 `x*x` overflows for `|x| > ~256` (max fp16 is 65504). bf16 has fp32 range but only 7 mantissa bits — squaring loses precision well before overflow.
- The reduction accumulator must be fp32 even when operands are bf16. `tl.sum` on a bf16 tensor without an explicit cast may accumulate in bf16 on some Triton versions.
- Eps is inside the sqrt: `rsqrt(mean_sq + eps)`. Matches PyTorch, HuggingFace, Apex.
- The weight tensor is 1D of shape (H,). Pointer arithmetic uses `col_offsets` only — never add `row_idx * H`. Most common RMSNorm bug.
- No bias parameter. Do not add a `bias_ptr` "for symmetry with LayerNorm".
- Pass `row_stride` (and `out_stride` if different) as kernel arguments. Do not hardcode `stride = H` — callers may pass non-contiguous slices.
- For fused residual + RMSNorm, the residual add must happen in fp32 after both inputs are cast, before squaring. Adding in fp16/bf16 then casting loses precision in the residual stream over many layers.
- Do not preload the weight into shared memory. Triton does not expose shared memory; L2 holds the (H,) weight after the first few rows.
- For H > BLOCK_SIZE, the two-pass kernel must use identical chunk loop bounds in both passes.

---

## Correctness requirements

- Mean denominator is H, not the count of in-bounds lanes. Masked-as-zero positions contribute 0 to the sum, so dividing by H is correct.
- `other=0.0` on the input load — not `-inf`, not `nan`. A non-zero `other` corrupts the sum-of-squares.
- Squaring must be done in fp32. `tl.sum(x * x, axis=0)` where `x` is fp16 produces incorrect (often `inf`) results for activations of moderate magnitude.
- Eps must be strictly positive and applied inside the sqrt. Reject or warn on `eps ≤ 0`.
- Weight is per-feature, not per-row: `weight_ptr + col_offsets`, no `row_idx`.
- Output store mask must equal the input load mask. Storing past the tail corrupts the next row when allocations are tightly packed.
- For training, the saved value must be `rrms` (the reciprocal), the same scalar the forward multiplied by. Saving `sqrt(mean_sq + eps)` and dividing in the backward forces an extra op and risks silent forward/backward mismatch.
- Backward `dx` must read the saved `rrms`, not recompute it from `x`. Recomputation is mathematically equivalent but introduces fp32-rounding drift between forward and backward and doubles the read cost.
- `dweight` reduces over the batch/time axes, producing a (H,) tensor. A common bug is reducing across H and producing a scalar, silently broadcastable in PyTorch and undetected by shape checks.
- For RMSNorm specifically there must be **no** mean-subtraction step. Accidentally including `x = x - mean` (e.g., copy-pasted from a LayerNorm template) silently changes the function the kernel computes.

---

## Performance requirements

The agent must reason through the following before finalizing:

- **Memory-bound regime.** RMSNorm reads x once, reads weight once (cached in L2 after the first row), writes y once. Arithmetic intensity is ~2 FLOPs/byte for fp16 — solidly memory-bound. Aim for ~70–90% of HBM bandwidth at H ≥ 1024.
- **Persistent vs two-pass.** When H fits in one BLOCK_SIZE, the persistent pattern keeps `x` in registers across the reduction and the scale step, saving one HBM read per row (e.g., 8 KB/row at H=4096 fp16; 32 MB total for 4096 rows).
- **Register pressure.** BLOCK_SIZE = 8192 fp16 cast to fp32 is 32 KB of registers per program. On NVIDIA SM80/90 the per-SM register file is 256 KB, capping occupancy at 8 programs/SM. For H ≥ 8192, prefer the two-pass loop with smaller BLOCK_SIZE (1024 or 2048).
- **`num_warps` choice.** BLOCK_SIZE = 1024 → 4. 4096 → 8. 8192 → 16 may help on H100 but check register spill (`ptxas -v`).
- **`rsqrt` instruction.** `tl.rsqrt(x)` lowers to `rsqrt.approx.f32` on NVIDIA (one cycle, ~22-bit mantissa). `1.0 / tl.sqrt(x)` is two instructions. Prefer `rsqrt`.
- **Benchmark targets.** Persistent Triton RMSNorm at H = 4096, fp16, batch = 4096 rows on H100 should reach ~1.5–2.5 TB/s effective bandwidth — within 10–20% of `apex.normalization.FusedRMSNorm`. Far below this suggests register spill or a redundant reload.
- **Fused residual.** One extra HBM read. Justified when it saves writing `x + residual` back only to re-read it in the next kernel — i.e., at the entry of a transformer block.

---

## Output format

The agent should produce:

1. **The forward Triton kernel** with `@triton.jit`, taking `x_ptr`, `y_ptr`, `weight_ptr`, `rrms_ptr` (optional, for training), `row_stride`, `out_stride`, `H`, `eps`, and `BLOCK_SIZE: tl.constexpr`.
2. **Backward kernel(s)** if requested — one for `dx` (per-row, mirrors the forward decomposition) and one for `dweight` (per-feature reduction across the batch).
3. **A Python wrapper** that flattens leading dims to (N_rows, H), checks contiguity along H, computes the grid as `(N_rows,)`, picks BLOCK_SIZE as the smallest power of 2 ≥ H (with a fallback to the two-pass kernel above the chosen cap), and exposes a `torch.autograd.Function` if training is needed.
4. **A correctness test** comparing against:
   ```python
   ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
   ```
   with `torch.allclose(atol=1e-3, rtol=1e-3)` for fp16, `atol=1e-2, rtol=1e-2` for bf16, `atol=1e-5` for fp32. Include at least one H not divisible by BLOCK_SIZE (e.g., H = 4097) to exercise the mask path.
5. **A short comment block** stating: dtype assumptions, eps convention (inside sqrt), weight shape (H,), absence of bias, and the saved-statistics convention (rrms not rms) if backward is supported.

---

## Common failure modes

- **Forgetting fp32 accumulation.** `tl.sum(x * x)` in fp16 produces `inf` for `|x| > ~256` and loses precision long before that. The single most common RMSNorm bug.
- **Mask tail bug.** When H is not a multiple of BLOCK_SIZE, masked lanes must load as 0.0 and must not be stored. Forgetting either mask produces wrong output and/or memory corruption.
- **Eps applied wrong.** `1 / (tl.sqrt(mean_sq) + eps)` instead of `1 / tl.sqrt(mean_sq + eps)` is a different function. Small difference for typical activations, but causes match failures vs PyTorch and can produce `inf` when `mean_sq = 0`.
- **Weight loaded with row offset, or as 2D.** `tl.load(weight_ptr + row_idx * H + col_offsets, ...)` or `weight_ptr + row_idx + col_offsets * stride_w` reads garbage memory. RMSNorm weight is always (H,); correct access is `weight_ptr + col_offsets`. Often produces plausible-looking but wrong output that passes shape checks.
- **Folding `rrms * w` carelessly.** `(x * rrms) * w`, `x * (rrms * w)`, and `(x * w) * rrms` are equivalent in exact arithmetic but cast points and rounding differ. Don't "optimize" between forms without preserving the fp32 multiply through `rrms`.
- **Saving rms instead of rrms.** Storing `sqrt(mean_sq + eps)` and dividing in the backward costs an extra op per element and risks subtle mismatch with the forward-applied scale.
- **Backward dx using recomputed rrms.** Mathematically equivalent but uses different rounding than the forward; can cause `gradcheck` to fail at fp32. Read the saved value.
- **dweight reduced over the wrong axis.** `(dy * x_normed).sum()` (scalar) instead of `.sum(dim=tuple(range(rank-1)))` (shape (H,)) is silently broadcast in downstream PyTorch ops and may go unnoticed until the model fails to train.
- **Copy-pasting from a LayerNorm template and leaving the mean subtraction.** Produces a kernel that looks like RMSNorm in shape and signature but computes LayerNorm. Hard to spot in code review; surfaces only as a numerics regression.
- **Treating bf16 as a free fp32 substitute.** bf16 has fp32 range but only 7 mantissa bits. Squaring a bf16 value loses precision well before any overflow.

---

## Review checklist

- [ ] Input is cast to fp32 immediately after loading, before any squaring.
- [ ] `tl.sum(x_f32 * x_f32, axis=0)` accumulates in fp32.
- [ ] Mean denominator is H (the full row length), not the count of valid lanes.
- [ ] Eps is added inside the sqrt: `rsqrt(mean_sq + eps)`.
- [ ] BLOCK_SIZE is a power of 2 and declared `tl.constexpr`; persistent pattern used when H ≤ chosen cap.
- [ ] For H > BLOCK_SIZE, the two-pass loop uses identical chunk bounds in both passes.
- [ ] Weight pointer arithmetic uses `col_offsets` only — no `row_idx` term.
- [ ] No bias argument, no bias load, no bias add.
- [ ] Output store mask matches the input load mask (`col_offsets < H`).
- [ ] Row stride (and output stride if different) is passed as a kernel argument, not assumed equal to H.
- [ ] For training: `rrms` (the reciprocal, not `rms`) is saved per row to a (N_rows,) buffer.
- [ ] Backward `dx` reads the saved `rrms` rather than recomputing it.
- [ ] Backward `dweight` reduces over the batch/time axes, producing a (H,) tensor.
- [ ] No mean-subtraction step anywhere in the kernel.
- [ ] Correctness test passes against a PyTorch reference at H not divisible by BLOCK_SIZE (e.g., 4097), at fp16, bf16, and fp32.
- [ ] No performance claims without a benchmark against `torch.nn.functional.rms_norm` or `apex.normalization.FusedRMSNorm`.
