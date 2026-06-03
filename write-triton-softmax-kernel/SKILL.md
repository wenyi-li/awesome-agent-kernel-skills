# Skill: Write a Triton Softmax Kernel

## Purpose

Guide the agent through implementing a numerically stable, performant row-wise softmax kernel in Triton. This covers single-program-per-row assignment, online max+sum reduction with `tl.max`/`tl.sum`, masking for rows wider than BLOCK_SIZE, fp32 accumulation to avoid overflow and precision loss, and the masked softmax variant for attention.

---

## Use this when

- You need a fused softmax that avoids a separate max-reduction kernel pass and a separate division pass — i.e., you want a single kernel that reads each row once.
- You need a masked softmax (attention mask applied before the exp) and PyTorch's built-in path is not fusing the mask application correctly.
- You are fusing softmax with a preceding or following elementwise operation (e.g., scale by `1/sqrt(d_k)` before softmax, or multiply output by values V immediately after).
- The row dimension is large enough that a per-row kernel is worthwhile (row_size >= 256 is a reasonable floor). Below this, `torch.nn.functional.softmax` is likely faster.

---

## Do not use this when

- The input is a standard 2D or 3D tensor with no mask and no fusion requirement. `torch.nn.functional.softmax` backed by cuDNN or `torch.compile` will handle this efficiently.
- The softmax dimension is across rows rather than within rows (i.e., column-wise softmax). The per-row strategy does not apply without transposing the problem.
- The row size is very small (< 64). A warp-level reduction in CUDA or a fused `torch.compile` graph is more efficient.
- You need a stable online-softmax for arbitrarily long sequences in a streaming fashion — this requires a more complex multi-block reduction strategy beyond a single-program-per-row approach.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Input shape** — (N, D) or (B, H, N, D) or similar. Which dimension is the softmax applied over?
2. **Row dimension size D** — is it fixed (compile-time constant), or dynamic? Is it always a power of 2?
3. **Input dtype** — fp16, bf16, or fp32. Accumulation dtype must be fp32 regardless.
4. **Whether a mask is applied** — additive mask (large negative values added before softmax) or boolean mask (invalid positions should be treated as -inf)?
5. **Whether to fuse a downstream operation** — e.g., multiply by a V matrix tile, apply dropout, or write to a specific output layout.
6. **Whether the row fits in one BLOCK_SIZE** — or whether a loop over multiple blocks per row is needed.
7. **Hardware target** — relevant for choosing BLOCK_SIZE and deciding whether to use `triton.autotune`.

---

## Required reasoning process

1. **Determine whether the row fits in a single block.** If `D <= BLOCK_SIZE` (and BLOCK_SIZE is a power of 2 >= D), one program handles the full row in a single pass with masking. If `D > BLOCK_SIZE`, the program must loop over chunks, maintaining a running max and running sum (online softmax).

2. **Assign one program per row.** `row_idx = tl.program_id(0)`. For a 2D input of shape (N, D), the grid is `(N,)`. For higher-rank inputs, flatten the batch and head dimensions into the row count.

3. **Compute the row base pointer.**
   ```python
   row_start_ptr = input_ptr + row_idx * input_row_stride
   col_offsets = tl.arange(0, BLOCK_SIZE)
   input_ptrs = row_start_ptr + col_offsets
   mask = col_offsets < D
   row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
   ```
   Using `other=-float('inf')` for out-of-bounds positions ensures they do not affect the max reduction.

4. **Compute max for numerical stability.** `row_max = tl.max(row, axis=0)`. For the multi-block case, accumulate running max across iterations before computing exp.

5. **Subtract max and exponentiate.** `row = tl.exp(row - row_max)`. Perform this in fp32. If inputs are fp16/bf16, cast before the subtraction: `row = row.to(tl.float32)`.

6. **Sum the exponentiated values.** `row_sum = tl.sum(row, axis=0)`. Again in fp32.

7. **Normalize.** `row = row / row_sum`.

8. **Store the output.**
   ```python
   tl.store(output_ptrs, row.to(output_dtype), mask=mask)
   ```

9. **For the multi-block (D > BLOCK_SIZE) case**, implement online softmax:
   - Pass 1 (compute max): loop over blocks, track `running_max = max(running_max, tl.max(chunk, axis=0))`.
   - Pass 2 (compute sum): loop over blocks, compute `running_sum += tl.sum(tl.exp(chunk - running_max), axis=0)`.
   - Pass 3 (normalize and store): loop over blocks, divide and write back.
   This is three passes over the row but requires no inter-program communication. Alternatively, use the single-pass online softmax update: `new_max = max(old_max, chunk_max); sum = sum * exp(old_max - new_max) + chunk_sum_relative_to_new_max` — reduces to two passes (one forward scan, one normalize).

10. **Apply additive mask (if required).** Add the mask to the raw logits before the max reduction, not after. Do not apply a boolean mask by zeroing after exp — this changes the sum and produces incorrect probabilities for masked positions.

---

## Kernel design rules

- BLOCK_SIZE must be a power of 2 and declared `tl.constexpr`. Choose the smallest power of 2 >= D for the single-block case.
- All reductions (`tl.max`, `tl.sum`) must operate on fp32 tensors. Cast fp16/bf16 inputs to fp32 before any reduction.
- Out-of-bounds loads must use `other=-float('inf')` (not 0.0) so they do not corrupt the max or sum.
- Out-of-bounds positions in the exp-sum must naturally collapse to 0 when loaded as -inf (exp(-inf) = 0). Verify this logic is correct before storing.
- Do not store fp32 intermediate values to global memory in the single-pass case. All intermediate state (running_max, running_sum) lives in registers.
- For the masked softmax, the additive mask (e.g., `-1e9` for invalid positions) must be added to the logits before the max reduction. Passing it as a separate load and adding it before `tl.max` is the correct pattern.
- `input_row_stride` must be passed as a kernel argument. Do not assume the row stride equals D (the tensor may be a slice of a larger allocation).

---

## Correctness requirements

- Subtracting the row max before exponentiation is mandatory. Without it, exp(logit) overflows to inf for logits > ~88 in fp32 and even smaller in fp16.
- The mask applied to out-of-bounds loads must use `other=-float('inf')`, not `other=0.0`. A value of 0.0 would contribute `exp(0) = 1.0` to the sum, producing incorrect probabilities for padded positions.
- For rows shorter than BLOCK_SIZE, the output store must apply the same mask (`col_offsets < D`) to avoid writing garbage to out-of-bounds memory.
- In the multi-block online softmax, the running max correction factor `exp(old_max - new_max)` must be applied to the running sum before adding the new chunk's contribution. Missing this rescaling produces incorrect sums.
- Additive attention masks must be added before the max reduction, not after the division. Adding after division is not a softmax.
- The output must sum to 1.0 per row (within numerical tolerance). Verify with `torch.allclose(output.sum(dim=-1), torch.ones(N))`.

---

## Performance requirements

The agent must reason through the following:

- **Memory bandwidth bound.** A softmax kernel reads each row element once (for max), once (for sum), and writes once (normalized values). Total traffic is 3 * N * D * sizeof(dtype) bytes. At A100 HBM bandwidth of ~2 TB/s, this limits achievable throughput — ensure the kernel is not significantly below this bound.
- **Single-block efficiency.** When D fits in one BLOCK_SIZE, the entire row is loaded once into registers, and all reductions happen there. No synchronization is needed. This is optimal for SRAM-bound problems.
- **BLOCK_SIZE vs occupancy.** Larger BLOCK_SIZE means more registers and fewer concurrent programs on the SM. For D=1024 with fp32 accumulation, BLOCK_SIZE=1024 uses ~8KB of register space per program. Profile occupancy at target D.
- **Vectorized loads.** Triton will attempt to vectorize `tl.load` to 128-bit loads (4 fp32 or 8 fp16 values per instruction). Ensure the input pointer is aligned to 16 bytes and BLOCK_SIZE is a multiple of 4 (for fp32) or 8 (for fp16). This typically holds for power-of-2 BLOCK_SIZE.
- **Do not autotune BLOCK_SIZE blindly.** For softmax, BLOCK_SIZE is tightly coupled to the row dimension D. The correct BLOCK_SIZE is the smallest power of 2 >= D (or a fixed large value with masking). Searching over arbitrary BLOCK_SIZE values will not change correctness but will change register pressure.
- **Comparison point.** Benchmark against `torch.nn.functional.softmax` with `torch.compile`. A well-written Triton softmax should match or slightly exceed this for large D (>= 4096) by avoiding kernel launch overhead for a separate reduction step.

---

## Output format

The agent should produce:

1. **The Triton kernel function** with `@triton.jit`, taking `input_ptr`, `output_ptr`, `input_row_stride`, `output_row_stride`, `D`, and `BLOCK_SIZE: tl.constexpr` as arguments.
2. **The Python launcher** that computes the grid as `(input.shape[0],)` after flattening non-softmax dimensions, extracts strides, and chooses BLOCK_SIZE as the next power of 2 >= D (up to a reasonable maximum like 65536).
3. **Masking strategy documented** — single-block vs multi-block, and why.
4. **A correctness test** comparing against `torch.nn.functional.softmax` using `torch.allclose(atol=1e-4)` for fp16 inputs.
5. **If masked softmax is requested**, show mask application in the load step, not as a post-processing step.

---

## Common failure modes

- **Overflow in exp without max subtraction.** For logits > 88 in fp32 (or > 11 in fp16), `exp(logit)` returns inf. The row becomes inf/inf = NaN. Always subtract the row max before exp.
- **Using `other=0.0` in masked loads.** Out-of-bounds positions load as 0.0, contributing `exp(0) = 1` to the sum. The output probabilities are diluted by phantom positions. Use `other=-float('inf')`.
- **Accumulating sum in fp16.** `tl.sum` on a fp16 tensor accumulates in fp16. For BLOCK_SIZE=4096 and all values near 1.0, the sum can reach 4096, which overflows fp16 (max ~65504). Cast to fp32 before summing.
- **Wrong handling of row stride.** Assuming stride equals D fails when the input is a non-contiguous slice (e.g., `x[:, :, :, :D]` of a padded tensor). Always pass `x.stride(-2)` (the row stride) as a kernel argument.
- **Multi-block running max not rescaling running sum.** When updating running_max in the second chunk, the running_sum from the first chunk must be multiplied by `exp(old_max - new_max)` before adding the new chunk's sum. Missing this produces a sum that corresponds to different max normalizations and gives incorrect output.
- **Applying boolean mask after division.** Setting masked positions to 0 after dividing by the sum changes the normalization. The remaining positions do not sum to 1. Apply additive masking (-inf) before the max reduction.
- **Grid dimension mismatch.** For a (B, H, N, D) input with softmax over D, the grid must be `(B * H * N,)` not `(B, H, N)`. Triton programs are indexed by a flat `program_id`, not a multi-dimensional block index.

---

## Review checklist

- [ ] BLOCK_SIZE is a power of 2 and declared `tl.constexpr`.
- [ ] Out-of-bounds loads use `other=-float('inf')`, not `other=0.0`.
- [ ] Input is cast to fp32 before any reduction (`tl.max`, `tl.sum`, `tl.exp`).
- [ ] Max is subtracted from logits before exponentiation.
- [ ] Output store applies the same boundary mask as the load.
- [ ] Row stride is passed as a kernel argument, not assumed equal to D.
- [ ] Grid dimension equals the total number of rows (all non-softmax dimensions flattened).
- [ ] For multi-block case: running sum is rescaled by `exp(old_max - new_max)` on each max update.
- [ ] For masked softmax: mask is added as large negative value before the max reduction.
- [ ] Correctness test verifies per-row sum equals 1.0 and values match `F.softmax` reference.
- [ ] No performance claims made without a benchmark comparison.
