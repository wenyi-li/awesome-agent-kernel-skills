# Skill: Write a Triton Fused Add+RMSNorm Kernel

## Purpose

Guide the agent through implementing a single Triton kernel that computes `y = rmsnorm(x + residual)` while also writing back `x + residual` for the next transformer block's residual stream. This fusion is the dominant pattern in LLaMA, Mistral, Qwen, and similar decoder blocks: every attention sub-block and every MLP sub-block ends with `residual_add -> rmsnorm`. Done correctly, the kernel saves one full read+write pass over the activation tensor compared to a naive `add` kernel followed by an `rmsnorm` kernel, and removes one launch.

---

## Use this when

- You are implementing a transformer inference path (LLaMA-family, Mistral, Qwen, Gemma, DeepSeek, etc.) and the block structure is `h = x + residual; out = rmsnorm(h)` with `h` becoming the residual into the next block.
- Profiling shows the unfused `add` and `rmsnorm` kernels each touching the full activation tensor and the pipeline is HBM-bandwidth bound.
- You need a custom Triton implementation because the framework path (PyTorch eager, vendor library) does not fuse these two ops, or `torch.compile` is unavailable, partial, or breaks the graph.
- You are matching the kernel surface of vLLM (`fused_add_rms_norm`), FlashInfer, or Liger-Kernel and need a Triton equivalent.
- You want to keep the residual write as part of the same kernel pass to avoid an extra launch and an extra read of `x` in the next block.

---

## Do not use this when

- The next operation is itself fusable with the normalization output (e.g., RMSNorm immediately followed by a QKV projection where the matmul's loader can ingest unnormalized values from the epilogue of a previous kernel). Prefer a larger fused block over a chain of small ones.
- Hidden dimension is small (< 512). The kernel is launch-overhead bound at that size, and a generic `add` + framework `rmsnorm` is fine — the saved bandwidth is dwarfed by launch latency.
- Training is required and you have not designed a backward that handles gradients through both the add and the norm. Forward fusion is straightforward; backward fusion is materially harder (see Common failure modes).
- Mean subtraction is needed (LayerNorm, not RMSNorm). Use the standard Triton LayerNorm skill instead — RMSNorm omits the mean.
- The residual is on a different dtype, layout, or device than `x`. Resolve the layout mismatch before fusing.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Input dtype** — fp16, bf16, or fp32 for `x` and `residual`. Both should match. The summed residual write-back keeps the input dtype; the normalized `y` typically keeps the same dtype as well.
2. **Hidden dimension H** — fixed or dynamic? Power of 2? Always divisible by `BLOCK_SIZE`? Does it fit in a single `BLOCK_SIZE` (single-pass) or require a loop (two-pass)?
3. **Batch shape** — (N, H), (B, T, H), or higher rank. Confirm normalization is over the last dimension only.
4. **Epsilon** — typically 1e-5 or 1e-6. Confirm placement (inside vs outside `sqrt`); PyTorch RMSNorm convention is inside.
5. **Weight (gamma) dtype** — most checkpoints store gamma in fp16/bf16. Multiply in fp32 then cast back.
6. **In-place vs out-of-place residual write** — does the caller want `x` overwritten with `x + residual`, or written to a separate output buffer? Aliasing rules matter.
7. **Backward needed?** — if yes, you must save `rstd` and the summed activation, and design a backward that splits the gradient into the residual path and the rmsnorm path.
8. **Eps double-counting** — confirm the downstream kernel (e.g., the next block's QKV) does not also add eps if it consumes the rstd you produce.

---

## Required reasoning process

1. **Assign one program per row.** `row_idx = tl.program_id(0)`. Grid is `(N_rows,)` where `N_rows = prod(batch_dims)`. Compute the row base pointer as `x_ptr + row_idx * row_stride` and similarly for `residual_ptr`, `y_ptr`, and the summed-output pointer.

2. **Load x and residual together with a mask.**
   ```python
   col_offsets = tl.arange(0, BLOCK_SIZE)
   mask = col_offsets < H
   x   = tl.load(x_ptr   + row_idx * stride + col_offsets, mask=mask, other=0.0)
   res = tl.load(res_ptr + row_idx * stride + col_offsets, mask=mask, other=0.0)
   ```
   `other=0.0` is correct: out-of-bounds positions contribute 0 to the sum and 0 to the variance.

3. **Cast to fp32 and compute the sum.**
   ```python
   x   = x.to(tl.float32)
   res = res.to(tl.float32)
   h   = x + res
   ```
   This is the single most important precision choice. Residual streams in deep transformers grow over 32+ layers; performing the add in fp16/bf16 loses bits before the squaring step and corrupts the variance estimate.

4. **Write the summed residual back immediately.**
   ```python
   tl.store(sum_ptr + row_idx * stride + col_offsets, h.to(input_dtype), mask=mask)
   ```
   Cast back to the input dtype on the way out. This is the value the next block consumes as its residual; it is not optional and must be written in the same kernel pass to realize the fusion benefit.

5. **Compute the mean square in fp32.**
   ```python
   var = tl.sum(h * h, axis=0) / H
   rstd = 1.0 / tl.sqrt(var + eps)
   ```
   Divide by `H`, not by the count of unmasked elements. Out-of-bounds positions loaded as 0.0 contribute 0 to the sum-of-squares; dividing by `H` is correct.

6. **Normalize and apply gamma.**
   ```python
   gamma = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
   y     = h * rstd * gamma
   ```
   Gamma is a 1D tensor of shape `(H,)`. Use `col_offsets` only — no row offset.

7. **Store the normalized output.**
   ```python
   tl.store(y_ptr + row_idx * stride + col_offsets, y.to(input_dtype), mask=mask)
   ```
   Same boundary mask as the load.

8. **For H > BLOCK_SIZE,** loop over chunks. Two paths:
   - **Two-pass:** pass 1 loads `x`, `res`, computes `h`, stores `h` to the sum buffer, accumulates `sum(h*h)`. Pass 2 reloads `h` from the sum buffer, normalizes, stores `y`. Requires reading `h` twice but avoids any extra storage.
   - **Single-pass with online accumulation:** keep a running `sum_sq` across chunks; on the second pass over chunks, recompute `h = x + res` (since `h` was not stashed) and normalize. This re-reads `x` and `res` and is usually worse than two-pass.
   Pick two-pass. It writes the residual once (which you must do anyway) and reads it back once.

9. **For training, save `rstd` and the summed residual.** The backward kernel needs both. The summed residual is already written by step 4; just store `rstd` to a `(N_rows,)` buffer.

10. **Backward (if required).** The forward output is `y = h * rstd * gamma` where `h = x + res`. The gradient of the loss with respect to `x` is `dL/dh` (since `dh/dx = 1`); the gradient with respect to `res` is also `dL/dh`. So `dL/dx = dL/dres = drmsnorm_dh`, and gradients on the residual stream pass through unchanged. Implement the rmsnorm backward to produce `drmsnorm_dh` once, then write it to both `dx` and `dres` (or accumulate into `dx` if `res` aliases `x`).

---

## Kernel design rules

- `BLOCK_SIZE` must be a power of 2 and declared `tl.constexpr`. Choose the smallest power of 2 >= `H` for the single-pass case; otherwise pick a fixed tile (e.g., 1024 or 2048) and loop.
- All arithmetic from the add through the variance must run in fp32. Cast inputs immediately after loading. Cast back to the input dtype only at the two stores.
- The summed residual write (`x + residual`) must happen in the same kernel pass. If you skip it, the fusion is meaningless — the next block will recompute the same sum.
- Epsilon placement is `1.0 / sqrt(var + eps)` (inside the sqrt). This matches the PyTorch RMSNorm convention. Document the choice if you deviate.
- Gamma is 1D, shape `(H,)`, shared across all rows. Pointer arithmetic uses `col_offsets` only.
- Row stride for `x`, `residual`, `y`, and the summed output must be passed as kernel arguments. Do not hardcode `stride = H`. They are usually equal but the contract should not assume it.
- If `x` and the summed-output pointer alias (in-place residual write), make sure you finish reading `x` before writing the summed residual. In Triton, since the load is a register copy, the in-place store after the add is safe.
- Do not add a bias term — RMSNorm has gamma but no beta. Adding beta silently turns the kernel into a malformed LayerNorm.

---

## Correctness requirements

- The summed residual must be written to memory in the same kernel pass. Skipping this write produces silently wrong outputs starting at layer 2, because every subsequent block reads a stale residual.
- The variance denominator is `H` (full row length), not the count of unmasked positions. Out-of-bounds positions contribute 0 because they are loaded with `other=0.0` and `0 * 0 = 0`.
- The add `h = x + residual` must execute in fp32. Performing the add in fp16/bf16 truncates the residual stream and produces a measurably wrong variance — visible as small but consistent drift versus a reference implementation, especially at layer counts > 16.
- Out-of-bounds load uses `other=0.0`. Using `other=-inf` or any nonzero value corrupts the sum-of-squares.
- The output store mask must match the input load mask exactly (same `col_offsets < H`).
- Gamma loads use `col_offsets` only — no row offset. Adding `row_idx * H` is a common bug that reads garbage.
- Eps must be strictly positive. Reject or warn on `eps <= 0`.
- If `x` and the summed-output buffer point to the same memory (in-place residual update), the kernel must read `x` into registers before writing the sum back. Do not interleave per-element loads and stores between `x` and the sum buffer — load the full tile first.
- The summed residual must be cast back to the input dtype on store. Writing fp32 into an fp16 buffer (or vice versa without a cast) is a memory-corruption bug.

---

## Performance requirements

The agent must reason through the following before finalizing:

- **Memory traffic.** Naive: read `x`, read `residual`, write `h` (the add kernel), read `h`, read `gamma`, write `y` (the rmsnorm kernel) — about `5 * N_rows * H * sizeof(dtype)` bytes plus one gamma read per row. Fused: read `x`, read `residual`, read `gamma`, write `h`, write `y` — about `4 * N_rows * H * sizeof(dtype)` bytes. The saving is one full read pass over the activation tensor. Quote the saving as `~20%` of the unfused traffic for the most common case, not as a speedup multiplier.
- **Single-pass vs two-pass.** Two-pass is correct for `H > BLOCK_SIZE` but adds one read of the summed residual. For `H <= BLOCK_SIZE` (the common case for `H` up to 8192 on modern hardware), single-pass keeps `h` in registers and avoids the reload.
- **Register pressure.** Holding fp32 versions of both `x` and `residual` plus their fp32 sum doubles register use compared to LayerNorm without fusion. For `BLOCK_SIZE = 8192` fp16 inputs, that is roughly 96 KB of register data per program if you keep all three live; you cannot. Reuse the registers: cast and add immediately, drop `x` and `res`, keep only `h`.
- **Gamma reuse.** Gamma is read per row but resides in L2 after the first few programs. Do not preload to shared memory.
- **Bandwidth ceiling.** A well-implemented fused kernel should approach HBM peak for moderate `H` (1024–8192). State the achieved bandwidth in benchmark output, not a speedup multiplier alone.
- **Launch overhead.** For very small workloads (e.g., decoding with `N_rows = 1`), the absolute saving is one launch — about 5–10 microseconds. Below that scale the fusion still helps but the relative speedup is dominated by Python and PyTorch overhead, not the kernel.

---

## Output format

The agent should produce:

1. **The Triton kernel function** with `@triton.jit`, taking `x_ptr`, `residual_ptr`, `gamma_ptr`, `y_ptr`, `sum_ptr` (the summed-residual output, may alias `x_ptr`), `rstd_ptr` (optional, training only), `stride`, `H`, `eps`, and `BLOCK_SIZE: tl.constexpr` as arguments.
2. **A Python launcher** that flattens the leading dimensions, computes the grid as `(N_rows,)`, asserts contiguity on the last dim, extracts strides, picks `BLOCK_SIZE` based on `H`, and exposes an in-place vs out-of-place flag for the residual.
3. **A correctness test** comparing against a reference `(x + residual).pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * (x + residual) * gamma`, plus a check that the summed residual buffer matches `x + residual` exactly. Use `torch.allclose(atol=1e-3, rtol=1e-3)` for fp16/bf16 and `atol=1e-5` for fp32.
4. **A bandwidth benchmark** comparing the fused kernel against an unfused baseline (PyTorch `add` + reference `rmsnorm`) using `triton.testing.do_bench` or `torch.utils.benchmark.Timer`. Report achieved GB/s, not just speedup.
5. **A note on backward** — if forward-only, state explicitly that this kernel is forward-only. If backward is included, state how the residual gradient is split.
6. **Comment block** documenting epsilon placement, dtype contract for gamma, and the aliasing assumption between `x` and the summed-output buffer.

---

## Common failure modes

- **Forgetting to write the summed residual.** The kernel produces the correct `y` for the current layer but the next layer reads a stale residual. The model output is silently wrong from layer 2 onward. Tests on a single layer pass; tests on the full stack fail. Always include an explicit check that the summed buffer equals `x + residual`.
- **fp16 add before squaring.** Performing `h = x + res` in fp16/bf16 before the cast loses precision in the residual stream. The variance estimate drifts and per-token logits diverge over many layers. Always cast to fp32 first, then add, then square.
- **Aliasing bug between `x` and `sum_ptr`.** If the caller passes `sum_ptr == x_ptr` for in-place residual update and the kernel writes the sum before fully reading `x`, the loaded `x` is corrupted. In Triton this is fine if you load the full tile first then add then store, but is a real bug if you tile within a single program.
- **Double-counting eps.** If a downstream kernel (e.g., a second normalization or a reused rstd) also adds eps to the variance, the effective eps is `2 * eps`. Document who owns eps in the contract.
- **Gamma cast at the wrong stage.** Loading gamma in fp16 and multiplying by fp32 `h * rstd` produces an implicit narrowing in some Triton versions. Cast gamma to fp32 explicitly before multiplying.
- **Wrong denominator for variance.** Dividing by `tl.sum(mask.to(tl.float32))` instead of `H` gives a different (incorrect) variance when `H` is not a multiple of `BLOCK_SIZE`.
- **Storing fp32 into an fp16 output buffer.** A forgotten `.to(input_dtype)` on either the summed-residual store or the normalized-output store either crashes (Triton type check) or corrupts memory (silent in some versions).
- **Backward forgetting the residual path.** A forward-fused kernel paired with a backward that only handles the rmsnorm path produces wrong gradients for `x` and `residual`. Both must receive `drmsnorm_dh`.
- **Using this for LayerNorm by accident.** RMSNorm has no mean subtraction and no beta. Adding either turns the kernel into a malformed LayerNorm and wastes compute.

---

## Review checklist

- [ ] `BLOCK_SIZE` is a power of 2 and declared `tl.constexpr`.
- [ ] Both `x` and `residual` are cast to fp32 immediately after loading, before the add.
- [ ] The add `h = x + residual` runs in fp32.
- [ ] The summed residual `h` is stored back to memory in the same kernel pass, cast to the input dtype.
- [ ] Out-of-bounds load uses `other=0.0` for both `x` and `residual`.
- [ ] Variance denominator is `H` (full row length), not count of valid elements.
- [ ] Variance and rstd are computed in fp32.
- [ ] Eps is placed inside the sqrt: `1.0 / sqrt(var + eps)`.
- [ ] Gamma loads use `col_offsets` only — no row offset added.
- [ ] Gamma is cast to fp32 before multiplying with `h * rstd`.
- [ ] No beta term and no mean subtraction (this is RMSNorm, not LayerNorm).
- [ ] Output store applies the same boundary mask as the input load.
- [ ] Row strides for all four buffers are passed as kernel arguments.
- [ ] Aliasing contract between `x` and the summed-output buffer is documented and respected (load full tile before storing).
- [ ] For training: `rstd` is stored to a `(N_rows,)` buffer; the summed residual is already in memory.
- [ ] Backward (if implemented) routes `drmsnorm_dh` to both `dx` and `dresidual`.
- [ ] Correctness test verifies both the normalized output and the summed residual buffer.
- [ ] Benchmark reports achieved bandwidth, not only a speedup multiplier, and compares against an unfused baseline.
- [ ] No claim that fusion is faster without a measured number on the target hardware.
