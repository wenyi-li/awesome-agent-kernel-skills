# Skill: Write a Triton Dequant Kernel (int4 / int8 → fp16 / bf16)

## Purpose

Guide the agent through implementing a Triton kernel that unpacks and dequantizes a quantized weight tensor (int4 or int8) into fp16 or bf16. This is the standalone building block underneath W4A16 / W8A16 schemes (AWQ, GPTQ, SqueezeLLM, bitsandbytes NF4, int8 per-channel). Covers bit-unpacking, per-group scale/zero arithmetic, NF4 codebook lookup, and — critically — when *not* to write a standalone dequant kernel because the operation should be fused into the matmul instead.

---

## Use this when

- You need to materialize a fp16/bf16 weight tile from a packed int4/int8 representation for a consumer that does not have a fused-dequant matmul (custom op, MoE expert hot-swap, LoRA merge into a quantized base).
- You need a baseline reference to validate a fused dequant-GEMM (Marlin, AWQ-fused-GEMM, ExLlama, GPTQ kernels) by comparing intermediate fp16 weights.
- You are debugging a quantized model's accuracy and need to inspect the dequantized weights for a specific scheme (AWQ packing order, GPTQ act-order permutation).
- You are implementing a one-off or research scheme and a fused kernel does not yet exist.
- You must dequantize on the fly because the quantized weight is being modified at runtime (online calibration, dynamic LoRA composition).

---

## Do not use this when

- You are running a standard W4A16 or W8A16 linear layer in production. Standalone dequant materializes the full fp16 weight to HBM and reads it back for the GEMM, doubling weight memory traffic and defeating most of the benefit of weight quantization. Use a fused dequant-GEMM instead (Marlin / AWQ kernels / GPTQ kernels / ExLlama / `torch.ops.aten._weight_int4pack_mm`).
- The downstream GEMM is bandwidth-bound on the weight (typical for batch size 1–32 LLM decode). A fused kernel is strictly better — standalone dequant turns one HBM read of int4 into one read of int4 + one write of fp16 + one read of fp16.
- You only need activation dequantization. This skill is about *weight* dequant; activation dequant lives in the GEMM epilogue.
- The scheme is symmetric int8 per-tensor and the framework already has `torch.dequantize` or equivalent.
- You want to "speed up" inference by replacing a fused GEMM with standalone dequant + cuBLAS fp16 GEMM. Almost always slower.

---

## Inputs the agent should gather first

1. **Quantization scheme** — AWQ, GPTQ, bitsandbytes NF4, SqueezeLLM, plain int8 per-channel, or a custom variant. Each has a different packing, scale/zero layout, and dequant formula.
2. **Bit width** — 3, 4, or 8. Determines packing density (8 int4 / uint32, ~10 int3 / uint32, 4 int8 / uint32) and unpacking arithmetic.
3. **Symmetric vs asymmetric** — symmetric (NF4, plain int8): `w = scale * q`. Asymmetric (AWQ, GPTQ): `w = scale * (q - zero)`.
4. **Group size** — typical 128 along K. Scales and zeros are stored at this granularity; the kernel must compute the correct scale/zero index per weight position.
5. **Scale and zero dtypes** — scales usually fp16/bf16. Zeros may be stored quantized (AWQ packs zeros as int4) or as fp16. Verify.
6. **Packing convention** — the order low-bit values are packed. AWQ: interleaved `[0,4,1,5,2,6,3,7]` (SIMT throughput). GPTQ default: sequential `[0,1,2,3,4,5,6,7]`. NF4: 2 nibbles per uint8, high-vs-low-nibble-first depends on library.
7. **GPTQ act-order (`desc_act`)** — if true, the K dimension is permuted; kernel must apply `g_idx[k]` to look up the group. Missing this produces silently wrong outputs.
8. **NF4 codebook** — the 16 fp16 levels. bitsandbytes, HuggingFace, and forks ship slightly different codebooks. Confirm exact values.
9. **Output dtype** — fp16 or bf16. Affects overflow clamping.
10. **Output target** — full tensor materialization (rare, justify), per-tile-on-demand (common for fused-matmul testing), or streaming.
11. **Storage layout** — AWQ `[K, N//8]` (pack along N) vs GPTQ `[K//8, N]` (pack along K). Unpacking arithmetic differs.

---

## Required reasoning process

1. **Decide whether the kernel should exist.** If the consumer is a fp16 GEMM, propose a fused dequant-GEMM instead and state the bandwidth penalty. Only proceed for a valid use case (debugging baseline, MoE hot-swap, runtime modification, scheme without a fused kernel).

2. **Write out the dequant formula as kernel comments.**
   - AWQ / GPTQ asymmetric: `w = scale * (q_int - zero_int)`, both unpacked from int4/int3.
   - Symmetric int8: `w = scale * q_int8`, `q` in `[-128, 127]`.
   - NF4: `w = scale * NF4_LUT[q_int4]`, fixed 16-entry table.

3. **Plan the bit-unpacking.** For int4 packed 8/uint32: `q[i] = (packed >> (4 * unpack_order[i])) & 0xF`. `unpack_order` is scheme-specific — AWQ: `[0,4,1,5,2,6,3,7]`, GPTQ default: `[0,1,2,3,4,5,6,7]`. Wrong order produces plausible-looking but incorrect output; the model degrades silently rather than crashing.

4. **Plan scale/zero indexing.** With group_size G along K, scales have shape `[K // G, N]`. For weight `(k, n)`, scale is `scales[k // G, n]`:
   ```
   group_idx = k_offsets // GROUP_SIZE
   scale = tl.load(scales_ptr + group_idx[:, None] * N + n_offsets[None, :])
   ```
   For GPTQ act-order, replace `k_offsets // GROUP_SIZE` with `tl.load(g_idx_ptr + k_offsets)`.

5. **Grid.** One program per `[BLOCK_K, BLOCK_N]` tile. Grid = `(cdiv(K, BLOCK_K), cdiv(N, BLOCK_N))`. Each program loads packed ints, matching scales and zeros, dequantizes, writes fp16.

6. **Packed tile load.** `BLOCK_K` int4 weights along K = `BLOCK_K // 8` uint32 loads. Each uint32 unpacks into 8 fp16 values via shifts and masks on the vector.

7. **Arithmetic precision.** For typical scales (1e-3 to 1e-1), fp16 dequant is fine. For outlier blocks or `|q - zero| * scale` approaching 65504, multiply in fp32 and cast back at store.

8. **NF4 codebook.** Pass as a `tl.constexpr` 16-entry tuple of fp16 values. Triton lacks direct constexpr-table gather; use an inline `tl.where` chain or constexpr indexing if your Triton version supports it. Confirm values match the source library exactly.

9. **Output layout.** Write to `[K, N]` fp16 (full materialization) or a tile buffer indexed by program coordinates. Do not transpose during dequant — adds smem traffic.

10. **Validate by roundtrip.** Quantize an fp16 reference, dequant with the kernel, compare against the reference Python dequant (`bitsandbytes.functional.dequantize_4bit`, `auto_gptq`, `awq`). Mismatches point to packing-order or scale-indexing bugs.

---

## Kernel design rules

- BLOCK_K must be a multiple of 8 (for int4) so packed loads align on word boundaries; 32 or 64 is typical for vectorized loads.
- BLOCK_K must be `<= GROUP_SIZE`, or the kernel must explicitly handle a tile spanning a group boundary (load multiple scales per tile row).
- Use `tl.constexpr` for BLOCK_K, BLOCK_N, GROUP_SIZE, and BIT_WIDTH so unpacking unrolls at compile time.
- Cast unpacked int values to int32 (not int8 / int16) before scale/zero arithmetic. fp16 conversion from int32 is well-defined across Triton versions; from int8 it varies.
- Apply zero subtraction in integer arithmetic before multiplying by scale: `(q_int32 - zero_int32) * scale_fp16`. Reversing the order requires pre-multiplying zero by scale and changes precision.
- For NF4, the codebook must be a `tl.constexpr` tuple, not a runtime-loaded tensor. Loading the LUT from HBM each tile is wasteful.
- Scale and zero loads must use the group index, not the raw K index. Common bug: `tl.load(scales_ptr + k_offsets * N + ...)` instead of `(k_offsets // GROUP_SIZE) * N + ...`.
- Mask all loads and stores against `K` and `N` bounds. Edge tiles are partially out of range.
- Do not fuse with subsequent ops (transpose, reshape) unless asked. Keep the kernel focused on dequant.
- Document the assumed packing order at the top of the kernel as a comment — this is the single most common source of confusion when reading or debugging dequant kernels.

---

## Correctness requirements

- Unpacking order must match the scheme's storage convention exactly. A roundtrip (`quantize -> store -> kernel-dequant -> compare to reference dequant`) must pass element-wise within `0.5 * scale`.
- Zero-point subtraction must happen in integer arithmetic before scale multiplication. `scale * (q - zero)` and `scale * q - scale * zero` are algebraically equivalent but differ in fp16 due to rounding.
- For GPTQ with act-order, `g_idx[k]` must determine the group, not `k // group_size`. Missing this produces a permuted output that the downstream matmul silently miscomputes; degradation drifts with sequence length.
- The NF4 codebook must match the source library bit-exactly. Use the exact fp16 values from `bitsandbytes/functional.py` for bitsandbytes compatibility — at least three codebook variants exist in the wild.
- Output dtype clamp: casting from fp32 to fp16 saturates outside `[-65504, 65504]`. For weights this should never trigger — if it does, the scale or zero is wrong.
- Symmetric int8 with `q == -128`: dequant gives `scale * -128`, not `scale * -127`. Some quantizers clamp to `[-127, 127]` for symmetry; verify which convention upstream used.
- Asymmetric zero stored as int4: it's an unsigned 4-bit value in `[0, 15]`. Treating it as signed int4 (range `[-8, 7]`) flips half the weights' signs.
- Group boundary handling: for `BLOCK_K > GROUP_SIZE` or unaligned offsets, rows within the tile may need different scales. Either constrain `BLOCK_K <= GROUP_SIZE` or load a per-row scale vector.

---

## Performance requirements

- **Bandwidth budget.** Reads `K * N / 2` bytes (int4), writes `K * N * 2` bytes (fp16). The write is 4x the read — bandwidth-bound on the write side.
- **Why fused matmul wins.** A fused dequant-GEMM never materializes the fp16 weight to HBM — dequant lives in registers and feeds the tensor core directly. Eliminates the fp16 write and subsequent read. Standalone cannot match this.
- **Vectorized loads.** Load packed weights as `tl.uint32` or wider, not `tl.uint8`. For int4 packed 8/uint32, 4 uint32 per thread = 32 weights per HBM transaction.
- **Scale/zero reuse.** With group_size=128 and BLOCK_N=64, each scale is reused 128x within a tile. Load once per tile into registers; do not reload per row.
- **NF4 LUT in registers.** 16 fp16 entries = 32 bytes. Compiler should keep it resident — verify in PTX. If it spills to local memory, restructure as `tl.where` chain.
- **Occupancy.** Dequant is not register-heavy. Prefer small blocks (64x64) for high occupancy since the kernel is bandwidth-bound.
- **Comparison baseline.** `bitsandbytes.functional.dequantize_4bit` (NF4), `auto_gptq` reference, or `awq` reference. Match these on bandwidth-limited shapes.
- **Honest framing.** Do not claim a speedup over fused matmul kernels — they solve a strictly easier problem. Benchmark only against reference Python dequant.

---

## Output format

The agent should produce:

1. **Scheme specification.** Explicit statement of the scheme (AWQ/GPTQ/NF4/int8), bit width, group size, packing order, and dequant formula as math comments.
2. **Storage layout note.** A short comment showing how `BLOCK_K * BLOCK_N` weights map to packed storage words, including the unpack permutation if non-sequential.
3. **The Triton kernel** with `@triton.jit`: packed weight ptr, scales ptr, zeros ptr (if asymmetric), output ptr, K, N, GROUP_SIZE, strides, and `BLOCK_K`, `BLOCK_N`, `BIT_WIDTH` as `tl.constexpr`. NF4 LUT as a constexpr tuple when applicable.
4. **A Python launcher** computing the grid, validating shapes and dtypes, asserting `K % GROUP_SIZE == 0`, and dispatching the correct scheme variant.
5. **A correctness test** comparing against the reference Python dequant for the scheme, with `torch.allclose(atol=scale.max() / 2)`.
6. **A "when not to use this" note** in the launcher docstring or header, pointing to fused-matmul alternatives (Marlin, AWQ-fused-GEMM, GPTQ kernels, ExLlama).

---

## Common failure modes

- **Wrong unpacking permutation.** GPTQ-style sequential unpacking on AWQ weights (or vice versa) produces correct magnitude distribution but each weight at the wrong physical position. The model loads, runs, produces fluent-looking but degraded output. Verify against the source library's pack function.
- **Sign-extending unsigned int4 zeros.** AWQ zeros are unsigned `[0, 15]`. Treating them as signed `[-8, 7]` flips signs on roughly half the weights.
- **fp16 overflow in scale arithmetic.** Degenerate scales (calibration bug, outlier block) overflow `(q - zero) * scale` to inf; downstream GEMM produces NaN. Multiply in fp32 and clamp, or validate scales at load.
- **Group-index off-by-one.** Boundary-shift bugs silently scale the first row of every group with the previous group's scale. Caught only via roundtrip.
- **Forgetting `g_idx` for GPTQ act-order.** `desc_act=True` permutes weight columns; a kernel ignoring `g_idx` produces a permuted weight that downstream attention/FFN silently miscomputes.
- **Wrong NF4 codebook.** bitsandbytes levels: `[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0]`. A uniform 16-level table produces noticeably degraded output.
- **Wrong scale pointer dtype.** Scales are usually fp16/bf16. Treating the pointer as fp32 doubles the stride and reads garbage.
- **Materializing the full fp16 weight in production.** A design error, not a bug. If the output feeds a fp16 GEMM, redirect to a fused kernel.
- **Dequant in-place into the input buffer.** int4 storage is smaller than fp16 output; overwriting corrupts later reads. Use a separate output buffer.

---

## Review checklist

- [ ] Standalone dequant is justified — debugging baseline, MoE hot-swap, runtime weight modification, or no fused kernel exists for the scheme.
- [ ] Scheme, bit width, group size, and packing order are documented as comments in the kernel.
- [ ] Unpacking permutation matches the source library's pack convention exactly.
- [ ] Scale indexing uses `k // GROUP_SIZE` (or `g_idx[k]` for GPTQ act-order), not raw `k`.
- [ ] Zero-point subtraction is in integer arithmetic before scale multiplication.
- [ ] Asymmetric zeros are interpreted as unsigned unless the scheme explicitly stores signed zeros.
- [ ] NF4 codebook (if used) matches the source library bit-exactly.
- [ ] BLOCK_K is a multiple of 8 (for int4) and either `<= GROUP_SIZE` or handles group-boundary crossings.
- [ ] All loads and stores are masked against `K` and `N` bounds.
- [ ] Scale and zero loads are hoisted out of the inner loop and reused across the tile.
- [ ] fp32 dequant arithmetic checked for fp16 overflow on extreme scales.
- [ ] Correctness test compares against the reference Python dequant with `atol = scale.max() / 2`.
- [ ] Launcher docstring points users to fused-matmul alternatives for production use.
- [ ] No claimed speedup over fused dequant-GEMM kernels — comparison baseline is reference Python dequant only.
