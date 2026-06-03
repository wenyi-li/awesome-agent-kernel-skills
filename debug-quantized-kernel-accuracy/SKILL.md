# Skill: Debug Quantized Kernel Accuracy

## Purpose
Guide the agent through a systematic process for diagnosing and isolating accuracy degradation in a quantized (INT8, FP8, or low-bit) kernel, from measuring the error to identifying the specific computational step responsible.

## Use this when
- A quantized kernel produces outputs that differ from the fp32 reference by more than the expected quantization error bound.
- A model using quantized kernels shows accuracy degradation that exceeds what is expected for the chosen quantization scheme.
- A quantization refactor introduced a regression and the specific step that broke is not obvious.
- Debugging a quantized kernel that works correctly on some input shapes or batch sizes but fails on others.

## Do not use this when
- The error is within the expected quantization error bound (approximately 0.5 * scale per element for well-calibrated INT8) and the downstream task accuracy loss is acceptable.
- The issue is clearly a non-accuracy bug (segfault, wrong shape, miscompilation) — fix the structural bug first.
- The degradation is due to model-level quantization sensitivity (certain layers or operators being inherently sensitive to quantization), which requires a quantization-aware training or mixed-precision approach rather than kernel debugging.

## Inputs the agent should gather first
- The exact mathematical specification of what the quantized kernel is supposed to compute, written in terms of the original unquantized operation.
- The quantization scheme: per-tensor, per-channel, or per-token; symmetric or asymmetric; INT8, INT4, or FP8; signed or unsigned range.
- The scale computation method: offline calibration, per-batch dynamic quantization, or per-token dynamic quantization.
- The accumulation dtype: INT32, FP32, FP16, or FP8.
- The dequantization epilogue: where is scale applied, in what order, and what is the output dtype.
- A fp32 reference output for the same inputs (required for comparison).
- Whether the error is consistent across runs (deterministic) or varies (stochastic — possible race condition or non-deterministic reduction).

## Required reasoning process
1. **Measure the error precisely.** Do not characterize accuracy failures as "wrong" without quantifying them:
   - Compute max absolute error (MAE_max), mean absolute error (MAE_mean), and relative error (RE = |output - reference| / (|reference| + epsilon)).
   - Compute per-element SNR if the output represents a signal: `20 * log10(rms(reference) / rms(output - reference))`.
   - Check whether the error is distributed uniformly across the output tensor or concentrated in specific rows, columns, or batch elements. Localized error strongly suggests a scale indexing bug.
   - Check whether the error scales with the magnitude of the input values. If error is proportional to input magnitude, the scale factor is wrong. If error is constant, the zero-point is wrong.

2. **Isolate the quantization step responsible.** The quantization pipeline is:
   ```
   fp32 input → quantize → int8 input
   int8 GEMM (or other op) with int32 accumulation
   int32 accumulation → dequantize → fp32/fp16 output
   ```
   Test each boundary:
   - **Quant-dequant roundtrip**: quantize the fp32 input to INT8, then immediately dequantize back to fp32. Compare to the original. Max error should be approximately 0.5 * scale per element. If it exceeds this, the scale computation or the quantization formula is wrong.
   - **INT8 kernel with exact INT8 inputs**: if you can construct inputs where the exact INT8 representation is known (e.g., all values are multiples of the scale), run the INT8 kernel and verify that INT32 accumulation produces the expected result before dequantization.
   - **Dequantization epilogue only**: take known-correct INT32 accumulation results, apply the dequantization epilogue, and verify the fp32 output. If this step introduces error, the scale application formula is wrong.

3. **Check scale computation for outliers.** A single large-magnitude input value can dominate the scale computation for a per-tensor or per-row quantization scheme, causing all other values to be mapped to a narrow range of INT8 values (near zero). This is called the outlier problem:
   - Compute the histogram of INT8 values for the weight tensor and the activation tensor. If most values cluster near 0 with only a few values near ±127, outliers are the cause of accuracy loss.
   - Check per-tensor vs per-channel granularity: if per-tensor quantization produces a scale dominated by one outlier channel, switching to per-channel quantization may resolve the accuracy issue.
   - Examine the distribution of absolute values in the activation tensor. If a significant fraction are near the max representable value in fp16, fp16 overflow may be affecting the scale computation itself.

4. **Verify zero-point handling for asymmetric quantization.** Zero-point bugs produce systematic bias in the output:
   - The correct dequantization is `x_fp32 = (q_int8 - zero_point) * scale`.
   - If `zero_point` is added instead of subtracted, every output element has a systematic bias of `2 * zero_point * scale`.
   - If `zero_point` is a signed quantity but stored as unsigned (or vice versa), the bias direction is wrong.
   - For symmetric quantization: zero-point must be exactly 0. If a non-zero zero-point leaks in from misconfiguration, it introduces a constant offset.

5. **Verify accumulation dtype is INT32.** INT8 accumulation overflows for dot products of length > ~10 with typical INT8 values. Symptoms: outputs are wildly wrong and the error grows with K (the reduction dimension). Verify by reading the kernel source and confirming the accumulator variable is `int32_t`, not `int8_t` or `int16_t`.

6. **Check scale application order for per-channel / per-token quantization.** The scale must be applied along the correct dimension:
   - For per-channel weight quantization with per-tensor activation quantization: `output[m][n] = int32_acc[m][n] * act_scale * weight_scale[n]`.
   - If `weight_scale[n]` is applied along the M dimension instead of N, every output row `m` is multiplied by `weight_scale[m]` instead of `weight_scale[n]`. The error pattern will show column-wise or row-wise distortion.
   - For per-token activation quantization: `output[m][n] = int32_acc[m][n] * act_scale[m] * weight_scale[n]`. Verify that `act_scale[m]` loads the scale for token m, not token m+1 or another token.

7. **Check INT32 accumulation overflow.** For very long dot products with INT8 inputs, INT32 can overflow:
   - Maximum accumulation value = `127 * 127 * K`. This exceeds INT32 max (~2.1B) when K > ~131,000.
   - For standard LLM linear layer sizes (K up to ~16384), overflow is not a concern for symmetric INT8. For larger K, check.
   - If INT32 overflow is detected, the fix is to split the K accumulation into segments and dequantize partially, or to accumulate in INT64 for the final segment.

8. **Check the dequantization epilogue sequence.** The order of operations in the epilogue matters:
   - Correct order: INT32 accumulation → cast to fp32 → multiply by scale → add bias (in fp32) → cast to output dtype.
   - Incorrect order: INT32 accumulation → multiply by scale (in INT32, which may overflow for large scales) → cast to fp32.
   - Incorrect order: INT32 accumulation → cast to fp32 → add bias (using wrong bias scale) → multiply by scale.
   - Write out the epilogue operations in order and verify each step against the mathematical definition.

9. **Check for shape-dependent failures.** If the kernel is correct on some shapes but wrong on others, the likely causes are:
   - Partial K tile (K not a multiple of dp4a width = 4): the tail elements are either skipped or computed incorrectly.
   - Per-token scale with variable sequence length: if the scale tensor is indexed incorrectly when sequence length changes, scales from the wrong position are applied.
   - Batch size 1 vs larger batch: if batch dimension handling has an off-by-one, batch size 1 may work correctly but larger batches fail.

## Kernel design rules
- Never accumulate in INT8. Always accumulate in INT32. This is non-negotiable.
- The dequantization step must be a separate, identifiable section of the kernel. Do not interleave scale application with the INT8 accumulation loop.
- Scale tensors must be loaded with explicit indexing that is reviewed against the mathematical formula. Do not use flat pointer arithmetic without verifying the index maps to the correct scale.
- For debugging: add an assert or `printf` in the kernel for the first output element to verify the INT32 accumulation value before dequantization. Compare this to a Python reference that computes `int32_t(a_int8) @ int32_t(b_int8)` (integer GEMM).
- Zero-point values must be stored and loaded in the same signed/unsigned convention throughout the pipeline. Mixing signed and unsigned zero-points is a correctness bug.

## Correctness requirements
- Quant-dequant roundtrip error must be bounded by 0.5 * scale per element. If it exceeds this, the scale computation or quantization formula is wrong and must be fixed before debugging the kernel.
- After confirming the roundtrip is correct, the INT8 kernel output must match the fp32 reference to within N * (0.5 * scale) absolute error, where N is the number of terms in the accumulation. In practice, well-implemented INT8 GEMM matches fp32 GEMM to within 1–3 ULPs at the output scale.
- Per-element error must be randomly distributed across the output tensor, not structured (not concentrated in specific rows, columns, or blocks). Structured error implies a scale indexing bug, not just quantization noise.
- The kernel must produce identical results across multiple runs on the same inputs (deterministic). Non-deterministic INT8 accumulation is a sign of a race condition or unsynchronized reduction.

## Performance requirements
- Debugging steps (quant-dequant roundtrip tests, INT32 accumulation verification) can use slow, reference-quality Python code. Do not optimize the debugging harness.
- Once the accuracy bug is fixed, re-run the performance baseline to confirm the fix did not introduce a regression. Accuracy fixes that add redundant loads or computations may degrade throughput.

## Output format
The agent should produce:

1. **Error characterization**: a table showing max absolute error, mean absolute error, relative error, and SNR (if applicable) compared to the fp32 reference.
2. **Error distribution analysis**: a statement of whether the error is uniform or structured (row/column/block-wise), with evidence (e.g., per-row error norms).
3. **Isolation test results**: results of the quant-dequant roundtrip test, the INT32 accumulation test, and the dequantization epilogue test, each compared to their expected values.
4. **Root cause statement**: a single-sentence identification of the specific bug (e.g., "scale is applied to the M dimension instead of the N dimension" or "zero-point subtraction has the wrong sign").
5. **Fix implementation**: the corrected code with a comment explaining what was wrong and why the fix is correct.
6. **Regression test**: a minimal test case that would have caught this bug, added to the test suite.

## Common failure modes
- **Scale dominated by outlier activations**: a single large-magnitude activation token causes the per-token scale to be large, quantizing all other tokens to near-zero INT8 values. Diagnosed by checking the per-token scale distribution — a few scales will be 10–100x larger than the median.
- **Wrong scale granularity**: using per-tensor quantization for a weight matrix where individual channels have very different magnitude ranges. Error is large and channel-dependent.
- **INT32 accumulation overflow for very long K**: rare in practice but occurs for K > ~131K with symmetric INT8. Diagnosed by testing with a linearly increasing K and observing when the error changes character.
- **Incorrect dequantization epilogue order**: multiplying scale into the INT32 accumulation before casting to fp32, causing overflow in the scale multiplication. Diagnosed by checking whether the INT32 value before scale application is correct.
- **Per-channel scale indexed along the wrong dimension**: the scale vector has shape (N,) and must be applied along the output channel dimension. If the kernel transposes the output or uses a non-standard layout, the index mapping may be wrong.
- **Asymmetric zero-point with wrong sign**: `(q + zero_point)` instead of `(q - zero_point)` introduces a constant bias of `2 * zero_point * scale` per output element. The error is constant and does not depend on the input values.
- **Scale computed from float16 statistics**: if `max_abs(x)` is computed in fp16 before being used to set the INT8 scale, fp16 overflow or rounding in the max computation propagates into every scale value. Always compute scales in fp32.
- **Bias added before dequantization scale**: if the bias is in the original fp32 space and the scale is applied after, `output = (int32_acc + bias_int32) * scale`. But if the bias is stored in fp32, adding it to int32_acc requires careful conversion and is almost always a bug. Apply bias after dequantization: `output = int32_acc * scale + bias_fp32`.

## Review checklist
- [ ] Error is quantified: max absolute error, mean absolute error, and relative error are computed and reported.
- [ ] Error distribution is characterized: uniform or structured (row/column/block)?
- [ ] Quant-dequant roundtrip error is verified to be < 0.5 * scale per element before debugging the kernel.
- [ ] INT32 accumulation correctness is verified independently of dequantization.
- [ ] Dequantization epilogue is verified independently with known-correct INT32 inputs.
- [ ] Scale computation is verified against the mathematical formula; units and signs are correct.
- [ ] Zero-point sign convention is verified to be consistent throughout the pipeline.
- [ ] Scale granularity matches the expected error magnitude; per-tensor scale with high-variance channels has been ruled out.
- [ ] Accumulation dtype is confirmed as INT32 by reading the kernel source directly.
- [ ] Shape-dependent failures have been investigated (partial K tiles, variable sequence length, batch size 1).
- [ ] A minimal regression test for the identified bug is added to the test suite.
