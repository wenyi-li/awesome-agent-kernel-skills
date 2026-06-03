# Skill: Write an INT8 Quantized Kernel

## Purpose
Guide the agent through designing and implementing an INT8 quantized matrix multiplication or linear layer kernel for inference, covering quantization scheme selection, scale computation, int32 accumulation, dequantization epilogue, and the decision between custom code and library solutions.

## Use this when
- Writing an inference kernel that needs to reduce memory bandwidth and/or increase throughput by operating on INT8 weights and activations.
- Implementing a quantized linear layer where scales and zero-points are known at kernel call time (static or dynamic).
- The hardware is Turing (sm_75) or later, where the `dp4a` INT8 dot product instruction is available.
- Evaluating whether a custom INT8 GEMM is justified versus using cuBLAS INT8 (`cublasGemmEx` with `CUDA_R_8I`) or CUTLASS INT8 GEMM templates.

## Do not use this when
- The target hardware is pre-Turing (sm_70 or earlier) — there is no hardware INT8 dot product instruction and the implementation falls back to emulation, which is unlikely to outperform fp16.
- The workload is training (not inference) — INT8 gradients require careful handling that is a separate design problem.
- The accuracy budget cannot tolerate INT8 quantization error. Evaluate accuracy first before committing to INT8.
- cuBLAS `cublasGemmEx` with `CUDA_R_8I` inputs and `CUDA_R_32I` compute already satisfies the performance requirement. Use the library unless there is a specific reason (custom epilogue, non-standard quantization scheme, latency constraints) that the library does not meet.

## Inputs the agent should gather first
- **Quantization granularity**: per-tensor (one scale for the entire tensor), per-channel (one scale per output channel of the weight), or per-token (one scale per token/row of the activation). This determines the dequantization operation.
- **Symmetric vs asymmetric quantization**: symmetric maps the range `[-127, 127]` to `[-max_abs, max_abs]` with zero-point = 0. Asymmetric maps `[0, 255]` (or `[-128, 127]`) to an arbitrary range with a non-zero zero-point. Symmetric is strongly preferred for inference because zero-point handling adds extra arithmetic to the hot loop.
- **Accumulation dtype**: INT32 is mandatory. INT8 accumulation overflows for any practical dot product length.
- **Output dtype**: FP32 or FP16 after dequantization. This determines the epilogue.
- **Scale dtype and location**: are scales stored as fp32 scalars, fp32 tensors, or fp16? Are they computed offline (static quantization) or at kernel launch time (dynamic quantization)?
- **Hardware target**: compute capability (sm_75 for Turing dp4a, sm_80 for Ampere, sm_89 for Ada).
- **Weight layout**: are weights already quantized and stored as INT8? Are they in row-major or column-major order?
- **Activation layout**: contiguous rows? Are activations quantized offline or online (at each forward pass)?

## Required reasoning process
1. **Choose the quantization scheme and verify it is implemented correctly.** Clarify the formula:
   - Symmetric quantization: `q = clamp(round(x / scale), -127, 127)`. Scale = `max_abs(x) / 127`. Zero-point = 0.
   - Asymmetric quantization: `q = clamp(round(x / scale) + zero_point, 0, 255)` (uint8) or `(-128, 127)` (int8). Scale = `(max(x) - min(x)) / 255`. Zero-point = `round(-min(x) / scale)`.
   - Write these formulas explicitly before touching kernel code. Incorrect scale computation is the most common source of quantized kernel accuracy bugs.

2. **Understand the INT32 accumulation math.** For a dot product of length K with INT8 inputs and INT32 accumulation:
   - `acc_int32 += int32(a_int8[k]) * int32(b_int8[k])` for k in 0..K-1.
   - Maximum value before overflow: `127 * 127 * K`. For K=4096: max = 127 * 127 * 4096 = ~66M, which fits in INT32 (max ~2.1B). For K >> 65536, overflow risk exists with per-tensor symmetric quantization — verify.
   - The `dp4a` instruction computes `int32 += dot(int8[4], int8[4])` — four INT8 multiplications and an INT32 accumulation in a single instruction. K must be a multiple of 4 for efficient use of dp4a.

3. **Decide between dp4a (custom CUDA), CUTLASS, or cuBLAS.**
   - cuBLAS `cublasGemmEx(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I)`: handles per-tensor symmetric quantization with alpha/beta scaling. Fast and reliable. Limited to column-major inputs in some versions — check the API requirements.
   - CUTLASS INT8 GEMM: provides per-channel scale epilogue support, mixed layout support, and custom epilogue fusion. More configuration work but handles non-standard epilogues.
   - Custom dp4a kernel: only justified when the quantization scheme or epilogue is not supported by cuBLAS or CUTLASS. Writing a custom INT8 GEMM that outperforms CUTLASS is very difficult.

4. **Design the dequantization epilogue.** After INT32 accumulation, convert to the output dtype:
   - Per-tensor: `out_fp32 = int32_result * scale_A * scale_B`. One multiply per output element.
   - Per-channel weights + per-tensor activations: `out_fp32[m][n] = int32_result[m][n] * scale_A * scale_B[n]`. One multiply per output element with a broadcast over the N dimension.
   - Per-token activations + per-channel weights: `out_fp32[m][n] = int32_result[m][n] * scale_A[m] * scale_B[n]`. One multiply per output element with no broadcast — both scale vectors must be loaded.
   - Asymmetric zero-point correction: for asymmetric quantization, `int32_result` must be corrected by a term that depends on the zero-points before dequantization. This correction is an additional INT32 reduction per output element and must be computed correctly.

5. **Design the kernel structure.** For a custom dp4a GEMM:
   - Load INT8 A (M x K) and INT8 B (N x K) tiles into shared memory.
   - Use dp4a in the inner loop: `__dp4a(a_frag, b_frag, acc)` (PTX intrinsic via `__dp4a` or via inline PTX).
   - Accumulate in INT32 throughout the K loop.
   - In the epilogue: load scales, dequantize, optionally apply bias (in fp32), cast to output dtype, store.
   - Do not cast from INT32 to INT8 at any intermediate stage — accumulation in INT8 is incorrect.

6. **Handle the quantization boundary.** K must be a multiple of 4 for `dp4a`. If K is not a multiple of 4, pad the input to the next multiple of 4 or use scalar fallback for the remaining elements.

7. **Verify accuracy.** Compare INT8 kernel output to fp32 reference using mean absolute error (MAE) and max absolute error as fractions of the quantization scale. For well-calibrated symmetric per-channel quantization, the max error should be bounded by roughly 0.5 * (scale / 127) per element.

## Kernel design rules
- Accumulate in INT32 throughout the entire dot product. Never cast the accumulator back to INT8 at any intermediate step.
- Use `dp4a` (four INT8 multiplications per cycle, INT32 accumulation) as the inner loop instruction. Do not use INT16 or INT32 multiplications for the inner loop — they are slower.
- Apply scale and zero-point corrections in fp32 in the epilogue, after all INT32 accumulation is complete.
- For per-channel or per-token scales: load scale vectors into registers before the epilogue loop, not inside the epilogue loop body.
- Clamp quantized values to the valid range before storing: `[-128, 127]` for signed INT8. Saturation on out-of-range values must be explicit, not assumed.
- K must be padded to a multiple of 4 in the input layout or handled via a scalar tail loop.
- Tile sizes for INT8 GEMM should be larger than for fp16 GEMM because INT8 has half the element size — more elements fit in the same smem budget, enabling more K-reuse.
- When using cuBLAS INT8 GEMM: the leading dimensions and layout must match the API's expectations (column-major by default). Passing row-major pointers without transposing produces silently wrong results.

## Correctness requirements
- INT32 accumulation must not overflow for the maximum possible K length with the given INT8 input range. Compute `127 * 127 * K_max` and verify it is below 2^31 - 1 (~2.1B).
- The dequantization formula must exactly invert the quantization formula. A quant → dequant roundtrip on a simple input must recover the original values to within quantization error (0.5 * scale).
- Zero-point corrections for asymmetric quantization must be applied correctly before scaling. Applying scale before zero-point correction produces wrong results.
- Scales must be applied in the correct order: for per-channel weight scales, the scale for output column `n` must be applied to output element `(m, n)`, not to a different column.
- For per-token activation quantization: the scale for row `m` must be loaded from the correct position in the scale tensor. Off-by-one indexing in the scale tensor is a common correctness bug.
- Verify the kernel against an fp32 reference with per-element error reporting. A single miscalculated scale can corrupt an entire row or column of output silently.

## Performance requirements
- On Ampere (sm_80), well-implemented INT8 GEMM should achieve approximately 2x the throughput of fp16 GEMM in terms of peak TFLOPS/s, since the hardware has 2x INT8 vs fp16 tensor core throughput. This assumes compute-bound operation.
- For small M (inference batch size 1–16), INT8 GEMM is often memory-bandwidth bound, not compute-bound. In this regime, INT8 provides benefit by reducing weight size and therefore memory traffic, but the throughput ratio over fp16 is much less than 2x.
- Profile the kernel with Nsight Compute. Check `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld` and `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st` for smem efficiency, and `smsp__inst_executed_pipe_tensor_op_hmma_pred_on` for tensor core utilization.
- Compare against cuBLAS INT8 GEMM on the same shape. Beating cuBLAS is difficult for general GEMM. A custom kernel is only justified if it provides capabilities cuBLAS lacks (e.g., custom epilogue).

## Output format
The agent should produce:

1. **Quantization scheme specification**: explicit formulas for quantization (forward and inverse), granularity, zero-point handling, and scale dtype, written in mathematical notation before any code.
2. **Scale computation code**: the offline or online code that computes scales from fp32 input tensors.
3. **INT8 GEMM kernel**: complete, compilable kernel code using dp4a or CUTLASS, with the dequantization epilogue.
4. **Dequantization epilogue code**: the scale application logic, labeled with which scale vector corresponds to which dimension.
5. **Accuracy test**: compare INT8 kernel output to fp32 reference, reporting MAE and max absolute error.
6. **Library alternative assessment**: explicit statement of whether cuBLAS or CUTLASS was considered and why a custom kernel was or was not chosen.

## Common failure modes
- **Accumulating in INT8 instead of INT32**: the most severe correctness bug. The accumulator overflows after only a few iterations for typical INT8 values. Always accumulate in INT32.
- **Incorrect scale application order for asymmetric quantization**: applying the scale before correcting for the zero-point produces a systematic bias in the output.
- **Wrong zero-point offset direction**: the correct dequantization is `x_fp32 = (q - zero_point) * scale`, not `(q + zero_point) * scale`. The sign of the zero-point correction is a frequent source of bugs.
- **Not clamping quantized values to [-128, 127]**: quantizing a value outside the representable range without clamping produces a wraparound INT8 value, which dequantizes to the wrong value.
- **Per-channel scale applied to the wrong dimension**: for a weight matrix of shape (N, K), per-channel scales have shape (N,) and apply along the N (output channel) dimension. Applying them along K is wrong.
- **K not a multiple of 4 for dp4a**: if K is not divisible by 4 and this is not handled, the tail elements are either skipped (incorrect) or accessed out of bounds (undefined behavior).
- **cuBLAS column-major assumption**: cuBLAS INT8 GEMM expects column-major matrices. Passing a row-major matrix A (M x K) as-is results in computing the wrong product. Transpose both inputs or pass A as a K x M column-major matrix with the operation transposed.
- **Scale computed from the wrong tensor**: computing the activation scale from the weight tensor or vice versa. Always verify which tensor each scale corresponds to.

## Review checklist
- [ ] Quantization formulas (forward and inverse) are written out explicitly before any code.
- [ ] Accumulation is in INT32 throughout; no intermediate cast to INT8 exists.
- [ ] `dp4a` is used as the inner loop instruction; K is verified to be a multiple of 4 or a tail loop handles the remainder.
- [ ] INT32 overflow is verified: `127 * 127 * K_max < 2^31 - 1`.
- [ ] Zero-point correction is applied before scale multiplication in the dequantization epilogue.
- [ ] Per-channel or per-token scales are applied along the correct dimension.
- [ ] Quantized values are clamped to `[-128, 127]` before storing as INT8.
- [ ] cuBLAS and CUTLASS were evaluated; the choice of custom kernel is explicitly justified.
- [ ] Accuracy is validated against an fp32 reference with per-element error reporting.
- [ ] Performance is compared to the cuBLAS INT8 GEMM baseline on the target shape.
- [ ] The dequantization epilogue is separated from the accumulation loop and operates in fp32.
