# Skill: Write an FP8 Kernel

## Purpose
Guide the agent through designing and implementing FP8 compute kernels for inference and training on NVIDIA Hopper (sm_90) and Ada Lovelace (sm_89) hardware, covering FP8 format selection, scaling strategy, tensor core usage via WGMMA or cuBLAS, and dequantization epilogue design.

## Use this when
- Writing a high-throughput GEMM, attention, or linear layer kernel targeting H100 or H200 (sm_90a) hardware where FP8 tensor cores provide 2x the throughput of FP16.
- Implementing FP8 training (forward pass in E4M3, gradient computation in E5M2) following the transformer engine or similar recipe.
- Evaluating whether FP8 achieves sufficient accuracy for a given model and activation distribution.
- Fusing quantization, GEMM, and dequantization into a single pass to avoid expensive fp32 ↔ fp8 conversion round-trips in memory.

## Do not use this when
- The target hardware is pre-Ada (sm_80 or earlier) — FP8 tensor core instructions do not exist. fp16 or bf16 is the correct choice.
- The activation range is highly dynamic or the per-tensor scale would need to be recomputed at sub-batch granularity with prohibitive overhead.
- Numerical accuracy has not been validated for the target model. FP8 E4M3 has a much narrower dynamic range than fp16; overflows and underflows silently corrupt outputs without per-tensor or per-channel scaling.
- The simpler option (FP16 GEMM via cuBLAS or CUTLASS with a Flash Attention kernel) already meets the performance target.

## Inputs the agent should gather first
- **Hardware target**: H100/H200 (sm_90a), RTX 4090/Ada (sm_89), or other. WGMMA instructions are H100-specific (sm_90a). Ada exposes FP8 via cuBLAS/CUTLASS but without WGMMA.
- **FP8 format**: E4M3 (4-bit exponent, 3-bit mantissa, max representable value ≈ 448) for forward pass weights and activations; E5M2 (5-bit exponent, 2-bit mantissa, max ≈ 57344) for gradients. Clarify which is needed.
- **Scaling strategy**: per-tensor (one scale for the whole tensor), per-channel/per-row (one scale per row or column), or delayed scaling (track amax history across steps and update scales at a fixed interval). Transformer Engine uses delayed scaling.
- **Accumulation dtype**: fp32 is mandatory for FP8 tensor core accumulation. DO NOT use fp16 accumulators with FP8 inputs — precision loss is severe.
- **Output dtype**: fp16, bf16, or fp32 after dequantization. This determines the epilogue.
- **Operation type**: GEMM only, GEMM + bias, attention (QK^T scaling), or custom fused operation.
- **Whether cuBLAS FP8 API or CUTLASS FP8 templates are acceptable**: these should be the default choice unless there is a specific reason for a custom kernel.

## Required reasoning process

1. **Select the FP8 format for each tensor role.** The industry-standard mapping is:
   - Forward pass weights: E4M3 (`__nv_fp8_e4m3`). Higher mantissa precision within a narrower range — suitable for static or slowly varying weight distributions.
   - Forward pass activations: E4M3. Same rationale.
   - Loss scale / gradients flowing backward: E5M2 (`__nv_fp8_e5m2`). Larger dynamic range is required for gradient values which can span many orders of magnitude.
   - Do not use E5M2 for weights or forward activations without explicit justification — the reduced mantissa precision (2 bits) degrades forward pass accuracy.

2. **Design the scaling strategy.** FP8 has a max representable magnitude of ~448 (E4M3). Any value exceeding this overflows to infinity or saturates. Scaling maps the input range to fit within FP8:
   - Compute `amax = max(|x|)` over the scaling granularity (tensor, channel, token).
   - Compute `scale = amax / fp8_max`, where `fp8_max = 448.0` for E4M3 or `57344.0` for E5M2.
   - Quantize: `x_fp8 = cast_fp8(x / scale)` (or equivalently `cast_fp8(x * scale_inv)` where `scale_inv = 1.0 / scale`).
   - The inverse scale `scale_inv` is used in the epilogue to dequantize: `out = acc_fp32 * scale_A_inv * scale_B_inv`.
   - For delayed scaling: use the amax from a previous iteration (e.g., the maximum over the last 16 steps) to set the scale at the start of each step. This avoids an all-reduce over the current batch but requires tracking amax history and checking for overflow after the fact.

3. **Choose between cuBLAS FP8, CUTLASS, and custom WGMMA.** In order of implementation effort:
   - **cuBLAS FP8 GEMM** (`cublasGemmEx` with `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2`): handles per-tensor scales via `cublasGemmStridedBatchedEx` or the `cublasLt` API with `CUBLASLT_MATMUL_DESC_SCALE_TYPE`. This is the right choice for standard GEMMs without custom epilogues.
   - **CUTLASS FP8 GEMM**: provides epilogue fusion support (per-channel scale, bias add, activation function). Use when the epilogue is non-standard or per-channel scales are required.
   - **Custom WGMMA kernel (Hopper only)**: use only when the operation is not supported by cuBLAS or CUTLASS, or when extreme fusion is required (e.g., fused attention with FP8 input). WGMMA programming requires PTX or CUTLASS's Cute abstractions and is architecturally specific to sm_90a.

4. **Design the quantization pass.** Quantization from fp32/fp16/bf16 to FP8 must:
   - Compute the amax over the correct granularity (per-tensor: full tensor; per-token: one pass per row).
   - Apply the scale factor and cast to FP8 using `__nv_cvt_float_to_fp8` (CUDA C++ API) or via PTX `cvt.rn.satfinite.e4m3x2.f32`.
   - Use `satfinite` saturation: values that overflow are clamped to the max FP8 value, not converted to infinity. Overflow to infinity propagates through the GEMM and produces NaN outputs silently.
   - For per-token (per-row) scaling: compute amax with a warp- or block-level reduction per row; write the inverse scale to a separate buffer.

5. **Design the GEMM and epilogue.** For a standard FP8 GEMM producing fp16 output:
   - Inputs: A (M x K, E4M3), B (N x K, E4M3), scale_A (scalar or [M] vector), scale_B (scalar or [N] vector).
   - Accumulation: fp32 tensor core accumulation.
   - Epilogue: `out[m][n] = fp16(acc_fp32[m][n] * scale_A_inv * scale_B_inv)` for per-tensor scales; `out[m][n] = fp16(acc_fp32[m][n] * scale_A_inv[m] * scale_B_inv[n])` for per-token/per-channel scales.
   - Optional: bias add in fp32 before final cast to fp16/bf16.

6. **Handle the amax feedback loop for training.** In training with delayed scaling:
   - After the forward GEMM, compute the amax of the output tensor.
   - Write this amax to a history buffer.
   - At the start of the next iteration, compute the new scale from `max(amax_history) / fp8_max`.
   - If the amax exceeds the current scale * fp8_max (overflow occurred): the step is invalid and must be skipped (loss scale reduction in gradient scaling parlance).
   - This loop is typically managed by Transformer Engine or similar library infrastructure; do not reimplement it from scratch for a training use case.

7. **Validate numerics.** After implementing:
   - Run a GEMM with known inputs (e.g., all-ones or random uniform in [-1, 1]) and compare to fp32 reference.
   - For well-scaled inputs, FP8 E4M3 GEMM should produce results within ~1% relative error compared to fp32 for typical weight/activation distributions encountered in transformer models.
   - Check for any +Inf, -Inf, or NaN in the output — these indicate overflow in the FP8 range (scale too small, or `satfinite` not used).
   - Sweep scale values and verify that the kernel handles `scale = 0` (degenerate case: input tensor is all zeros) without division by zero.

## Kernel design rules
- Always use `satfinite` (saturate to finite) conversion mode when casting to FP8. Overflow to infinity produces NaN in accumulation and silently corrupts the entire output.
- Accumulate in fp32, not fp16. FP8 tensor cores on Hopper produce fp32 accumulation by design; using fp16 intermediate accumulation throws away precision without any throughput benefit.
- Do not cast intermediate accumulation results back to FP8 at any point during the GEMM. Quantization applies only to the input tensors, not to intermediate or output values.
- For per-channel or per-token scales: load both scale vectors into registers before the epilogue loop and apply them as a fused multiply in the epilogue. Do not reload per-element from global memory inside the epilogue loop body.
- FP8 inputs should be loaded using 128-bit vectorized loads (16 bytes = 16 FP8 elements per load instruction) to saturate memory bandwidth.
- Tile sizes for FP8 GEMM should be at least as large as for INT8 (e.g., 128x128xK with K-tile ≥ 64) to hide memory latency and keep tensor cores fed.
- On Hopper (sm_90a), use the asynchronous copy pipelines (TMA / `cp.async`) to overlap A/B tile loads with WGMMA compute. Manual smem double-buffering without TMA is significantly less efficient.
- Do not use `__nv_fp8_e4m3` arithmetic operators (addition, multiplication) between FP8 values directly — no hardware supports FP8 element-wise arithmetic. All FP8 arithmetic goes through tensor core instructions; scalar FP8 arithmetic is emulated in software and is extremely slow.

## Correctness requirements
- The dequantization formula must be: `out[m][n] = acc_fp32[m][n] * (1.0f / scale_A) * (1.0f / scale_B)` for per-tensor scales, applied in this form to avoid loss of precision from computing `scale_A * scale_B` first when the scales differ by large factors.
- For per-row activation scales and per-column weight scales, the scale indexing must match: `scale_A[m]` for row m, `scale_B[n]` for column n.
- FP8 tensors must use `satfinite` saturation on conversion. Verify by checking that the output contains no `+Inf` or `NaN` when inputs are in the representable range.
- The amax computation must cover the full tensor (or the correct granularity). An amax computed over a partial view of the tensor produces an incorrect scale.
- INT32 overflow cannot occur in FP8 accumulation because the accumulation is in fp32. Verify only that fp32 accumulation is not silently downcast to fp16 in the epilogue before the scale multiplication.

## Performance requirements
- On H100 (sm_90a), FP8 tensor core peak throughput is 3958 TFLOPS (e8m0 / fp8 dense), roughly 2x fp16 TF32, and 4x the tf32 throughput. A well-implemented FP8 GEMM should reach >65% of this peak for large M, N, K (≥ 4096).
- The quantization pass (fp16 → FP8) should be memory-bandwidth-bound and not introduce significant compute overhead. Fuse the quantization into the previous operation's epilogue when possible rather than launching a separate kernel.
- The amax reduction for per-tensor scaling has O(N) cost. For very small tensors (N < 8192), the amax overhead can dominate. In this case, consider per-tensor delayed scaling.
- Dequantization in the epilogue is a scalar multiply per output element — this is compute-cheap and should not be the bottleneck.
- Measure: compare FP8 kernel throughput (TFLOPS/s) against cuBLAS fp16 GEMM at the same M/N/K to confirm the expected ~2x benefit. If FP8 is not faster than fp16, the kernel is likely memory-bandwidth bound (small batch) or incorrectly implemented (fp32 copies being made of the fp8 data).

## Output format
The final response must include:
1. **FP8 format choice**: E4M3 or E5M2 for each tensor, with justification.
2. **Scaling strategy**: per-tensor, per-channel, or delayed — with the amax computation and scale update formula written out explicitly.
3. **Implementation path**: cuBLAS, CUTLASS, or custom WGMMA, with justification.
4. **Quantization code**: explicit conversion with `satfinite` and scale application.
5. **GEMM and epilogue code**: accumulation in fp32, scale multiply, output cast.
6. **Accuracy validation**: comparison against fp32 reference with expected error bound.
7. **Performance estimate**: expected TFLOPS/s relative to hardware peak and fp16 baseline.

## Common failure modes
- **Overflow to infinity without `satfinite`**: values slightly outside the FP8 range are converted to `+Inf` instead of being clamped. This produces NaN in the first accumulation that involves an Inf value and silently corrupts the output. Always use `satfinite`.
- **Scale computed on the wrong granularity**: computing amax per-tensor when per-channel is needed, or computing amax on the weights instead of the activations. This produces a scale that is either too large (many overflows) or too small (quantization precision wasted on the outlier range).
- **Forgetting `scale_inv` in the epilogue**: dividing by scale instead of multiplying by `scale_inv = 1.0f / scale` introduces a second division instruction in the epilogue hot loop. Precompute and store `scale_inv` on the host.
- **Using fp16 accumulation**: explicitly or implicitly interpreting the fp32 accumulator as fp16 (e.g., from a spurious cast in the epilogue template parameter). The resulting output has quantization errors far larger than expected for FP8.
- **WGMMA descriptor initialization errors (Hopper)**: the WGMMA instruction requires carefully formatted shared memory descriptors. An incorrect descriptor field (wrong stride, wrong swizzle mode, or incorrect base pointer alignment) produces silent wrong results or hangs. Validate with a reference CUTLASS FP8 kernel first.
- **Per-row scale buffer indexing off by stride**: the scale tensor for per-token quantization has shape [M] but is accessed with a stride of `M * sizeof(float)` instead of `sizeof(float)` when the buffer is part of a larger allocation.
- **FP8 arithmetic operators used for element-wise ops**: writing `fp8_val * fp8_val` to compute an element-wise square triggers slow software emulation, not hardware FP8. All FP8 element-wise arithmetic should be done by upcasting to fp32 first.

## Review checklist
- [ ] Is E4M3 used for forward pass tensors and E5M2 for gradients (or is there a justified exception)?
- [ ] Is `satfinite` saturation mode used in every FP8 conversion call?
- [ ] Is accumulation in fp32 throughout the entire GEMM (no fp16 intermediate accumulators)?
- [ ] Does the amax computation cover the correct granularity (per-tensor, per-row, per-column)?
- [ ] Is `scale_inv` precomputed on the host and passed to the kernel (not recomputing `1.0 / scale` inside the kernel loop)?
- [ ] Is the epilogue applying `scale_A_inv * scale_B_inv` in the correct order with the correct indices?
- [ ] Has the kernel been validated against an fp32 reference for correctness (no NaN, no Inf, error within expected quantization bounds)?
- [ ] For Hopper WGMMA: have the shared memory descriptors been validated against a reference CUTLASS FP8 implementation?
- [ ] Has throughput been measured and compared to fp16 baseline to confirm the expected ~2x improvement?
- [ ] Is the quantization pass fused or is it a separate memory-bandwidth-bound kernel pass?
