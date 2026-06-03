# Skill: Write CUDA LayerNorm Kernel

## Purpose
Guide the agent through designing and implementing a correct, efficient CUDA LayerNorm (and RMSNorm) kernel, covering mean/variance computation strategies, Welford online accumulation, epsilon placement, affine transform application, backward pass structure, and decomposition for non-power-of-two hidden dimensions.

## Use this when
- You need a custom LayerNorm kernel with a non-standard normalization axis, fused activation, or non-standard epilogue
- You are implementing RMSNorm (no mean subtraction, only variance normalization) and need a custom kernel
- You need the backward pass and require explicit control over the gradient computation for numerical accuracy or fusion
- The hidden dimension size does not match the assumptions of library implementations, or you need custom handling of non-power-of-two sizes
- You are fusing LayerNorm with a preceding or following operation and need to avoid extra memory round-trips

## Do not use this when
- Standard forward-only LayerNorm with affine parameters on common hidden dimensions: use `torch.nn.LayerNorm` (which calls into cuDNN or a well-tuned CUDA kernel) or APEX's `FusedLayerNorm`
- You are using PyTorch with autograd: the built-in LayerNorm already has a correct, optimized backward pass
- The normalization is applied over batch dimensions (BatchNorm): this skill covers LayerNorm (normalization over the last D dimensions within a single sample)

## Inputs the agent should gather first
- **Input shape**: [batch, seq, hidden] or [N, D] where D is the normalization dimension. What is the hidden dimension size? Is it always the same or variable at runtime?
- **Normalization axis**: which dimension(s) are normalized? LayerNorm normalizes over the last D dimensions; confirm this assumption.
- **Dtype**: fp32, fp16, or bf16 for input/output. What dtype are the gamma/beta parameters? (Usually fp32 even when input is fp16.)
- **Affine parameters**: are gamma (scale) and beta (shift) parameters present (LayerNorm) or absent (plain normalization)?
- **Epsilon**: what value? Typically 1e-5 for LayerNorm, 1e-6 for RMSNorm. Is it applied inside or outside the sqrt?
- **Forward only vs forward + backward**: does the kernel need to support gradient computation? If yes, what tensors need to be saved from the forward pass?
- **RMSNorm variant**: should mean subtraction be skipped (RMSNorm only divides by the root mean square, not the standard deviation)?
- **Hardware target**: SM architecture.

## Required reasoning process
1. **Determine the normalization work per row.** Each row has D elements (the hidden dimension). A single row is normalized independently of all other rows. Assign one thread block (or one warp for small D) to each row. Grid size = number of rows = N * seq_len (or batch * seq if 3D input).

2. **Choose the accumulation strategy for mean and variance.**
   - **Two-pass**: first pass computes the mean `mu = sum(x) / D`; second pass computes the variance `var = sum((x - mu)^2) / D`. Requires two reads of the row. Straightforward to implement correctly.
   - **Welford online algorithm**: single pass computes both mean and variance simultaneously using the recurrence `mean_new = mean_old + (x - mean_old) / n`, `M2_new = M2_old + (x - mean_old) * (x - mean_new)`, with `var = M2 / D`. Requires one read of the row. Numerically more stable than the naive `E[x^2] - E[x]^2` formula, which suffers from catastrophic cancellation when variance is small relative to the mean.
   For production kernels: use Welford for its single-pass and numerical stability properties. For pedagogical or simple cases: two-pass is easier to verify.

3. **Design the warp+block reduction for Welford.** Welford accumulators `(count, mean, M2)` must be merged at the warp and block level. Parallel Welford merge: given two accumulators `(n_a, mean_a, M2_a)` and `(n_b, mean_b, M2_b)`, the combined accumulator is:
   ```
   n = n_a + n_b
   delta = mean_b - mean_a
   mean = mean_a + delta * n_b / n
   M2 = M2_a + M2_b + delta * delta * n_a * n_b / n
   ```
   This merge is used at each step of the warp shuffle reduction and the smem block reduction. For simple sum reductions (two-pass approach), standard sum reductions suffice.

4. **Handle non-power-of-two D.** The reduction pattern must correctly handle D values that are not multiples of the block size. Threads with index `threadIdx.x + k * blockDim.x >= D` must not read from global memory and must contribute the identity accumulator to the Welford merge.

5. **Compute the normalization.** After the reduction:
   ```
   mean = accumulated_sum / D   (or from Welford directly)
   var = M2 / D                 (from Welford, or second-pass sum-of-squared-deviations / D)
   inv_std = rsqrtf(var + epsilon)
   ```
   Each element is then normalized: `y_i = (x_i - mean) * inv_std`. For RMSNorm: skip mean computation; `inv_std = rsqrtf(mean_of_squares + epsilon)`.

6. **Apply the affine transform.** If gamma and beta are present: `y_i = y_i * gamma[i] + beta[i]`. The gamma and beta arrays have size D; each thread reads the gamma/beta value for its assigned position. These are typically loaded from global memory once per thread per call.

7. **Epsilon placement.** Epsilon must be added inside the sqrt (i.e., `rsqrtf(var + epsilon)`, not `rsqrtf(var) + epsilon`). The latter formula does not prevent division by zero and provides incorrect normalization when variance is near zero.

8. **Handle the backward pass (if required).** The LayerNorm backward computes:
   - `dL/dgamma_i = sum over rows of (normalized_x_i * dL/dy_i)`
   - `dL/dbeta_i = sum over rows of dL/dy_i`
   - `dL/dx_i = (1/D) * inv_std * (D * dL/dy_i_hat - sum(dL/dy_hat) - x_hat_i * sum(dL/dy_hat * x_hat))`
   where `x_hat_i = (x_i - mean) * inv_std` is the normalized input.
   The backward pass needs the normalized input `x_hat` or the saved `mean` and `inv_std` from the forward pass. Save those in the forward pass if backward is needed.

9. **Handle fp16/bf16 inputs.** Convert fp16/bf16 inputs to fp32 immediately on load. All accumulation (sum, sum-of-squares, mean, variance) must be in fp32. Gamma and beta should be fp32 regardless of input dtype. The output is converted back to fp16/bf16 at the store step.

## Kernel design rules
- Always use `rsqrtf(var + epsilon)` — epsilon inside the sqrt argument. Never add epsilon outside the sqrt.
- For fp16/bf16 inputs: accumulate entirely in fp32. Convert on load, convert back on store.
- Welford parallel merge must be used when combining warp-level accumulators into block-level accumulators. Do not sum means independently — it produces the wrong result when different threads hold different counts.
- Thread block size: 128 or 256 threads for D >= 128. For D < 32, use one warp per row (blockDim.x = 32, grid = N). For D in [32, 128], one warp or 128-thread block per row.
- Gamma and beta reads should be coalesced: consecutive threads read consecutive elements of gamma and beta. This is naturally the case when threadIdx.x indexes directly into the D dimension.
- Do not use `__syncthreads` more than necessary. For Welford with a pure warp (D <= 32), no smem is needed at all.
- If the hidden dimension D exceeds the block size, each thread processes multiple elements in a stride loop: `for (int i = threadIdx.x; i < D; i += blockDim.x)`. This must be done consistently for both the accumulation pass and the normalization write pass.
- RMSNorm: compute `mean_sq = sum(x_i^2) / D`, then `inv_rms = rsqrtf(mean_sq + epsilon)`. Apply `y_i = x_i * inv_rms * gamma[i]`. No beta term for RMSNorm.

## Correctness requirements
- **Epsilon placement**: epsilon must be inside the variance sqrt argument. Verify by inspection that the formula is `rsqrtf(var + eps)`.
- **Welford merge correctness**: the parallel merge formula must be used when combining partial Welford accumulators across warps or threads. Averaging the means without weighted combination is incorrect and produces wrong variance for non-uniform load distributions.
- **fp16 variance underflow**: for fp16 inputs with small activation values, the variance computed in fp16 may underflow to 0, making `inv_std = rsqrtf(0 + epsilon) = 1/sqrt(epsilon)`, which is a valid but imprecise result. This is prevented by accumulating in fp32.
- **Thread synchronization**: the smem write (warp leaders writing Welford partials) must be followed by `__syncthreads()` before any reads.
- **Boundary handling**: for D not a multiple of blockDim.x, threads with out-of-range indices must not read gamma/beta or the input tensor, and must contribute identity accumulators (count=0 for Welford).
- **Gamma/beta indexing**: gamma[i] and beta[i] are indexed by position within the D dimension, not by the global thread index. Verify that the index into gamma/beta is `threadIdx.x + k * blockDim.x` (the position within the row), not the global thread index.
- **Backward pass saved tensors**: if the backward pass recomputes `x_hat` from saved `mean` and `inv_std`, verify that the saved tensors use the same epsilon and have the same dtype as used in the backward computation.

## Performance requirements
- LayerNorm is memory-bandwidth-bound for large hidden dimensions. Target at least 80% of device peak memory bandwidth.
- A single read of the input row is preferred (Welford). Two reads (two-pass) is acceptable for clarity but costs roughly 2x memory bandwidth for the input.
- Vectorized loads: use `float4` for fp32 inputs where D is divisible by 4, or `half2`/`half4` for fp16 inputs where alignment allows. This reduces load instruction count and improves effective bandwidth.
- For the forward pass, gamma and beta reads add D * 2 extra reads (assuming they fit in L2 cache across calls). For large batch sizes, they will be cached; for single-sample inference, they may not be. This is typically a small fraction of total memory traffic.
- Occupancy: LayerNorm kernels use a small amount of smem (one Welford triplet per warp per block). Occupancy should be high (50–100% theoretical). If register usage is high due to Welford state, consider annotating with `__launch_bounds__`.
- For the common transformer case (D = 768, 1024, 2048, 4096): ensure the thread block size divides D or handles the non-divisible case with a stride loop. Power-of-two block sizes (128, 256) are preferred because they simplify the reduction tree.

## Output format
The final response must include:
1. **Decomposition decision**: rows-per-block assignment, thread block size, justification.
2. **Algorithm choice**: Welford online or two-pass, with reasons.
3. **Kernel code**: complete, compilable CUDA kernel for the forward pass. If backward is requested, a second kernel.
4. **Warp+block Welford reduction helper**: device function implementing the parallel Welford merge.
5. **Host dispatch**: grid/block dims, kernel call, saved tensor list for backward.
6. **Correctness notes**: epsilon placement, fp16 accumulation strategy, boundary handling.
7. **RMSNorm variant** (if requested): separate kernel or a compile-time template flag.
8. **Known limitations**: maximum D for the chosen block size, non-divisible D behavior, backward pass numerical sensitivity.

## Common failure modes
- **Variance underflow at fp16**: computing variance in fp16 when input values are small (e.g., near-zero activations) causes `var = 0`, making `inv_std = 1/sqrt(epsilon)`. This is mathematically valid but loses all information. Fix: always accumulate variance in fp32.
- **Incorrect epsilon placement (outside sqrt)**: `1.0f / (sqrtf(var) + epsilon)` does not prevent division by zero when `var` is exactly 0 — epsilon must be inside the argument to `rsqrtf`. This is a subtle but important difference.
- **Wrong gamma/beta broadcast**: gamma/beta are 1D tensors of size D, not size `N * D`. A common bug indexes them with the global thread index instead of the position-within-row index, causing gamma[0] to be applied to the first element of every row, gamma[1] to the second element, etc. — but only if the global thread index happens to match the D index, which it may not for all rows.
- **Non-parallel-merge of Welford accumulators**: summing partial means (each warp's local mean) without weighting by the count produces a biased estimate of the true mean when each warp processes a different number of elements (which happens when D is not a multiple of `blockDim.x * numWarps`). Fix: use the weighted parallel Welford merge formula.
- **Missing `__syncthreads` between smem write and read**: warp leaders write Welford partials to smem, but the first warp starts reading before all warps have written. Fix: `__syncthreads()` after the write, before the read.
- **Two-pass mean drift**: computing the variance in the second pass using `(x_i - mean)^2` with a mean that was computed in floating-point can accumulate errors for large D. Welford avoids this by maintaining a numerically stable running estimate.
- **Backward pass shape mismatch**: gamma gradients must be summed over the batch and sequence dimensions (all rows). A common bug reduces only over the batch but not the sequence, producing wrong gamma gradients for 3D inputs.

## Review checklist
- [ ] Is epsilon placed inside the `rsqrtf` argument — `rsqrtf(var + epsilon)` — not outside?
- [ ] For fp16/bf16 inputs: is all accumulation (sum, variance, Welford state) in fp32?
- [ ] Is the Welford parallel merge formula used when combining partial accumulators across warps?
- [ ] Is gamma/beta indexed by the position within the D dimension (not the global thread index)?
- [ ] Are threads beyond position D excluded from global reads and contributing identity accumulators?
- [ ] Is there a `__syncthreads()` after warp leaders write Welford partials to smem?
- [ ] For D not divisible by blockDim.x: does the stride loop handle the final partial tile correctly?
- [ ] For RMSNorm: is mean subtraction absent, and is the accumulator computing sum-of-squares (not sum)?
- [ ] For the backward pass: are `mean` and `inv_std` saved from the forward pass, or is `x_hat` explicitly saved?
- [ ] Has the kernel been validated against `torch.nn.LayerNorm` on: D=64, D=128, D=512, D=768, D=1024, D=2049 (non-power-of-two), small variance (near-constant rows), zero-variance rows?
- [ ] Is APEX FusedLayerNorm or cuDNN LayerNorm considered and explicitly noted for standard use cases?
