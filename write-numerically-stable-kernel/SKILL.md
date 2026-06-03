# Skill: Write a Numerically Stable Kernel

## Purpose
Guide the agent through identifying numerical instability risks in a kernel's computation path and applying the correct stabilization strategy for each risk class.

## Use this when
- Writing or reviewing a kernel that contains reductions, accumulations, exponentials, logarithms, or divisions over floating-point inputs.
- A kernel produces correct results in fp32 but diverges when run in fp16 or bf16.
- A kernel computes variance, softmax, log-softmax, cross-entropy, or layer normalization — all of which have standard stable formulations that differ from the naive algebraic form.
- A kernel accumulates a large number of values (e.g., dot products over long sequences, large reduction trees).
- Results show inf, NaN, or unexpectedly large relative error relative to a double-precision reference.

## Do not use this when
- The computation is already fp32 or fp64 throughout, operates on bounded inputs, and correctness has been validated against a reference. Do not add unnecessary stabilization steps that cost performance without improving correctness.
- The instability is caused by a bug (wrong indexing, wrong reduction tree, missing synchronization) rather than a precision limitation. Fix the bug first.
- The application explicitly accepts approximate computation (e.g., stochastic rounding for training with intentional noise). Understand the tolerance before adding stabilization overhead.

## Inputs the agent should gather first
- The mathematical definition of the computation, written out explicitly — not just "softmax" but the exact formula being implemented.
- Input dtype (fp16, bf16, fp32, fp64) and whether that dtype is fixed or configurable.
- Expected input value range: are inputs bounded, potentially large, or potentially near zero?
- Accumulation length: how many values are summed or dot-producted? Longer accumulations accumulate more rounding error.
- Whether the output is consumed by a loss function, an activation, or another reduction — downstream consumers may have their own precision requirements.
- Hardware: which compute capability? On Hopper (sm_90), fp8 and bf16 tensor core paths have different precision characteristics than on Ampere.
- Whether correctness is validated against a double-precision reference or only against another fp16 run.

## Required reasoning process
1. **Enumerate all accumulation and reduction paths.** Walk through the kernel's computation and list every place where values are summed, multiplied, or combined iteratively. Each is a candidate for precision loss.

2. **Classify each risk.** Apply the following classification:
   - **Catastrophic cancellation**: subtraction of two nearly equal large numbers. Example: variance as `E[x^2] - E[x]^2`. Risk is high when x values cluster near a common mean.
   - **Overflow/underflow in exp or log**: `exp(x)` overflows for `x > ~89` in fp32, `x > ~11` in fp16. `log(x)` is undefined for `x <= 0`. Risk is high whenever inputs are unbounded.
   - **Accumulation error in fp16/bf16**: summing N values in fp16 accumulates O(sqrt(N) * eps) relative error for random inputs, but can be much worse for sorted or adversarial inputs. fp16 epsilon is ~1e-3; bf16 epsilon is ~8e-3.
   - **Inf propagation**: a single inf in a tensor propagates through multiplications and additions, masking the true result. Common in attention with large logits.
   - **Subnormal underflow**: values smaller than ~6e-8 (fp32) or ~6e-5 (fp16) become subnormal and lose relative precision. Products of small values can flush to zero.

3. **Choose mitigation for each risk.** Apply the correct strategy, not a generic "use fp32" instruction:
   - Catastrophic cancellation in variance: use the Welford online algorithm (single-pass, numerically stable) rather than the two-pass `E[x^2] - E[x]^2` formulation.
   - exp/log overflow: apply the shift-by-max trick before `exp`. For softmax: subtract `max(x)` before `exp(x - max(x))`. This does not change the mathematical result but prevents overflow.
   - Long accumulation in fp16: accumulate in fp32, cast input elements to fp32 before adding, cast the final result back to fp16. This adds a small number of register ops but prevents catastrophic error growth.
   - log-sum-exp: use `max(x) + log(sum(exp(x - max(x))))` rather than `log(sum(exp(x)))`.
   - Large attention logits causing inf: apply logit clipping or ensure Q/K are scaled by `1/sqrt(d_k)` before the softmax, not after.
   - bf16 vs fp16 choice: bf16 has the same dynamic range as fp32 (8 exponent bits) so it is far less prone to overflow/underflow than fp16 (5 exponent bits). When overflow is the primary risk and mantissa precision is secondary, bf16 is preferable to fp16.

4. **Write the stabilized formulation explicitly.** Do not write pseudocode — write the exact mathematical expressions that will appear in the kernel, annotated with dtypes at each step.

5. **Identify whether the stabilization requires additional kernel passes.** The log-sum-exp trick and stable softmax both require computing the maximum before the sum, which means either a two-pass design or an online algorithm. State this explicitly in the design.

6. **Verify with a double-precision reference.** Numerical stability can only be confirmed by comparing to a higher-precision reference, not by comparing two fp16 implementations against each other.

## Kernel design rules
- Never accumulate into fp16 or bf16 for reductions over more than ~32 elements. Accumulate in fp32. The additional register cost is almost always worth it.
- Never compute variance as `mean_of_squares - square_of_mean` (the naive two-pass formula). Use Welford's online algorithm for single-pass variance, or compute variance as `mean((x - mean(x))^2)` with a two-pass design that subtracts the mean first.
- For softmax and attention, always subtract the per-row maximum before applying `exp`. This is not optional — it is required for fp16 correctness on large inputs.
- For log-softmax, compute `x - max(x) - log(sum(exp(x - max(x))))` explicitly. Do not compute `log(softmax(x))` — the composition loses precision near zero.
- For cross-entropy loss, use the log-sum-exp stable formulation directly rather than computing softmax first and then taking the log.
- When combining bf16 and fp32 in the same kernel, cast to fp32 before any accumulation that involves more than a few terms. Cast back to bf16 only at the final store.
- Avoid branching on `isnan` or `isinf` inside hot loops as a stability measure — it does not fix the root cause and adds divergence overhead. Fix the upstream computation instead.
- For attention kernels: implement online softmax (the flash attention pattern) to compute the stable softmax and the weighted sum in a single pass without materializing the full attention matrix. This is both numerically correct and memory-efficient.

## Correctness requirements
- The kernel output must match a double-precision (fp64) reference implementation within a tolerance appropriate for the output dtype:
  - fp32 output: relative tolerance 1e-5, absolute tolerance 1e-8.
  - fp16 output: relative tolerance 1e-2, absolute tolerance 1e-3. Test with inputs that stress the fp16 range, including values near the fp16 max (~65504).
  - bf16 output: relative tolerance 5e-2, absolute tolerance 1e-3.
- Test with adversarial inputs: large values (>1000), small values (<1e-4), mixed signs, repeated identical values, and sequences where the maximum is at the last position (not the first).
- Confirm that the kernel produces no NaN or Inf for inputs that are finite (within the dtype's representable range), unless the mathematical result is genuinely undefined (e.g., `log(0)`).
- For online/streaming algorithms (Welford, online softmax), verify that the result matches the two-pass reference exactly on small test cases before trusting it on large inputs.

## Performance requirements
- Quantify the cost of the stabilization strategy before implementing it. Accumulating in fp32 instead of fp16 adds register pressure — estimate the register count increase and check whether it reduces occupancy.
- For two-pass algorithms (compute max, then compute sum), the second pass may benefit from L2 cache reuse if the input fits in L2. For inputs larger than L2, the two-pass approach costs one extra full read of the input.
- Online algorithms (single-pass Welford, online softmax) avoid the second pass at the cost of more arithmetic per element. For memory-bandwidth-bound kernels on large tensors, online algorithms are usually preferable.
- fp32 accumulation in a tensor core kernel has direct hardware support on sm_80+ (Ampere) via the `HMMA.F32` instruction variant. Do not implement fp32 accumulation manually in WMMA-based code — use the correct API.
- State the overhead honestly: "fp32 accumulation increases register usage by approximately X registers per thread, which is expected to reduce occupancy by Y% on sm_86."

## Output format
The agent should produce:

1. **Risk classification table**: a table listing each identified risk (catastrophic cancellation, overflow, underflow, accumulation error, inf propagation), where in the kernel it occurs, and which mitigation is applied.
2. **Stabilized mathematical formulation**: the exact formulas to be implemented, with dtypes annotated at each step, written before any code.
3. **Kernel implementation**: complete, compilable code implementing the stabilized formulation.
4. **Reference comparison test**: a test that computes the same operation in fp64 on CPU and compares to the kernel output using appropriate tolerances. The test must include at least one adversarial input case.
5. **Performance cost statement**: a brief explicit statement of what the stabilization costs (extra passes, registers, arithmetic) relative to the naive implementation.

## Common failure modes
- **Naive softmax overflow**: computing `exp(x)` before subtracting the row maximum. Produces Inf for any logit above ~89 (fp32) or ~11 (fp16). The fix is always to subtract `max(x)` first.
- **Variance via E[x^2] - E[x]^2**: catastrophic cancellation when x has low variance relative to its mean. Produces large relative error or negative variance values. Use Welford instead.
- **log(softmax(x)) for log-softmax**: `softmax(x)` produces values near 0 for the non-maximum classes, and `log` of near-zero values in fp16 is numerically poor. Use the direct log-sum-exp formula.
- **fp16 accumulation in long dot products**: a dot product of length 4096 in fp16 accumulates enough rounding error to degrade accuracy by 1–2 orders of magnitude relative to fp32. Always accumulate in fp32 for sequences longer than ~64.
- **bf16 mistaken for fp16**: bf16 does not have higher mantissa precision than fp16 (both have about 7–10 decimal digits effective precision for individual values), but bf16 has significantly larger dynamic range. Choosing bf16 to avoid overflow is correct; choosing it to improve mantissa precision is not.
- **inf propagation from attention masking**: applying a large negative mask value (e.g., -1e9) to padding positions before softmax can produce -inf + inf = NaN when a row is entirely masked. Use a finite but sufficiently large mask, or handle the all-masked case explicitly.
- **Online softmax implementation bug**: the online update rule for max and log-sum-exp has a specific correction factor. An incorrect online softmax update is hard to spot without testing on inputs where the max is encountered late in the sequence.
- **Missing test for adversarial input range**: testing only on `randn()` inputs with mean 0 and std 1 does not stress the precision limits. Explicitly test with scaled inputs in the range [100, 1000] and [-1000, -100].

## Review checklist
- [ ] Every reduction and accumulation path has been identified and its precision risk classified.
- [ ] No reduction over more than 32 elements accumulates in fp16 or bf16.
- [ ] Softmax and attention logits have the row maximum subtracted before `exp`.
- [ ] Variance is computed via Welford or two-pass (mean first, then mean of squared deviations), never as `E[x^2] - E[x]^2`.
- [ ] Log-softmax uses the direct log-sum-exp formula, not `log(softmax(x))`.
- [ ] The stabilized mathematical formulation is written out explicitly before the code, with dtypes annotated.
- [ ] The kernel is tested against a double-precision reference, not only against another fp16 implementation.
- [ ] Adversarial inputs (large values, small values, all-same values, mixed signs) are included in the test suite.
- [ ] The performance cost of each stabilization step is stated explicitly.
- [ ] No NaN or Inf appears in the output for finite inputs within the dtype's representable range (unless the math is genuinely undefined).
- [ ] bf16 vs fp16 choice is explicitly motivated by the dominant risk (dynamic range vs mantissa precision).
