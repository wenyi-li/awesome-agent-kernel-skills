# Skill: Write a Triton Sampling Kernel

## Purpose

Guide the agent through implementing a Triton kernel for LLM decode-time token sampling: take a `[batch, vocab]` logits tensor, apply per-request temperature, top-k, and top-p (nucleus) filtering, renormalize, and draw one token per request. This is the last hot kernel on every decode step — it runs once per generated token, so latency directly translates into tokens/second.

---

## Use this when

- You need a sampling strategy that vLLM, SGLang, TGI, or FlashInfer do not expose (typical-p, mirostat, classifier-free guidance, fused repetition penalty, structured-generation logit bias, contrastive decoding).
- You need heterogeneous per-request sampling — each request has its own `T`, `k`, `p`, seed, and possibly its own logit-bias mask — and you want one fused kernel rather than N samplers.
- You are willing to special-case the greedy path (T == 0 or top-k == 1) to skip softmax and sort.
- Decode batch size is large enough (B >= 8) that one-program-per-request is worthwhile. For B == 1, a CPU-side argmax/multinomial is usually fine.

---

## Do not use this when

- A vendor sampler covers your case. vLLM's `Sampler` and FlashInfer's `top_k_top_p_sampling_from_probs` are heavily tuned and handle edge cases (extremely peaked distributions, ties, deterministic argmax fallback). Re-implementing without a concrete reason is a likely source of subtle bias.
- You need only argmax. `logits.argmax(-1)` from PyTorch is competitive and avoids every numerical pitfall in this skill.
- The strategy requires global communication across the batch (beam search, speculative decoding verification). Those are not multinomial-per-request.
- You need provably uniform reproducibility across hardware. RNG semantics, cumsum reduction order, and sort tie-breaking are all platform-dependent.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Vocab size V.** Typical: 32k (Llama-2), 128k (Llama-3), 256k (Gemma). Determines whether the row fits in one BLOCK or needs multi-block streaming.
2. **Batch size B.** Number of concurrent requests in the decode step. Each request maps to one program.
3. **Per-request sampling params.** Are `T`, `k`, `p` scalars (uniform) or tensors of shape `[B]` (heterogeneous)? Heterogeneous is the realistic case in continuous-batching servers.
4. **Logits dtype.** Almost always fp16 or bf16 from the LM head. Sampling internally promotes to fp32.
5. **RNG source.** Stateful Philox seed/offset (advanced once per decode step) or a precomputed `[B]` tensor of uniforms. Stateful is more flexible; precomputed is simpler and easier to test.
6. **Greedy fallback policy.** Is `T == 0` legal? Is `top_k == 1` legal? Both must short-circuit to argmax.
7. **Logit bias / mask.** Per-request additive bias (e.g., grammar-constrained decoding) is added to logits before temperature scaling.
8. **Maximum top_k.** A hard upper bound (e.g., `K_MAX = 1024`) lets you pick a sort strategy at compile time. Without a bound, you cannot size a fixed on-chip sort buffer.

---

## Required reasoning process

1. **Parallelism axis.** Sampling is embarrassingly parallel across the batch. One Triton program per request: `req_idx = tl.program_id(0)`, grid `(B,)`. All work for one request — temperature, softmax, top-k, top-p, multinomial draw — happens inside that program. No inter-program communication.

2. **Vocab tiling.** Choose `BLOCK_V` as a power of 2 (4096 or 8192 typical). For V <= BLOCK_V, the row fits in one block. For V > BLOCK_V (common at V >= 32k), the program loops over `ceil(V / BLOCK_V)` chunks. Running state (max, sum, top-k buffer) lives in registers across iterations.

3. **Short-circuit greedy.** Read `T = T_ptr[req_idx]` and `k = k_ptr[req_idx]`. If `T == 0.0` or `k == 1`, run an argmax-only pass: streaming reduction over V tracking `(max_logit, max_idx)`, then store `max_idx`. Skip softmax, sort, and RNG. Tie-break deterministically by smaller index.

4. **Temperature and logit bias before softmax.** Inside each chunk: `chunk = (chunk + bias_chunk) / T`. Promote to fp32 first — when T is small (e.g., 0.1 multiplies logits by 10), fp16 dynamic range is insufficient.

5. **Numerically stable softmax over V.** Same problem as `write-triton-softmax-kernel`. For V <= BLOCK_V, single-pass online softmax. For V > BLOCK_V, online algorithm: track `running_max` and `running_sum`, rescale `running_sum` by `exp(old_max - new_max)` whenever the max updates. Accumulate in fp32.

6. **Top-k before top-p.** Top-p alone requires sorting all V probabilities — too expensive at V >= 32k. Apply top-k first with a moderate K (50–1024), then top-p on the K survivors. If the user's k is unset, treat it as `K_MAX` — caps sort cost without changing semantics for typical p.

7. **Top-k as streaming partial sort.** Maintain a fixed-size sorted buffer of size K in registers. For each streamed value, if it exceeds the current min of the buffer, replace the min and re-sort. For `K <= 256`, an in-register sorted array with binary insertion is competitive on Triton. For larger K, prefer radix-select or fall back to FlashInfer — pure Triton `tl.sort` over the full vocab is too slow at V = 128k. Output: `top_probs[K]` and `top_indices[K]`, sorted descending.

8. **Top-p on the K survivors.** Read `p = p_ptr[req_idx]`. Cumsum over `top_probs` (already sorted descending). Find the smallest prefix with `cumsum >= p`; keep that prefix, zero the rest. Always keep at least one token — if `top_probs[0] > p`, the top-1 alone forms the nucleus. Missing this guard produces an all-zero distribution.

9. **Renormalize.** After top-k and top-p masking, surviving probs no longer sum to 1. Divide by `sum(top_probs_after_masking)`. Skipping this is the single most common sampling-bias bug.

10. **Multinomial draw.** Generate `u ~ U(0, 1)`. Walk the sorted, masked, renormalized `top_probs` accumulating cumsum; pick the first index `j` with `cumsum >= u`. Chosen token is `top_indices[j]`. With Philox: `tl.rand(seed, offset + req_idx)` or precomputed `u_ptr[req_idx]`. The offset must advance every decode step — reusing the same offset gives identical samples each step.

11. **Store.** `tl.store(out_ptr + req_idx, token_id)`.

---

## Kernel design rules

- One program per request. No cross-request communication. Grid is `(B,)`.
- All reductions, softmax, cumsum, and renormalization run in fp32. Logits arrive as fp16/bf16; cast on load. Keep `probs` in registers — do not write back to global memory.
- `BLOCK_V` and `K_MAX` are `tl.constexpr`. `K_MAX` is the compile-time upper bound on top-k; the runtime `k` is masked against it. Without a compile-time bound, the sort buffer cannot be sized.
- The greedy path (`T == 0` or `k == 1`) is a separate `if` branch. Do not share code with the sampling path — determinism, dtype handling, and exit conditions differ.
- Out-of-bounds vocab loads must use `other=-float('inf')` so they collapse to zero probability.
- Logit bias is added in fp32 before division by T. Adding after softmax is incorrect — it biases probabilities directly, not the energy.
- The cumsum used for the multinomial draw must be over the renormalized array. A stale cumsum (computed before renormalization) is one of the easiest bugs to ship.
- RNG: Philox via `tl.rand(seed, offset)`. The `(seed, offset)` pair must be unique per `(request, decode_step)`. Common pattern: `seed = global_seed`, `offset = step_idx * B + req_idx`.
- For heterogeneous batches, load every per-request param (`T`, `k`, `p`, `seed`) from a `[B]` tensor inside the program. Do not pass them as kernel scalars unless the batch is genuinely homogeneous.

---

## Correctness requirements

- **Renormalization after filtering is mandatory.** After top-k and top-p, surviving probs do not sum to 1. A cumsum-vs-uniform draw against an unnormalized vector is silently biased. Always divide by the post-filter sum.
- **Top-p must keep at least one token.** If `top_probs[0] >= p`, the nucleus is the top-1 alone. Use a guard so the top-1 is never masked out.
- **Greedy path must be exact.** When `T == 0.0`, no softmax, no RNG, no sort — pure argmax with smallest-index tie-breaking. Softmax with T = 0 is undefined; short-circuit before division.
- **Max subtraction before exp.** Same as softmax: subtract running max before `exp` to prevent fp32 overflow above ~88. After temperature scaling with small T, raw logits often exceed this.
- **Top-k masking is consistent.** Pick one representation: temperature → softmax → top-k on probs → top-p on probs → renormalize → sample. Applying top-k as a logit mask before softmax is also valid but requires `-inf` masking, with max-subtraction running after.
- **RNG offset advances every decode step.** A common bug: same offset every step, same `u` every step. The driver must increment between calls.
- **Cumsum precision.** Cumsum over K = 1024 entries in fp16 drifts: each value ~1/K and `K * eps_fp16` is order 0.5. The final entry can be 0.5 instead of 1.0. Cumsum must run in fp32.
- **Tie-breaking in argmax.** Equal logits are common with bias masks that set allowed tokens to a fixed value. Define and implement smallest-index-wins; `tl.max`'s tie order is not part of its contract.

---

## Performance requirements

The agent must reason through:

- **Memory bandwidth.** Sampling reads each logit at most twice (max, then masked exp/sort) and writes one int per request. At B = 64, V = 128k, fp16 logits this is ~16 MB per pass — small enough that per-request overhead, not HBM bandwidth, dominates at high B.
- **Sort cost dominates.** Streaming partial top-k over V is O(V log K). At V = 128k, K = 1024 that is ~2M comparisons per request. Tightening K via a hard upper bound is the highest-leverage optimization.
- **Per-request divergence.** Heterogeneous params mean programs do different work — one request greedy, another full top-k+top-p. SM-level load balancing suffers. If the batch is bimodal (half greedy, half sampling), dispatch them as two kernels.
- **fp32 cost.** All reductions and the cumsum are fp32. Non-negotiable for correctness; do not "optimize" by dropping to fp16.
- **Compare against vendor.** Benchmark against vLLM's `Sampler` or FlashInfer's `top_k_top_p_sampling_from_probs`. Matching their throughput at V = 128k is a real result; significantly slower usually means top-k is being implemented as a full `tl.sort`.
- **Latency budget.** End-to-end decode for a 7B model on H100 is ~10 ms/token. The sampler should be < 5% of that — sub-0.5 ms per decode step for the whole batch.

---

## Output format

The agent should produce:

1. **The Triton kernel** with `@triton.jit`, taking: `logits_ptr`, `out_token_ids_ptr`, per-request param pointers (`T_ptr`, `k_ptr`, `p_ptr`, optionally `seed_ptr`, `offset`), optional `logit_bias_ptr`, `B`, `V`, `logits_row_stride`, and `BLOCK_V: tl.constexpr`, `K_MAX: tl.constexpr`.
2. **The greedy short-circuit path** as an explicit branch on `T == 0.0` or `k == 1`.
3. **The sampling path**: temperature scale → online softmax → streaming partial top-k → top-p prefix-cumsum mask → renormalize → multinomial draw via cumsum-vs-uniform.
4. **The Python launcher** that builds the per-request param tensors, picks `BLOCK_V` and `K_MAX` based on V and the user-specified max k, and computes the grid as `(B,)`. Driver code increments the RNG offset between calls.
5. **A correctness test** comparing against a reference PyTorch implementation: for fixed seed, the chosen token must match the reference for several `(T, k, p)` settings. Also test the limit cases: `T = 0` matches argmax; `k = 1` matches argmax; `k = V` and `p = 1.0` matches plain multinomial sampling from softmax.
6. **A statistical test** for the sampling path: with a known logit distribution, draw N >> 1 samples and verify the empirical histogram matches the post-filter probabilities within chi-squared tolerance.
7. **Documented assumptions**: max supported K, behavior on `T < 0` or `p > 1` (reject or clamp), tie-breaking policy.

---

## Common failure modes

- **Skipped renormalization.** Top-k or top-p applied, surviving probs sent directly to the cumsum-vs-uniform draw without dividing by the post-filter sum. Outputs drift from the intended distribution; invisible without statistical tests.
- **Stale RNG offset.** Same `(seed, offset)` every decode step. Same `u`, same sample relative to the same probabilities. Manifests as repetition or as outputs that look almost-greedy without being greedy.
- **Greedy path bypassed at T = 0.** `logits / T` produces inf/NaN, softmax produces NaN, multinomial picks whatever lane resolves first. Always test `T == 0.0` and route to argmax.
- **Top-k mask before max-subtraction with wrong fill.** Masked-out entries left at original logits or set to 0 corrupt the max reduction. Use `-inf` (or a large negative) so masked entries collapse to zero probability and do not skew the max.
- **fp16 cumsum.** See Correctness — drifts to ~0.5 over K = 1024. Always fp32.
- **Top-p with no min-keep guard.** Peaked distribution + tight `p` (e.g., `top_probs[0] = 0.95`, `p = 0.9`) with naive `cumsum > p` excludes the top-1 and produces an all-zero nucleus. Always keep at least one token.
- **Full `tl.sort` over the vocab.** Compile time and runtime blow up at V = 128k. Use a streaming partial sort with a fixed `K_MAX` buffer.
- **Heterogeneous batching ignored.** Kernel takes a scalar `T` and applies it to every request. The moment requests have different temperatures (normal case), all but one request samples wrong. Thread per-request params through `[B]` tensors.
- **Logit bias added after softmax.** Adding to probabilities does not produce the bias-conditioned distribution. Symptom: structured-generation grammars allow tokens that should be masked.
- **Non-deterministic argmax tie-breaking.** Equal logits (common with bias masks). The kernel returns whichever lane resolved first, which can vary across runs. Implement smallest-index-wins explicitly.

---

## Review checklist

- [ ] Greedy path (`T == 0.0` or `k == 1`) short-circuits to argmax with deterministic tie-breaking; no softmax, no RNG, no sort on this path.
- [ ] Logits are cast to fp32 before temperature scaling, max-subtraction, exp, sum, cumsum, and renormalization.
- [ ] Out-of-bounds vocab loads use `other=-float('inf')`.
- [ ] Top-k is implemented as a fixed-size streaming partial sort, with K bounded by a compile-time `K_MAX`.
- [ ] Top-p is applied after top-k, on the sorted survivors, and always keeps at least the top-1 token.
- [ ] Probabilities are renormalized after top-k and top-p masking, before the multinomial draw.
- [ ] The cumsum used for the draw is over the renormalized, masked probs — not a stale pre-filter cumsum.
- [ ] RNG `(seed, offset)` is unique per `(request, decode step)`; the driver advances the offset between calls.
- [ ] Logit bias, if any, is added on logits before division by T, never on probabilities.
- [ ] Per-request `T`, `k`, `p`, and seed are loaded from `[B]` tensors inside the kernel; no scalar params for heterogeneous batches.
- [ ] Correctness test covers: `T = 0` matches argmax, `k = 1` matches argmax, `k = V, p = 1.0` matches reference multinomial-from-softmax, and a chi-squared empirical histogram check.
- [ ] Behavior on illegal inputs (`T < 0`, `p > 1`, `k = 0`) is documented and either rejected at the launcher or clamped consistently.
- [ ] Performance is compared against vLLM `Sampler` or FlashInfer top-k/top-p sampling, and the result is reported as a measurement, not a claim.
