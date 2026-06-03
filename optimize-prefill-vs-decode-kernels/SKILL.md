# Skill: Optimize Prefill vs Decode Kernels

## Purpose
Guide the agent through choosing and tuning kernels for the prefill phase versus the decode phase of LLM inference. The two phases have fundamentally different arithmetic intensity, occupy different sides of the roofline, and respond to different optimizations. Continuous batching and speculative decoding shift the balance and must be reasoned about explicitly.

## Use this when
- Designing or selecting kernels for an LLM inference engine (matmul, attention, KV cache access, normalization fusion).
- An existing engine reuses a single kernel across both phases and decode latency or throughput is poor.
- Tuning tile shapes, split-K, or persistent kernels separately for prefill and decode.
- Adding speculative decoding, lookahead decoding, EAGLE, or Medusa, where decode shifts toward a small-batch GEMM regime.
- Implementing chunked prefill or hybrid prefill/decode batches and choosing per-batch kernels.
- Diagnosing why TTFT is good but inter-token latency is bad, or vice versa.

## Do not use this when
- An off-the-shelf kernel (cuBLASLt heuristics, vendor attention) already saturates the relevant resource. Trust the library until profiling shows a gap.
- The bottleneck is outside the kernel layer (Python overhead, scheduler, RPC, host-device copies). Fix the host path first.
- Training, not inference. Training runs in a different regime — large M dominates throughout, so this skill's decode reasoning does not apply.
- A non-batched, latency-insensitive offline workload where one configuration is good enough.

## Inputs the agent should gather first
- **Phase**: prefill, decode, or mixed (continuous batching). For mixed, the per-request M distribution.
- **Effective M per request**: prefill M = prompt length (typically 256–32k+); decode M = 1, or M = k for speculative decoding (typical k = 2–8, lookahead trees up to 32–64).
- **Batch size B**: number of concurrent requests in this kernel call. With continuous batching, the kernel sees the *aggregated* M across all requests.
- **Model dimensions and dominant K**: hidden size N, head dim d_h, num heads h, intermediate FFN size; K = hidden for QKV/output proj, K = intermediate for FFN down proj, K = head_dim for attention.
- **dtype** of weights, activations, KV cache, and matmul accumulator (fp16, bf16, fp8, int8, int4 weight-only).
- **KV cache layout**: paged (PagedAttention), contiguous, MQA, GQA group size. Cache bytes per token = `2 * num_layers * num_kv_heads * head_dim * dtype_bytes`.
- **Target hardware**: HBM bandwidth (H100 SXM ≈ 3.35 TB/s, H200 ≈ 4.8 TB/s, MI300X ≈ 5.3 TB/s, A100 80GB ≈ 2.0 TB/s), tensor core gen, FP8/INT8 support.
- **Whether speculative decoding is active**, draft model, verified-tokens-per-step distribution (acceptance rate × tree size).
- **SLO targets**: TTFT, p50/p99 inter-token latency, throughput. The right kernel for throughput is rarely the right one for tail latency.

## Required reasoning process
1. **Identify the phase and resolve the effective M.** Prefill: M = seq_len (per request) or sum of seq_lens (mixed batch). Decode: M = 1 per request, M = B for plain decode batch of B requests, M = k * B for speculative decode with tree size k. Write down M, K, N for the dominant matmul before going further.

2. **Compute arithmetic intensity for each dominant matmul.** For an M×K @ K×N matmul: FLOPs ≈ `2*M*K*N`, bytes (fp16) ≈ `(M*K + K*N + M*N) * 2`, intensity = FLOPs / Bytes.
   - Prefill, M=2048, K=N=8192, fp16: ~1300 FLOPs/byte.
   - Decode, M=1, K=N=8192, fp16: ~1 FLOP/byte (dominated by weight load).
   - Spec-decode, M=8, K=N=8192, fp16: ~8 FLOPs/byte — still memory-bound, but closer to the knee.

3. **Locate against the hardware roofline.** Compute the machine balance: `peak_FLOPs / peak_bandwidth`. For H100 SXM (fp16 tensor core): ~989 TFLOP/s / 3.35 TB/s ≈ 295 FLOPs/byte. For MI300X: ~1300 TFLOP/s / 5.3 TB/s ≈ 245. Any matmul with intensity below the machine balance is memory-bound. Decode (~1 FLOP/byte) sits roughly two orders of magnitude below the ridge — it cannot be made compute-bound by tuning alone.

4. **Pick the kernel family by phase.**
   - **Prefill matmul**: tiled GEMM with large output tiles (BLOCK_M=128–256, BLOCK_N=128–256, BLOCK_K=32–64). Tensor cores fully utilized. cuBLASLt / CUTLASS / Triton matmul / hipBLASLt are all reasonable.
   - **Prefill attention**: FlashAttention-2 / FA-3 (Hopper) or vendor equivalents. M = seq_len fills the MMA along the query dim; reductions are fp32 on-chip.
   - **Decode matmul**: BLOCK_M = 16 (MMA minimum) even when M=1; BLOCK_N = 64–128. Use **split-K** along the reduction dimension to fill SMs — without split-K the grid is `1 × ceil(N/BLOCK_N)` blocks, far too small.
   - **Decode attention**: FlashDecoding / split-KV attention. Split the KV cache axis across blocks, each computes a partial softmax, a final reduction combines partial sums via LSE merge. Persistent kernels avoid relaunch overhead per layer.

5. **Decide split-K (or split-KV) factor for decode.** Target grid size ≥ `numSMs * 2–4`. Split factor S = `ceil(numSMs * target_blocks_per_SM / (M_blocks * N_blocks))`; for decode, S is often 4–32. Each split adds a partial output and a final reduction — clamp S so each split processes ≥ ~512–1024 K elements, otherwise the reduction dominates. For attention, split along KV-cache sequence with LSE-correct combine.

6. **Choose tile shapes and dtypes.**
   - Prefill: aggressive tiles (128×128×64 fp16, or 128×256×64 fp8 on Hopper). fp32 accumulator. num_stages 3–5.
   - Decode matmul: BLOCK_M = 16 (MMA min). At M=1, 15/16 of the MMA's M dim is "wasted" — but the op is memory-bound, so this is not the bottleneck. Do not contort the kernel to chase MMA lanes for decode.
   - Decode attention: BLOCK_M = M (1, 4, 8, 16 depending on spec decode), BLOCK_N (KV tile) = 64–128. The K dim fills the MMA, so cores are usable there.
   - Quantized weight-only (W4A16, W8A16): dequant cost is per K element, amortized over only one row at M=1. Verify dequant is not the new bottleneck; fuse it into the matmul main loop with vectorized loads.

7. **Account for KV cache traffic explicitly.** Per decode step, every layer reads the entire per-request KV history. KV bytes per step = `2 * num_layers * num_kv_heads * head_dim * past_len * dtype_bytes`. For a 70B model with GQA at past_len=8k this is hundreds of MB *per request per step* and frequently exceeds weight-load bandwidth. KV intensity in attention is ~1 FLOP/byte; reason about it as a first-class bottleneck, not an afterthought.

8. **Reason about continuous batching heterogeneity.** A single batch may mix prefilling (large M) and decoding (M=1) requests. Two strategies:
   - **Chunked prefill**: split each prefill into chunks sized like a decode batch slot, so the kernel sees uniform M. Simpler kernels, but TTFT for long prompts grows (prefill spread across steps).
   - **Separate prefill and decode batches** (vLLM-style, TGI flash-decoding): each batch is homogeneous; pick the right kernel per batch. Better per-phase tuning, more scheduler complexity, risk of one phase starving the other.
   The kernel is selected per *batch*, not per *model*. State which strategy the engine uses before tuning.

9. **Account for speculative decoding.** With k draft tokens verified per step, M jumps from 1 to k:
   - k=2–4: still memory-bound, split-K factor can drop, split-KV grid shrinks.
   - k=8–16: intensity rises 8–16×; attention starts to fill the MMA along M. Consider switching to the prefill kernel with BLOCK_M=k.
   - k>16 (lookahead/EAGLE/Medusa with large trees): treat as small-M prefill.
   - Tradeoff: spec decode amortizes weight bandwidth per accepted token but burns FLOPs on rejected drafts. If acceptance rate < ~0.5 the FLOPs cost can exceed the bandwidth saving — verify empirically.

10. **Evaluate tradeoffs and decide.** Write down: chosen kernel family per phase, split factors, expected occupancy, expected memory-bandwidth utilization, and what is being measured (TTFT, ITL, throughput). State the conditions under which the choice would flip — these are the conditions that warrant retuning later.

## Kernel design rules
- Never reuse a prefill matmul kernel unchanged for decode. A 128×128 output tile at M=1 leaves 127/128 of the MMA's M dim idle and produces a grid of 1 × ceil(N/128) blocks — far below SM count on any modern GPU.
- Decode matmul kernels must use split-K (or a persistent design) to fill SMs. The grid for a non-split decode matmul is roughly `ceil(N/BLOCK_N)` blocks; on H100 (132 SMs) this is below SM count whenever `N < 132 * BLOCK_N`.
- For decode attention, split along the KV-cache dimension, never along M (M=1 by definition). FlashDecoding's split-KV with LSE combine is the standard pattern.
- Tensor core MMA fragments have minimum shapes (16×8×16 for fp16 on Ampere/Hopper). Choosing BLOCK_M < 16 saves nothing — the MMA still issues with M=16 and the compiler pads. Use BLOCK_M = 16 for M=1 decode.
- For W4A16 / W8A16 decode kernels, fuse dequantization into the matmul main loop. A separate dequant + GEMM pass doubles weight bandwidth (read quantized → write fp16 → read fp16 in matmul) and erases the quantization win.
- KV cache reads should be vectorized (128-bit loads) and contiguous along the sequence dimension. Paged KV adds page-table indirection — verify this does not break coalescing.
- For chunked prefill, chunk size should align to the matmul BLOCK_M and the FlashAttention BLOCK_M. Mismatched chunking forces partial-tile masking on every chunk.
- Persistent kernels matter in decode: with N layers × multiple kernel launches/layer × ~5 μs/launch, launch overhead accumulates into a significant share of inter-token latency.
- Do not chase tensor-core utilization in decode. The phase is memory-bound; idle tensor cores are a *symptom* of the regime, not a problem to fix.

## Correctness requirements
- Split-K and split-KV reductions must match the single-pass implementation within fp16/bf16 tolerance. Test with split factor 1, 2, 4, 8, and a non-power-of-2 split.
- For attention split-KV with LSE combine: partial max and partial sum-exp must be carried in fp32 and combined with the online-softmax merge. Combining naive partial softmax outputs without the LSE correction silently produces wrong probabilities.
- KV cache append must order correctly relative to attention reads. For sequential decode, write new K/V *after* attention reads the past. For tree-style spec decode, ensure within-tree dependencies are visible as required.
- Mixed-batch kernels must not cross-contaminate between requests. The attention mask must isolate each request's tokens to its own KV history; continuous-batching bugs here produce subtle quality regressions.
- Speculative decoding's verification step must use the same numerics the target model would have used at that position. Any mismatch (e.g., different fp16 reduction order) breaks the rejection-sampling correctness guarantee.
- Quantized decode kernels must be validated against the unquantized reference on a held-out perplexity or task suite. Per-channel/per-group scale errors compound across layers; see the debug-quantized-kernel-accuracy skill.

## Performance requirements
- State the roofline position of every kernel: arithmetic intensity, machine balance of the target GPU, and whether the kernel is memory- or compute-bound. If the answer is "I don't know," compute it before tuning.
- Decode metric: achieved HBM bandwidth as a fraction of peak. Target ≥ 60% on weight-load-dominated matmuls; < 40% suggests grid undersubscription or non-coalesced loads.
- Prefill metric: achieved tensor-core throughput. Target ≥ 50% of peak fp16 TC on large GEMMs; < 30% suggests poor tile shape or small K.
- Report TTFT and inter-token latency separately. Throughput-tokens-per-second alone hides regressions in either phase.
- For continuous batching, report throughput at fixed p99 latency. Unconstrained throughput is achieved by long batches with terrible tail latency.
- Speculative decoding gain should be reported as wall-clock tokens/s *and* accepted tokens / verification step. The first is what the user sees; the second tells whether the draft model is well-calibrated for this workload.

## Output format
The agent should produce:
1. **Phase identification**: prefill, decode, or mixed; effective M, K, N for each dominant matmul; KV cache bytes per step per request.
2. **Roofline placement**: arithmetic intensity per kernel, machine balance of the target HW, classification (memory-bound / compute-bound / at the knee).
3. **Kernel selection table**: per phase × per operation (QKV proj, attention, MLP up/gate, MLP down, output proj), the chosen kernel family, tile shape, split factor, expected bottleneck.
4. **Continuous batching strategy**: chunked prefill vs separate batches, with rationale tied to SLO targets.
5. **Speculative decoding adjustment** (if applicable): how the selection changes when M jumps from 1 to k, and the breakeven acceptance rate.
6. **Tradeoff summary**: 2–3 explicit tradeoffs.
7. **What to measure to validate**: TTFT, ITL p50/p99, achieved bandwidth on decode, achieved TFLOPs on prefill, and the test workloads.

## Common failure modes
- **Reusing a single kernel implementation for both phases**: a kernel tuned for prefill at BLOCK_M=128 will leave decode at single-digit percent of HBM bandwidth. Symptoms: good prefill throughput, terrible inter-token latency, low SM occupancy in decode.
- **Forgetting split-K in decode matmul**: the grid is too small to fill SMs and most of the GPU sits idle while a handful of blocks stream weights through HBM. If `gridDim.x * gridDim.y < numSMs`, decode is undersubscribed.
- **Assuming speculative decoding is always a win**: at low draft acceptance rates, spec decode burns more compute on rejected drafts than it saves in bandwidth. Measure end-to-end on realistic prompts; do not extrapolate from a benchmark suite.
- **Treating continuous batching as homogeneous**: tuning for a fixed M and ignoring that real batches mix prefilling and decoding requests. Either pick a chunked-prefill design with uniform M, or accept that two kernel paths and a scheduler are needed.
- **Measuring only token throughput**: a configuration with 2× throughput but 5× p99 inter-token latency is a regression for chat workloads. Carry TTFT and ITL distributions through tuning.
- **Ignoring KV cache bandwidth**: at long context, KV traffic exceeds weight traffic. Optimizing the matmul further while KV access is the bottleneck yields nothing. Profile the actual bandwidth split before deciding what to tune.
- **Quantized weight kernels with unfused dequant**: quantized weights are read, dequantized to a temporary buffer, then re-read by the matmul. Doubles weight traffic and erases the quantization bandwidth win.
- **Persistent kernels with broken exit conditions**: a persistent decode kernel that waits for all KV splits before exiting can deadlock if the scheduler adds new requests mid-step. Verify the exit condition matches the scheduler's batching contract.
- **Tuning with BLOCK_M < 16 for decode**: the MMA fragment is 16×8×16 minimum on Ampere/Hopper. Smaller BLOCK_M saves nothing; the compiler pads. Set BLOCK_M = 16 and move on.

## Review checklist
- [ ] Are prefill and decode analyzed separately, with M, K, N stated for each?
- [ ] Is arithmetic intensity computed and compared to the target HW machine balance for each dominant matmul?
- [ ] Is the decode matmul using split-K (or a persistent design) sized to fill SMs?
- [ ] Is decode attention using FlashDecoding / split-KV with an LSE-correct combine?
- [ ] Is BLOCK_M for decode set to MMA-aligned (typically 16) rather than chasing M=1 literally?
- [ ] Is KV cache bandwidth quantified per step per request and compared to weight bandwidth?
- [ ] Is the continuous batching strategy (chunked prefill vs separate batches) stated and justified against the SLO?
- [ ] If spec decoding is in scope, is the kernel adjusted for M=k and is the breakeven acceptance rate stated?
- [ ] Is the dominant bottleneck identified per kernel (HBM bandwidth, KV traffic, dequant, compute)?
- [ ] Are TTFT, ITL p50/p99, and throughput reported separately with a fixed-latency throughput target?
- [ ] For quantized weight kernels: is dequant fused into the matmul main loop, not a separate pass?
- [ ] Are the conditions under which the kernel choice would flip stated (longer context, larger spec tree, different HW)?
