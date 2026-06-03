# Skill: Write a vLLM Custom Op Integration Plan

## Purpose
Guide the agent through planning the integration of a custom CUDA or Triton kernel into the vLLM inference engine before any integration code is written — covering where the op plugs into the engine, paged KV cache and continuous batching compatibility, CUDA graph capture constraints, tensor parallelism implications, and the testing and benchmarking strategy. This skill produces an integration plan, not a kernel implementation.

## Use this when
- A custom CUDA or Triton kernel exists (or is being designed) and the goal is to ship it inside vLLM rather than as a standalone library.
- Replacing an existing vLLM op (e.g., a faster RMSNorm, a fused activation, a new attention variant) and the model file has to be modified to call the new path.
- Adding a new quantization scheme that must be exposed through vLLM's `LinearMethodBase` and quantization config registry.
- The kernel needs to coexist with paged attention, continuous batching, CUDA graphs, and tensor parallelism without breaking any of them.

## Do not use this when
- The kernel is being prototyped outside any serving engine; integration concerns are premature until correctness and standalone performance are established.
- vLLM already ships a competitive kernel for the same operation (FlashAttention-3, FlashDecoding, vllm-flash-attn, Marlin) and there is no novel reason to replace it. Integration cost is real and only justified by novel kernels, unsupported quantization formats, or custom serving features (constrained decoding, retrieval-augmented attention, RAG-fused ops).
- The kernel only helps in a regime that does not match vLLM's deployment profile (tiny batches, no KV cache, single-request offline). vLLM's throughput-oriented design will not surface the kernel's win.
- The user wants a kernel implementation. This skill is for the integration plan only.

## Inputs the agent should gather first
- **Op identity and intent**: which op is being added or replaced? Is it a drop-in replacement (same numerics, faster), a new fusion (combining two existing ops), a new algorithm (e.g., a different attention variant), or a new quantization scheme?
- **Target phase**: prefill only, decode only, or both? Does the kernel handle chunked prefill (mixed prefill+decode in one batch)?
- **Supported dtypes**: fp16, bf16, fp8, int8, int4? Does it require a specific accumulation dtype?
- **Quantization involvement**: is the kernel itself quantized? Does it consume quantized weights, quantized activations, or both? What is the scale/zero-point layout?
- **Tensor parallelism requirement**: must the op work under TP=1, TP=2, TP=4, TP=8? Is the op TP-trivial (elementwise, normalization) or TP-sensitive (matmul, attention)?
- **CUDA graph mode**: must the kernel work under captured graphs (the default `enforce_eager=False`), or is eager-only acceptable? Eager-only severely limits production usefulness.
- **vLLM version**: v0 and v1 have very different code paths (v1 reworks the scheduler, model runner, and KV cache manager). Plans must target one explicitly.
- **Hardware target**: SM generations (sm_80, sm_89, sm_90, sm_100), AMD CDNA support requirement, and whether the kernel needs to ship on multiple architectures.
- **Reference implementation**: is there an existing PyTorch eager reference, a paper, or a public kernel to validate against?

## Required reasoning process

1. **Orient to vLLM's call stack.** Before placing the op anywhere, the plan must state where it plugs in. The relevant layers, top-down:
   - **Engine** (`vllm/engine/`): scheduling, request lifecycle, output streaming. Custom ops do not live here.
   - **Worker** (`vllm/worker/`): one per GPU; owns the model and KV cache for that rank. Custom ops do not live here either, but the worker is what executes them.
   - **ModelRunner** (`vllm/worker/model_runner.py` or v1 equivalent): builds the input tensors, drives CUDA graph capture, and calls into the model. Custom ops affect this layer only if they need extra metadata (e.g., a new attention kernel that needs a different `block_table` format).
   - **Layer modules** (`vllm/model_executor/layers/`): `Attention`, `RMSNorm`, `LinearMethodBase`, `RotaryEmbedding`, etc. Most custom ops replace or subclass one of these.
   - **Model files** (`vllm/model_executor/models/`): each model (LLaMA, Mistral, Qwen, etc.) is a normal `nn.Module` composed from the layer modules. A new fusion that crosses two existing layers is wired in here.
   - **C++ / CUDA ops** (`csrc/`): hand-written CUDA and C++ extensions, bound via PyTorch's C++ extension system (`torch::Library`). New CUDA ops live here.
   - **Triton ops**: typically Python files under `vllm/model_executor/layers/` or `vllm/attention/ops/`. They are normal Triton kernels invoked from layer code.

2. **Decide where the op plugs in (decision tree).**
   - **Replacing an existing op** (faster RMSNorm, faster SwiGLU, faster RoPE): write the kernel under `csrc/` (CUDA) or `vllm/.../ops/` (Triton). Replace the call site in the existing layer module — do not modify every model file. If the layer module is shared across models, one change reaches all of them.
   - **Adding a fused op** (fused-add-rmsnorm, fused-bias-gelu): the kernel goes in the same place, but the model's `forward` must be changed to call the fused path. Check `vllm/compilation/` first — vLLM has fusion hooks for known patterns and a graph compiler that may already do the fusion if the underlying primitives are exposed correctly.
   - **Adding a new attention kernel**: subclass or register a new `AttentionBackend` under `vllm/attention/backends/`. The backend must produce the metadata the kernel needs (block tables, slot mappings, sequence lengths) and must be selectable through the engine config.
   - **Adding a quantization scheme**: implement a `LinearMethodBase` subclass under `vllm/model_executor/layers/quantization/`. Register it in the quantization config registry. Existing model files will pick it up through the standard `--quantization` argument; no per-model code changes required.
   - **Adding a brand-new layer**: extend the relevant model file directly. This is the highest-friction path and should be reserved for genuinely new architectures.

3. **Verify paged KV cache compatibility.** vLLM's KV cache is paged: physical layout is roughly `[num_blocks, block_size, num_kv_heads, head_dim]` (with variants per backend). Any kernel that reads or writes K/V must:
   - Accept `block_table` (per-sequence list of block indices) and `slot_mapping` (per-token destination slot) as inputs, not raw token positions.
   - Write into K/V via gather/scatter through `slot_mapping`. A kernel that assumes contiguous K/V buffers will silently corrupt other requests' cache slots.
   - The canonical reference is `reshape_and_cache_flash` in `csrc/cache_kernels.cu` (and its Triton analogue). Read it before designing a new K/V-touching kernel.
   - Document the exact KV layout the kernel expects (e.g., FlashAttention-style vs. xFormers-style) and confirm it matches the configured attention backend.

4. **Plan for continuous batching.** Within one batch, every request can be at a different position in its generation. The kernel will see a flattened representation:
   - `query` shape: `[num_total_query_tokens, num_heads, head_dim]`, not `[batch, seq, heads, head_dim]`.
   - Per-request boundaries are described by metadata such as `cu_seqlens_q` (cumulative sequence lengths) and `seq_lens` (per-request KV lengths).
   - With chunked prefill enabled, prefill tokens and decode tokens coexist in the same batch. The kernel must either handle both (preferred) or the integration plan must require the scheduler to split phases (`--enable-chunked-prefill=False`), which forfeits a major vLLM feature.
   - Decode-only kernels should be honest about this in the plan: they will be selected only when the scheduler runs decode batches, and the engine must still have a prefill path.

5. **Audit CUDA graph capture compatibility.** vLLM defaults to `enforce_eager=False` and captures CUDA graphs for a bucketed set of batch sizes (typically 1, 2, 4, 8, 16, 24, 32, ...; configurable). Each bucket produces one captured graph. Inside a captured region, the kernel must satisfy:
   - **No dynamic shapes**. The kernel must work for every batch size the engine captures. Shapes are fixed at capture time.
   - **No host-side memory allocation**. No `cudaMalloc`, no `torch.empty` inside the captured region, no Python list/dict construction that triggers allocator calls.
   - **No Python control flow that branches on tensor values**. Branches that depend on `.item()` or `.cpu()` reads break capture.
   - **No JIT compilation during capture**. Triton kernels with autotuning must have their autotune table populated *before* graph capture. The standard pattern is a warm-up pass that runs the kernel on every captured shape with realistic inputs, forcing autotune to resolve and JIT to compile, before the engine enters its capture loop.
   - **Stable workspace memory**. If the kernel needs scratch space, allocate it once at engine init and pass it in as a tensor argument; do not allocate per-call.
   - The integration plan must list, for each of these constraints, how the kernel either satisfies it or why an exception is justified. Eager-only kernels must be flagged as a deployment limitation.

6. **Analyze tensor parallelism behavior.** vLLM shards weights across TP ranks at the layer level:
   - Column-parallel linear: weight sharded along output dim; each rank produces a slice; all-gather (or fused all-reduce) at the layer boundary.
   - Row-parallel linear: weight sharded along input dim; each rank produces a partial sum; all-reduce at the layer boundary.
   - For each custom op, the plan must answer: **is this op TP-trivial or TP-sensitive?**
     - TP-trivial: elementwise ops, normalizations (when applied to per-rank-sized hidden), activations. The kernel sees a per-rank-sized tensor and does the right thing automatically.
     - TP-sensitive: matmuls (must integrate with the column/row-parallel split), attention (must shard heads), quantization (scales must follow the sharded weight). These ops must explicitly declare their TP behavior, and the plan must specify how scales, biases, and any auxiliary tensors are sharded alongside weights.
   - Common bug: kernel hardcodes `hidden_size` from the model config but receives a per-rank-sized input. Always read the size from the input tensor, not the config.

7. **Acknowledge pipeline parallelism.** PP is less common in vLLM v0 and is more central in v1. For most custom ops, the plan only needs to confirm that the op does not break PP — i.e., it operates on per-rank tensors and does not assume the entire model is on one device. If the op crosses pipeline boundaries (rare), the plan must be explicit about it.

8. **Define the testing strategy.** Two layers, both required:
   - **Unit test under `tests/kernels/`**: invoke the kernel directly, compare against a PyTorch eager reference (or a SciPy/NumPy reference for non-Torch ops). Cover: dtype matrix (fp16, bf16, fp8 if applicable), shape matrix (small, medium, large; aligned and unaligned), batch sizes, sequence lengths including odd values, and TP-shard equivalents (rank-shaped inputs).
   - **Integration test under `tests/models/`**: run a small model end-to-end through the engine with the new op enabled, and verify the output matches a reference run (typically the same model with the original op). Do this with both `enforce_eager=True` and `enforce_eager=False` to catch CUDA-graph-only bugs.
   - **Numerical tolerance**: bf16/fp16 ops typically need `rtol≈1e-2, atol≈1e-2` against an fp32 reference. Tighter tolerances are usually wrong for low-precision kernels and will produce flaky tests. Document the chosen tolerance and justify it.

9. **Define the benchmarking strategy.** All numbers must be measured against the un-customized vLLM baseline, not against a non-vLLM reference (the engine overhead changes the picture entirely). Use the in-tree harnesses:
   - `benchmarks/benchmark_latency.py` for single-request latency at fixed batch and sequence length.
   - `benchmarks/benchmark_throughput.py` for offline batched throughput.
   - `benchmarks/benchmark_serving.py` for production-like serving simulation against an open dataset (ShareGPT, ArXiv summary, or similar).
   - For each benchmark, the plan must specify: hardware, vLLM commit, model, dtype, TP size, batch sizes, dataset, and the metric reported (TTFT, TPOT, throughput). A single number with no context is not a benchmark.

## Kernel design rules
- The custom op must accept per-rank-sized tensors and read sizes from inputs, never from the global model config.
- Any kernel that touches K/V cache must take `block_table` and `slot_mapping` as explicit inputs and follow the configured KV layout exactly.
- No Python-side `.item()`, `.cpu()`, or shape-dependent control flow in any path that runs inside CUDA graph capture.
- Triton autotune tables must be primed during a warm-up pass before graph capture begins, with realistic inputs at every captured shape.
- Workspace and scratch tensors are allocated once at engine init and passed in; never allocated per-call.
- Custom ops must be registered through PyTorch's op registration machinery (`torch.library`), not as raw Python functions, so they participate correctly in `torch.compile` and graph capture.
- Quantization scales, zero points, and any auxiliary tensors must be sharded in lockstep with the weight they accompany; document the sharding rule explicitly.

## Correctness requirements
- The integrated kernel must produce outputs that match a reference within the documented tolerance, both standalone (unit test) and end-to-end (integration test).
- End-to-end correctness must be verified under both `enforce_eager=True` and `enforce_eager=False`. A kernel that passes only in eager mode is not integrated correctly.
- TP correctness must be verified at every TP size the integration claims to support. A kernel that is correct at TP=1 and broken at TP=4 is a regression.
- Chunked prefill correctness must be verified if the kernel claims to support it; mixed prefill+decode batches are the failure mode that unit tests miss.
- Paged KV writes must be verified against a non-paged reference: write to the cache, read it back through the standard attention path, and confirm content matches the unpaged reference.

## Performance requirements
- The plan must specify a baseline (un-customized vLLM at the same commit) and a target metric (latency reduction, throughput increase, or memory reduction) with a numeric goal.
- Performance must be reported on at least two batch sizes — one captured-graph bucket near the lower end (e.g., 1, 2) and one near the upper end (e.g., 32, 64) — because optimizations that win at one size frequently lose at the other.
- If the kernel adds compilation or warm-up cost (Triton autotune, graph capture overhead), the plan must report engine startup time before and after, and confirm the cost is acceptable for the deployment scenario.
- Performance claims that are not measured against vLLM's own baseline are not acceptable. Comparing a vLLM-integrated kernel against a non-vLLM reference is meaningless.
- Memory footprint changes (additional scratch buffers, extra cache layouts) must be reported at realistic batch and sequence configurations, not just at the test sizes.

## Output format
The integration plan must include:
1. **Op summary**: what is being added or replaced, target phase, dtypes, hardware.
2. **Plug-in point**: which file(s) under `csrc/`, `vllm/model_executor/layers/`, `vllm/attention/backends/`, or `vllm/model_executor/models/` will be touched, with the rationale for each.
3. **KV cache compatibility analysis**: confirms paged layout assumptions (or states the op does not touch K/V).
4. **Continuous batching analysis**: prefill, decode, chunked prefill behavior; cu_seqlens handling.
5. **CUDA graph compatibility audit**: each constraint (no dynamic shapes, no host alloc, no JIT-during-capture, stable workspace) addressed point by point.
6. **Tensor parallelism analysis**: TP-trivial vs. TP-sensitive classification, sharding rules for weights and auxiliary tensors.
7. **Testing plan**: unit tests under `tests/kernels/`, integration tests under `tests/models/`, both eager and captured-graph modes, tolerance rationale.
8. **Benchmarking plan**: harnesses used, hardware, model, baseline, target metrics.
9. **Risk register**: known risks (TP edge cases, dtype gaps, graph capture pitfalls) with mitigations.
10. **Out-of-scope statement**: what the integration explicitly does not cover (e.g., AMD support deferred, fp8 deferred), so reviewers do not expect it.

## Common failure modes
- **Wrong KV cache slot writes**: kernel assumes contiguous K/V and writes through `seq_pos` instead of `slot_mapping`. Other requests' cached tokens get corrupted, producing subtly wrong outputs that pass single-request tests.
- **CUDA graph capture failure due to dynamic alloc**: kernel allocates a workspace tensor on first call. Capture fails with a cryptic CUDA error, or worse, captures a graph that points at a now-freed allocation.
- **Triton autotune fires inside graph capture**: the kernel is correct standalone but the engine fails at capture time because Triton's JIT compiles a new variant. The fix is a pre-capture warm-up; without it the integration is broken.
- **TP breakage from hardcoded sizes**: kernel reads `hidden_size` from the global config instead of the input tensor, and at TP=4 it processes a tensor 4x smaller than expected. Outputs are silently wrong.
- **Eager-only ship**: the kernel is tested only with `enforce_eager=True` because that is the development default, and the production deployment with captured graphs blows up on first request.
- **Chunked prefill ignored**: the kernel works for pure-prefill and pure-decode batches but produces wrong outputs when the scheduler mixes them. Disabled by default in tests, hit immediately in production.
- **Benchmarking against the wrong baseline**: numbers are reported against a non-vLLM PyTorch reference, making the kernel look great. Once integrated, the engine overhead dominates and the win disappears.
- **Quantization scale sharding mismatch**: the weight is sharded correctly but the per-channel scale tensor is not, so each rank applies the wrong scale to its weight slice. Outputs are wrong only at TP > 1.
- **vLLM v0 vs. v1 confusion**: the plan was written against v0 internals and the target deployment is v1; the model runner, attention backend, and KV cache manager have all moved.

## Review checklist
- [ ] Is the plug-in point named precisely (file path or module), with a justification for not choosing alternative locations?
- [ ] Does the plan explicitly state v0 or v1 as the target?
- [ ] If the kernel touches K/V, does it consume `block_table` and `slot_mapping`, and is the KV layout matched to the configured attention backend?
- [ ] Does the plan address chunked prefill, or explicitly require it to be disabled with the consequence stated?
- [ ] Has every CUDA graph capture constraint been audited (no dynamic shapes, no host alloc, no JIT during capture, stable workspace, autotune pre-warmed)?
- [ ] Is the TP behavior classified, and are sharding rules for all auxiliary tensors (scales, biases) specified?
- [ ] Are unit and integration tests both planned, and is `enforce_eager=False` covered by the integration test?
- [ ] Is the numerical tolerance documented and justified for the dtype?
- [ ] Are benchmarks defined against a real vLLM baseline (not a non-vLLM reference) at multiple batch sizes?
- [ ] Has the deployment limitation of any eager-only path been flagged?
- [ ] Is there a risk register, and does each risk have a mitigation?
- [ ] Has the integration been justified against existing vLLM kernels (FlashAttention-3, vllm-flash-attn, Marlin), with a clear reason this op is novel or necessary?
