# Skill: Write a Triton KV Cache Append Kernel

## Purpose

Guide the agent through implementing a Triton kernel that writes newly computed K and V tensors into a pre-allocated KV cache during LLM inference. This covers two cache layouts (contiguous and paged / vLLM-style PagedAttention), unified prefill and decode handling via a `slot_mapping` tensor, GQA/MQA where the cache stores fewer heads than Q, optional fp8/int8 quantized KV cache with scaling, coalesced versus scattered write patterns, and the boundary checks required to avoid corrupting other requests' cache regions. The kernel runs once per layer per forward step; correctness is the dominant concern, throughput is secondary because the data volume per call is small.

---

## Use this when

- You are building a custom serving stack and need a KV cache append step that is not provided by vLLM or FlashInfer.
- You need a paged KV cache layout (block_table indirection) with a non-standard block_size, an exotic block layout, or a sharding scheme like RingAttention that the existing libraries do not support.
- You need fp8 or int8 quantized KV cache with per-block or per-tensor scales, and the in-tree kernels do not match your scaling convention.
- You want a unified kernel that handles prefill (many new tokens per request) and decode (one new token per request) through the same `slot_mapping` interface.
- You are fusing the append into a custom QKV epilogue and need a reference standalone kernel to verify against.

---

## Do not use this when

- You are inside vLLM and the existing `reshape_and_cache_flash` (or `reshape_and_cache`) kernel already covers your layout. It is heavily tuned and maintained.
- You are using FlashInfer — its `append_paged_kv_cache` is well optimized and integrates with the rest of the FlashInfer attention API.
- You only need a contiguous cache and a simple PyTorch indexing assignment is fast enough. For small batch sizes the launch overhead of a custom kernel can exceed `cache[:, :, pos, :] = new_k`.
- You are tempted to fuse the append into the QKV projection in your first iteration. The projection's tile shape constraints and the cache's scatter pattern rarely align cleanly. Write the standalone kernel first, profile, then consider fusion if the launch overhead actually matters.
- You need a backward pass. KV cache append is inference-only; there is no autograd-relevant version of this kernel.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Cache layout** — contiguous `[batch, num_kv_heads, max_seq_len, head_dim]`, or paged. If paged, which paged layout: `[num_blocks, block_size, num_kv_heads, head_dim]` (vLLM v1 flash style) or `[num_blocks, num_kv_heads, head_dim // x, block_size, x]` (older vLLM with vectorized inner dim) or block_v1 dual-stacked `[num_blocks, 2, num_kv_heads, block_size, head_dim]`.
2. **block_size** — paged layout only. Typically 16, 32, or 64. Must divide the cache evenly.
3. **num_kv_heads, num_heads, head_dim** — confirm GQA grouping. The cache only stores `num_kv_heads`; the kernel never replicates K to `num_heads`.
4. **Input dtype and cache dtype** — usually identical (fp16 or bf16). If fp8 or int8 KV cache, confirm the scale dtype (fp32) and granularity: per-tensor, per-token, per-head, or per-block.
5. **Batch shape of new K/V** — `[num_tokens, num_kv_heads, head_dim]` where `num_tokens` is the total number of new tokens across all requests in this step. For decode, `num_tokens == batch_size`. For prefill, `num_tokens` is the sum of prompt lengths.
6. **slot_mapping** — int64 tensor of shape `[num_tokens]` giving the destination slot index for each new token. For contiguous layout, this is a flat slot index into `[batch * num_kv_heads * max_seq_len]` or equivalent. For paged layout, it is the global slot index `block_id * block_size + slot_in_block`.
7. **block_table** — paged layout only. Int32 tensor `[batch, max_blocks_per_req]` mapping logical block index to physical block ID. Usually consumed in Python to build `slot_mapping`; confirm whether the kernel takes `block_table` directly or only `slot_mapping`.
8. **Quantization scales** — required if cache dtype is fp8/int8. Shape, dtype, and indexing convention.
9. **Stride information** — strides of the new K/V tensors and the cache tensor. Never assume contiguity, especially for K/V which may come from a transposed projection.

---

## Required reasoning process

1. **Pick the kernel interface.** The clean, unified interface is: take new K and V tensors of shape `[num_tokens, num_kv_heads, head_dim]`, a `slot_mapping` tensor of shape `[num_tokens]`, the cache tensors, and (optional) quantization scales. The Python launcher is responsible for translating block_table + per-request positions into a flat slot_mapping. This makes prefill and decode go through the same kernel.

2. **Choose the grid.** One program per (token, kv_head) pair is the simplest and most flexible:
   ```python
   grid = (num_tokens, num_kv_heads)
   ```
   Each program writes one head_dim vector to the cache. For very small `head_dim` (e.g., 64) and large `num_kv_heads`, fusing along the head dimension can reduce launch overhead, but the (num_tokens, num_kv_heads) decomposition is what most production kernels use.

3. **Resolve program ids and the destination slot.**
   ```python
   token_idx = tl.program_id(0)
   head_idx  = tl.program_id(1)
   slot      = tl.load(slot_mapping_ptr + token_idx)
   ```
   If `slot < 0`, the launcher has marked this token as a no-op (used in CUDA-graph capture with padded batches). Skip the write:
   ```python
   if slot < 0:
       return
   ```

4. **Compute the source pointer for the new K/V vector.**
   ```python
   d_offsets = tl.arange(0, BLOCK_D)
   d_mask    = d_offsets < head_dim
   k_src_ptr = k_new_ptr + token_idx * stride_k_token + head_idx * stride_k_head
   k_vec     = tl.load(k_src_ptr + d_offsets * stride_k_d, mask=d_mask, other=0.0)
   ```
   Same for V. `BLOCK_D` is the next power of 2 >= head_dim (typically 64, 128, or 256).

5. **Compute the destination pointer based on cache layout.**

   **Contiguous layout** `[batch, num_kv_heads, max_seq_len, head_dim]`. The launcher must encode `(req_idx, position)` into `slot`, e.g. `slot = req_idx * max_seq_len + position`, and the kernel reconstructs:
   ```python
   req_idx  = slot // max_seq_len
   position = slot %  max_seq_len
   k_dst = k_cache_ptr
       + req_idx  * stride_kc_b
       + head_idx * stride_kc_h
       + position * stride_kc_n
   ```

   **Paged layout** `[num_blocks, block_size, num_kv_heads, head_dim]`:
   ```python
   block_id      = slot // BLOCK_SIZE   # constexpr divisor
   slot_in_block = slot %  BLOCK_SIZE
   k_dst = k_cache_ptr
       + block_id      * stride_kc_block
       + slot_in_block * stride_kc_slot
       + head_idx      * stride_kc_h
   ```
   For block_v1 dual-stacked `[num_blocks, 2, num_kv_heads, block_size, head_dim]`, K and V share the same buffer and are selected by an extra index 0 (K) or 1 (V).

6. **Apply quantization if cache dtype is fp8 or int8.** For per-tensor scale:
   ```python
   k_q = (k_vec / k_scale).to(cache_dtype)   # fp8 or int8
   ```
   For per-token or per-block scale, load the appropriate scalar first. For fp8 with `e4m3` or `e5m2`, use `tl.cast(..., tl.float8e4nv)` (or the bf16 equivalent), and never quantize before applying the scale — scale division must happen in fp32.

7. **Store the vectors.**
   ```python
   tl.store(k_dst + d_offsets * stride_kc_d, k_q, mask=d_mask)
   tl.store(v_dst + d_offsets * stride_vc_d, v_q, mask=d_mask)
   ```
   Use the same `d_mask` as the load. No other masking is required because the program is a no-op when `slot < 0` and slot_mapping is otherwise expected to be valid.

8. **Decide K and V kernel sharing.** K and V have identical layouts and slot mappings; do them in the same kernel to halve launch overhead. The only divergence is the optional separate scale tensors.

9. **In the launcher, build slot_mapping correctly.** This is where most bugs live — see the failure modes section. For paged decode with one new token per request:
   ```python
   # context_lens[req] = number of tokens already in cache (i.e., position of the new token)
   logical_block = context_lens // block_size
   slot_in_block = context_lens %  block_size
   block_id      = block_table[torch.arange(batch), logical_block]
   slot_mapping  = block_id * block_size + slot_in_block
   ```
   For prefill, expand per-request positions into per-token slots.

---

## Kernel design rules

- `BLOCK_D` must be `tl.constexpr`, a power of 2, and >= `head_dim`. Common values: 64, 128, 256. Do not loop over the head dimension; head_dim is small enough to fit in one block.
- `BLOCK_SIZE` (paged block size) must be `tl.constexpr` so the integer divide and modulo by `BLOCK_SIZE` are compiled to shifts and masks when it is a power of 2.
- Pass all strides as kernel arguments. K/V often come from a fused QKV projection and may not be contiguous; cache strides depend on layout. Hardcoding strides will silently corrupt other heads or other requests' cache regions.
- The slot_mapping load must be a scalar load: `slot = tl.load(slot_mapping_ptr + token_idx)`. Do not vectorize across tokens in the same program — different tokens map to different physical blocks and the destination pointers are not contiguous.
- The `slot < 0` early exit must be present if you support CUDA-graph padding. vLLM and most production stacks pad the batch up to a captured size and mark padding tokens with `slot = -1`. Without this guard, padding tokens write to slot `-1` interpreted as a huge unsigned offset, corrupting random GPU memory.
- Use the same dtype for the load and the destination unless quantizing. Mixing fp16 source and bf16 cache without an explicit cast produces undefined results.
- For fp8 KV cache, the scale must be applied in fp32 before the cast. Casting to fp8 first and then multiplying loses range.
- For GQA, the kernel does not replicate K across query head groups. The cache stores `num_kv_heads`, period. Replication happens inside the attention kernel via index arithmetic.
- Do not emit a barrier or reduction. This kernel is pure point-to-point copy plus optional per-element quantization.

---

## Correctness requirements

- Each new token is written to exactly one cache slot. The slot must correspond to the token's logical position in its request's KV sequence. Verify by reading the cache back and checking `cache[req, head, pos] == new_k[token_for(req, pos), head]` for several positions.
- For paged layout, the `(block_id, slot_in_block)` decomposition must use the kernel's compile-time `BLOCK_SIZE`. Mismatching the `BLOCK_SIZE` between the launcher's slot_mapping construction and the kernel's modulo writes to the wrong physical block.
- For contiguous layout, the slot encoding must match what the kernel decodes. If the launcher uses `slot = req * max_seq_len + pos` but the kernel does `req = slot // num_kv_heads`, the writes go to a completely wrong region. Pin one convention and document it at the kernel's argument list.
- The `slot < 0` no-op path must not load from the K/V source pointer either, because invalid token slots may also have padding K/V tensors that are not necessarily safe to read. Place the early exit before any load.
- Prefill writes for one request must not overlap with another request's cache region. If the launcher's `slot_mapping` is wrong (e.g., off-by-one because the new tokens' starting position was computed before incrementing `context_lens`), this is the bug — the kernel itself cannot detect cross-request corruption.
- For fp8/int8 quantization, the dequantized cache must reproduce the original K/V within the quantization grid resolution. Verify with a roundtrip test: write fp16 K, read back from the quantized cache with the inverse scale, and check the L_inf error is within one quantization step.
- GQA correctness: with `num_kv_heads = 8` and `num_heads = 32`, the new K/V tensor has 8 heads, the cache has 8 heads, and `head_idx` ranges over 8. Do not accidentally use `num_heads = 32` anywhere in this kernel.
- Idempotence: appending the same token twice (same slot) must produce the same cache state as appending it once. Confirm there is no read-modify-write with stale data — this kernel is pure write.

---

## Performance requirements

The agent must reason through the following:

- **Volume per call is small.** For decode with batch 32, num_kv_heads 8, head_dim 128, fp16: total bytes written is `32 * 8 * 128 * 2 * 2 (K and V) = 128KB` per layer. On an H100 with 3TB/s HBM, this is ~40ns of pure transfer. Launch overhead and program scheduling dominate. Do not over-engineer the inner loop; minimize launch overhead.
- **Prefill is the bandwidth case.** With prefill_tokens = 4096 and the same head config, total bytes are `4096 * 8 * 128 * 2 * 2 = 16MB`. This is bandwidth-limited and benefits from coalesced writes. Because the slot_mapping for a single request's prefill tokens is contiguous (consecutive positions in the same block, then jumping to the next block), the writes are coalesced within a request.
- **Decode writes are scattered.** Each request writes to a different physical block. There is no coalescing across requests. This is fine — the volume is too small for it to matter.
- **Quantization adds compute.** fp8 conversion is a few extra instructions per element. For bandwidth-bound prefill, this is hidden by memory latency. For decode, the compute is irrelevant — launch overhead still dominates.
- **CUDA graphs.** This kernel must work inside a captured CUDA graph. That means the grid must be deterministic for a given captured shape, and the early-exit on `slot < 0` is what handles padded-out batch slots. Do not use dynamic grids that depend on the slot_mapping contents.
- **`num_warps`.** For BLOCK_D = 128 and one program writing one head_dim vector, `num_warps=1` is sufficient — the program writes a single 128-element vector. Higher num_warps wastes warps. For BLOCK_D = 256, `num_warps=2`.
- **Do not benchmark in isolation.** This kernel's cost is meaningful only relative to the per-step end-to-end latency. A 5us improvement here is invisible behind a 10ms attention call. Measure the impact in the full decode loop.

---

## Output format

The agent should produce:

1. **The Triton kernel** with `@triton.jit`, parameterized by `BLOCK_D: tl.constexpr`, `BLOCK_SIZE: tl.constexpr` (paged layout only), all relevant strides, `head_dim`, and a `IS_FP8: tl.constexpr` flag (or separate kernels) for quantized variants.
2. **The Python launcher** that constructs `slot_mapping` from `block_table` and `context_lens` (or accepts a precomputed slot_mapping), validates K/V shapes, asserts `slot_mapping.dtype == torch.int64` (or int32 with documented range), and launches with `grid = (num_tokens, num_kv_heads)`.
3. **A docstring on the launcher** stating: cache layout, slot_mapping convention (paged: `block_id * BLOCK_SIZE + slot_in_block`; contiguous: `req * max_seq_len + pos`), and the no-op behavior for `slot == -1`.
4. **A correctness test** that:
   - For contiguous layout, writes random K/V at known positions and reads back via PyTorch indexing.
   - For paged layout, writes through the kernel and reads back through the same `slot_mapping` indirection in PyTorch.
   - Includes a `slot == -1` padding token in the batch and verifies the cache is untouched at no other position.
5. **Optional fp8 variant** with a roundtrip-error check after dequantization.
6. **No fused QKV-projection version on the first pass.** Note explicitly that fusion is a separate optimization gated on profiling.

---

## Common failure modes

- **Off-by-one in slot_mapping for new tokens.** The new token's position is `context_lens[req]` *before* incrementing, not after. Using `context_lens[req] - 1` writes to the previous token's slot and corrupts it; using `context_lens[req] + 1` skips a slot and the next attention call reads garbage at that position.
- **Block_size mismatch between launcher and kernel.** The launcher computes `slot_in_block = pos % 16` but the kernel was compiled with `BLOCK_SIZE = 32`. Every write goes to a wrong slot in the wrong block.
- **Block_table indexing bug.** Using `block_table[req][logical_block]` where `logical_block` is computed in physical units (already including block_size) instead of logical block index. Result: writes scatter into random other requests' blocks. This is the most common source of cross-request corruption.
- **Missing `slot < 0` early exit.** Production decode batches are CUDA-graph captured at a maximum batch size. Smaller real batches are padded and the padding tokens have `slot = -1`. Without the guard, `slot = -1` becomes a huge unsigned offset in pointer arithmetic and the kernel writes to random GPU memory. This often manifests as silent NaN propagation in unrelated layers.
- **GQA / MQA confusion.** Using `num_heads` (the Q head count) instead of `num_kv_heads` to compute strides in the cache. The cache has fewer heads; using num_heads strides over-indexes by a factor of the GQA group size and corrupts other requests' regions.
- **Quantizing before scaling.** `tl.cast(k, fp8) / scale` rather than `(k / scale).to(fp8)` loses dynamic range entirely because fp8 saturates aggressively before the division.
- **Non-contiguous source K/V with assumed contiguous strides.** K/V from a fused QKV projection are typically non-contiguous along the head dim or the head index. Hardcoding `stride_k_token = num_kv_heads * head_dim` reads from the wrong rows of the projection output.
- **Writing both K and V to the K cache.** Easy in block_v1 dual-stacked layout where K and V share the buffer and the only difference is the `kv_idx` (0 or 1). Forgetting to flip this for V doubles the K writes and silently zeros V.
- **Race condition with attention reading the cache.** The attention kernel for the next token must run after the append finishes. If both are on the same stream this is automatic, but custom multi-stream pipelines must add an event sync. This kernel does not handle the sync; the caller does.
- **Forgetting to synchronize block_table updates.** When a new block is allocated mid-step, the block_table must be updated on the GPU before the append kernel reads it. A CPU-side update without a `.copy_(non_blocking=False)` or explicit sync produces a stale block_table on the GPU.

---

## Review checklist

- [ ] Kernel takes new K, new V, slot_mapping, K cache, V cache, all relevant strides, `BLOCK_D: tl.constexpr`, and (paged) `BLOCK_SIZE: tl.constexpr`.
- [ ] Grid is `(num_tokens, num_kv_heads)`; one program writes one head_dim vector for K and one for V.
- [ ] `slot < 0` early-exit is present and placed before any K/V load.
- [ ] `BLOCK_D` is a power of 2 and >= head_dim; `d_mask = d_offsets < head_dim` is applied to both load and store.
- [ ] All strides are kernel arguments. No hardcoded `head_dim * num_kv_heads` arithmetic.
- [ ] For paged layout, `slot // BLOCK_SIZE` and `slot % BLOCK_SIZE` use the `tl.constexpr` `BLOCK_SIZE`.
- [ ] For contiguous layout, the `(req, position)` decoding in the kernel matches the encoding in the launcher.
- [ ] Launcher constructs `slot_mapping` using `context_lens` (pre-increment) for the new token's position.
- [ ] GQA: kernel uses `num_kv_heads`, never `num_heads`. K is not replicated.
- [ ] fp8/int8 quantization: scale division happens in fp32, then cast. Roundtrip correctness verified.
- [ ] CUDA-graph compatibility: grid is shape-determined, padded batch slots use `slot = -1`.
- [ ] Correctness test covers contiguous, paged, and a padding-token case.
- [ ] No claim that this kernel is faster than vLLM's `reshape_and_cache_flash` without a measurement against it.
- [ ] No premature fusion with the QKV projection on the first iteration.
