# Skill: Write a Triton RoPE Kernel

## Purpose

Guide the agent through implementing a correct Triton kernel that applies Rotary Position Embeddings (RoPE) to query and key tensors before attention. This covers the two incompatible layout conventions (GPT-NeoX/HuggingFace-LLaMA vs GPT-J/original-paper), pre-computed cos/sin table consumption, per-token position handling for continuous batching, partial-RoPE masking, and the precision discipline required to keep cos/sin in fp32 while applying to fp16/bf16 activations. RoPE is the dominant positional encoding in LLaMA, Mistral, Qwen, Gemma, GPT-NeoX, and most decoder-only LLMs trained after 2022, so getting this kernel right is load-bearing for inference correctness.

---

## Use this when

- You are building an inference serving stack (vLLM-style, TGI-style, custom) that does not use FlashAttention-3's fused RoPE-in-attention path, and you need a standalone RoPE op for prefill or decode.
- You need a decode-time RoPE kernel: Q/K of length 1 per request, where the launch overhead of a fused FA3-style attention kernel exceeds the cost of a tiny dedicated RoPE kernel.
- You need a custom RoPE variant (NTK-aware scaling, YaRN, longrope, partial RoPE on the first N dims only) where the framework's stock kernel does not match the model definition.
- You need to support continuous batching where each request has a distinct position offset and the standard contiguous-position kernel cannot be used.
- You are porting a model whose RoPE layout (NeoX vs GPT-J) does not match what your inference framework provides.

---

## Do not use this when

- You are using FlashAttention-3 or a similar fused attention kernel that already applies RoPE inside attention. Adding a separate RoPE pass duplicates work and rotates Q/K twice.
- You are running training or inference where HuggingFace's `apply_rotary_pos_emb` is fast enough — for non-tight loops the Python-level reference is fine and avoids a custom kernel surface.
- The model uses a different positional encoding (ALiBi, T5 relative bias, learned absolute embeddings). RoPE is not a drop-in substitute.
- The tensor layout is exotic and you have not yet decided whether RoPE is applied to the (B, N, H, D) or (B, H, N, D) view. Resolve layout first; the kernel structure depends on it.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Layout convention.** Two conventions exist and they are not equivalent:
   - **GPT-NeoX / HuggingFace-LLaMA**: rotate halves. The vector is split into a first half `x_lo = x[..., :D/2]` and a second half `x_hi = x[..., D/2:]`. The rotated component is `concat(-x_hi, x_lo)`. Cos/sin tables have shape `[max_seq_len, D/2]` and are broadcast across the two halves (i.e., `cos` is duplicated to length D as `concat(cos, cos)`).
   - **GPT-J / original RoPE**: rotate adjacent pairs. For each `i`, the pair `(x[2i], x[2i+1])` is rotated together. FlashAttention-2 uses this layout.
   Picking the wrong layout silently produces broken model output. Confirm by reading the reference model code, not by guessing from the dtype or shape.
2. **Head dimension D.** Must be even. If the model uses partial RoPE (rotate only the first `rot_dim` dimensions, leave `D - rot_dim` untouched), get `rot_dim` explicitly.
3. **Number of heads** for Q (`H_q`) and K (`H_kv`). Often `H_kv < H_q` for GQA — the kernel must handle both head counts independently (they share the cos/sin table but loop over their own head range).
4. **Cos/sin table dtype.** Should be fp32 at table-build time. If the framework hands you fp16 cos/sin tables, flag it — long-position precision loss is a real bug at sequence lengths > 8K.
5. **Q/K dtype.** Usually fp16 or bf16. The kernel reads in this dtype, casts to fp32 for the rotation, and stores back in the original dtype.
6. **Position layout.** Two cases:
   - Contiguous prefill: positions `[0, 1, 2, ..., N-1]` for each sequence, sequences padded to a uniform length.
   - Continuous batching: a `positions` array of shape `[total_tokens]` giving each token's true position in its request's sequence. Token `i` may be position 5 in request A while token `i+1` is position 137 in request B.
7. **Partial RoPE ratio.** Some models (e.g., certain DeepSeek variants) rotate only a fraction of `D`. If `rot_dim < D`, the kernel must leave the tail untouched.
8. **In-place vs out-of-place.** HuggingFace and most serving stacks apply RoPE in place to save memory. Confirm whether the caller expects `Q` and `K` to be overwritten or written to separate output buffers.
9. **Whether Q and K share a launch.** A combined kernel applies RoPE to both in one launch (avoids two launches' overhead but slightly complicates pointer arithmetic). Two separate launches is simpler and only marginally slower for typical shapes.

---

## Required reasoning process

1. **Pin down the layout first, then the formula.** Write out the rotation in the chosen layout before touching the kernel:
   - **NeoX (rotate halves):** `out_lo = x_lo * cos - x_hi * sin`, `out_hi = x_hi * cos + x_lo * sin`, where `cos`, `sin` have shape `[D/2]` and `x_lo`, `x_hi` are each shape `[D/2]`.
   - **GPT-J (rotate pairs):** for each `i`, `out[2i] = x[2i]*cos[i] - x[2i+1]*sin[i]`, `out[2i+1] = x[2i+1]*cos[i] + x[2i]*cos[i]` — wait: `out[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]`. Write this out and double-check signs against a reference implementation before coding.

2. **Choose the program decomposition.** A typical mapping is one program per `(token, head)` pair:
   ```python
   grid = (total_tokens, num_heads)
   pid_t = tl.program_id(0)   # token index in the flat (B*N) layout
   pid_h = tl.program_id(1)   # head index for this Q or K tensor
   ```
   For continuous batching, `total_tokens` is the sum of all request lengths, not `B * N`.

3. **Look up the position for this token.** Load `pos = tl.load(positions_ptr + pid_t)`. Do not compute `pos = pid_t % seq_len` — this is wrong for continuous batching and silently produces incorrect positional information for all tokens after the first request.

4. **Load the cos/sin row for this position.** The cos/sin tables have shape `[max_seq_len, D/2]`. Compute the row pointer:
   ```python
   d_offsets = tl.arange(0, BLOCK_D_HALF)         # BLOCK_D_HALF >= D/2
   d_mask    = d_offsets < (rot_dim // 2)         # for partial RoPE
   cos_row = tl.load(cos_ptr + pos * (D // 2) + d_offsets, mask=d_mask, other=1.0)
   sin_row = tl.load(sin_ptr + pos * (D // 2) + d_offsets, mask=d_mask, other=0.0)
   ```
   `other=1.0` for cos and `other=0.0` for sin: at masked-out (non-rotated) positions, the rotation reduces to identity, which is what partial RoPE wants.

5. **Load the activation halves (NeoX layout).**
   ```python
   x_ptr_base = X + pid_t * stride_token + pid_h * stride_head
   x_lo = tl.load(x_ptr_base + d_offsets,                mask=d_mask, other=0.0).to(tl.float32)
   x_hi = tl.load(x_ptr_base + d_offsets + (D // 2),     mask=d_mask, other=0.0).to(tl.float32)
   ```
   Cast to fp32 immediately. The rotation arithmetic must be fp32; doing it in fp16 loses precision at large positions (sin/cos of large angles are tiny).

6. **Apply the rotation in fp32.**
   ```python
   out_lo = x_lo * cos_row - x_hi * sin_row
   out_hi = x_hi * cos_row + x_lo * sin_row
   ```
   Sign convention: `out_lo = x_lo*cos - x_hi*sin`. If you write `+ x_hi*sin` instead of `- x_hi*sin`, you have rotated in the wrong direction — the model will produce nonsense early-token attention but may not crash.

7. **Store results back.** Cast to the activation dtype, apply the same boundary mask:
   ```python
   tl.store(out_ptr_base + d_offsets,            out_lo.to(x_dtype), mask=d_mask)
   tl.store(out_ptr_base + d_offsets + (D // 2), out_hi.to(x_dtype), mask=d_mask)
   ```
   For partial RoPE, the unrotated tail (`d_offsets >= rot_dim/2`) must either be skipped (in-place: leave as-is, do not store) or copied through (out-of-place: explicit copy load/store).

8. **GPT-J layout variant.** Replace step 5 with strided loads:
   ```python
   x_even = tl.load(x_ptr_base + 2 * d_offsets,     mask=d_mask, other=0.0).to(tl.float32)
   x_odd  = tl.load(x_ptr_base + 2 * d_offsets + 1, mask=d_mask, other=0.0).to(tl.float32)
   ```
   And step 6 produces `out_even`, `out_odd` stored back at the corresponding strided offsets. The cos/sin table indexing is the same.

9. **Apply to Q and K, not V.** RoPE rotates Q and K. V is left untouched. If the kernel takes a single tensor pointer, the launcher calls it once for Q and once for K. If the kernel takes both Q and K pointers and processes them together, ensure V is never passed.

10. **Handle the head-count mismatch for GQA.** Q has `H_q` heads, K has `H_kv` heads, where often `H_kv = H_q / 4` or `H_kv = H_q / 8`. They share the cos/sin tables (same position, same `D`). The grid for the Q launch is `(total_tokens, H_q)`; for K it is `(total_tokens, H_kv)`.

---

## Kernel design rules

- `BLOCK_D_HALF` must be a power of 2, declared `tl.constexpr`, and at least `D / 2`. For typical `D = 128`, use `BLOCK_D_HALF = 64`. For `D = 64`, use `BLOCK_D_HALF = 32`.
- All rotation arithmetic in fp32. Cast inputs to fp32 right after the load; downcast only at the final store.
- The cos/sin table must be loaded as fp32. If the table is stored as fp32 in memory, load directly. If the framework provides fp16/bf16 tables, either accept the precision loss (document it) or cast inside the kernel — but the underlying table-build must be fp32 to avoid loss at large positions.
- `positions` is an int32 or int64 tensor of shape `[total_tokens]`. Pass its dtype consistently. Do not compute positions inside the kernel from `tl.program_id(0)` — that breaks continuous batching.
- All strides (token stride, head stride, D-dimension stride for Q/K; row stride for cos/sin) must be passed as kernel arguments. Do not assume contiguous `(total_tokens, H, D)` layout.
- For partial RoPE, the rotated portion must be the **first** `rot_dim` dimensions in the standard convention. Confirm against the reference model — some research models rotate the last `rot_dim` instead.
- In-place writes are valid because each `(token, head, dim)` element is read once and written once with no cross-element dependence beyond the `(x_lo, x_hi)` pair, which is loaded fully before any store. There is no race within a program.
- A combined Q+K kernel (one launch handles both tensors) saves a launch's worth of latency for decode (length-1 Q/K), which matters in tight serving loops. For prefill, two launches are fine.

---

## Correctness requirements

- Layout must match the reference model's layout exactly. The standard test: feed a known input through the reference (HuggingFace `apply_rotary_pos_emb` for NeoX-layout models, the original RoPE paper code for GPT-J-layout models) and your kernel; outputs must match within fp16 tolerance (`atol=1e-3`).
- Sign convention: `out_lo = x_lo * cos - x_hi * sin`. If swapped to `+`, the rotation goes the wrong direction. Test against the reference, not against a hand-derived formula.
- The position used for token `i` must come from the `positions` array, not from `i % seq_len`. Continuous batching breaks the latter silently.
- For partial RoPE, the unrotated tail must be unchanged. Verify by checking that `out[..., rot_dim:] == x[..., rot_dim:]` exactly (bit-identical for in-place; equal for out-of-place copy).
- V must not be passed to this kernel. Apply RoPE to Q and K only.
- The cos/sin table must be built with `theta_i = base ** (-2i / D)` for `i in [0, D/2)`, then `angle = position * theta_i`, then `cos = cos(angle)`, `sin = sin(angle)`. Mismatch in the base (10000 vs 1000000 vs an NTK-scaled value) silently changes the embedding and breaks pretrained models.
- The boundary mask (`d_mask`) must be applied identically on load and store. Asymmetric masking corrupts memory or produces partial outputs.
- Cos/sin computation must be fp32 at table-build time. Storing the table in fp16 loses precision at long sequence positions because `sin(p * theta_i)` for large `p` and small `theta_i` produces values whose fp16 representation is much coarser than the true rotation.

---

## Performance requirements

The agent must reason about:

- **Memory traffic.** RoPE reads Q and K once and writes them back. For Q of shape `(total_tokens, H_q, D)` in fp16 and partial RoPE on the full `D`, the traffic is `2 * total_tokens * H_q * D * 2` bytes (read + write). The kernel is fully memory-bound; no amount of math optimization helps.
- **Cos/sin table reuse.** Every `(token, head)` program for the same token loads the same cos/sin row. With `H_q + H_kv` heads per token, this is read `H_q + H_kv` times per token. L2 cache absorbs this — do not preload the table into shared memory manually, Triton's tile-level caching handles it.
- **Decode-path sensitivity.** For decode (length-1 Q/K), the kernel processes `B` tokens (one per request). Launch latency dominates execution time; aim for one launch handling both Q and K, or fuse RoPE into the attention kernel via FA3-style fusion.
- **Combined vs separate launches.** A single kernel processing both Q and K in one launch saves one launch's worth of latency (~5-10 μs on H100). For decode, this matters; for prefill, the kernel runtime dwarfs launch overhead and either approach works.
- **Head dimension coverage.** With `D = 128` and `BLOCK_D_HALF = 64`, the kernel loads `D/2 = 64` elements in one tile — fits easily in registers, no inner loop needed. For unusually large head dims (`D = 256` on some models), `BLOCK_D_HALF = 128` still fits, but verify register pressure does not exceed occupancy targets.
- **Do not write a custom RoPE kernel if FA3 fuses it.** FlashAttention-3 fuses RoPE into the attention kernel, eliminating a full read-write round-trip on Q and K. A standalone Triton RoPE always loses to this fusion for prefill on H100.

---

## Output format

The agent should produce:

1. **The Triton kernel function** with `@triton.jit`, taking pointers (X or Q+K, output, cos, sin, positions), strides for each tensor's token/head/dim axes, shape parameters (`D`, `rot_dim`, `H_q` and/or `H_kv`), and `BLOCK_D_HALF: tl.constexpr`.
2. **A Python launcher** that computes the grid as `(total_tokens, H)`, validates that `D` is even and that `rot_dim <= D`, extracts strides, and invokes the kernel separately for Q and K (or jointly, if combined).
3. **An explicit comment naming the layout** (NeoX or GPT-J) at the top of the kernel. This single comment prevents 90% of layout-confusion bugs in downstream usage.
4. **A correctness test** comparing against `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` (for NeoX layout) at fp16 with `torch.allclose(atol=1e-3)`. For GPT-J layout, compare against a reference implementation derived from the original RoPE paper.
5. **A continuous-batching test** with at least two requests of different lengths and non-zero starting positions, verifying that each token's rotation uses its correct position.
6. **Documentation of any partial-RoPE handling**, including which dimensions are rotated and which are passed through.

---

## Common failure modes

- **Wrong layout (NeoX vs GPT-J).** The most insidious bug. The kernel runs, no NaN, no shape mismatch — but the model produces garbled or off-distribution output. Tests against a reference model in the right layout are the only reliable check.
- **Wrong sign of sin.** `out_lo = x_lo * cos + x_hi * sin` instead of `- x_hi * sin` rotates in the opposite direction. The model often still produces plausible-looking text early but degrades catastrophically with longer context. Catches: explicit sign comparison against reference at multiple positions, not just position 0.
- **Applying RoPE to V.** V must remain unrotated. Applying RoPE to V destroys the value semantics; the model output collapses to gibberish.
- **Using token index instead of true position.** With continuous batching, `pid_t` is a flat index into a packed batch, not a position within a sequence. Computing the position as `pid_t` or `pid_t % seq_len` produces wrong rotations for every token after the first request. Always read from a `positions` array supplied by the scheduler.
- **fp16 cos/sin precision loss at long positions.** If the cos/sin table is stored in fp16, then for position `p = 8000` and a small `theta_i ≈ 1e-4`, the angle `p * theta_i ≈ 0.8` is fine — but for `p = 100000` (extended-context models with YaRN), precision in fp16 is insufficient. Always build the table in fp32 and store it in fp32, or accept the precision loss explicitly for short-context models only.
- **Head-count mismatch silently scaling wrong heads.** If the launcher passes Q's `H_q` as the head count for both Q and K (where K should use `H_kv`), the K kernel reads beyond its valid head range and writes garbage. Pass the correct head count for each tensor.
- **Partial RoPE applied to the wrong half.** Some models rotate the first `rot_dim` dimensions; some rotate the last `rot_dim`. Check the reference model's slicing convention before coding.
- **Storing fp32 results to a fp16 buffer without cast.** Forgetting `out_lo.to(x_dtype)` before `tl.store` either crashes (Triton type-checks) or silently writes the fp32 bit pattern into the fp16 buffer. Always explicit cast at the store boundary.
- **Off-by-one in cos/sin row stride.** Cos/sin tables have shape `[max_seq_len, D/2]` — row stride is `D/2`, not `D`. Using `pos * D` reads the wrong row.
- **Forgetting that `D/2` may not be a power of 2.** For `D = 192` (used in some research models), `D/2 = 96` is not a power of 2. `BLOCK_D_HALF` must still be a power of 2 (e.g., 128) with masking on the unused tail.

---

## Review checklist

- [ ] Layout convention (NeoX or GPT-J) is named explicitly in a kernel-level comment.
- [ ] Cos and sin tables are built in fp32 and have shape `[max_seq_len, D/2]`.
- [ ] Position for each token is read from the `positions` array, not derived from `program_id`.
- [ ] All rotation arithmetic is done in fp32 after immediate cast from input dtype.
- [ ] Sign convention `out_lo = x_lo*cos - x_hi*sin`, `out_hi = x_hi*cos + x_lo*sin` (NeoX) is verified against a reference.
- [ ] RoPE is applied to Q and K only — V is never passed to this kernel.
- [ ] For GQA, Q uses `H_q` heads and K uses `H_kv` heads in their respective launches.
- [ ] Partial RoPE (if applicable) leaves the unrotated tail bit-identical.
- [ ] `BLOCK_D_HALF` is a power of 2 and declared `tl.constexpr`.
- [ ] Boundary masking on load and store is symmetric, with `other=1.0` for cos and `other=0.0` for sin.
- [ ] The output cast `to(x_dtype)` is present immediately before every `tl.store`.
- [ ] Strides for the token, head, and dim axes are kernel arguments, not hardcoded.
- [ ] A correctness test against the reference HuggingFace or original-paper implementation passes at fp16 with `atol=1e-3`.
- [ ] A continuous-batching test with at least two requests of differing lengths and non-zero start positions passes.
- [ ] No claim of speedup over FlashAttention-3-fused RoPE without a benchmark.
