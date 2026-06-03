# Skill: Write a Triton SiLU-Mul (SwiGLU) Kernel

## Purpose

Guide the agent through implementing a correct, numerically stable Triton kernel that computes `y = silu(a) * b`, the elementwise activation step inside SwiGLU MLPs used by LLaMA, Mistral, Qwen, Gemma, and similar modern LLMs. The full MLP is `down_proj( silu(gate_proj(x)) * up_proj(x) )`; this skill covers the fused activation that sits between the two GEMMs. It also generalizes to GeGLU (`gelu(a) * b`) and ReGLU (`relu(a) * b`), which share the same kernel structure with a different activation.

---

## Use this when

- You need a fused elementwise kernel that reads `a` and `b` once and writes `y` once, instead of materializing `silu(a)` as a separate tensor.
- You are writing an inference path where `gate_proj` and `up_proj` are computed separately (or as a single fused matmul producing `[gate, up]`) and the activation is a distinct kernel call between the matmuls.
- The matmul backend (cuBLAS, CUTLASS without a custom epilogue, or a vendor library) does not allow you to fuse the activation into the matmul epilogue.
- The intermediate tensor is wide enough (e.g., `intermediate_size` of 14336, 28672, or larger) that the bandwidth cost of materializing `silu(a)` separately is measurable.
- You want a GeGLU or ReGLU variant — same kernel skeleton, different activation function.

---

## Do not use this when

- `torch.nn.functional.silu(a) * b` under `torch.compile` already fuses the chain on your PyTorch build. Validate this with `TORCH_COMPILE_DEBUG=1` before writing a custom kernel — modern inductor handles this case well.
- A working CUDA implementation already exists in your serving stack. vLLM ships `silu_and_mul` in `csrc/activation_kernels.cu`; SGLang and TensorRT-LLM have equivalents. Re-implementing in Triton is only worth it if you need backend portability or kernel-level fusion with an adjacent op.
- You can fuse the activation into the matmul epilogue (CUTLASS epilogue visitor, Triton matmul with custom epilogue). A standalone elementwise kernel always pays an extra round trip to HBM; the epilogue does not.
- The shape is small enough that kernel launch overhead dominates (e.g., `B*T*intermediate_size < 1M elements`). At that size, any reasonable implementation is fine.
- You need the backward pass for training. The forward kernel is straightforward, but the backward must save `a` and `b` (or `silu(a)` and `b`) and recompute `silu'(a)`. Plan the autograd function before writing the forward.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **Activation choice** — SiLU (LLaMA, Mistral, Qwen, Gemma), GELU (some PaLM-style models), or ReLU. SiLU is `x * sigmoid(x)`. Confirm which one the model was trained with — using GELU when the model expects SiLU silently produces wrong outputs.
2. **GELU variant (if GELU)** — erf-based (`0.5 * x * (1 + erf(x/sqrt(2)))`) or tanh-approximation. These are not interchangeable. LLaMA does not use GELU at all; Gemma uses GeGLU with the exact erf form.
3. **Input shape** — typically `(B, T, intermediate_size)` or `(B*T, intermediate_size)`. Is the activation dimension the last (contiguous) dimension?
4. **Layout of `a` and `b`** — separate tensors of identical shape, or interleaved as a single `(B, T, 2*intermediate_size)` tensor with `a = x[..., :intermediate_size]` and `b = x[..., intermediate_size:]`? The interleaved form is common when `gate_proj` and `up_proj` are fused into a single GEMM.
5. **Input dtype** — fp16, bf16, or fp32. fp16 and bf16 require fp32 sigmoid; fp32 inputs do not.
6. **Output dtype** — usually matches input dtype, but the next op (`down_proj`) may want a specific dtype. Confirm.
7. **In-place vs out-of-place** — can `y` overwrite `b`? In inference, in-place is common to save memory.
8. **Backward needed** — forward-only is the inference case; backward requires saving inputs.
9. **`intermediate_size`** — typical values: 11008 (LLaMA-7B), 14336 (LLaMA-13B), 28672 (LLaMA-70B), 18944 (Qwen2-7B). These are not powers of 2; masking matters.

---

## Required reasoning process

1. **Pick the program decomposition.** Treat `a` and `b` as flat 1D buffers of length `N = prod(shape)`. One program handles a `BLOCK_SIZE`-element tile. Grid is `(triton.cdiv(N, BLOCK_SIZE),)`. This is bandwidth-bound; the 1D decomposition is the simplest correct choice.

2. **Compute tile offsets and mask.**
   ```python
   pid = tl.program_id(0)
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   mask = offsets < N
   ```
   Mask must be applied on every load and the final store.

3. **Load `a` and `b` with masking.**
   ```python
   a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
   b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
   ```
   `other=0.0` is safe here because masked positions are not written.

4. **Cast to fp32 for the activation.**
   ```python
   a_f32 = a.to(tl.float32)
   ```
   `b` does not need fp32 for the multiply itself, but cast it if you want to keep the multiply in fp32 before downcasting the result.

5. **Compute SiLU in fp32.**
   ```python
   silu_a = a_f32 * tl.sigmoid(a_f32)
   ```
   Equivalently `a_f32 / (1.0 + tl.exp(-a_f32))`. `tl.sigmoid` is preferred — it is numerically stable for large negative inputs.

6. **Multiply and downcast.**
   ```python
   y = silu_a * b.to(tl.float32)
   tl.store(y_ptr + offsets, y.to(a.dtype), mask=mask)
   ```
   Cast the final result back to the storage dtype right before the store.

7. **For GeGLU**, replace step 5 with the GELU formula. Use the erf form unless the model card specifies tanh-approximation:
   ```python
   gelu_a = 0.5 * a_f32 * (1.0 + tl.erf(a_f32 * 0.7071067811865475))
   ```

8. **For the interleaved layout**, the kernel takes a single input pointer and computes `a` and `b` offsets inside the kernel:
   ```python
   row = offsets // intermediate_size
   col = offsets % intermediate_size
   a = tl.load(in_ptr + row * (2 * intermediate_size) + col, mask=mask)
   b = tl.load(in_ptr + row * (2 * intermediate_size) + intermediate_size + col, mask=mask)
   ```
   This is more arithmetic per element but avoids splitting the tensor.

9. **For backward**, compute and save `sigmoid(a)` (or `silu(a)`) during forward. The backward kernel reads `a`, `b`, and `dy`, and produces:
   ```
   dx_a = dy * b * silu'(a)        where silu'(a) = silu(a) + sigmoid(a) * (1 - silu(a))
   dx_b = dy * silu(a)
   ```
   Recomputing `silu(a)` and `sigmoid(a)` from saved `a` is usually cheaper than storing extra activations.

---

## Kernel design rules

- `BLOCK_SIZE` must be a power of 2 and declared `tl.constexpr`. Start at 1024 or 2048; tune via `triton.testing.Benchmark`.
- All sigmoid and GELU computation must run in fp32 when inputs are fp16 or bf16. fp16 sigmoid saturates around `|x| >= 10` and silently loses precision around `|x| ~ 5`, where SiLU still has meaningful curvature.
- Use `tl.sigmoid(x)` rather than `1.0 / (1.0 + tl.exp(-x))`. The library form is numerically stable for very negative `x`; the manual form overflows `exp(-x)` for large negative `x` in some dtype paths.
- Cast back to the storage dtype only at the final store. Doing intermediate downcasts inside the activation chain reintroduces the precision loss the fp32 cast was meant to avoid.
- Pass `N` (total element count) and `BLOCK_SIZE` as kernel arguments. Do not hardcode shape-derived constants.
- Do not assume `intermediate_size` is divisible by `BLOCK_SIZE`. Real model sizes (11008, 14336, 18944, 28672) are not powers of 2, and even when they are, batch and sequence dimensions usually aren't. Always mask.
- For the interleaved layout, derive the pointer arithmetic from the tensor's strides, not from a hardcoded `2 * intermediate_size`. Strided inputs (sliced views) will break otherwise.
- `num_warps` typically 4 or 8 for this kernel. More warps do not help a memory-bandwidth-bound op — they only increase scheduling overhead.

---

## Correctness requirements

- `silu(a) * b` must match `torch.nn.functional.silu(a) * b` within `atol=1e-2, rtol=1e-2` for fp16/bf16 and `atol=1e-5, rtol=1e-5` for fp32. The fp16/bf16 tolerance is loose because the reference itself is fp16/bf16; an fp32 reference would tighten this.
- The activation must be applied to `a`, not `b`. Swapping these is a silent correctness bug — the output looks reasonable but does not match the trained model.
- For fused `gate_proj`/`up_proj` matmuls producing a `[gate, up]` interleaved tensor, the gate half (first `intermediate_size`) is `a` (gets SiLU); the up half (second `intermediate_size`) is `b` (no activation). Confirm the layout convention with the model checkpoint — some implementations swap these.
- Boundary masking must cover the last partial tile. With `intermediate_size = 11008` and `BLOCK_SIZE = 1024`, the last tile of every row (or of the flat buffer) has masked positions. Storing without a mask corrupts adjacent memory.
- The activation choice must match the model's training. SiLU and GELU(erf) differ by up to ~0.05 in the `[-2, 2]` range — small enough that bugs aren't obvious from sample outputs, but large enough to degrade benchmarks.
- For backward, `silu'(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))`, which is algebraically equivalent to `silu(a) + sigmoid(a) * (1 - silu(a))`. Either form is correct; pick one and verify against `torch.autograd.gradcheck` on fp64 inputs.

---

## Performance requirements

The agent must reason through the following before finalizing:

- **Arithmetic intensity.** Per element: 2 loads + 1 store = 3 memory ops. Roughly 4 FLOPs (sigmoid as one op, two multiplies, one add). For fp16, that is `4 FLOPs / 6 bytes ≈ 0.67 FLOP/byte` — firmly memory-bandwidth-bound on every modern GPU. State this explicitly: tuning compute (more warps, more unrolling) will not help.
- **Achievable bandwidth.** A correctly written kernel should hit 70–90% of peak HBM bandwidth on H100/A100 for large `N`. If you measure significantly less, the bottleneck is launch overhead (`N` too small), occupancy collapse (`BLOCK_SIZE` too large), or non-contiguous loads.
- **`BLOCK_SIZE` tuning.** Sweep over `{512, 1024, 2048, 4096}` and `num_warps in {4, 8}`. Larger `BLOCK_SIZE` reduces launch overhead but increases register pressure; on H100, `BLOCK_SIZE=1024, num_warps=4` is a strong default.
- **Versus epilogue fusion.** A standalone activation kernel reads `a` and `b` from HBM after the matmul wrote them there. Epilogue fusion in the matmul keeps `a` and `b` in registers/shared memory and writes only `y`. The epilogue form saves roughly `2 * N * dtype_bytes` of HBM traffic. For `intermediate_size=14336` and `B*T=4096`, that is ~470 MB saved per layer at fp16. State explicitly: "this kernel is preferred only when epilogue fusion is not available."
- **Versus `torch.compile`.** Benchmark `torch.compile`'d `silu(a) * b` first. If it is within 5–10% of your custom kernel, do not ship the custom kernel — the maintenance cost is not justified.
- **Interleaved layout cost.** The interleaved-layout variant adds a divmod per element. For `intermediate_size` that is a power of 2, the compiler may strength-reduce these to shifts/masks; for non-power-of-2 sizes, expect a small (~5–10%) throughput hit versus the split-tensor form.

---

## Output format

The agent should produce:

1. **Activation and layout summary** — one short paragraph: which activation (SiLU/GELU-erf/GELU-tanh/ReLU), which layout (split vs interleaved), input/output dtypes, in-place flag.
2. **The Triton kernel** with `@triton.jit`, taking `a_ptr`, `b_ptr`, `y_ptr`, `N`, and `BLOCK_SIZE: tl.constexpr`. For the interleaved layout, take `in_ptr`, `y_ptr`, `intermediate_size`, `N`, `BLOCK_SIZE`.
3. **A Python launcher** that flattens `a` and `b` to 1D, computes the grid, validates `a.shape == b.shape` and dtype, and handles contiguity (call `.contiguous()` if needed, and document that this allocates).
4. **The variant kernel(s)** if GeGLU or ReGLU is also requested — same skeleton, different activation expression, clearly labeled.
5. **A correctness test** comparing against `torch.nn.functional.silu(a) * b` with appropriate tolerances per dtype, on shapes that include non-power-of-2 `intermediate_size` (e.g., 11008).
6. **A benchmark** comparing the kernel against `torch.compile(lambda a,b: F.silu(a)*b)` and reporting GB/s achieved and percent of theoretical peak HBM bandwidth.
7. **An explicit statement** of when the user should NOT ship this kernel: "if torch.compile fusion is within 10%, or if matmul epilogue fusion is available."

---

## Common failure modes

- **Sigmoid in fp16.** `tl.sigmoid` on an fp16 tensor without an explicit fp32 cast saturates at `|x| ~ 6` (one-ulp error grows fast there), and the multiply `x * sigmoid(x)` near zero loses several bits. Symptom: model perplexity drifts up by ~0.05–0.2 with no obvious crash. Always cast to fp32 before sigmoid.
- **Wrong activation.** Using GELU when the model trained with SiLU, or vice versa. The output looks plausible but is mathematically wrong. Always confirm the activation against the model config (`config.json` `hidden_act` field for HuggingFace models — `silu` or `gelu`).
- **GELU variant mismatch.** Using tanh-approximation GELU when the model uses erf-GELU (or vice versa). The tanh form is `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`; the erf form uses `erf(x/sqrt(2))`. They differ by up to ~5e-4 in absolute terms — small per element, large enough to shift quality metrics on long generations.
- **Swapped `a` and `b`.** Applying SiLU to the up projection instead of the gate projection. Both halves are the same shape and dtype, so there is no shape error — only a quality regression.
- **Off-by-one mask.** `mask = offsets <= N` instead of `offsets < N`. Reads one byte past the end; on most allocations this happens to return zero or a benign value, masking the bug until a later allocation pattern exposes it.
- **In-place store before the load completes.** If `y_ptr == b_ptr` (in-place over `b`), the kernel must not store any element before all reads of `b` for that tile are complete. In Triton, `tl.load` of the full tile happens before any `tl.store`, so per-tile in-place is safe — but only within a single program. Cross-program aliasing is fine because each program owns disjoint offsets.
- **Hardcoding `2 * intermediate_size` for interleaved layout.** If the input tensor came from `torch.cat([gate, up], dim=-1)` followed by a `.transpose()` or slicing, the stride is no longer `2 * intermediate_size`. Read strides from the tensor metadata.
- **Forgetting to call `.contiguous()`.** A view of a transposed tensor has the right shape but wrong strides. The kernel reads the wrong elements with no error. Either enforce contiguity in the launcher or thread strides through the kernel.
- **Backward without saving `a`.** The backward needs `a` (or `silu(a)` and `sigmoid(a)`). Saving only `silu(a) * b` is insufficient — `silu'(a)` cannot be recovered from the product.

---

## Review checklist

- [ ] `BLOCK_SIZE` is a power of 2 and declared `tl.constexpr`.
- [ ] Sigmoid and any GELU call run on an fp32 tensor; inputs are cast immediately after `tl.load`.
- [ ] `tl.sigmoid` is used rather than a manual `1 / (1 + exp(-x))` formulation.
- [ ] Final downcast to storage dtype happens only at the `tl.store`.
- [ ] Boundary mask `offsets < N` is applied on every load and the store.
- [ ] The activation is applied to `a` (gate), not `b` (up). Confirmed against the model's `hidden_act`.
- [ ] For GELU variant, the erf vs tanh form matches the model's training; documented in a comment.
- [ ] Interleaved-layout kernel reads strides from tensor metadata, not from `2 * intermediate_size`.
- [ ] Launcher validates `a.shape == b.shape` and matching dtypes; calls `.contiguous()` or threads strides explicitly.
- [ ] Correctness test covers fp16, bf16, and fp32, with at least one non-power-of-2 `intermediate_size` (e.g., 11008 or 14336).
- [ ] Benchmark reports GB/s and percent of theoretical HBM peak; comparison against `torch.compile` is included.
- [ ] No claim of "faster than X" without a measurement.
- [ ] Skill of when NOT to ship this kernel (torch.compile is close, or epilogue fusion is available) is stated in the output.
- [ ] Backward, if implemented, saves `a` (not `silu(a)`) and verified against `torch.autograd.gradcheck` at fp64.
