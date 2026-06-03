# Skill: Write a Triton GEMM Kernel

## Purpose

Guide the agent through implementing a correct, performant blocked matrix multiplication kernel in Triton. This covers tile assignment via `program_id`, pointer arithmetic for A/B/C tiles, accumulation with `tl.dot`, boundary masking for non-divisible shapes, swizzled tile ordering for L2 reuse, and autotuning for BLOCK_M/BLOCK_N/BLOCK_K/num_stages/num_warps.

---

## Use this when

- You need a custom GEMM or batched GEMM that fuses an epilogue (bias add, activation, scaling, etc.) that cuBLAS or CUTLASS cannot express without a separate kernel.
- You need a GEMM on a dtype or layout combination that vendor libraries do not natively support efficiently (e.g., mixed-precision accumulation, custom quantized formats).
- You are building a research kernel and need full visibility into the tiling strategy.
- The matmul is not on the hot path and you want a single portable Triton implementation rather than a CUTLASS build dependency.

---

## Do not use this when

- The operation is a standard fp16/bf16/fp32 GEMM with no epilogue fusion requirements. Use `torch.compile`, `torch.mm`, or cuBLAS directly — they will match or beat a hand-written Triton GEMM at most shapes.
- The required shapes are very small (M or N < 64). cuBLAS handles these with batched or grouped GEMM routines that are difficult to match in Triton.
- You need int8 or fp8 tensor core throughput with fused dequantization. Prefer CUTLASS or cuDNN unless you have a specific reason to own this kernel.
- Latency matters more than throughput and the problem is memory-bandwidth-bound at small batch. Profiling should drive this decision — do not assume Triton wins.

---

## Inputs the agent should gather first

Before writing any code, confirm:

1. **M, N, K** — exact values or the range of values expected at runtime (static vs dynamic shapes).
2. **Input dtype** — fp16, bf16, fp32, or mixed (e.g., bf16 inputs, fp32 accumulation).
3. **Layout of A and B** — row-major or column-major. If transposed, clarify whether the caller passes the transpose or the kernel should handle it internally.
4. **Output dtype** — same as input or upcast.
5. **Epilogue** — plain C = A @ B, or is there a scaling factor alpha, bias addition, activation function, or in-place accumulation into an existing C?
6. **Batch dimension** — standard 2D matmul, batched (B, M, K) x (B, K, N), or broadcasted batch?
7. **Hardware target** — A100, H100, or other. This determines tensor core eligibility and the optimal pipeline depth.
8. **Whether autotuning is allowed** — production kernels that ship with a fixed config need to justify that choice; autotuned kernels need a representative benchmark shape.

---

## Required reasoning process

1. **Confirm tensor core eligibility.** For fp16/bf16 inputs on A100/H100, `tl.dot` maps to tensor core instructions when BLOCK_K >= 16 and BLOCK_M, BLOCK_N >= 16. Confirm the dtypes support this path.

2. **Determine tile assignment.** Each Triton program handles one (BLOCK_M, BLOCK_N) output tile. Compute the total number of tiles: `num_pid_m = ceil(M / BLOCK_M)`, `num_pid_n = ceil(N / BLOCK_N)`. Assign via:
   ```
   pid = tl.program_id(0)
   num_pid_in_group = GROUP_SIZE_M * num_pid_n
   group_id = pid // num_pid_in_group
   first_pid_m = group_id * GROUP_SIZE_M
   group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
   pid_m = first_pid_m + (pid % group_size_m)
   pid_n = (pid % num_pid_in_group) // group_size_m
   ```
   This swizzled ordering groups tiles that share rows of A into a cache group, improving L2 reuse for B tiles.

3. **Compute base pointers.** For row-major A (shape M x K, stride_am=K, stride_ak=1):
   ```
   offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
   offs_k  = tl.arange(0, BLOCK_K)
   a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
   ```
   Advance similarly for B. Update pointers inside the K loop with `a_ptrs += BLOCK_K * stride_ak`.

4. **Accumulate in fp32.** Initialize `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`. Call `acc += tl.dot(a_tile, b_tile)`. Downcast to output dtype only after the loop ends.

5. **Apply boundary masking.** When M, N, or K are not divisible by the tile size:
   - Load A with `mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)`.
   - Pass `other=0.0` to `tl.load` so out-of-bounds positions do not contribute to the dot product.
   - Mask the C store with `(offs_cm[:, None] < M) & (offs_cn[None, :] < N)`.

6. **Apply epilogue.** After the K loop, apply alpha scaling, bias, or activation to `acc` before the store.

7. **Configure autotuning.** Define a list of configs over BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps. Key parameters for cache invalidation should include M, N, K so the autotuner re-runs when shapes change.

8. **Validate correctness against a reference.** Run `torch.mm` on cpu or with `torch.float64` and compare with `torch.allclose(tol=1e-2)` for fp16 kernels.

---

## Kernel design rules

- BLOCK_M, BLOCK_N, BLOCK_K must all be powers of 2 and declared `tl.constexpr`.
- BLOCK_K must be >= 16 for `tl.dot` to use tensor cores on fp16/bf16 inputs.
- Accumulate in `tl.float32` regardless of input dtype. Accumulating in fp16 introduces catastrophic cancellation on long K dimensions.
- Use swizzled tile ordering (GROUP_SIZE_M grouping) when L2 reuse for A tiles matters — typically when N is large relative to L2 cache.
- Do not use `tl.atomic_add` for C writes unless implementing a split-K variant. Standard GEMM writes each output tile exactly once.
- For split-K: each program computes a partial sum over a K slice, then atomically accumulates into a workspace buffer. Requires a separate reduction pass or atomic writes. Only implement split-K if K >> M*N and parallelism is otherwise insufficient.
- Pointer strides must be passed as kernel arguments — never hardcode strides computed from shapes, as this breaks non-contiguous tensors.
- Set `tl.load` eviction policy to `evict_last` for the B tile when B is reused across the M tile group (swizzled layout). Use `evict_first` for A when L2 pressure from B is high.

---

## Correctness requirements

- Every load in the K loop must have a mask that checks both the row/col index and the K boundary. A missing K-boundary mask causes reads from out-of-bounds memory when K is not a multiple of BLOCK_K.
- The output store must mask both M and N dimensions.
- Pointer updates inside the K loop must use the stride arguments, not hardcoded values, or non-contiguous tensors will silently produce wrong results.
- Accumulation must be in fp32. Verify this by inspecting the `dtype` argument of `tl.zeros`.
- The tile assignment formula must produce a bijection: every (pid_m, pid_n) pair must be assigned to exactly one program, with no gaps and no overlaps. Verify by checking `total_programs = num_pid_m * num_pid_n` matches the kernel launch grid.
- For batched GEMM, stride_batch_a and stride_batch_b must be separate arguments. Broadcasting (one matrix shared across the batch) requires stride = 0, not reuse of the M/K stride.

---

## Performance requirements

The agent must reason through all of the following before finalizing the kernel:

- **Tile shape vs occupancy.** Larger tiles (BLOCK_M=128, BLOCK_N=128) improve arithmetic intensity but require more shared memory and registers, reducing occupancy. On A100, 128x128x32 tiles with fp16 use ~96KB smem, filling one SM. Trade this against parallelism for small M/N.
- **BLOCK_K and pipeline depth.** A100 L2 bandwidth is ~2TB/s. Pipelining `num_stages=3` or `4` hides the latency of async loads from global memory. BLOCK_K=32 or 64 is typical; larger BLOCK_K increases smem pressure.
- **num_warps.** 4 or 8 warps per block. More warps improve latency hiding but consume more registers. For 128x128 tiles, `num_warps=8` is typical on A100.
- **Swizzling benefit.** Quantify whether your problem is B-bandwidth-bound. If N is small, swizzling is irrelevant. If N is large and M is large, GROUP_SIZE_M=8 is a reasonable starting point.
- **Arithmetic intensity.** A GEMM tile of (BLOCK_M x BLOCK_N x BLOCK_K) has 2*BLOCK_M*BLOCK_N*BLOCK_K FLOPs and loads BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N elements. For 128x128x64 fp16: 2M FLOPs / 49KB data = ~41 FLOPs/byte. This exceeds A100's fp16 arithmetic threshold (~14 FLOPs/byte), so the kernel is compute-bound at this tile size.
- **Do not claim a speedup over cuBLAS without measurement.** Triton GEMMs at standard shapes typically underperform cuBLAS by 5-20% due to lack of fine-grained ldmatrix/stmatrix scheduling. This gap may matter for production throughput.

---

## Output format

The agent should produce:

1. **The Triton kernel function** annotated with `@triton.autotune` and `@triton.jit`, with all tile parameters as `tl.constexpr`, all strides and dimensions as regular arguments.
2. **The Python launcher function** that computes the grid, handles contiguity checks (`tensor.contiguous()`), extracts strides, and calls the kernel.
3. **A brief design note** (inline comment block or docstring) explaining: chosen tile sizes and why, accumulation dtype, masking strategy, and epilogue handling.
4. **A correctness test** using `torch.mm` as reference with `torch.allclose`.
5. **An autotune config list** with at least 6 configs covering the likely Pareto frontier for the target shape range.

---

## Common failure modes

- **Wrong pid mapping.** Using `pid_m = pid // num_pid_n` and `pid_n = pid % num_pid_n` is correct for row-major tile ordering but misses swizzling. Forgetting to recompute `num_pid_n` when using GROUP_SIZE_M causes tile overlaps or gaps.
- **Accumulating in fp16.** `tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)` silently degrades precision. The error compounds across K and may not be visible on short K but becomes severe for K >= 2048.
- **Incorrect K boundary mask.** When K is not a multiple of BLOCK_K, the final iteration reads beyond the valid range of A and B. The mask must check `offs_k < K - k_start` (the remaining K), not `offs_k < K`.
- **BLOCK_K < 16 for tl.dot.** Triton will fall back to a slow SIMD path. BLOCK_K must be >= 16 for tensor core dispatch.
- **Hardcoded stride as shape product.** `a_ptr + row * K + col` breaks when A is a non-contiguous slice. Always pass `stride_am`, `stride_ak` as kernel arguments.
- **Missing contiguous() call.** If the input tensor is non-contiguous (e.g., from a transpose or slice), strides will be non-trivial. Either call `.contiguous()` before launch or handle arbitrary strides in the kernel — not both.
- **Forgetting alpha/beta semantics.** If the API contract includes alpha*A@B + beta*C (BLAS GEMM), omitting the beta*C term or applying it before the K loop produces incorrect results.
- **Grid size off by one.** `grid = (ceil_div(M, BLOCK_M) * ceil_div(N, BLOCK_N),)` must use ceiling division, not floor. `M // BLOCK_M` drops the last partial tile.

---

## Review checklist

- [ ] BLOCK_M, BLOCK_N, BLOCK_K are powers of 2 and declared `tl.constexpr`.
- [ ] BLOCK_K >= 16 for tensor core dispatch on fp16/bf16.
- [ ] Accumulation uses `tl.float32` regardless of input dtype.
- [ ] K loop boundary mask checks both the static dimension bounds and the loop-variable K offset.
- [ ] Output store is masked on both M and N dimensions.
- [ ] All strides are passed as kernel arguments and not recomputed from shapes.
- [ ] Grid size uses ceiling division.
- [ ] Tile assignment produces a bijection over all (pid_m, pid_n) pairs.
- [ ] Swizzle grouping is present and GROUP_SIZE_M is justified or explicitly set to 1 with reasoning.
- [ ] Autotune config list covers at least 6 configs and includes variation in BLOCK_K and num_stages.
- [ ] A correctness test against `torch.mm` is present.
- [ ] The launcher calls `.contiguous()` or explicitly documents stride handling.
- [ ] Epilogue is applied after accumulation, not inside the K loop.
- [ ] No performance claims are made without a benchmark call to back them up.
