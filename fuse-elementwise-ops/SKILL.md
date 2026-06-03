# Skill: Fuse Elementwise Operations

## Purpose
Guide the agent through deciding whether to fuse multiple elementwise operations into a single kernel pass, and if so, how to implement the fusion correctly and efficiently in CUDA or Triton.

## Use this when
- Two or more consecutive elementwise operations are applied to the same tensor and the intermediate results are not reused elsewhere.
- Profiling shows the pipeline is memory-bandwidth bound and the operations between loads and stores are cheap arithmetic.
- Epilogue fusion into an existing GEMM or convolution kernel is being considered (e.g., bias add + activation after GEMM).
- The operation chain is simple enough that a fused kernel remains readable and maintainable.
- torch.compile/inductor is unavailable or insufficient (e.g., custom op, non-standard dtype, or strict latency requirements).

## Do not use this when
- Any operation in the chain requires inter-element communication — softmax, layernorm, and reductions are not elementwise and require separate synchronization steps or multi-pass designs.
- The operations involve different tensor shapes that require broadcasting logic complex enough to obscure correctness.
- torch.compile with `mode="reduce-overhead"` or `mode="max-autotune"` already fuses the chain adequately — validate this before writing a manual kernel.
- The chain is so long that register pressure in the fused kernel would reduce occupancy below the unfused baseline.
- Development and maintenance cost of a custom kernel is not justified by the measured speedup.

## Inputs the agent should gather first
- The ordered list of operations to fuse, with their mathematical definitions.
- Input and output dtypes for each operation; flag any dtype transitions (e.g., fp16 input, fp32 accumulation, fp16 output).
- Tensor shapes and memory layouts (contiguous, strided, transposed) for all inputs.
- Whether any intermediate tensor is consumed by a path other than the next operation in the chain. If yes, fusion is not valid.
- Target hardware (compute capability, memory bandwidth, L2 size) and whether the workload is bandwidth-bound or compute-bound on that hardware.
- Whether torch.compile has already been tried and what the result was.
- Epilogue context: if this follows a GEMM, what CUTLASS or cuBLAS epilogue API is available.

## Required reasoning process
1. **Classify each operation.** Confirm every operation is strictly elementwise: output at index `i` depends only on inputs at index `i`. Operations like `exp`, `relu`, `mul`, `add`, `cast`, `clamp` qualify. Softmax, cumsum, and reductions do not.

2. **Count memory traffic before and after fusion.** For an unfused chain of N operations over a tensor of size S bytes:
   - Unfused traffic = 2 * N * S (N reads + N writes, ignoring L2 reuse).
   - Fused traffic = 2 * S (one read pass, one write pass).
   - Estimate bandwidth saving as (2N - 2) * S. If S is small enough to fit in L2, unfused kernels may already benefit from cache reuse — adjust accordingly.

3. **Check arithmetic intensity of the fused kernel.** If the fused kernel has very low arithmetic intensity (few FLOPs per byte), it will remain bandwidth-bound. Fusion still helps by reducing total bytes transferred, but do not expect compute-bound behavior.

4. **Assess register pressure.** Each live intermediate value in the fused operation chain consumes registers. Enumerate the intermediates that must be live simultaneously at the peak of the computation. If this count is high, check occupancy impact before proceeding.

5. **Check dtype consistency.** If operations transition from one dtype to another, insert explicit casts. Never silently widen or narrow at the wrong stage — state where each cast occurs and why.

6. **Decide on CUDA vs Triton vs torch.compile.**
   - If the chain is standard PyTorch ops, try torch.compile first. Verify fusion via `torch.compile` trace or `TORCH_COMPILE_DEBUG=1`.
   - If the chain includes custom ops, non-standard dtypes, or requires strict latency control, write a manual kernel.
   - Prefer Triton for rapid prototyping; prefer CUDA if the kernel needs to be embedded in a C++ inference stack.

7. **Design the kernel structure.**
   - CUDA: one thread per output element (or vectorized: one thread per 4 elements using `float4`/`half2`). Load once, apply all ops, store once.
   - Triton: one program per tile. Use `tl.load` with mask, apply ops as tensor expressions, `tl.store` with mask.

8. **Handle epilogue fusion into GEMM/conv separately.** This requires integrating with CUTLASS epilogue visitors or cuBLAS workspace epilogue APIs, not writing a standalone elementwise kernel. Treat this as a distinct design path.

## Kernel design rules
- Load all required inputs for a tile/thread in a single coalesced pass before beginning computation. Do not interleave loads and stores.
- Use vectorized loads (`float4`, `half2`, `__nv_bfloat162`) when the tensor is contiguous and aligned to the vector width. Do not assume alignment — check pointer alignment at runtime or enforce it via allocation.
- In Triton, use `tl.load(ptr, mask=mask, other=0.0)` for boundary tiles. Never load without a mask on the last tile.
- In CUDA, guard out-of-bounds threads with `if (idx < N) { ... }` before any load or store.
- Apply dtype casts explicitly between operations that change precision. Cast after load if input is fp16/bf16 and accumulation should be fp32. Cast before store if output dtype differs from accumulation dtype.
- Do not re-derive or re-compute the same subexpression more than once in a fused kernel. Assign to a named variable.
- For epilogue fusion into GEMM: the fused ops must depend only on the GEMM output element at the current thread's index. Any operation that requires a global reduction (e.g., layer norm after GEMM) cannot be fused into the GEMM epilogue in a single pass.

## Correctness requirements
- The fused kernel must produce output bitwise identical (or within expected floating-point rounding tolerance) to the sequential unfused execution of each operation on the reference device (CPU or fp32 GPU).
- Verify correct handling of the last partial tile. Every boundary thread must either compute a valid output or be masked out — never write garbage to out-of-bounds positions.
- When fusing operations with different dtypes, confirm that all intermediates are in the correct dtype at each stage. A cast placed in the wrong order can change numerical results.
- If any input tensor is non-contiguous (strided), the indexing logic must account for strides explicitly. Treating a strided tensor as contiguous is a correctness bug.
- For epilogue fusion, verify that the fused output matches the unfused output across a range of GEMM shapes, including non-power-of-2 sizes.

## Performance requirements
- Measure achieved memory bandwidth against the theoretical peak for the target GPU. A well-implemented fused elementwise kernel should achieve 70–90% of peak bandwidth on modern hardware for large tensors.
- Compare fused vs unfused wall-clock time using `torch.utils.benchmark.Timer` or `cudaEvent` timing. Do not rely on roofline estimates alone — measure.
- Verify occupancy using `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (CUDA) or Triton's compiled metadata. Low occupancy from register pressure can negate bandwidth savings.
- For vectorized loads, confirm the tensor pointer is aligned to 16 bytes (for `float4`) before using vectorized instructions. Misaligned vectorized loads fall back to scalar on some architectures.
- If using Triton, tune `BLOCK_SIZE` for the target GPU. Common starting points: 1024 or 2048 elements per program instance for large 1D tensors. Use `triton.testing.Benchmark` to confirm.
- State explicitly: "this kernel is bandwidth-bound at this arithmetic intensity and should not be expected to improve further from compute-side tuning."

## Output format
The agent should produce:

1. **Fusion decision summary**: one paragraph stating which ops are fused, why fusion is valid, estimated bandwidth reduction, and whether torch.compile was considered.
2. **Dtype and cast map**: a table or annotated expression showing the dtype at each stage of the computation chain.
3. **Kernel implementation**: complete, compilable kernel code with launch parameters. Include the scalar fallback for non-aligned inputs if using vectorized loads.
4. **Test harness**: a short test that runs both the fused kernel and a reference sequential implementation, compares outputs with `torch.allclose`, and reports timing.
5. **Known limitations**: explicit statements about what shapes, dtypes, or use cases are not covered by this implementation.

## Common failure modes
- **Fusing a non-elementwise op**: softmax, layernorm, cumsum, and scatter/gather require inter-element communication. Fusing them as if they were elementwise produces wrong outputs silently.
- **Missing boundary guard in CUDA**: when `N` is not a multiple of block size, the last block has threads with `idx >= N`. Without a guard, these threads load garbage and may write garbage to adjacent memory.
- **Dtype mismatch across op chain**: fusing `fp16 input -> fp32 multiply -> fp16 relu` without explicit casts leads to implicit narrowing or widening at unexpected stages.
- **Strided input treated as contiguous**: a tensor that has been transposed or sliced is not contiguous. Using flat linear indexing on it reads the wrong elements.
- **Register spill from a long fusion chain**: if more than ~20–30 intermediate values are live simultaneously, the compiler may spill to local memory, degrading performance significantly. Check the PTX or cubin with `cuobjdump --dump-sass` if performance is unexpectedly poor.
- **Epilogue fusion incorrectly includes a reduction**: normalizing the GEMM output (e.g., dividing by a row sum) inside the epilogue requires a two-pass design, not a single-pass fused epilogue.
- **Assuming torch.compile always fuses**: torch.compile may break fusion due to graph breaks (data-dependent control flow, custom ops, Python side effects). Validate the fusion using the compiler trace.

## Review checklist
- [ ] Every operation in the fusion chain is strictly elementwise; no inter-element dependencies exist.
- [ ] Memory traffic reduction is quantified before and after fusion.
- [ ] All dtype transitions are explicit and correctly ordered.
- [ ] Boundary conditions are handled for non-aligned tensor sizes.
- [ ] Vectorized loads are conditioned on pointer alignment.
- [ ] Non-contiguous (strided) inputs are handled via stride arithmetic, not flat indexing.
- [ ] Register pressure has been assessed for the fused chain length.
- [ ] torch.compile was evaluated as an alternative and the decision to write a manual kernel is justified.
- [ ] The fused kernel output is verified against a reference implementation using `torch.allclose` with appropriate tolerances.
- [ ] Timing comparison between fused and unfused is included, not just a theoretical argument.
- [ ] For GEMM epilogue fusion: the epilogue op depends only on the current output element, not on neighboring elements or global state.
