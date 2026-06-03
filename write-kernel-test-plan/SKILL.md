# Skill: Write a Kernel Test Plan

## Purpose
Guide the agent through constructing a systematic, coverage-complete test plan for a compute kernel, covering correctness, numerical precision, boundary conditions, layout variations, and performance regression.

## Use this when
- Writing a new CUDA or Triton kernel that needs a test suite before it is used in production.
- Reviewing an existing kernel where the test coverage is unknown or suspected to be incomplete.
- A kernel has exhibited silent correctness failures and the root cause needs to be isolated through systematic testing.
- Preparing a kernel for open source release or external contribution.

## Do not use this when
- The kernel is a trivial wrapper around a well-tested library call (e.g., a single cuBLAS invocation) with no added logic.
- The "kernel" is a Python-level composition of existing tested ops with no custom GPU code.

## Inputs the agent should gather first
- The mathematical specification of the kernel: what function does it compute, exactly?
- Input and output dtypes.
- The set of tensor shapes the kernel is expected to handle, including whether shapes are static or dynamic.
- Memory layout assumptions: does the kernel require contiguous input? Does it handle strided tensors?
- Whether the kernel has stochastic behavior (e.g., dropout) that requires special handling in tests.
- The target hardware: some kernels have architecture-specific code paths (e.g., tensor core paths vs fallback paths) that must be tested separately.
- Whether there is an existing reference implementation (PyTorch op, numpy, or a simpler known-correct version) to compare against.

## Required reasoning process
1. **Define the reference implementation.** Before writing any test, identify the ground truth:
   - Prefer a CPU reference in fp64 or fp32 for maximum precision.
   - For kernels that match an existing PyTorch op, use `torch.<op>` as the reference.
   - For kernels with no direct PyTorch equivalent, write a simple, unoptimized reference in Python/numpy that is obviously correct.
   - The reference must not share code with the kernel under test. It must be independently correct.

2. **Choose numerical tolerances.** Do not use a single tolerance for all dtypes:
   - fp64 reference vs fp32 kernel: rtol=1e-5, atol=1e-8.
   - fp64 reference vs fp16 kernel: rtol=1e-2, atol=1e-3.
   - fp64 reference vs bf16 kernel: rtol=5e-2, atol=1e-3.
   - fp32 reference vs int8 quantized kernel: measure max absolute error as a fraction of the quantization scale; typically < 0.5 * scale is acceptable.
   - For kernels with known numerical instability mitigations (log-sum-exp, Welford variance): use tighter tolerances and verify that the mitigation is actually working.

3. **Design the shape coverage matrix.** For every independent dimension of the kernel:
   - Size 1 (edge case: single element).
   - A small non-power-of-2 (e.g., 7, 13, 37) to stress boundary handling.
   - A power-of-2 that fits in one tile (e.g., 64, 128).
   - A power-of-2 that requires multiple tiles (e.g., 512, 4096).
   - A non-power-of-2 requiring partial tiles (e.g., 100, 1000, 4097).
   - A large prime (e.g., 1009, 8191) to stress tile boundary handling.
   - A size near the maximum expected input size.
   - For multi-dimensional kernels (GEMM): test at least (M, N, K) combinations where each dimension independently hits the above cases. A full Cartesian product is not required, but corner cases (e.g., M=1, large N and K) must be included.

4. **Design the dtype coverage.** Test every dtype the kernel claims to support:
   - fp32, fp16, bf16 each independently.
   - For kernels with mixed-precision paths (fp16 input, fp32 accumulation, fp16 output), verify that the accumulation is actually in fp32 by using inputs where fp16 accumulation would produce a measurably wrong result.
   - For quantized kernels: test each quantization scheme (per-tensor, per-channel, per-token) independently.

5. **Design the layout coverage.** Test all layout variants:
   - Contiguous input (standard).
   - Transposed input (the tensor is non-contiguous; strides are reversed).
   - Sliced input (a view into a larger tensor; stride in the sliced dimension is 1, but the tensor is not contiguous).
   - For GEMM: test row-major A x row-major B, row-major A x column-major B, and column-major A x row-major B if the kernel claims to support them.

6. **Design the boundary coverage.** These tests must verify that the partial tile handling is correct:
   - A shape that produces exactly one partial tile at the end of each dimension.
   - A shape where the entire tensor fits in less than one tile.
   - A shape where the last tile has exactly one valid element.
   - For attention/softmax: a shape where the sequence length is one.

7. **Design stress and adversarial tests.** The correctness tests above use controlled inputs. Adversarial inputs expose edge cases in numerical stability:
   - All-zeros input.
   - All-ones input.
   - Large values (e.g., `torch.full(shape, 1e4)` for fp32, `torch.full(shape, 1e2)` for fp16).
   - Small values near underflow (e.g., `torch.full(shape, 1e-38)` for fp32).
   - Mixed signs (alternating +1 and -1) to expose cancellation bugs.
   - Random input with a fixed seed (seed must be set explicitly for reproducibility).
   - For robustness: explicitly test that the kernel handles a single NaN or Inf in the input without producing silent corruption for other output elements (document expected behavior — some kernels are allowed to propagate NaN; others should clamp or ignore it).

8. **Design the performance regression test.** Correctness is necessary but not sufficient. Include:
   - A timing test using `torch.utils.benchmark.Timer` (preferred for CUDA; handles warmup and synchronization automatically).
   - A baseline measurement: either a library call (e.g., `torch.nn.functional.softmax`) or a prior version of the kernel.
   - A threshold: "this kernel must not be more than 10% slower than the baseline on the standard benchmark shape."
   - The standard benchmark shape should be a representative production input, not the smallest test case.

9. **For CUDA kernels: integrate compute-sanitizer.** Run the test suite under:
   - `compute-sanitizer --tool memcheck` to detect out-of-bounds memory accesses and uninitialized reads.
   - `compute-sanitizer --tool racecheck` to detect shared memory race conditions.
   - These tools add significant overhead; run them as part of a CI check, not in the hot test loop.

10. **For Triton kernels: use triton.testing.** Use `triton.testing.allclose` for dtype-aware tolerance comparison. Use `triton.testing.Benchmark` for performance measurement.

## Kernel design rules
- Every test must set a random seed before generating random inputs. An unseeded test that fails intermittently is harder to debug than a failing test with a fixed seed.
- Reference implementation must be written in a separate function, not inline with the kernel call. It must be readable and obviously correct without understanding the kernel.
- Tests must explicitly check output shapes and dtypes before comparing values. A kernel that returns the wrong shape should fail before the numerical comparison.
- Performance tests must use GPU-synchronized timing. CPU-side timing with `time.perf_counter` does not account for GPU execution time.
- Do not test only the "happy path" (large, power-of-2, contiguous inputs). The boundary and adversarial cases are where most kernel bugs live.

## Correctness requirements
- The test suite must include at least one test that verifies each of the following independently: correct output values, correct output shape, correct output dtype, correct behavior on a partial-tile shape, and correct behavior on a non-contiguous input.
- For kernels that use synchronization primitives (`__syncthreads`, `tl.debug_barrier`), racecheck under compute-sanitizer must pass.
- Tests must fail detectably (not silently pass) when the kernel produces wrong results. Use `torch.allclose` with the appropriate tolerance, and fail the test if it returns False.
- For reduction kernels, verify both the correctness of the reduced value and the correctness of the unreduced dimensions (shapes and strides of the output tensor).

## Performance requirements
- The performance test must report GB/s (memory bandwidth) or TFLOPS/s (for compute-bound kernels) alongside raw latency. Raw latency alone is not comparable across shapes or hardware.
- Include at least one performance test on the largest expected production input shape.
- Do not include the first kernel invocation in timing measurements — the first invocation may incur JIT compilation overhead (especially for Triton) that is not representative of steady-state performance.

## Output format
The agent should produce:

1. **Reference implementation**: a standalone Python function implementing the kernel's mathematical specification in an obviously correct way.
2. **Shape/dtype/layout test matrix**: a table or parameterized test listing all combinations of shapes, dtypes, and layouts to be tested.
3. **Adversarial input test cases**: explicit list of adversarial input patterns with the expected behavior for each.
4. **Correctness test code**: complete, runnable pytest or unittest code that compares kernel output to reference output with appropriate tolerances.
5. **Performance test code**: complete, runnable timing harness with GB/s or TFLOPS reporting.
6. **compute-sanitizer invocation**: the exact command to run the test suite under memcheck and racecheck.
7. **Coverage gaps statement**: explicit enumeration of any test scenarios that are known to be missing and why.

## Common failure modes
- **Only testing power-of-2 shapes**: shapes like 128, 256, 512 are rarely partial-tile cases and hide boundary bugs. The test suite that only covers these shapes will not catch the most common class of CUDA kernel bugs.
- **Testing fp16 precision only against an fp16 reference**: comparing a custom fp16 kernel to PyTorch's fp16 implementation does not detect systematic numerical errors that both share. Always compare to an fp32 or fp64 reference for precision validation.
- **Not testing the last partial tile explicitly**: if the kernel has a tile size of 128, test with shapes of 127, 129, 193, and 257 — not only 128, 256, 512.
- **Unseeded random inputs**: a test that fails only on specific input values but uses `torch.randn` without a fixed seed will not reproduce consistently, making debugging very difficult.
- **CPU-side timing for GPU kernels**: measuring wall time with `time.perf_counter` around a GPU kernel call without synchronization reports only the kernel launch time, not the execution time.
- **Not testing non-contiguous layouts**: most kernel bugs involving stride handling are only triggered by non-contiguous inputs. A test that always passes contiguous tensors will not catch these bugs.
- **Reference and kernel share the same numerical algorithm**: if the reference and the kernel both implement the same (potentially buggy) algorithm, their outputs will match even when both are wrong relative to the true mathematical result.
- **Performance test on warm cache only**: if the performance test fits entirely in L2, the reported bandwidth will be much higher than achievable on real workloads. Include at least one test where the input is larger than L2.
- **Not testing batch size 1**: batch-size-1 inputs often expose off-by-one errors in batch-dimension indexing.

## Review checklist
- [ ] A reference implementation exists that is independent of the kernel and obviously correct.
- [ ] Numerical tolerances are set per dtype (fp32/fp16/bf16/int8) and documented with rationale.
- [ ] The shape matrix includes size 1, small non-power-of-2, power-of-2 (single tile), power-of-2 (multi-tile), large prime, and a partial-tile case for each dimension.
- [ ] All supported dtypes are tested independently.
- [ ] Non-contiguous (strided) and transposed layouts are tested.
- [ ] Adversarial inputs are included: all-zeros, large values, small values, mixed signs, NaN/Inf.
- [ ] Every test uses a fixed random seed.
- [ ] Output shape and dtype are verified before numerical comparison.
- [ ] Performance tests use GPU-synchronized timing (cudaEvent or torch.utils.benchmark.Timer).
- [ ] Performance tests report bandwidth or throughput, not just latency.
- [ ] For CUDA: compute-sanitizer memcheck and racecheck invocations are documented.
- [ ] Coverage gaps are explicitly enumerated.
