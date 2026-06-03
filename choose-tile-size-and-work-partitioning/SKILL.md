# Skill: Choose Tile Size and Work Partitioning

## Purpose
Guide the agent through selecting tile sizes and work partitioning strategies for a CUDA or Triton kernel, based on shared memory budget, register pressure, occupancy targets, problem shape, and access pattern.

## Use this when
- Designing a new tiled kernel and the tile size and launch configuration have not yet been chosen.
- An existing kernel has suboptimal performance and the cause may be a poor tile size, low occupancy, or inefficient work partitioning.
- The problem shape has an irregular or non-power-of-2 size that makes default tile choices potentially wasteful.
- Writing a Triton kernel where `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages` must be chosen.
- Profiling shows low SM utilization, high idle warp cycles, or memory throughput below roofline estimates.

## Do not use this when
- Using a library (cuBLAS, cuDNN, CUTLASS with auto-tuning) that handles tile size selection internally. Trust the library's tuner unless profiling shows a clear gap.
- The kernel is purely streaming (one pass, no reuse) and tiling provides no shared memory reuse benefit. In that case, work partitioning reduces to choosing a block size that achieves good occupancy and coalesced access, which is a simpler problem.
- The problem is so small (total work fits in one or two blocks) that tile size selection is irrelevant compared to launch overhead.

## Inputs the agent should gather first
- Problem dimensions (e.g., M, N, K for GEMM; sequence length and head dimension for attention; reduction length for softmax).
- Element dtype: determines element size in bytes (fp16 = 2B, fp32 = 4B, bf16 = 2B, int8 = 1B).
- Target GPU architecture: shared memory capacity per SM (48 KB–228 KB depending on architecture and configuration), L2 size, number of SMs, warp size (always 32 for CUDA).
- Maximum shared memory per block (for sm_86: up to 100 KB with `cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`).
- Number of registers available per SM (65536 for most modern architectures), and target occupancy.
- Whether the problem shape is static or dynamic at kernel launch time.
- Whether the input access pattern is coalesced in the tile dimension being chosen.
- For Triton: whether autotuning will be used to sweep tile sizes, or if a fixed tile must be chosen.

## Required reasoning process
1. **Determine what is loaded into shared memory per block.** For a GEMM-style kernel with tiles `BLOCK_M x BLOCK_N` and accumulation tile `BLOCK_K`:
   - smem per block = `(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * element_size`
   - For an elementwise kernel with no smem reuse, smem per block = 0 (tiles reside in registers only).
   - For a softmax kernel (row-wise): smem per block = `BLOCK_N * element_size` (one row of the softmax).

2. **Compute smem usage for candidate tile sizes.** Start from the hardware smem limit and work downward. A common starting constraint:
   - smem per block <= 48 KB (safe default, no `cudaFuncSetAttribute` needed)
   - smem per block <= 100 KB (requires explicit dynamic smem request on sm_80+)
   - smem per block <= 228 KB (sm_90 Hopper maximum with smem partitioning)
   - Reject tile sizes that exceed the smem budget.

3. **Estimate register usage per thread.** A precise count requires compiling and inspecting with `--ptxas-info-registers` or `cuobjdump`. As a first estimate:
   - Each live fp32 value requires one register. Each live fp16 value typically still uses one register.
   - Tile accumulation in a GEMM with a `BLOCK_M x BLOCK_N` output tile and 128-bit vectorized loads requires roughly `(BLOCK_M * BLOCK_N) / 32` registers per thread for the accumulator alone (assuming warp-level decomposition).
   - Total register pressure = accumulator + input tiles in registers + loop induction variables + addressing offsets. Budget ~4–8 registers for overhead per active load.
   - Registers per SM = 65536. At 32 threads per warp and W warps per block: registers per thread budget = 65536 / (W * 32 * max_blocks_per_SM).

4. **Compute occupancy for candidate configurations.** Occupancy = (active warps per SM) / (max warps per SM). Constraints that limit occupancy:
   - smem: max_blocks_per_SM = floor(smem_per_SM / smem_per_block). Warps = max_blocks_per_SM * warps_per_block.
   - registers: max_blocks_per_SM = floor(regs_per_SM / (regs_per_thread * threads_per_block)). Then same warp calculation.
   - hardware block limit: typically 32 blocks per SM maximum (sm_86), sometimes 16.
   - Take the minimum across all constraints. This is the achieved occupancy.
   - Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (CUDA) for accurate computation after writing the kernel.

5. **Check warp efficiency in the last tile.** If the problem dimension is not a multiple of the tile size, the last block has fewer valid elements than the tile size. The fraction of useful work in that block = `(N % TILE_SIZE) / TILE_SIZE`. If this fraction is very small (e.g., < 0.1) and many blocks hit this case, the waste is significant. Pad to a tile multiple or choose a smaller tile size.

6. **Check alignment and divisibility for vectorized loads.** Vectorized loads (float4, half2) require the pointer to be aligned to the vector width. For a tile of width W loaded with 128-bit (float4) loads, W must be a multiple of 4 (for float4). If W is not aligned, scalar fallback loads are needed, which reduces memory throughput.

7. **Choose the work partitioning shape.** Options:
   - **1D grid**: one block per output vector (row). Simple. Good for row-wise ops (softmax, layernorm). Poor if the number of rows is small relative to SM count.
   - **2D grid**: one block per output tile `(block_row, block_col)`. Standard for GEMM, attention. Grid dimensions are `ceil(M/BLOCK_M) x ceil(N/BLOCK_N)`.
   - **3D grid**: adds a batch dimension. Standard for batched GEMM, multi-head attention. Grid = `ceil(M/BLOCK_M) x ceil(N/BLOCK_N) x batch`.
   - **Persistent kernel**: one block per SM; each block loops over multiple tiles. Reduces launch overhead, improves load balance for irregular shapes. Required for reduction-then-broadcast patterns where a second kernel launch would be expensive.
   - **Streaming decomposition**: for problems too large to hold all intermediate state, decompose along the K or sequence dimension and accumulate across kernel calls. Used in multi-pass attention or tiled reduction.

8. **For Triton: choose num_warps and num_stages.** 
   - `num_warps`: number of warps per program instance. For compute-bound kernels, 4–8 warps is typical. For memory-bound kernels, 1–2 warps can be sufficient.
   - `num_stages`: pipeline stages for the software prefetch loop. More stages hide memory latency but consume more smem for the prefetch buffers. For A100-class hardware, 3–5 stages is a common range. Verify smem budget: each additional stage adds one tile's worth of smem.
   - Use `triton.testing.Benchmark` to sweep configurations and confirm empirically.

## Kernel design rules
- Tile sizes must be powers of 2 in each dimension when using shared memory and warp-level reductions, to avoid non-uniform warp partitioning and bank conflicts.
- The tile width (innermost dimension) must be a multiple of 32 (one warp) to avoid underutilizing warp lanes. A tile width of 16 means half the warp is idle at each step.
- For GEMM-style kernels, `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` must individually not exceed the smem budget divided by the number of tiles resident in smem simultaneously.
- For Triton, `BLOCK_SIZE` must be a `tl.constexpr` to enable compile-time specialization. Do not pass it as a dynamic argument unless the kernel explicitly handles dynamic tile sizes.
- Do not choose tile sizes that result in a grid with fewer blocks than the number of SMs on the target GPU — this leaves SMs idle for the entire kernel duration.
- For persistent kernels, distribute work to blocks in tile index order, not in arbitrary order, so that adjacent tiles share L2 cache lines when possible.
- When using Triton autotuning, define the autotune search space explicitly. Do not leave it as a fully open search — constrain to architecturally valid combinations (e.g., smem budget respected, num_warps divides threads evenly).

## Correctness requirements
- The tile decomposition must cover all output elements exactly once. Verify that `ceil(N/TILE_SIZE) * TILE_SIZE >= N` (no output position is skipped) and that no two tiles write to the same output position.
- Partial tiles at the boundary must be masked correctly. Refer to the handle-boundary-conditions skill for the masking strategy.
- For reductions split across tiles or kernel calls, the accumulation across tiles must produce the same result as a single-pass reduction. Verify with small test cases where the split is at a tile boundary.
- Thread assignments within a block must not alias: two threads must not read from or write to the same shared memory location simultaneously without synchronization.

## Performance requirements
- Target at least 50% theoretical occupancy as a starting point. Below 25% occupancy, latency hiding is severely limited. Above 75% occupancy, returns on further increases are often marginal — prioritize smem and arithmetic throughput.
- For memory-bound kernels, the dominant metric is memory throughput, not occupancy. A low-occupancy kernel that achieves high memory bandwidth is preferable to a high-occupancy kernel with poor coalescing.
- For compute-bound kernels (GEMM), maximize the ratio of arithmetic to memory traffic. Large tiles improve reuse but increase register and smem pressure — find the crossover point empirically.
- Document the roofline position of the kernel: compute the arithmetic intensity (FLOPs / bytes) and compare to the hardware's compute-to-bandwidth ratio. This determines whether the bottleneck is memory or compute, and thus what tile size changes will help.
- State the expected occupancy for the chosen tile size configuration and the hardware-imposed constraint responsible for limiting it.

## Output format
The agent should produce:

1. **Tile size analysis table**: for each candidate tile size configuration, a table showing: smem per block, estimated registers per thread, occupancy estimate, warp efficiency on the last tile, alignment compatibility.
2. **Selected configuration with justification**: the chosen tile size(s) and launch configuration, with explicit reasoning for why it was selected over alternatives.
3. **Work partitioning description**: the grid decomposition (1D/2D/3D/persistent), grid dimensions as a function of problem size, and block/warp dimensions.
4. **Kernel implementation skeleton**: the launch configuration and thread-to-data mapping code, even if the full kernel is not yet written.
5. **Autotuning plan** (if applicable): the parameter space to sweep and the metric to optimize (throughput, latency).

## Common failure modes
- **Tile width not a multiple of 32 (warp size)**: a tile width of 16, 24, or 48 leaves some warp lanes idle during each step, reducing warp efficiency. Always round tile widths up to a multiple of 32.
- **smem exceeds hardware limit without explicit request**: using more than 48 KB of shared memory per block on CUDA without calling `cudaFuncSetAttribute` causes the kernel launch to fail silently (0 blocks launched) or error.
- **Grid smaller than SM count**: launching fewer blocks than there are SMs on the GPU means some SMs are idle for the entire kernel. This is a hard floor on performance.
- **Register pressure from large tile sizes**: choosing BLOCK_M=128 x BLOCK_N=128 in a Triton GEMM requires a 128x128 register accumulator tile, which exceeds the register file on most hardware. The compiler will spill to local memory. Check the compiled kernel's register count.
- **num_stages too large in Triton**: each pipeline stage adds one tile's smem. With BLOCK_K=64 and fp16 (2B), each additional stage adds 64*2B = 128B per warp. For large BLOCK_M/BLOCK_N tiles, multiple stages can push over the smem budget.
- **Persistent kernel with uneven work distribution**: if the number of tiles is not evenly divisible by the number of blocks (SMs), some blocks will process one extra tile. This is acceptable but must be handled correctly — the last few blocks must not process past the tile count.
- **Choosing tile size to avoid boundary handling rather than for performance**: using BLOCK_SIZE=prime-number-close-to-N to make the problem fit in one tile is not a strategy; it produces different suboptimal behavior for different input shapes.
- **Ignoring L2 reuse in tile ordering**: for a 2D GEMM grid, iterating over (row, col) in row-major order reuses the A matrix tile across columns. Iterating in column-major order reuses the B matrix tile. Choosing the wrong iteration order reduces L2 hit rate.

## Review checklist
- [ ] Shared memory per block is computed for the chosen tile size and verified to fit within the hardware limit (with or without explicit dynamic smem request as appropriate).
- [ ] Register pressure is estimated (or measured after compilation) for the chosen configuration.
- [ ] Occupancy is computed for the chosen configuration and the limiting resource is identified.
- [ ] The grid covers all output elements exactly once.
- [ ] The tile width is a multiple of 32 (warp size) in the innermost dimension.
- [ ] Partial boundary tiles are handled correctly (refer to handle-boundary-conditions skill).
- [ ] The grid size is at least equal to the number of SMs on the target GPU.
- [ ] For Triton: `num_warps` and `num_stages` are chosen with smem budget verified.
- [ ] The arithmetic intensity of the kernel is computed and the roofline position is stated.
- [ ] The chosen tile size is justified by occupancy, smem reuse, or register analysis — not just by convention.
- [ ] An autotuning or benchmarking plan is included if the tile size is not verified empirically.
