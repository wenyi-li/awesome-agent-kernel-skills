# TileLang Examples

A collection of TileLang kernel examples demonstrating the DSL's capabilities across GPU architectures (NVIDIA Hopper/Blackwell, AMD ROCm), covering matrix operations, attention mechanisms, Mixture-of-Experts, quantization, and more.

---

## Getting Started

When an operator family has a dedicated example here, prefer it over deriving
the operator from a generic reference template. The reference docs explain
primitives; the examples show the intended tile-level decomposition.

- **quickstart.py**: The entry-point example — a tiled matrix multiplication kernel with a fused ReLU activation. Demonstrates `@tilelang.jit`, shared memory tiling, fragment allocation, pipelined `T.copy`, `T.gemm` for tensor core dispatch, and `T.Parallel` for elementwise ops, with profiling and correctness validation against PyTorch.

- **gemm/**: The canonical introduction to TileLang. Starts with a basic tiled GEMM demonstrating all core DSL primitives, then progresses through layout annotations, swizzle-based rasterization, autotuning, fine-grained MMA intrinsics (`ldmatrix`, `mma`, `stmatrix`), and persistent kernels. Includes a detailed README tutorial.

- **eager_jit/**: Notebooks (English and Chinese) demonstrating TileLang's eager JIT execution mode, where kernels are compiled and launched on-the-fly at call time for a more interactive, Pythonic development experience without explicit pre-compilation.

- **dynamic_shape/**: A fully parameterized tiled GEMM kernel that accepts dynamic (runtime-determined) M, N, K dimensions via `T.dynamic()` and supports transposed operand layouts. Demonstrates how `T.symbolic` dimension variables enable a single compiled kernel to handle variable problem sizes.

---

## Elementwise & Utility Kernels

- **elementwise/**: A minimal "hello world" vector addition kernel demonstrating fundamental TileLang concepts: `@tilelang.jit`, shared memory allocation, fragment allocation, `T.copy` for data movement, and `T.Parallel` for elementwise work.

- **cast/**: Type-casting kernels converting FP16/BF16 to FP8 (E4M3) with per-token-group quantization and per-group scale factors. Includes batch variable-length sequence support and a Triton reference implementation, illustrating `T.reduce_absmax`, `T.Parallel`, `T.if_then_else`, and the `T.float8_e4m3fn` datatype.

- **norm/**: Full LayerNorm (forward and backward) and RMSNorm (split-K and non-split variants) kernels. Demonstrates reduction primitives (`T.reduce_sum`, `T.reduce_max`), shared memory tiling, and `torch.autograd.Function` integration for end-to-end training.

- **online_softmax/**: A numerically stable online softmax kernel using the two-pass maximum-and-log-sum-exp algorithm with base-2 `exp2`/`log2`. This is a canonical starting point for learning how TileLang expresses row-wise operators with loop-carried state. Demonstrates `T.Pipelined` tiled iteration, `T.reduce_max`, `T.reduce_sum`, and mixed-precision (float16 input, float32 accumulation) within a single kernel.

- **topk/**: A top-K selection kernel using iterative `T.reduce_max` over rows, recording maxima then zeroing them out for the next iteration. Supports autotuning over block size and thread count.

- **rand/**: A 1D RNG kernel using TileLang's built-in `T.rng_init` and `T.rng_rand` primitives, with a Triton comparison for correctness. Demonstrates scalar operations alongside block-parallel tiled computation.

---

## GEMM Variants

- **gemm_fp8/**: FP8 GEMM kernels using native FP8 MMA instructions on Hopper (SM89+). Includes a 2xAcc variant that periodically promotes partial sums for numeric stability, an SM100 variant with `T.tcgen05_gemm` and TMEM, and AMD-specific variants with MFMA instructions and preshuffle optimization.

- **gemm_int4/**: An INT4 GEMM kernel demonstrating TileLang's support for sub-byte integer types via `T.int4`. Includes packing utilities for converting between PyTorch INT8/INT32 and packed INT4 formats.

- **gemm_sm100/**: Progressive Blackwell (SM100) GEMM examples introducing TCGEN05MMA tensor cores — from baseline MMA, to raw `T.tcgen05_gemm` with TMEM and mbarrier synchronization, to warp-specialized persistent kernels with CLC (Cooperative Load/Compute) 2-CTA cluster scheduling.

- **gemm_sp/**: Semi-structured sparse GEMM (2:4 sparsity) using `T.gemm_sp` to accelerate sparse-dense matmul on Tensor Cores with pipelined data movement and swizzle rasterization.

- **gemm_splitk/**: Split-K GEMM where the K-dimension is partitioned across multiple thread blocks, with partial results accumulated via per-element or vectorized `T.atomic_add` reductions.

- **gemm_streamk/**: The Stream-K scheduling algorithm, balancing work unevenly across SMs by assigning each SM a contiguous range of K-iteration tiles, eliminating a separate reduction grid. Uses dynamic iteration ranges and warp-level atomics.

- **gemv/**: Matrix-vector multiplication kernels exploring multiple parallelization strategies — naive, split-K, and warp-shuffle vectorized — for this bandwidth-bound operation.

- **dequantize_gemm/**: A family of dequantization GEMM kernels that unpack low-bit storage formats (FP4, MXFP4, INT4, W4A8) into higher-precision compute formats on-the-fly, targeting both Hopper (NVIDIA) and CDNA4 (AMD) architectures.

- **grouped_gemm/**: Grouped GEMM examples covering two different contracts: a segmented-row formulation where one packed matrix A is multiplied by per-group weights with variable-length row groups, and a ptr-table formulation where each group keeps its own tensor storage. Read the grouped-GEMM file headers carefully before tuning: the padding, metadata, and pipeline constraints differ between the two variants.

- **convolution/**: A 2D convolution using the im2col + GEMM strategy with `T.im2col` for spatial unrolling and `T.Pipelined` for overlapping data movement with GEMM computation. Includes autotuned variants with architecture-specific heuristics.

- **blockscaled_gemm_sm100/**: MXFP8 block-scaled GEMM kernels for NVIDIA Blackwell using 1D-1D block scaling. Includes 1CTA baseline, 2CTA cluster-based cooperative MMA, and persistent 2CTA variants with warp specialization (TMA producer, MMA compute, SF transposer warps).

---

## Attention Mechanisms

- **flash_attention/**: Full FlashAttention (MHA and GQA, forward and backward) for Hopper GPUs using WGMMA-based `T.gemm`. Demonstrates online safe-softmax with rescaling, pipelined tiled GEMM, causal masking, TMA-based data movement, and autotune infrastructure for runtime kernel selection.

- **flash_attention_sm100/**: FlashAttention for NVIDIA Blackwell (SM100) using TCGEN05MMA tensor core instructions with TMEM accumulators. Includes shared-to-shared, TMEM-to-shared, and warp-specialized variants.

- **flash_decoding/**: GQA and MHA decode-phase kernels optimized for inference (query length = 1) using the split-KV strategy — partitioning the KV cache into splits processed in parallel, then combining partial outputs via log-sum-exp reduction.

- **blocksparse_attention/**: Block-sparse FlashAttention with top-K and threshold-based sparse masking. Covers prefill-style attention, GQA decode with variable-length sequences, and paged-attention for LLM inference, using `T.if_then_else` on boolean block masks for conditional execution.

- **linear_attention/**: Chunked linear attention and state-space variants with forward and backward passes, including Mamba-style chunk state and cumulative-sum decay kernels. Demonstrates advanced `T.Pipelined`, `T.atomic_add`, and multi-dimensional grid launch patterns.

- **minference/**: Microsoft-inspired vertical-and-slash sparse attention on H100 using TMA hardware. Leverages warp specialization (producer/consumer), TMA asynchronous copy with mbarrier multicast, and online softmax rescaling, achieving 1.2x–1.7x speedup over Triton.

- **attention_sink/**: FlashAttention with learnable attention sink tokens, requiring extra rescaling and `dsinks` gradient computation. Supports MHA and GQA forward/backward with variable-length sequences, achieving 1.21x–1.35x speedup over Triton on H800.

- **seer_attention/**: Block-sparse FlashAttention with online softmax using base-2 `exp2` scaling and conditional execution based on a coarse block-sparse mask from top-k selection or thresholding.

---

## DeepSeek Model Implementations

- **deepseek_deepgemm/**: FP8 GEMM with 2x accumulation promotion based on DeepGEMM, using block-wise and per-token FP8 quantization, swizzled L2 access patterns, and promotion to a second accumulation buffer for higher-precision results.

- **deepseek_mhc/**: DeepSeek's Multi-Head Codebook (mHC) transformer layers as deeply fused kernels — forward pre-processing (RMS norm, sigmoid split-mix, Sinkhorn normalization, residual application), forward post-processing, and a backward Sinkhorn gradient kernel.

- **deepseek_mla/**: DeepSeek's Multi-Head Latent Attention decode kernel for Hopper GPUs, benchmarked against FlashMLA and FlashInfer. Uses warpgroup specialization (`FullCol`) to partition the accumulator and avoid register spilling, achieving FlashMLA-comparable performance in ~80 lines of code.

- **deepseek_nsa/**: DeepSeek's Native Sparse Attention (NSA) with TileLang kernels for forward and backward. Processes only top-k KV blocks per query head using block-level sparse indices, with online softmax, causal masking, and GQA support.

- **deepseek_v32/**: The full DeepSeek V3.2 sparse attention pipeline — Lightning Indexer (FP8 index vector encoding with ReLU-aggregated multi-head dot products), Top-K Selector (radix-sort-based selection), and Sparse MLA forward/backward with manual warpgroup specialization achieving ~600 TFlops on H800.

- **deepseek_v4/**: DeepSeek V4 components — FP8/FP4 block-wise quantization with IEEE 754 bit manipulation for fast power-of-2 scale computation, and sparse MQA attention on SM90+ with partial RoPE and attention sink support.

- **dsa_hisa/**: HISA (Hierarchical Sparse Attention) — a two-stage plug-and-play indexer: coarse block-level selection via FP8 block-mean pooling and pool MQA, followed by fine-grained token-level scoring within top-k blocks.

- **dsa_sparse_finetune/**: A differentiable sparse attention pipeline for fine-tuning. Includes a GPU bitonic-sort-based top-k indexer with log-softmax scores, sparse MLA forward/backward with atomic sparse gradient updates, and end-to-end autograd integration.

---

## Architecture-Specific

- **amd/**: FlashAttention for AMD ROCm/HIP GPUs with RDNA architecture, using WMMA instructions instead of NVIDIA Tensor Cores, with RDNA-specific constraints and autotuning.

- **hadamard_transform/**: A blocked Hadamard transform kernel exploiting the transform's recursive structure via three hierarchical stages — in-thread butterfly ops, warp-level shuffle exchange, and block-level shared memory exchange — using `T.macro`, `T.tvm_warp_shuffle`, and `T.vectorized` loops.

---

## Model-Specific Kernels

- **bitnet-1.58b/**: BitNet b1.58 (1.58-bit LLM) with custom INT8xINT2 GEMM using inline PTX `lop3.b32` instructions for packed 2-bit weight decoding. Integrates with vLLM for production inference, achieving up to 25x speedup with CUDA graphs.

- **fusedmoe/**: A fully fused Mixture-of-Experts forward pass combining gate logits, top-k expert routing with logits clipping, and SwiGLU activation into a single kernel. Demonstrates `T.Pipelined` with swizzle layout and token-level parallelization.

- **gdn/**: Gated Delta Net (GDN) — a state-space/linear attention model with chunked forward and backward kernels for delta-rule hidden state updates, WY representation, and cumulative-sum gate signals, validated against flash-linear-attention.

- **kda/**: Kernelized Delta Attention (KDA / Gated Delta Rule) with multi-kernel chunked computation — chunk-order, WY-fast algorithm, backward delta/state computation, and intra-chunk token-parallel forward, all using `T.Pipelined` and autotuning.

---

## Tools & Visualization

- **analyze/**: A performance analysis toolkit (`Analyzer`) that estimates FLOPs, memory traffic, and execution time for TileLang-compiled TVM IR modules using a roofline model across pre-configured NVIDIA architectures (A100, RTX 3080, RTX 4090).

- **autodd/**: AutoDD (Automatic Delta Debugging), a built-in tool that uses probabilistic delta debugging to automatically reduce complex TileLang programs to the minimal code needed to reproduce a specific error, via AST-based rewrite rules.

- **plot_layout/**: Scripts that generate and visualize memory layouts (data-to-thread and data-to-register mappings) using TileLang's `plot_layout` tool, covering CUDA MMA, AMD MFMA, shared memory bank-swizzle patterns, and layout transform functions.

- **visual_layout_inference/**: Demonstrates TileLang's layout visualization pass, which infers and renders data-to-thread mappings of fragment buffers during compilation as SVG or text representations.

- **warp_specialize/**: Multiple examples of warp-specialized GEMM kernels where different warp groups handle data copy vs. computation with barrier synchronization, including a full FlashAttention MLA kernel with decoupled RoPE using interleaved producer-consumer patterns.

---

## Infrastructure

- **conftest.py**: Pytest configuration setting deterministic random seeds and marking a curated list of known-failing tests as xfail when targeting the `cutedsl` target.
