# Awesome Agent Kernel Skills

A curated, high-signal index of agent skills for writing, testing, profiling, debugging, and optimizing GPU kernels.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](./CONTRIBUTING.md)

This list focuses on reusable skill packages for AI coding agents that work on GPU kernels and accelerator programming. Entries are grouped by the primary workflow they support.

## Contents

- [What Are Agent Kernel Skills?](#what-are-agent-kernel-skills)
- [Skills](#skills)
  - [CUDA Kernel Development](#cuda-kernel-development)
  - [Triton and Inference Kernels](#triton-and-inference-kernels)
  - [TileLang, CuTe, TileIR, and Ascend](#tilelang-cute-tileir-and-ascend)
  - [ROCm, HIP, AMD, and Portability](#rocm-hip-amd-and-portability)
  - [Profiling and Performance Analysis](#profiling-and-performance-analysis)
  - [Debugging and Correctness](#debugging-and-correctness)
  - [Testing, Benchmarking, and Optimization Workflows](#testing-benchmarking-and-optimization-workflows)
  - [Knowledge Bases and Reference Skills](#knowledge-bases-and-reference-skills)
- [Using These Skills](#using-these-skills)
- [Creating Kernel Skills](#creating-kernel-skills)
- [Contributing](#contributing)
- [Resources](#resources)

## What Are Agent Kernel Skills?

Agent skills are reusable instruction packages that teach an AI coding agent how to perform a focused class of work. A typical skill contains a `SKILL.md` file with metadata and execution guidance, plus optional `scripts/`, `references/`, or `assets/` for deterministic helpers and longer documentation.

Agent kernel skills specialize that pattern for low-level GPU and accelerator programming. They can encode workflows for authoring kernels, reviewing memory access patterns, profiling with tools such as NVIDIA Nsight Compute, debugging generated kernels, validating forward/backward operators, or optimizing framework-specific kernels.

## Skills

### CUDA Kernel Development

- [cuda](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/cuda_skill) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - CUDA programming skill from `tilelang-cuda-skills`; useful as a general CUDA kernel authoring and review reference.
- [cuda-knowledge](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/cuda-knowledge) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - CUDA kernel development, debugging, optimization, linear algebra, and multi-GPU communication guidance.
- [cuda-code-generator](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/cuda-code-generator) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - Generate optimized CUDA kernels from algorithm descriptions, reports, or existing implementations.
- [cuda-optimizer](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/cuda-optimizer) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - Profiling-driven CUDA optimization loop from validation to bottleneck analysis and iteration.
- [cuda-samples](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/cuda-samples) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - Curated index of NVIDIA CUDA Samples for common kernel and API patterns.
- [write-cuda-gemm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/write-cuda-gemm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing CUDA GEMM kernels.
- [write-cuda-layernorm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/write-cuda-layernorm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing CUDA LayerNorm kernels.
- [write-cuda-reduction-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/write-cuda-reduction-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing CUDA reduction kernels.
- [write-cuda-softmax-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/write-cuda-softmax-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing CUDA softmax kernels.
- [choose-launch-configuration](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/choose-launch-configuration) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Choose CUDA launch configuration for occupancy, memory behavior, and workload shape.
- [optimize-global-memory-access](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/optimize-global-memory-access) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Improve global memory coalescing and bandwidth use in CUDA kernels.
- [optimize-shared-memory-tiling](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/optimize-shared-memory-tiling) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Apply shared-memory tiling while managing bank conflicts and synchronization.
- [avoid-warp-divergence](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/avoid-warp-divergence) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Reduce branch divergence and warp-level inefficiency in CUDA code.
- [cuda-kernels](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/cuda-kernels) ![GitHub stars](https://img.shields.io/github/stars/huggingface/kernels?style=social&cacheSeconds=86400) - Write and benchmark CUDA kernels for Hugging Face model workloads.
- [agent_workdir](https://github.com/BytedTsinghua-SIA/CUDA-Agent/tree/main/agent_workdir) ![GitHub stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/CUDA-Agent?style=social&cacheSeconds=86400) - Expert-designed CUDA-Agent skill and generation environment.
- [kernel-creator](https://github.com/Adversarr/kernel-dsl/tree/main/skills/kernel-creator) ![GitHub stars](https://img.shields.io/github/stars/Adversarr/kernel-dsl?style=social&cacheSeconds=86400) - Author reusable GPU kernel implementations across CUDA, Triton, TileLang, CuTe DSL, and similar DSLs.
- [kernel-designer](https://github.com/mindspore-ai/akg/tree/master/akg_agents/workspace/.opencode/skills/kernel-designer) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Design operator kernel sketches for Triton CUDA, Triton Ascend, C++, CUDA C, TileLang CUDA, and PyPTO backends.
- [kernel-generator](https://github.com/mindspore-ai/akg/tree/master/akg_agents/workspace/.opencode/skills/kernel-generator) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Generate operator kernel code across CUDA, Triton, TileLang, Ascend, C++, and PyPTO DSLs from task descriptions.

### Triton and Inference Kernels

- [kernel-triton-writing](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-triton-writing) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - TensorRT-LLM guidance for writing Triton kernels.
- [triton-kernel-optimization](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/triton-kernel-optimization) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Tune Triton kernels with block-size, memory-access, reduction, and fused-op strategies.
- [triton-kernel-reflection-prompts](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/triton-kernel-reflection-prompts) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Self-review prompts for fixing AMD-targeted Triton kernels after generation or test failures.
- [triton-hip-reference-kernel-search](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/triton-hip-reference-kernel-search) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Search and adapt Triton/HIP kernel patterns for AMD GPUs.
- [write-triton-attention-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/triton/write-triton-attention-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing Triton attention kernels.
- [write-triton-gemm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/triton/write-triton-gemm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing Triton GEMM kernels.
- [write-triton-layernorm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/triton/write-triton-layernorm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing Triton LayerNorm kernels.
- [write-triton-softmax-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/triton/write-triton-softmax-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Skill for writing Triton softmax kernels.
- [optimize-triton-block-parameters](https://github.com/KrxGu/kernel-skills/tree/main/skills/triton/optimize-triton-block-parameters) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Tune Triton block parameters for occupancy, memory behavior, and shape coverage.
- [write-triton-dequant-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-dequant-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton dequant kernels for int4/int8 to fp16/bf16 paths.
- [write-triton-fused-add-rmsnorm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-fused-add-rmsnorm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton fused add plus RMSNorm inference kernels.
- [write-triton-kv-cache-append-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-kv-cache-append-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton KV-cache append kernels for serving workloads.
- [write-triton-rmsnorm-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-rmsnorm-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton RMSNorm kernels.
- [write-triton-rope-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-rope-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton RoPE kernels.
- [write-triton-sampling-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-sampling-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton sampling kernels for inference pipelines.
- [write-triton-silu-mul-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-triton-silu-mul-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write Triton SiLU-Mul and SwiGLU kernels.
- [optimize-prefill-vs-decode-kernels](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/optimize-prefill-vs-decode-kernels) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Optimize different kernel paths for prefill and decode phases.
- [write-tensorrt-plugin-integration-plan](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-tensorrt-plugin-integration-plan) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Plan TensorRT plugin integration for custom kernels.
- [write-vllm-custom-op-integration-plan](https://github.com/KrxGu/kernel-skills/tree/main/skills/inference/write-vllm-custom-op-integration-plan) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Plan vLLM custom-op integration for inference kernels.
- [add-jit-kernel](https://github.com/sgl-project/sglang/tree/main/.claude/skills/add-jit-kernel) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Add a lightweight JIT CUDA kernel to SGLang.
- [add-sgl-kernel](https://github.com/sgl-project/sglang/tree/main/.claude/skills/add-sgl-kernel) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Add a heavyweight AOT CUDA/C++ kernel to `sgl-kernel`.
- [sglang-skill](https://github.com/slowlyC/agent-gpu-skills/tree/main/sglang_skill) ![GitHub stars](https://img.shields.io/github/stars/slowlyC/agent-gpu-skills?style=social&cacheSeconds=86400) - Develop, debug, and optimize the SGLang serving engine and kernel stack.
- [add-cuda-kernel](https://github.com/flashinfer-ai/flashinfer/tree/main/.claude/skills/add-cuda-kernel) ![GitHub stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social&cacheSeconds=86400) - Step-by-step tutorial for adding new CUDA kernels to FlashInfer.
- [triton-kernels](https://github.com/Adversarr/kernel-dsl/tree/main/skills/triton-kernels) ![GitHub stars](https://img.shields.io/github/stars/Adversarr/kernel-dsl?style=social&cacheSeconds=86400) - Write optimized Triton kernels from vector ops through FlashAttention, persistent matmul, fused normalization, and quantized GEMM.
- [xpu-kernels](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/xpu-kernels) ![GitHub stars](https://img.shields.io/github/stars/huggingface/kernels?style=social&cacheSeconds=86400) - Write, optimize, and benchmark Triton kernels for Intel XPU GPUs with Xe-Forge workflows.
- [flashinfer](https://github.com/dtunai/agent-skills-for-compute/tree/main/skills/flashinfer) ![GitHub stars](https://img.shields.io/github/stars/dtunai/agent-skills-for-compute?style=social&cacheSeconds=86400) - Reference FlashInfer attention, KV-cache, sampling, GEMM, MoE, FP8/FP4, integration, and performance patterns for LLM inference kernels.
- [sglang-diffusion-ako4all-kernel](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-ako4all-kernel) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Optimize SGLang diffusion kernels with AKO4ALL, custom microbenchmarks, NCU-guided iteration, and denoise validation.
- [sglang-diffusion-performance](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Choose SGLang diffusion performance flags for a model, GPU, and VRAM budget.
- [sglang-sota-performance](https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-sota-performance) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Run an end-to-end SGLang performance loop against vLLM and TensorRT-LLM baselines, including profiling and kernel/fusion fixes.
- [ad-add-fusion-transformation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/ad-add-fusion-transformation) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Implement or extend TensorRT-LLM AutoDeploy fusion transforms, preferring existing kernels and custom ops before Triton.

### TileLang, CuTe, TileIR, and Ascend

- [writing-tilelang-kernels](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/writing-tilelang-kernels) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Guidance for writing TileLang kernels.
- [optimizing-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/optimizing-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Optimization workflow for TileLang programs.
- [profiling-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/profiling-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Profiling workflow for TileLang programs.
- [torch-profiling-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/torch-profiling-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Profiling TileLang programs in PyTorch-facing workflows.
- [debugging-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/debugging-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Debugging workflow for TileLang programs.
- [tilelang-developer](https://github.com/yzlnew/infra-skills/tree/main/tilelang-developer) ![GitHub stars](https://img.shields.io/github/stars/yzlnew/infra-skills?style=social&cacheSeconds=86400) - Write, optimize, and debug high-performance AI compute kernels with TileLang.
- [tilelang-to-flydsl](https://github.com/EmbeddedLLM/tilelang-to-flydsl-skills/tree/main/.claude/skills/tilelang-to-flydsl) ![GitHub stars](https://img.shields.io/github/stars/EmbeddedLLM/tilelang-to-flydsl-skills?style=social&cacheSeconds=86400) - Convert TileLang workflows to FlyDSL and AMD ROCm-oriented kernels.
- [kernel-cute-writing](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-cute-writing) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - TensorRT-LLM guidance for writing CuTe DSL kernels.
- [kernel-tileir-optimization](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-tileir-optimization) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - TileIR and Blackwell kernel optimization guidance.
- [tilegym-adding-cutile-kernel](https://github.com/NVIDIA/TileGym/tree/main/.claude/skills/adding-cutile-kernel) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Add cuTile GPU kernel operators to TileGym.
- [tilegym-adding-cutile-kernel](https://github.com/NVIDIA/skills/tree/main/skills/tilegym-adding-cutile-kernel) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/skills?style=social&cacheSeconds=86400) - NVIDIA skills catalog entry for adding cuTile kernels to TileGym.
- [tilelang-op-design](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/external/cannbot/ops-lab/tilelang/skills/tilelang-op-design) ![GitHub stars](https://img.shields.io/github/stars/ascend-ai-coding/awesome-ascend-skills?style=social&cacheSeconds=86400) - Generate TileLang Ascend operator design documents from operator requirements.
- [tilelang-op-developer](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/external/cannbot/ops-lab/tilelang/skills/tilelang-op-developer) ![GitHub stars](https://img.shields.io/github/stars/ascend-ai-coding/awesome-ascend-skills?style=social&cacheSeconds=86400) - Implement and test TileLang Ascend operators from design documents.
- [tilelang-review](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/external/cannbot/ops-lab/tilelang/skills/tilelang-review) ![GitHub stars](https://img.shields.io/github/stars/ascend-ai-coding/awesome-ascend-skills?style=social&cacheSeconds=86400) - Review and format TileLang Ascend kernel code for CI compliance.
- [tilelang-programming-model-guide](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/external/cannbot/ops-lab/tilelang/skills/tilelang-programming-model-guide) ![GitHub stars](https://img.shields.io/github/stars/ascend-ai-coding/awesome-ascend-skills?style=social&cacheSeconds=86400) - Guide Developer and Expert mode selection for TileLang Ascend kernels.
- [tilelang-api-best-practices](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/external/cannbot/ops-lab/tilelang/skills/tilelang-api-best-practices) ![GitHub stars](https://img.shields.io/github/stars/ascend-ai-coding/awesome-ascend-skills?style=social&cacheSeconds=86400) - Best practices for TileLang Ascend memory, data movement, compute, synchronization, and scheduling APIs.
- [tilegym-cutile-python](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-cutile-python) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Write, validate, debug, and optimize high-performance cuTile Python GPU kernels.
- [tilegym-cutile-autotuning](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-cutile-autotuning) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Add and debug cuTile autotuning search spaces, cache patterns, architecture filters, and launch logic.
- [tilegym-improve-cutile-kernel-perf](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-improve-cutile-kernel-perf) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Iteratively optimize cuTile kernels with profiling, IR comparison, tile-size tuning, TMA, persistent scheduling, and benchmarks.
- [tilegym-converting-cutile-to-triton](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-converting-cutile-to-triton) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Convert cuTile Python kernels to Triton while handling launch, shape, memory, and numerical mismatch issues.
- [tilegym-converting-cutile-to-julia](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-converting-cutile-to-julia) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Convert cuTile Python kernels to cuTile.jl, including indexing, memory layout, type mapping, and launch API differences.
- [tilegym-monkey-patch-kernels-to-transformers](https://github.com/NVIDIA/TileGym/tree/main/skills/tilegym-monkey-patch-kernels-to-transformers) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TileGym?style=social&cacheSeconds=86400) - Integrate TileGym kernels into Hugging Face Transformers by replacing or patching model submodules and methods.
- [tilelang-ascend-tile-api](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-ascend-tile-api) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Add Ascend-specific `T.tile.xxx` APIs across Python frontend, lowering, codegen, docs, and tests.
- [tilelang-ascend-kernel](https://github.com/jd-opensource/xllm/tree/main/.agents/skills/tilelang-ascend-kernel) ![GitHub stars](https://img.shields.io/github/stars/jd-opensource/xllm?style=social&cacheSeconds=86400) - Add, modify, debug, and review xLLM TileLang Ascend kernels, wrappers, generated Ascend-C source, CMake wiring, and NPU tests.
- [tilelang-op-test-design](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-op-test-design) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Design TileLang-Ascend operator tests from design documents, implementations, manual requirements, or coverage gaps.
- [tilelang-pass-analyzer](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-pass-analyzer) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Analyze TileLang compiler pass behavior, implementation, IR transformations, and pass relationships.
- [tilelang-pass-design](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-pass-design) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Design new TileLang-Ascend compiler passes with pipeline placement, IR transforms, implementation plans, and tests.
- [tilelang-pass-generate](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-pass-generate) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Generate TileLang-Ascend pass implementations from pass design documents and workflow analysis.
- [tilelang-pass-workflow-analyzer](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/.agents/skills/tilelang-pass-workflow-analyzer) ![GitHub stars](https://img.shields.io/github/stars/tile-ai/tilelang-ascend?style=social&cacheSeconds=86400) - Analyze TileLang pass workflow placement, dependencies, and pipeline integration before implementation.

### ROCm, HIP, AMD, and Portability

- [rocm-kernels](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills/rocm-kernels) ![GitHub stars](https://img.shields.io/github/stars/huggingface/kernels?style=social&cacheSeconds=86400) - Write and benchmark Triton kernels for AMD GPUs on ROCm.
- [amd-rocm-porting](https://github.com/amdpilot-org/amdpilot-skills/tree/main/amd-rocm-porting) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Port CUDA-oriented implementations to AMD ROCm.
- [amd-kernel-optimization](https://github.com/amdpilot-org/amdpilot-skills/tree/main/amd-kernel-optimization) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Optimize kernels for AMD GPU execution.
- [flydsl-kernel-authoring](https://github.com/amdpilot-org/amdpilot-skills/tree/main/flydsl-kernel-authoring) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Author GPU kernels with FlyDSL.
- [rocm-crash-debug](https://github.com/amdpilot-org/amdpilot-skills/tree/main/rocm-crash-debug) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Debug ROCm crashes and runtime failures.
- [rocm-profiler-analysis](https://github.com/amdpilot-org/amdpilot-skills/tree/main/rocm-profiler-analysis) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Analyze ROCm profiling results.
- [rocprofv3-profiler](https://github.com/amdpilot-org/amdpilot-skills/tree/main/rocprofv3-profiler) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Profile AMD GPU kernels with `rocprofv3` and identify bottlenecks.
- [gpu-architecture-fundamentals](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/gpu-architecture-fundamentals) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Reason about GPU architecture fundamentals for optimization choices.
- [mi300-cdna3-architecture](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/mi300-cdna3-architecture) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - MI300/CDNA3 architecture guide for HIP and Triton optimization.
- [mi300-hip-programming-insights](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/mi300-hip-programming-insights) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - MI300 HIP programming insights for cache, matrix cores, sparsity, and memory coherency.
- [mi300-hip-vs-nvidia](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/mi300-hip-vs-nvidia) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Compare MI300 HIP programming with NVIDIA CUDA assumptions.
- [hip-kernel-optimization](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/hip-kernel-optimization) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Write and tune HIP kernels with memory, tiling, occupancy, and warp/wavefront guidance.
- [aiter-reflection](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/aiter-reflection) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Optimize MI300 kernels using the AITER project workflow.
- [port-cuda-kernel-to-hip](https://github.com/KrxGu/kernel-skills/tree/main/skills/portability/port-cuda-kernel-to-hip) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Port CUDA kernels to HIP.
- [port-cuda-kernel-to-triton](https://github.com/KrxGu/kernel-skills/tree/main/skills/portability/port-cuda-kernel-to-triton) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Port CUDA kernels to Triton.
- [write-backend-agnostic-kernel-plan](https://github.com/KrxGu/kernel-skills/tree/main/skills/portability/write-backend-agnostic-kernel-plan) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Plan kernels across backend-specific implementations.
- [opus-kernel-best-practice](https://github.com/ROCm/aiter/tree/main/.claude/skills/opus-kernel-best-practice) ![GitHub stars](https://img.shields.io/github/stars/ROCm/aiter?style=social&cacheSeconds=86400) - Reduce HIP/C++ OPUS kernel compile time through header, template, and device-pass best practices.
- [opus-module-build-optimization](https://github.com/ROCm/aiter/tree/main/.claude/skills/opus-module-build-optimization) ![GitHub stars](https://img.shields.io/github/stars/ROCm/aiter?style=social&cacheSeconds=86400) - Optimize first-call JIT build wall time for opus-based AITER modules with multiple kernel translation units.

### Profiling and Performance Analysis

- [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) ![GitHub stars](https://img.shields.io/github/stars/mit-han-lab/ncu-report-skill?style=social&cacheSeconds=86400) - Analyze NVIDIA Nsight Compute reports for kernel performance bottlenecks.
- [ncu-cuda-profiling](https://github.com/maxiaosong1124/ncu-cuda-profiling-skill) ![GitHub stars](https://img.shields.io/github/stars/maxiaosong1124/ncu-cuda-profiling-skill?style=social&cacheSeconds=86400) - Automated Nsight Compute profiling workflow with full metrics collection and persistent storage.
- [ncu-rep-analyzer](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/ncu-rep-analyzer) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - Profile CUDA kernels with NCU and analyze `.ncu-rep` reports.
- [perf-nsight-compute-analysis](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-nsight-compute-analysis) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Nsight Compute performance analysis for TensorRT-LLM and kernel work.
- [kernel-profile](https://github.com/fmh66/kernel-opt-agent/tree/main/skills/kernel-profile) ![GitHub stars](https://img.shields.io/github/stars/fmh66/kernel-opt-agent?style=social&cacheSeconds=86400) - Standalone profiling skill for CUDA, CUTLASS, CuTe DSL, and Triton kernels.
- [rocprof-compute](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/rocprof-compute) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Profile AMD GPU kernels with `rocprof-compute` metrics and roofline analysis.
- [generate-profile](https://github.com/sgl-project/sglang/tree/main/.claude/skills/generate-profile) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Generate end-to-end SGLang server profiling traces.
- [benchmark-kernel](https://github.com/flashinfer-ai/flashinfer/tree/main/.claude/skills/benchmark-kernel) ![GitHub stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social&cacheSeconds=86400) - Benchmark FlashInfer kernels with CUPTI timing.
- [perf-analysis](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-analysis) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Coordinate performance profiling, bottleneck classification, and structured analysis reports.
- [perf-nsight-systems](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-nsight-systems) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Use Nsight Systems CLI to profile timelines, GPU idle time, distributed training, and NCCL overlap.
- [perf-optimization](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-optimization) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Route performance optimization work across specialist skills for kernel generation, TileIR, and safe iterative changes.
- [perf-torch-cuda-graphs](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-torch-cuda-graphs) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Apply CUDA Graphs to PyTorch workloads using framework-specific capture and replay APIs.
- [perf-torch-sync-free](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-torch-sync-free) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Identify and remove host-device synchronization points in PyTorch performance paths.
- [perf-workload-profiling](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-workload-profiling) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Instrument training loops, standalone ops, and kernels with timing and NVTX annotations.
- [perf-host-analysis](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-host-analysis) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Analyze host-side performance issues that block GPU work or serving throughput.
- [perf-host-optimization](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/perf-host-optimization) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social&cacheSeconds=86400) - Optimize host-side bottlenecks that limit accelerator utilization.
- [llm-torch-profiler-analysis](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-torch-profiler-analysis) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Analyze SGLang, vLLM, and TensorRT-LLM torch-profiler traces with kernel, overlap, and fusion-pattern tables.
- [sglang-diffusion-benchmark-profile](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Benchmark denoise latency and profile diffusion bottlenecks before SGLang kernel work.
- [tilelang-torch-profiler](https://github.com/Adversarr/kernel-dsl/tree/main/skills/tilelang-torch-profiler) ![GitHub stars](https://img.shields.io/github/stars/Adversarr/kernel-dsl?style=social&cacheSeconds=86400) - Profile TileLang and mixed PyTorch plus TileLang workloads with `torch.profiler`.

### Debugging and Correctness

- [debug-cuda-kernel-correctness](https://github.com/KrxGu/kernel-skills/tree/main/skills/cuda/debug-cuda-kernel-correctness) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Debug CUDA kernel correctness issues.
- [debug-quantized-kernel-accuracy](https://github.com/KrxGu/kernel-skills/tree/main/skills/quantization/debug-quantized-kernel-accuracy) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Debug accuracy regressions in quantized kernels.
- [handle-boundary-conditions](https://github.com/KrxGu/kernel-skills/tree/main/skills/patterns/handle-boundary-conditions) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Handle boundary conditions in tiled kernels.
- [write-numerically-stable-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/patterns/write-numerically-stable-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Design numerically stable kernel implementations.
- [debug-cuda-crash](https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Debug CUDA crashes in SGLang using kernel API logging.
- [debug-cuda-crash](https://github.com/flashinfer-ai/flashinfer/tree/main/.claude/skills/debug-cuda-crash) ![GitHub stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social&cacheSeconds=86400) - Debug CUDA crashes in FlashInfer using API logging.

### Testing, Benchmarking, and Optimization Workflows

- [testing-fwd-bwd-kernels](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/testing-fwd-bwd-kernels) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social&cacheSeconds=86400) - Testing workflow for forward and backward kernels.
- [kernel-benchmarker](https://github.com/ForceInjection/cuda-code-skill/tree/main/skills/kernel-benchmarker) ![GitHub stars](https://img.shields.io/github/stars/ForceInjection/cuda-code-skill?style=social&cacheSeconds=86400) - Compile, validate, and benchmark CUDA kernels against Python references.
- [kernel-benchmark](https://github.com/fmh66/kernel-opt-agent/tree/main/skills/kernel-benchmark) ![GitHub stars](https://img.shields.io/github/stars/fmh66/kernel-opt-agent?style=social&cacheSeconds=86400) - Benchmark CUDA, CUTLASS, CuTe DSL, and Triton implementations.
- [kernel-loop](https://github.com/fmh66/kernel-opt-agent/tree/main/skills/kernel-loop) ![GitHub stars](https://img.shields.io/github/stars/fmh66/kernel-opt-agent?style=social&cacheSeconds=86400) - Iterative measured optimization loop for CUDA, CUTLASS, CuTe DSL, and Triton kernels.
- [cuda-auto-tune](https://github.com/Bruce-Lee-LY/cuda_auto_tune/tree/main/cuda-auto-tune) ![GitHub stars](https://img.shields.io/github/stars/Bruce-Lee-LY/cuda_auto_tune?style=social&cacheSeconds=86400) - NCU-driven autotuning workflow for CUDA, CUTLASS, Triton, and CuTe DSL kernels.
- [auto-benchmark-rocm](https://github.com/amdpilot-org/amdpilot-skills/tree/main/auto-benchmark) ![GitHub stars](https://img.shields.io/github/stars/amdpilot-org/amdpilot-skills?style=social&cacheSeconds=86400) - Benchmark ROCm kernels and compare performance across iterations.
- [pytorch-kernel-optimization](https://github.com/AMD-AGI/Apex/tree/main/tools/skills/pytorch-kernel-optimization) ![GitHub stars](https://img.shields.io/github/stars/AMD-AGI/Apex?style=social&cacheSeconds=86400) - Optimize PyTorch models and kernels with tensor, compile, extension, and mixed-precision guidance.
- [choose-tile-size-and-work-partitioning](https://github.com/KrxGu/kernel-skills/tree/main/skills/patterns/choose-tile-size-and-work-partitioning) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Choose tile sizes and work partitioning for GPU kernels.
- [fuse-elementwise-ops](https://github.com/KrxGu/kernel-skills/tree/main/skills/patterns/fuse-elementwise-ops) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Fuse elementwise operations into efficient kernels.
- [write-kernel-test-plan](https://github.com/KrxGu/kernel-skills/tree/main/skills/patterns/write-kernel-test-plan) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write test plans for kernel correctness and edge cases.
- [write-fp8-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/quantization/write-fp8-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write FP8 kernels.
- [write-int8-quantized-kernel](https://github.com/KrxGu/kernel-skills/tree/main/skills/quantization/write-int8-quantized-kernel) ![GitHub stars](https://img.shields.io/github/stars/KrxGu/kernel-skills?style=social&cacheSeconds=86400) - Write INT8 quantized kernels.
- [kernel-verifier](https://github.com/mindspore-ai/akg/tree/master/akg_agents/workspace/.opencode/skills/kernel-verifier) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Verify generated operator kernels with static checks, import or compile checks, and numerical comparison.
- [kernel-workflow](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/kernel-workflow) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Run a multi-backend AI kernel generation, validation, and optimization workflow for Triton, CUDA C, C++, and TileLang.
- [llm-serving-auto-benchmark](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-serving-auto-benchmark) ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social&cacheSeconds=86400) - Compare SGLang, vLLM, TensorRT-LLM, and other serving frameworks under the same workload and latency SLA.
- [vllm-ascend-operator-fusion](https://github.com/mindspore-ai/akg/tree/master/akg_agents/workspace/.opencode/skills/vllm-ascend-operator-fusion) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Analyze vLLM-Ascend model operator paths and generate fusion kernel strategies.
- [flaggems-pr-submit-flagos](https://github.com/flagos-ai/skills/tree/main/skills/flaggems-pr-submit-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Prepare and submit FlagGems PRs in FlagOS workflows.
- [flaggems-pr-review-flagos](https://github.com/flagos-ai/skills/tree/main/skills/flaggems-pr-review-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Review FlagGems PRs in FlagOS workflows.
- [kernelgen-flagos](https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Generate kernels in FlagOS workflows.
- [perf-test-flagos](https://github.com/flagos-ai/skills/tree/main/skills/perf-test-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Run performance tests in FlagOS workflows.
- [tle-developer-flagos](https://github.com/flagos-ai/skills/tree/main/skills/tle-developer-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Develop TLE kernels and workflows for FlagOS.
- [vllm-plugin-fl-setup-flagos](https://github.com/flagos-ai/skills/tree/main/skills/vllm-plugin-fl-setup-flagos) ![GitHub stars](https://img.shields.io/github/stars/flagos-ai/skills?style=social&cacheSeconds=86400) - Set up vLLM plugin workflows for FlagOS.

### Knowledge Bases and Reference Skills

- [KernelWiki](https://github.com/mit-han-lab/KernelWiki) ![GitHub stars](https://img.shields.io/github/stars/mit-han-lab/KernelWiki?style=social&cacheSeconds=86400) - Blackwell and Hopper kernel optimization knowledge base.
- [kernel-KBS](https://github.com/fmh66/kernel-opt-agent/tree/main/skills/kernel-KBS) ![GitHub stars](https://img.shields.io/github/stars/fmh66/kernel-opt-agent?style=social&cacheSeconds=86400) - Corpus-backed kernel knowledge base for CUDA, Triton, CuTe, CUTLASS, and modern NVIDIA architectures.
- [cuda](https://github.com/technillogue/ptx-isa-markdown/tree/main/cuda_skill) ![GitHub stars](https://img.shields.io/github/stars/technillogue/ptx-isa-markdown?style=social&cacheSeconds=86400) - PTX ISA and CUDA reference skill.
- [cuda-skill](https://github.com/slowlyC/agent-gpu-skills/tree/main/cuda_skill) ![GitHub stars](https://img.shields.io/github/stars/slowlyC/agent-gpu-skills?style=social&cacheSeconds=86400) - CUDA, PTX ISA, Nsight Compute, Nsight Systems, and CUDA API reference skill.
- [cutlass-skill](https://github.com/slowlyC/agent-gpu-skills/tree/main/cutlass_skill) ![GitHub stars](https://img.shields.io/github/stars/slowlyC/agent-gpu-skills?style=social&cacheSeconds=86400) - Write, debug, and optimize CUTLASS and CuTe DSL kernels.
- [triton-skill](https://github.com/slowlyC/agent-gpu-skills/tree/main/triton_skill) ![GitHub stars](https://img.shields.io/github/stars/slowlyC/agent-gpu-skills?style=social&cacheSeconds=86400) - Write, debug, and optimize Triton and Gluon GPU kernels.
- [accelerated-computing-cudf](https://github.com/NVIDIA/skills/tree/main/skills/accelerated-computing-cudf) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/skills?style=social&cacheSeconds=86400) - NVIDIA-authored guidance for cuDF, pandas acceleration, dask-cuDF, and GPU DataFrame workflows.
- [tilelang-wiki](https://github.com/Adversarr/kernel-dsl/tree/main/skills/tilelang-wiki) ![GitHub stars](https://img.shields.io/github/stars/Adversarr/kernel-dsl?style=social&cacheSeconds=86400) - Local, code-grounded TileLang reference for DSL semantics, API lookup, examples, tuning, debugging, and compiler behavior.
- [cuda-c-api](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/cuda-c/guides/cuda-c-api) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - CUDA C API reference skill for kernel-generation agents.
- [cuda-c-basics](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/cuda-c/guides/cuda-c-basics) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - CUDA C basics reference for agent-authored operators.
- [cuda-c-optimization](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/cuda-c/guides/cuda-c-optimization) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - CUDA C optimization reference for memory access, tiling, occupancy, and kernel performance.
- [cuda-c-patterns](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/cuda-c/guides/cuda-c-patterns) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - CUDA C kernel pattern reference for generated operator implementations.
- [triton-cuda-api](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-api) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA API reference skill.
- [triton-cuda-basics](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-basics) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA basics reference for operator-kernel generation.
- [triton-cuda-optimization](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-optimization) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA optimization reference for memory, tiling, occupancy, and fusion choices.
- [triton-cuda-patterns](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-patterns) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA kernel pattern reference.
- [triton-cuda-debugging](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-debugging) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA debugging reference for generated kernels.
- [triton-cuda-memory](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-memory) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA memory-reference skill for load/store, masks, tiling, and memory hierarchy.
- [triton-cuda-matmul](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-matmul) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA matmul implementation reference.
- [triton-cuda-attention](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-cuda/guides/triton-cuda-attention) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton CUDA attention-kernel reference.
- [tilelang-cuda-api](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-api) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA API reference skill.
- [tilelang-cuda-basics](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-basics) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA basics reference for writing kernels.
- [tilelang-cuda-optimization](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-optimization) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA optimization reference for agent-generated kernels.
- [tilelang-cuda-memory](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-memory) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA memory hierarchy and movement reference.
- [tilelang-cuda-synchronization](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-synchronization) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA synchronization reference for safe kernel generation.
- [tilelang-cuda-patterns](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/tilelang-cuda/guides/tilelang-cuda-patterns) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - TileLang CUDA kernel pattern reference.
- [triton-ascend-basics](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/fundamentals/triton-ascend-basics) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton Ascend basics reference for operator generation.
- [triton-ascend-optimization](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/fundamentals/triton-ascend-optimization) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton Ascend optimization reference.
- [triton-ascend-ascend-hardware-constraints](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/fundamentals/triton-ascend-ascend-hardware-constraints) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Ascend hardware constraint reference for Triton-Ascend kernel generation.
- [triton-ascend-memory](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/fundamentals/triton-ascend-memory) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton Ascend memory reference for agent-written kernels.
- [triton-ascend-matmul](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/guides/triton-ascend-matmul) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton Ascend matmul reference.
- [triton-ascend-attention](https://github.com/mindspore-ai/akg/tree/master/akg_agents/python/akg_agents/op/resources/skills/triton-ascend/guides/triton-ascend-attention) ![GitHub stars](https://img.shields.io/github/stars/mindspore-ai/akg?style=social&cacheSeconds=86400) - Triton Ascend attention-kernel reference.
- [cudaq-guide](https://github.com/NVIDIA/cuda-quantum/tree/main/skills/cudaq-guide) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/cuda-quantum?style=social&cacheSeconds=86400) - CUDA-Q onboarding reference for GPU simulation, QPU hardware, Python, C++, and quantum application workflows.

## Using These Skills

Most agent skill systems (Claude Code/Codex/Cursor/...) expect each skill to live in a dedicated folder with a `SKILL.md` file. For Codex, install the desired skill directory under:

```bash
$CODEX_HOME/skills
```

If `CODEX_HOME` is unset, the default is commonly:

```bash
~/.codex/skills
```

After installing a skill, restart the agent so it reloads metadata from `SKILL.md`, then describe the kernel task naturally or mention the skill name.

## Creating Kernel Skills

A minimal kernel skill should look like this:

```text
skill-name/
|-- SKILL.md
|-- scripts/
|-- references/
`-- assets/
```

Only `SKILL.md` is required. Keep the trigger metadata precise, keep the execution steps focused, and move long reference material into `references/` so the agent can load it only when needed.

Useful kernel-skill topics include:

- Kernel authoring patterns for CUDA, Triton, TileLang, HIP, and ROCm.
- Profiling workflows for Nsight Compute, Nsight Systems, PyTorch Profiler, and rocprof.
- Correctness checks, numerical tolerance policies, and reference implementations.
- Debugging generated code, race conditions, synchronization bugs, and memory errors.
- Optimization playbooks for memory coalescing, occupancy, tiling, vectorization, fusion, and tensor cores.

## Contributing

PRs are welcome. Please add real, reusable skills rather than general articles, and include:

- Skill name.
- Source URL.
- Short description.
- Suggested category.
- Install notes if the skill is not at the repository root.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the lightweight contribution checklist.

## Resources

- [Codex Skills](https://developers.openai.com/codex/skills/)
- [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Triton Documentation](https://triton-lang.org/main/index.html)
- [TileLang Documentation](https://tilelang.com/)
