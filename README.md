# Awesome Agent Kernel Skills

A curated, high-signal index of agent skills for writing, testing, profiling, debugging, and optimizing GPU kernels.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](./CONTRIBUTING.md)

This list focuses on reusable skill packages for AI coding agents that work on GPU kernels and accelerator programming. Entries are grouped by the primary workflow they support.

## Contents

- [What Are Agent Kernel Skills?](#what-are-agent-kernel-skills)
- [Skills](#skills)
  - [CUDA Kernel Development](#cuda-kernel-development)
  - [TileLang Kernel Development](#tilelang-kernel-development)
  - [Profiling and Performance Analysis](#profiling-and-performance-analysis)
  - [Debugging and Correctness](#debugging-and-correctness)
  - [Testing](#testing)
- [Using These Skills](#using-these-skills)
- [Creating Kernel Skills](#creating-kernel-skills)
- [Contributing](#contributing)
- [Resources](#resources)

## What Are Agent Kernel Skills?

Agent skills are reusable instruction packages that teach an AI coding agent how to perform a focused class of work. A typical skill contains a `SKILL.md` file with metadata and execution guidance, plus optional `scripts/`, `references/`, or `assets/` for deterministic helpers and longer documentation.

Agent kernel skills specialize that pattern for low-level GPU and accelerator programming. They can encode workflows for authoring kernels, reviewing memory access patterns, profiling with tools such as NVIDIA Nsight Compute, debugging generated kernels, validating forward/backward operators, or optimizing framework-specific kernels.

## Skills

### CUDA Kernel Development

- [cuda](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/cuda_skill) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - CUDA programming skill from `tilelang-cuda-skills`; useful as a general CUDA kernel authoring and review reference.

### TileLang Kernel Development

- [writing-tilelang-kernels](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/writing-tilelang-kernels) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Guidance for writing TileLang kernels.
- [optimizing-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/optimizing-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Optimization workflow for TileLang programs.

### Profiling and Performance Analysis

- [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) ![GitHub stars](https://img.shields.io/github/stars/mit-han-lab/ncu-report-skill?style=social) - Analyze NVIDIA Nsight Compute reports for kernel performance bottlenecks.
- [profiling-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/profiling-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Profiling workflow for TileLang programs.
- [torch-profiling-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/torch-profiling-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Profiling TileLang programs in PyTorch-facing workflows.

### Debugging and Correctness

- [debugging-tilelang-programs](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/debugging-tilelang-programs) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Debugging workflow for TileLang programs.

### Testing

- [testing-fwd-bwd-kernels](https://github.com/sablin39/tilelang-cuda-skills/tree/main/skills/tilelang/testing-fwd-bwd-kernels) ![GitHub stars](https://img.shields.io/github/stars/sablin39/tilelang-cuda-skills?style=social) - Testing workflow for forward and backward kernels.

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
