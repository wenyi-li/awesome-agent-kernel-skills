---
name: krnopt-cuda-domain-optimization
description: Domain-specific CUDA kernel optimization guidance for choosing and planning kernel designs by workload family. Use when Codex needs to optimize a CUDA kernel using domain patterns such as MoE routing, grouped GEMM, fused expert MLPs, sparse dispatch/combine, persistent MoE scheduling, padding elimination, or future domain-specific CUDA playbooks under this skill.
---

# CUDA Domain Optimization

Use this skill when the kernel problem is best understood by workload domain,
not just by generic memory/coalescing/tile advice. The skill currently covers
MoE kernel design first; add future domains as sibling reference files.

## Workflow

1. Identify the domain and workload shape.
   - For MoE, classify the kernel as routing/dispatch, grouped GEMM, expert
     MLP fusion, combine/reduction, whole-operator fusion, or communication
     fusion.
   - Record target architecture, dtype/scale format, top-k, expert count,
     token shape, local expert count, and whether the run is prefill, decode,
     training, or contest-style fixed geometry.
2. Load the relevant domain reference.
   - MoE kernel design: `references/moe-kernel-design.md`
3. Decide which domain boundary is actually expensive before proposing a
   rewrite.
   - Prefer measured bottlenecks: routing/permutation, padding, GEMM1,
     activation, GEMM2, combine, launch overhead, communication, or scheduler
     imbalance.
4. Compose the domain pattern with the hardware-aware and coding skills.
   - Use `krnopt-hw-aware-optimization` for architecture-specific surfaces
     such as SM90 TMA/WGMMA or SM100 TMEM/UMMA.
   - Use `krnopt-cuda-coding` for implementation mechanics, including stream,
     handle, workspace, build, and synchronization checks when a library or
     generated-kernel path is selected.
   - Use the local repository's existing workflow for variant creation,
     workload selection, timing, and validation.
   - When the selected domain pattern depends on architecture-specific CUDA
     features, require the implementation and timing path to use the
     specialized build target such as `sm_90a` or `sm_100a`, not only general
     targets such as `sm_90` or `sm_100`.
5. Output a concrete optimization plan.
   - Name the current bottleneck hypothesis.
   - Name the selected domain pattern and why it fits.
   - Specify what kernel boundary changes.
   - Specify the required architecture-specialized build target when relevant.
   - Specify correctness gates, timing workloads, accept/reject criteria, and a
     fallback if the domain-specific pattern is too expensive.

## Domain References

- `references/moe-kernel-design.md`: MoE-specific kernel design patterns,
  decision tree, measurement checklist, and implementation task templates.

## Guardrails

- Do not recommend domain fusion before identifying which boundary costs time
  or memory.
- Do not treat MoE as only "make grouped GEMM faster"; routing, dispatch,
  activation, combine, padding, and scheduling often dominate.
- Do not mix architecture-specific mechanisms across generations. Domain
  patterns are portable ideas; implementation surfaces still depend on the
  target architecture.
- Do not copy paper headline speedups into claims about a local kernel. Convert
  them into hypotheses and validate against local workloads.
- Do not add future domains into `SKILL.md`; add one focused file under
  `references/` and link it from the Domain References section.
