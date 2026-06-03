---
name: kernel-loop
description: Iterative GPU kernel optimization orchestrator for CUDA/CUTLASS/CuTe DSL/Triton kernels. Use for measured, one-change-at-a-time optimization loops with correctness, NCU profiling, KBS evidence, hypothesis discipline, hard iteration gates, final benchmarking, and a traceable report.
---

# kernel-loop

Run a measured kernel optimization loop. The loop is valid only when every version is correct, profiled, evidence-backed, and accepted by the gate script before the next version is created.

This is an orchestration skill. It does not own profiling, KBS search, or benchmarking scripts. It owns only the loop rules, output layout, local report templates, and the iteration gate.

## Skill Ownership

Use the existing skills below for their own responsibilities. Paths in this table are relative to that skill's root directory. Do not assume a fixed install location.

| Skill | Role |
|---|---|
| `kernel-loop` | Orchestrates version order, artifact layout, hypothesis discipline, final report shape, and the gate script at `scripts/check_iteration_gate.sh`. |
| `kernel-profile` | Owns environment readiness, optional GPU clock config, correctness checks, NCU profiling, bottleneck evidence, and generated artifacts such as `env_check.md`, `correctness.md`, `ncu_summary.md`, and `ncu_details.md`. See its `SKILL.md`, `scripts/scripts.md`, and `reference/NCU.md`. |
| `kernel-KBS` | Owns read-only evidence search for optimization patterns, hardware features, prior PRs, and implementation examples. See its `SKILL.md` and `scripts/scripts.md`. |
| `kernel-benchmark` | Owns final correctness/timing comparison against PyTorch eager, torch.compile, or FlashInfer baselines, and writes `benchmark.md`. See its `SKILL.md` and `scripts/scripts.md`. |

The file names under `<output_dir>` are artifacts produced or written during the loop. They are not skill source paths.

## Inputs

Required: `<kernel>`, `<ref.py>`, `<implementation>`, `<dims>`.

Defaults: `<gpu>=0`, `<N>=3`.

Keep these fixed for the whole run: implementation, dimensions, seed, tolerances, GPU, pointer sizing, warmups, and timing trials.

## Version Model

`v0` is the baseline. `v1..vN` are optimization attempts. There are `N` transitions:

```text
v0 -> v1 -> ... -> vN
```

Each version directory must contain measured artifacts:

```text
<output_dir>/
+-- ref.py
+-- env_check.md
+-- current_iteration.txt        (managed only by the gate)
+-- v0/
|   +-- correctness.md
|   +-- ncu_summary.md
|   +-- ncu_details.md
|   +-- kbs_evidence.md
|   +-- hypothesis.txt           (required only when creating v1)
|   +-- kernel.py | kernel.cu | <single .py/.cu kernel>
+-- v1/
+-- ...
+-- vN/                          (final version: no next-change hypothesis required)
+-- benchmark.md
+-- final_report.md
```

Do not pre-populate future version directories. Empty placeholders are allowed but unnecessary. A future `vK+1` kernel, correctness file, NCU file, KBS file, or hypothesis before `vK` passes the gate invalidates the loop.

## Non-Negotiable Rules

1. Work on exactly one current version: the version named in `<output_dir>/current_iteration.txt`. If the file does not exist, the first valid gate run must be `v0`.
2. Correctness must pass before NCU profiling. Do not profile or optimize a failing kernel.
3. Write `kbs_evidence.md` after NCU profiling and before `hypothesis.txt`.
4. Write `hypothesis.txt` before creating the next kernel version. It belongs to the source version, for example `v1/hypothesis.txt` explains `v1 -> v2`.
5. Each transition changes exactly one independent performance variable. Mechanical edits that support that one change are allowed; do not bundle a second tactic.
6. Never parallelize or batch-create future iterations. Do not create `vK+1` artifacts while `current_iteration.txt` still says `vK`.
7. After any context compaction, resume, or long interruption, first read `current_iteration.txt` and continue only from that version. If later version artifacts already exist, stop and treat the loop state as drifted.
8. The gate script is the only accepted proof that an iteration is complete. If it exits non-zero, stop, fix the current version, and rerun it. Do not continue by reasoning around the failure.

## Loop

For `v0`:

1. Use `kernel-profile` to run environment check and write `<output_dir>/env_check.md`; copy `ref.py`.
2. Place the baseline kernel in `v0`.
3. Use `kernel-profile` correctness tooling to write `<output_dir>/v0/correctness.md`.
4. Use `kernel-profile` NCU tooling to write `<output_dir>/v0/ncu_summary.md` and `<output_dir>/v0/ncu_details.md`.
5. Use `kernel-KBS` to query measured NCU symptoms. Minimum: one kernel-specific query and one bottleneck-pattern query.
6. Write `kbs_evidence.md`, then `hypothesis.txt`.
7. Run the transition gate on `v0`. Only after it passes, create `v1`.

For each `vK` where `0 < K < N`, repeat steps 3-7. Preserve regressions; never overwrite an older version.

For final `vN`, use `kernel-profile` for correctness and NCU artifacts, use `kernel-KBS` for evidence, then run the final gate. Do not write a next-change hypothesis unless the user extends `N`.

## Gate Commands

Transition gate, before creating `vK+1`:

```bash
bash <kernel-loop-skill>/scripts/check_iteration_gate.sh <output_dir>/v<K> --verbose
```

Final pre-completion audit, before benchmarking and reporting:

```bash
bash <kernel-loop-skill>/scripts/check_iteration_gate.sh <output_dir>/v<N> --final --verbose
```

`<kernel-loop-skill>` means the resolved root directory of this skill, not a hard-coded repository path.

The gate enforces:

- required files are present and non-empty; their contents are not parsed
- NCU files exist; their content is generated by profiling scripts and is not re-parsed by this gate
- KBS and hypothesis files use the expected title hierarchy; body text, tables, metrics, query counts, and evidence values are not parsed
- kernel file exists, is non-empty, is unambiguous, and differs from the previous version when `K > 0`
- `current_iteration.txt` matches the checked version
- future versions are not pre-populated before the current gate passes

## KBS Evidence

Use `references/kbs_evidence.md` from this skill as a suggested output format. Query evidence through `kernel-KBS`; do not treat this template as a KBS retrieval script. The gate checks only that `<output_dir>/vK/kbs_evidence.md` exists, is non-empty, and has the expected H1/H2 title order.

## Hypothesis

Use `references/hypothesis.md` from this skill as a suggested output format. The hypothesis should describe the next one-variable change. The gate checks only that `<output_dir>/vK/hypothesis.txt` exists, is non-empty, and has the expected top-level section order before the next version is created.

## Final Report

Use `kernel-benchmark` only after the final gate passes. Write `final_report.md` from measured artifacts only, using `references/report_template.md` from this skill.

The report must compare every version, summarize every transition hypothesis, list KBS evidence actually used, identify the best correct version, and cite `benchmark.md` for final speedups. Use `N/A` for uncollected metrics and `unknown` only when an expected value cannot be located in artifacts.
