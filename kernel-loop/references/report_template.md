# Report Template

Use this template when writing `<output_dir>/final_report.md`.

Fill values from measured artifacts only:

| Artifact | Use |
|---|---|
| `env_check.md` | GPU, CUDA, nvcc, ncu, nsight-python, Triton, PyTorch |
| `v*/correctness.md` | Correctness status |
| `v*/ncu_summary.md` | Runtime, throughput, occupancy, bottleneck, high-level stalls |
| `v*/ncu_details.md` | Detailed metrics not present in the summary |
| `v*/kbs_evidence.md` | KBS queries, selected doc ids/paths, confidence labels, applicability notes |
| `v*/hypothesis.txt` | Strategy, rationale, expected gain, decision rule |
| `benchmark.md` | Final timing and baseline speedups |

Use `N/A` for metrics that were not collected. Use `unknown` for expected values that cannot be located in artifacts. Do not invent missing measurements.
Do not write `KBS docs: N/A` or `No KBS queries were made` unless the corresponding `kbs_evidence.md` exists.

In strategy tables, list only strategies that were actually used in at least one version. Use `yes` for the version where the listed strategy was applied and `no` for versions where that same strategy was not applied. Do not include unused strategy rows.

```markdown
# CUDA Optimization Final Report - `<kernel_name>` (`<date>`)

## Environment

| Item | Value |
|---|---|
| GPU | `<name>` (CC `<x.y>`) |
| CUDA / nvcc | `<version>` |
| Kernel file | `<path>` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | ... | best |
|---|---|---|---|---|---|---|
| Correctness | | | | | | |
| Execution Time (ms) | | | | | | |
| Speedup (x) | 1.00 | | | | | |
| Memory Throughput (%) | | | | | | |
| Compute Throughput (%) | | | | | | |
| SM Active Cycles (%) | | | | | | |
| Bottleneck | | | | | | |
| Achieved Occupancy (%) | | | | | | |
| Active Warps / SM | | | | | | |
| Registers / Thread | | | | | | |
| Warp Stall - Long SB (%) | | | | | | |
| Warp Stall - Short SB (%) | | | | | | |
| Branch Divergence (%) | | | | | | |

---

## Optimization Strategies per Version

Only include strategies that were actually used in at least one version. Add rows dynamically from `hypothesis.txt` and code changes instead of keeping a fixed full catalog.

| Strategy | v1 | v2 | v3 | ... |
|---|---|---|---|---|
| `<used strategy>` | yes/no | yes/no | yes/no | |
| `<used strategy>` | yes/no | yes/no | yes/no | |

---

## Hypothesis Outcomes

| Transition | NCU Symptom | KBS Pattern | Hypothesis | Result | Evidence |
|---|---|---|---|---|---|
| v0 -> v1 | `<measured bottleneck/stall>` | `<doc-id: pattern>` | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |
| v1 -> v2 | `<measured bottleneck/stall>` | `<doc-id: pattern>` | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |
| v2 -> v3 | `<measured bottleneck/stall>` | `<doc-id: pattern>` | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |

---

## KBS Evidence

| Version | Query | Doc ID | Canonical path | Confidence | Used for / Applicability |
|---|---|---|---|---|---|
| v1 | `<query string>` | `<doc-id>` | `<path>` | `<verified/source-reported/inferred/experimental>` | `<how it maps to NCU symptom and selected change>` |
| v2 | `<query string>` | `<doc-id>` | `<path>` | `<confidence>` | `<how it maps to NCU symptom and selected change>` |

For rejected KBS results, summarize briefly below the table when they affected the decision:

- `vK`: `<doc-id>` rejected because `<architecture/layout/dtype/bottleneck mismatch>`.

---

## NCU + KBS Synthesis

| Version | NCU fact set | KBS evidence | Decision |
|---|---|---|---|
| v1 | `<runtime + bottleneck + key metrics>` | `<doc ids/patterns>` | `<why the one-variable change follows from both>` |
| v2 | `<runtime + bottleneck + key metrics>` | `<doc ids/patterns>` | `<why the one-variable change follows from both>` |

---

## Final Benchmark

| Item | Value |
|---|---|
| Best kernel | `<path>` |
| Baselines | `<pytorch-eager, torch-compile, flashinfer, ...>` |
| Best execution time | `<ms>` |
| Baseline execution time | `<ms>` |
| Benchmark speedup | `<x>` |
| Benchmark artifact | `benchmark.md` |

---

## Best Version Conclusion

**Best version:** `v<N>` - execution time reduced from `<v0>` ms to `<vN>` ms, speedup `<x>`.

Key gains: `<primary optimization strategies>`.

Stopping reason: `<max iterations reached / performance target met / bottleneck saturated>`.

**Remaining optimization opportunities:** `<potential improvements for the next round, or N/A>`
```
