# Evidence Report Subskill

Use this integrated subskill when the goal is not only to run profilers but to
produce a reusable diagnosis handoff.

This is the generic profiling diagnosis layer. Repository-specific workflows
may wrap it with target-specific execution context, baseline timing, or
artifact inventory, but they should not redefine the profiling diagnosis
sections themselves.

## Expected Diagnosis Surface

A good profiling handoff should include at minimum:

- target and execution context
- profiling build target, including any architecture-specialized target such as
  `sm_90a` or `sm_100a` when relevant
- command shapes used for `nsys` and `ncu`
- `mlsys-cli eval timing` runner mode when that command surface is profiled;
  use and report `--runner persistent` for `nsys` or `ncu`
- hotspot summary from `nsys`
- selected kernels for focused diagnosis
- bottleneck classification
- NVTX correlation notes when applicable
- code-block or code-segment analysis when NVTX ranges, wrapper annotations, or
  source views exist
- source-attribution notes when applicable
- evidence artifact paths or references
- broad follow-on areas or next-skill guidance

## Reporting Discipline

- Distinguish measured findings from broad follow-on areas.
- Keep broad follow-on areas distinct from concrete rewrite plans.
- Do not present raw counters without interpretation.
- Do not present a speculative code change as proven unless it was implemented
  and remeasured elsewhere.
- Keep the next action concrete: what to inspect next, or which skill should
  own the follow-on work.
- If NVTX was inserted or already present, include a code-segment analysis that
  maps `nsys` phase evidence to the wrapper, launch path, and likely source
  segment that deserves inspection next.
- If source views exist, localize the likely hot block or region and state the
  broad inspection area without prescribing the rewrite.

## Recommended Generic Profiling Diagnosis Template

```text
Target:
Question:
Execution Context:
Profiling Build Target:
Profiler Scope:
nsys Command Shape:
ncu Command Shape:
MLSys Runner Mode:
Hotspot Summary:
Selected Kernels:
Bottleneck Class:
Key Counters:
NVTX Correlation:
Code Segment Analysis:
Source Attribution:
Evidence Files:
Measured Findings:
Broad Follow-on Areas:
Suggested Follow-on Skill:
```

## Wrapper Guidance For Repository Workflows

When a repository workflow needs profiling output inside a larger report, keep
this diagnosis block intact and wrap it with workflow-specific sections around
it. Typical wrapper sections include:

- scope
- target-specific execution surface
- baseline timing
- artifact inventory
- optimization or next-run handoff

The wrapper skill owns those outer sections. This subskill owns the profiling
diagnosis block.

## Template Notes

- `Code Segment Analysis` becomes required when NVTX ranges, wrapper
  annotations, or launch-path labels exist. Use it to bridge:

  ```text
  NVTX phase/wrapper -> dominant kernel(s) -> source segment to inspect next
  ```

- Keep `Measured Findings` separate from `Broad Follow-on Areas`.
- Prefer one dominant bottleneck class per kernel unless evidence clearly shows
  a staged bottleneck shift.

This makes the output consumable by later optimization or runner skills instead
of trapping the profiling session in one-off prose.
