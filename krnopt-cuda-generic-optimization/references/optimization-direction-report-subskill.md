# Optimization Direction Report Subskill

Use this integrated subskill when the user wants a reusable handoff or a clean
analysis shape for a CUDA kernel or source region.

This report is general CUDA-kernel guidance. It maps source plus evidence to
ranked optimization directions. Do not include repository-specific variant
lifecycle, prepared-cache, before/after research-run, or official-kit fields
here; those belong in the local repository workflow, not this generic direction
report.

## Recommended Report Template

```markdown
# CUDA Kernel Optimization Direction Report

## Target

- kernel or function:
- source region(s):
- GPU or architecture:
- workload or phase context:
- input evidence:

## Evidence Posture

- evidence level: source only | source + NVTX | source + nsys | source + ncu | line-attributed
- confidence: high | medium | low
- main evidence gaps:

## Measured Findings

- runtime-dominant kernel or phase:
- bottleneck indicators:
- source-attributed hotspots:
- relevant counters or symptoms:

## Localized Cause Candidates

| code region | observed symptom | likely cause | confidence |
| --- | --- | --- | --- |
|  |  |  |  |

## Bottleneck Summary

- primary bottleneck family:
- secondary bottleneck family:
- why this classification fits:
- what remains uncertain:

## Ranked Optimization Directions

### Direction 1: <short name>

- target region:
- direction summary:
- why it fits the evidence:
- expected effect:
- risks or trade-offs:
- validation needed:
- reject if:
- confidence:

### Direction 2: <short name>

- target region:
- direction summary:
- why it fits the evidence:
- expected effect:
- risks or trade-offs:
- validation needed:
- reject if:
- confidence:

### Direction 3: <short name>

- target region:
- direction summary:
- why it fits the evidence:
- expected effect:
- risks or trade-offs:
- validation needed:
- reject if:
- confidence:

## Directions Rejected For Now

- idea:
  reason:

## Recommended Next Step

- next owner or skill:
- next action:
- extra evidence needed:
```

## Reporting Discipline

- Lead with measured findings before hypotheses.
- Tie every direction to a specific code region or structural choice.
- Use conditional language when the evidence is only static inspection.
- Say what extra profiler evidence would disambiguate the next step.
- Keep registered-variant execution details out of this report.

## Decision-Knowledge Pattern

A well-formed optimization-direction report is not a free-text narrative; it
is a *decision-knowledge artifact*. Two memory layers make this explicit, and
both should be visible in the report.

**Cross-task decision knowledge.** This is the reusable part: the evidence-to-
direction mapping, the named scenarios (memory-bound access patterns,
underutilized parallelism, naive GEMM structure, launch overhead, register
pressure, etc.), and the deterministic rules that narrow candidate methods
from evidence. Converting raw technique catalogs into operational decision
knowledge follows three steps that the report should mirror:

1. **Scenario abstraction** — group candidate techniques into task-independent
   situations rather than listing them as flat lore.
2. **Evidence formalization** — define how each decision factor is measured
   from normalized profiling metrics, runtime features, and static code
   features.
3. **Rule materialization** — encode evidence-to-method logic as predicates,
   priority rules, veto rules, and a decision table that returns *allowed*
   methods, not a single forced answer.

In a single report this surfaces as: the "Bottleneck Summary" names the
scenario; "Measured Findings" and "Localized Cause Candidates" encode the
evidence formalization; "Ranked Optimization Directions" are the allowed
methods under the current rules, each tagged with the rule that admits it.

**Per-task trajectory memory.** This is the disposable part: optimization
plans tried so far, repair steps after compile or correctness failures,
profiler feedback at each iteration, latency or speedup values, and the
generated kernel versions. Two structural shapes matter:

- **Repair memory** is chain-shaped. Once a generated kernel fails compile
  or correctness, each repair step uses the latest failed kernel as the base,
  but the diagnoser sees the full chain of previous failed attempts and their
  outcomes so repairs do not oscillate.
- **Optimization memory** is organized around a correctness-passing *base*
  kernel. New candidates are evaluated against that base. The base changes
  only when a new kernel exceeds a configured speedup threshold (a deliberate
  noise filter; small improvements are remembered but do not become the next
  anchor).

In a single report this surfaces in two places: the "Directions Rejected For
Now" section carries what was tried and why it was rejected, and the
"Recommended Next Step" distinguishes between continuing on the current base
kernel (optimization-memory move) and reverting to an earlier working version
(repair-memory move).

## Gate Doctrine For Each Direction

Good reports treat each proposed direction as a decision that must pass a
small gate before it is written down. The gate is:

- cite at least one measured metric or one concrete source-structure clue
- cite at least one section-level analysis conclusion (bottleneck class,
  stall family, cache pattern, SOL gap)
- explicitly rule out at least one alternative bottleneck the evidence could
  plausibly imply
- state the allowed-method rule that admits this direction so the reasoning
  is auditable
- propose exactly one incremental change when the evidence supports a single
  step; avoid bundling unrelated edits into one direction

When the gate cannot be passed, the report should say so and list the missing
measurement instead of promoting a weaker direction.

## Acceptance Thresholds And Base Promotion

The report should make acceptance logic explicit so the reader knows when a
measured result should replace the base kernel versus merely be recorded.

A typical pattern:

- **Accept**: correctness passes and the improvement clears a configured
  threshold (for example a relative speedup above some percentage, or an
  absolute time gain above some minimum). Below-threshold improvements are
  noise; keep them in history but do not re-anchor.
- **Reject**: correctness fails, or runtime regresses. Revert to the current
  base.
- **Early stop**: large regressions (for example several-fold slowdown) or
  repeated reverts in a row are signal that the current direction is
  exhausted; move on rather than spend more budget.

State the thresholds used (or being assumed) in the "Recommended Next Step"
block so a downstream owner can re-run with the same rules.

## Coverage And Fallback

If the rule table or the evidence posture does not admit a direction, say so
and fall back to the next-best owner rather than inventing one:

- missing or under-specified measurement → escalate to the profiling skill,
  naming the specific counter, section, or NVTX mark needed
- missing or under-specified hardware decision → escalate to the hardware-
  aware optimization skill, naming the specific architecture feature at stake
- evidence is real but still does not support a definite direction → do
  targeted docs or source research rather than guessing

A good report is honest about its coverage gaps; a silent "laundry list" of
directions without a rule behind each one is a regression from the decision-
knowledge pattern.
