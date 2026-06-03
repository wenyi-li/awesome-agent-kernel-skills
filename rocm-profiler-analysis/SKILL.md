---
name: rocm-profiler-analysis
description: >
  Analyze SGLang and vLLM profiler traces on AMD ROCm systems, especially MI355X/gfx950 nodes.
  Adapted from the SGLang torch-profiler workflow: triage kernel breakdown, overlap headroom,
  and fuse opportunities, then write structured artifacts that can be attached to amdpilot
  experiments, trials, and dashboard views. Use when a run needs profiling, when an optimization
  trial should produce machine-readable profiling artifacts, or when the user asks why a ROCm
  workload is slow.
---

# ROCm Profiler Analysis

Use this skill when you need to turn a profiling run into **structured optimization evidence**
instead of a raw trace file.

This skill is the AMD/ROCm/MI355X adaptation of SGLang's torch-profiler analysis workflow. It is
designed for our current amdpilot stack:

- MI355X / gfx950 nodes
- ROCm 7.2
- SGLang / vLLM issue-driven runs
- dashboard artifacts, not just terminal output

## Why This Exists

Raw traces are not good enough for agents or dashboards. They tell you that time was spent
somewhere, but they do not directly answer:

1. Which kernel families dominate prefill or decode on ROCm?
2. Which kernels still have overlap headroom?
3. Which hotspots map back to Python or operator-level code paths?
4. Which results are actually relevant to **gfx950 / MI355X**, and which are only generic?
5. Which profiling outputs should be written into our canonical experiment/trial schema?

This skill standardizes that path.

## Main Workflow

Preserve the same four subcommands as the upstream SGLang profiler skill:

- `triage`
- `breakdown`
- `overlap`
- `perfetto-fix`

For normal agent use, default to `triage`.

### `triage`

Use this when you want one compact answer with three main outputs:

- kernel table
- overlap-opportunity table
- fuse-opportunity table

### `breakdown`

Use this when you need one-trace category share analysis without overlap reasoning.

### `overlap`

Use this when you have both:

- a graph-off mapping trace
- a graph-on formal trace

and need to tie overlap headroom back to code paths.

### `perfetto-fix`

Use this only when Perfetto renders overlapped lanes incorrectly and you need a repaired trace for
human inspection.

## Recommended Inputs

This skill supports two input shapes:

1. **Existing trace directory / trace file**
   - `trace.json`
   - `trace.json.gz`
   - profiler output directory

2. **Live server / live experiment**
   - trigger profiling against a running SGLang or vLLM server
   - then immediately analyze the result and attach artifacts back to the run

For amdpilot integration, prefer the second path for optimization-stage profiling and the first
path for post-hoc investigation.

## AMD / ROCm Adaptation Rules

### 1. Use ROCm-native kernel categories

Do not reuse CUDA/H100/B200 assumptions. On our nodes, category tables should explicitly account
for ROCm-specific paths:

- RCCL / communication
- Triton kernels
- CK / composable kernel paths
- AITER paths
- hipBLASLt / rocBLAS GEMM
- MIOpen / attention runtime kernels
- quantization
- normalization
- memory / copy / scheduler overhead

See [references/rocm-kernel-categories.md](references/rocm-kernel-categories.md).

### 2. Keep hardware relevance explicit

Every profiling result must declare whether it is truly relevant to our MI355X node:

- `observed_arch`: actual arch from the run
- `arch_match`: `exact | compatible | unknown`
- `hardware_relevance_reason`: short human-readable explanation

Do not hide `gfx942` vs `gfx950` differences.

### 3. Treat profiling as structured artifacts

Do not stop at stdout tables. Write stable artifacts that can be attached to an experiment or
trial. Minimum recommended outputs:

- `profile_summary.md`
- `profile_metadata.json`
- `kernel_table.json`
- `overlap_opportunities.json`
- `fuse_opportunities.json`
- `perfetto_fixed_trace.json` if used

See [references/artifact-contract.md](references/artifact-contract.md).

## Canonical Metadata Contract

`profile_metadata.json` should contain enough information to tie profiling results back to the
dashboard and DB.

Minimum fields:

- `experiment_id`
- `trial_id`
- `observed_arch`
- `arch_match`
- `hardware_relevance_reason`
- `rocm_version`
- `base_image`
- `resource_class`
- `gpu_device_ids`
- `gpu_clocks_mhz`
- `preflight_passed`
- `server_flags`
- `benchmark_config_hash`
- `model_name`
- `profile_stage`
- `source_trace_path`

This is the difference between "a useful local notebook" and "a reusable profiling artifact".

## Dashboard / DB Integration

The intended downstream path is:

1. run profile
2. emit structured artifacts
3. attach artifacts to experiment / trial
4. surface the summary and tables in dashboard

This skill should feed:

- experiment detail page profiling section
- trial-level artifact list
- trajectory context for optimization retries
- future data-flywheel / SFT signals

See [references/dashboard-integration.md](references/dashboard-integration.md).

## MI355X / gfx950 Specific Guidance

On our node, prefer profiling plans that stay grounded in actual machine facts:

- 8x MI355X
- gfx950
- ROCm 7.2
- explicit GPU ID allocation from the experiment
- exact Docker image tag, not just "ROCm 7.2"

When you compare profiles across runs, never compare them without also checking:

- `base_image`
- `resource_class`
- `gpu_device_ids`
- `server_flags`
- `benchmark_config_hash`

Otherwise the comparison is not trustworthy.

## Suggested Rollout

### Phase A: Analysis adaptation

Make the kernel classification and overlap heuristics ROCm-aware.

### Phase B: Artifactization

Write the profiling outputs into stable JSON + Markdown artifacts.

### Phase C: Live integration

Trigger profiling from real optimization stages and surface the artifacts in dashboard.

## Relationship to Other AMD Skills

- **rocprofv3-profiler**
  Use that skill when you need low-level AMD hardware counters or kernel-level bottleneck data.
  Use this skill when you need SGLang/vLLM trace triage tied back to Python/operator semantics.

- **env-probe**
  Run env-probe before profiling if you suspect hidden runtime defaults are skewing results.

- **rocm-crash-debug**
  Use crash-debug when the run is failing. Use this skill when the run is healthy enough to
  generate profiling evidence.

## Reviewer Checklist

Before calling this skill "done", verify:

1. It is explicitly MI355X / gfx950 / ROCm-aware
2. It produces structured artifacts, not just console output
3. It carries experiment/trial linkage fields
4. It distinguishes `arch_match` programmatically
5. It can be attached to dashboard and DB without ad-hoc parsing
