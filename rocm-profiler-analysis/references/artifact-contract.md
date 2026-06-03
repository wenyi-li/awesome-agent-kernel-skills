# Profiling Artifact Contract

This skill is only useful if its outputs can be attached to an experiment and rendered in the
dashboard without custom parsing per run.

## Required Files

- `profile_summary.md`
- `profile_metadata.json`
- `kernel_table.json`
- `overlap_opportunities.json`
- `fuse_opportunities.json`

Optional:

- `perfetto_fixed_trace.json`

## `profile_metadata.json`

Recommended shape:

```json
{
  "experiment_id": "issue-vllm-33123",
  "trial_id": "trial-3",
  "model_name": "Qwen/Qwen3.5-32B",
  "profile_stage": "decode",
  "observed_arch": "gfx950",
  "arch_match": "exact",
  "hardware_relevance_reason": "Collected on the target MI355X node",
  "rocm_version": "7.2.0",
  "base_image": "rocm/vllm-dev:rocm7.2.0_vllm_0.14.0_20260401",
  "resource_class": "multi-gpu",
  "gpu_device_ids": [0, 1, 2, 3],
  "gpu_clocks_mhz": {
    "0": 1785,
    "1": 1785,
    "2": 1785,
    "3": 1785
  },
  "preflight_passed": true,
  "server_flags": {
    "attention_backend": "aiter",
    "tp_size": 4
  },
  "benchmark_config_hash": "sha256:...",
  "source_trace_path": "results/issue-vllm-33123/profile/trace.json.gz"
}
```

## Table Files

The three tables should be valid JSON arrays so the dashboard can render them directly.

### `kernel_table.json`

Each row should contain:

- `stage`
- `kernel_name`
- `category`
- `time_share_pct`
- `python_location`
- `notes`

### `overlap_opportunities.json`

Each row should contain:

- `stage`
- `priority`
- `kernel_name`
- `python_scope`
- `recommendation`
- `dependency_risk`

### `fuse_opportunities.json`

Each row should contain:

- `stage`
- `pattern`
- `time_share_pct`
- `current_location`
- `candidate_fused_path`

## Why This Contract Matters

If we keep profiling outputs machine-readable:

- dashboard can render them without custom per-run logic
- experiment summaries can reference them consistently
- optimization retries can consume them as structured hints
- later SFT pipelines can use them as aligned signals
