# amdpilot Integration Guide for ROCm Crash Debug Skill

## How Crash Artifacts Flow Through amdpilot

```
Container (trial execution)
  ├── kernel_api.log          ← SGLANG_KERNEL_API_LOGDEST
  ├── crash_dumps/            ← SGLANG_KERNEL_API_DUMP_DIR
  │   ├── inputs.pt
  │   └── metadata.json
  └── crash_diagnostics.json  ← post-trial collection script

        │
        ▼
Orchestrator (main.py)
  ├── Reads crash_diagnostics.json from container
  ├── Extracts last_kernel_boundary → failure_reason
  ├── Writes to scoreboard.jsonl
  └── Calls db.insert_trial(failure_reason=...)

        │
        ▼
Dashboard DB (amdpilot.db)
  ├── trials.failure_reason       ← last kernel boundary + error
  ├── events.detail_json          ← full crash_diagnostics.json
  └── experiments.error_message   ← if experiment-level failure

        │
        ▼
Dashboard UI
  ├── Trial detail panel     ← shows failure_reason
  ├── Trajectory viewer      ← shows events timeline
  └── System info panel      ← shows GPU state at failure time
```

## Container Environment Variable Injection Points

### 1. task.yaml (user-configurable)
```yaml
container:
  env:
    SGLANG_KERNEL_API_LOGLEVEL: "1"
    SGLANG_KERNEL_API_LOGDEST: "/workspace/.amdpilot/kernel_api.log"
```

### 2. ContainerConfig.env (code-level)
In `src/amdpilot/orchestrator/config.py`, the `ContainerConfig.env` dict
is converted to Docker `-e` flags via the `env_flags` property.

### 3. Hardcoded in docker_manager.py
`start_container()` already sets `HIP_VISIBLE_DEVICES`, `KIMI_STATUS_INTERVAL`,
`GPU_COREDUMP_ENABLE`. Add kernel API logging defaults here for always-on level 1.

## DB Schema Fields Used

### trials table
```sql
failure_reason  TEXT   -- "hipErrorAssert in flash_attn_varlen_fwd (shapes: q=[2,32,128], k=[2,32,128])"
status          TEXT   -- "failed", "verification_failed", etc.
```

### events table
```sql
event_type      TEXT   -- "crash_diagnostic"
detail_json     TEXT   -- full crash_diagnostics.json content
```

## Supervisor Escalation Logic

```python
# In supervisor decision loop:
if trial.exit_code != 0:
    # Parse last trial's failure_reason
    if "hipErrorOutOfMemory" in failure_reason or trial.exit_code == 137:
        # OOM: reduce batch size, don't escalate logging
        action = "reduce_batch_size"
    elif trial.exit_code == 139 or "hipErrorIllegalAddress" in failure_reason:
        # Memory corruption: escalate to level 10
        next_env["SGLANG_KERNEL_API_LOGLEVEL"] = "10"
        next_env["SGLANG_KERNEL_API_DUMP_DIR"] = "/workspace/.amdpilot/crash_dumps"
        action = "retry_with_crash_dumps"
    elif metric == 0.0 and "Kernel API Call:" not in kernel_log:
        # Test harness not reaching kernel path at all
        action = "fix_test_harness"
    else:
        # General failure: escalate to level 3
        next_env["SGLANG_KERNEL_API_LOGLEVEL"] = "3"
        action = "retry_with_expanded_logging"
```

## Data Flywheel Value

Crash artifacts are high-value SFT training data:
- **Input**: crash_diagnostics.json + kernel_api.log + failure context
- **Output**: the fix that resolved the crash (from agent trajectory)
- **Label**: metric improvement from 0.00 to >0.00

Store crash dumps in `amdpilot-logs/` for training data collection.
The trajectory viewer shows the full agent reasoning chain from
crash observation to fix, which is the core SFT signal.
