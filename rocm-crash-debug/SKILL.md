---
name: rocm-crash-debug
description: >
  Debug ROCm/HIP kernel crashes in SGLang and vLLM on AMD GPUs (MI300X/MI325X/MI355X).
  Adapts SGLang's @debug_kernel_api kernel boundary logging to ROCm: captures input tensors
  before crash, tracks shapes/dtypes/values, dumps crash artifacts for offline analysis.
  Integrates with amdpilot executor failure_reason field and dashboard trajectory viewer.
  Triggered by: CUDA/HIP errors, illegal memory access, device-side assert, OOM kills,
  signal 137/139, NaN/Inf in outputs, "debug crash", "why did the trial fail".
---

# ROCm/HIP Crash Debug Skill for AMD GPUs

**Use this when an amdpilot trial fails with a GPU error, OOM kill, or produces wrong results.**
AMD ROCm crashes often leave minimal diagnostic information. This skill captures structured
crash evidence at kernel boundaries so failures become reproducible offline samples.

## Why This Exists

Problem: ROCm/HIP errors on MI300X/MI325X/MI355X crash the process before normal debug output
is flushed. A trial that exits with code 137 (OOM kill) or produces 0.00 metric tells you
nothing about what went wrong. The agent can't learn from a failure it can't observe.

Solution: Instrument kernel call boundaries with pre-execution logging. When a crash occurs,
the last logged boundary tells you exactly which op failed and what inputs caused it. Combined
with tensor dumps, a crash becomes an offline-analyzable sample.

This approach is adapted from SGLang's `@debug_kernel_api` decorator (PR #20910), which was
inspired by FlashInfer's API logging. The concepts are backend-agnostic; this skill specializes
them for ROCm/HIP and integrates with amdpilot's trial/experiment infrastructure.

## Step 1: Enable Kernel API Logging in Container

Set environment variables when launching the Docker container. These work with SGLang's
existing `kernel_api_logging.py` infrastructure on ROCm builds.

### Level 1 — Function Names Only (minimal overhead)
```bash
SGLANG_KERNEL_API_LOGLEVEL=1
SGLANG_KERNEL_API_LOGDEST=stderr
```
Use this as the default for all amdpilot trials. Overhead is negligible and it captures
the last kernel boundary before any crash.

### Level 3 — Input Metadata (shapes, dtypes, devices)
```bash
SGLANG_KERNEL_API_LOGLEVEL=3
SGLANG_KERNEL_API_LOGDEST=/workspace/.amdpilot/kernel_api.log
```
Use this when a trial has failed and you need to understand what data caused the crash.
Shows tensor shapes, dtypes, device placement, and contiguity.

### Level 5 — Tensor Statistics (min/max/mean/NaN/Inf counts)
```bash
SGLANG_KERNEL_API_LOGLEVEL=5
SGLANG_KERNEL_API_LOGDEST=/workspace/.amdpilot/kernel_api.log
```
Use this for numerical correctness issues (NaN/Inf in outputs, wrong results). Adds
statistical summaries of every input tensor at each kernel boundary.

### Level 10 — Full Crash Dumps (inputs.pt + metadata.json)
```bash
SGLANG_KERNEL_API_LOGLEVEL=10
SGLANG_KERNEL_API_DUMP_DIR=/workspace/.amdpilot/crash_dumps
SGLANG_KERNEL_API_LOGDEST=/workspace/.amdpilot/kernel_api.log
```
Use this for hard-to-reproduce crashes. Saves complete input tensors and metadata before
each kernel call. When the process crashes, the last dump directory contains a reproducible
snapshot.

**Note on HIP Graphs**: When `HIP_GRAPH_CAPTURE` is active, level-10 tensor dumps are
automatically skipped (same as CUDA Graph behavior), but boundary logging at levels 1-5
continues. This prevents dump operations from corrupting graph capture.

## Step 2: Inject via amdpilot Container Config

In `task.yaml`, add the logging environment variables:

```yaml
container:
  env:
    SGLANG_KERNEL_API_LOGLEVEL: "3"
    SGLANG_KERNEL_API_LOGDEST: "/workspace/.amdpilot/kernel_api.log"
```

For the orchestrator's Docker container startup, these flow through `ContainerConfig.env`
to `docker run -e` flags. No code changes needed — just config.

For amdpilot executor integration, the recommended approach:
- **Default**: Level 1 on all trials (always capture last boundary)
- **After first failure**: Supervisor escalates to level 3+ for retry trials
- **Known crash patterns**: Level 10 for targeted dump collection

## Step 3: ROCm-Specific Diagnostic Commands

When a trial fails, collect these AMD-specific diagnostics from inside or outside the container:

### GPU State (run from host after container exits)
```bash
# GPU memory and utilization at time of failure
rocm-smi --showuse --showmeminfo vram

# Check for GPU hangs or resets
dmesg | grep -i "amdgpu\|drm\|gpu" | tail -20

# Check for Xnack / page fault issues (MI300X/MI355X specific)
dmesg | grep -i "retry fault\|xnack" | tail -10
```

### Container Exit Analysis
```bash
# Get exit code and OOM details
docker inspect --format='{{.State.ExitCode}} {{.State.OOMKilled}}' <container>

# Get last N lines of container stderr
docker logs --tail 50 <container> 2>&1
```

### ROCm Error Codes (common on MI300X/MI325X/MI355X)
| Error | Meaning | What to Check |
|-------|---------|---------------|
| `hipErrorIllegalAddress` | Out-of-bounds GPU memory access | Tensor shapes, index bounds, contiguity |
| `hipErrorAssert` | Device-side assert triggered | Input validation in kernel, index range |
| `hipErrorOutOfMemory` | GPU VRAM exhaustion | Batch size, model size, KV cache config |
| `hipErrorLaunchFailure` | Kernel launch failed | Shared memory size, block dimensions |
| `hipErrorNoBinaryForGpu` | No binary for target GPU arch | Check gfx target matches (gfx942 vs gfx950) |
| Signal 137 | OOM killed by host kernel | Container memory limit, host swap pressure |
| Signal 139 | Segmentation fault | Usually host-side pointer corruption |

### rocgdb Quick Attach (for interactive debug)
```bash
# Inside container — attach to running process
rocgdb -p <pid>

# Or launch with rocgdb
rocgdb --args python your_script.py
(gdb) catch throw
(gdb) run
```

## Step 4: Multi-GPU / Multi-Process Debugging

MI355X nodes typically run 8 GPUs. Use per-process log files:

```bash
SGLANG_KERNEL_API_LOGDEST="/workspace/.amdpilot/kernel_api_rank%i.log"
SGLANG_KERNEL_API_DUMP_DIR="/workspace/.amdpilot/crash_dumps/rank%i"
```

The `%i` placeholder is replaced by the process rank. This prevents log interleaving
across TP/DP workers.

For RCCL (ROCm collective communication) hangs:
```bash
# Enable RCCL debug logging
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,COLL

# Set timeout for collective operations
NCCL_TIMEOUT=300
```

## Step 5: Filter Dumps to Reduce Disk Usage

Level-10 dumps can consume significant disk space. Filter to specific ops:

```bash
# Only dump attention-related ops
SGLANG_KERNEL_API_DUMP_INCLUDE="*attention*,*flash*,*sdpa*"

# Exclude high-frequency trivial ops
SGLANG_KERNEL_API_DUMP_EXCLUDE="*elementwise*,*copy*,*fill*"
```

## Step 6: Common ROCm Crash Patterns on MI300X/MI355X

### Pattern 1: gfx950 Binary Mismatch
**Symptom**: `hipErrorNoBinaryForGpu` or silent wrong results
**Cause**: Container built for gfx942 (MI300X) but running on gfx950 (MI355X)
**Fix**: Verify Docker image target matches hardware
```bash
# Check GPU architecture
rocminfo | grep "Name:" | head -1
# Should show: gfx950 for MI355X, gfx942 for MI300X/MI325X
```

### Pattern 2: VRAM Fragmentation OOM
**Symptom**: Signal 137 or `hipErrorOutOfMemory` despite sufficient total VRAM
**Cause**: HBM3e memory fragmentation after multiple allocation/free cycles
**Fix**: Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`

### Pattern 3: Flash Attention FP8 Crash
**Symptom**: Crash in attention kernel with FP8 quantized models
**Cause**: FP8 flash attention path not supported on all ROCm versions
**Fix**: Check `env-probe` skill output for FP8 flash attn availability; fall back to
non-FP8 attention if flagged

### Pattern 4: Triton Kernel Compilation Failure
**Symptom**: Hang during first forward pass, or `ModuleNotFoundError` in Triton
**Cause**: Triton cache corruption or version mismatch
**Fix**: Clear Triton cache: `rm -rf ~/.triton/cache`

### Pattern 5: RCCL Collective Timeout
**Symptom**: Process hangs on `all_reduce` or `all_gather`, eventually killed
**Cause**: GPU-to-GPU communication failure, often RDMA fabric issue
**Fix**: Check `rocm-smi --showtopo` for link health; set `NCCL_TIMEOUT`

## Step 7: Integration with amdpilot Dashboard

Crash artifacts flow into the dashboard through existing fields:

| Artifact | DB Field | Dashboard View |
|----------|----------|----------------|
| Last kernel boundary | `trials.failure_reason` | Trial detail panel |
| Input shapes/dtypes | `trials.failure_reason` (structured) | Trial detail panel |
| Full crash dump path | `events.detail_json` | Trajectory viewer |
| GPU state at crash | `events.detail_json` | System info panel |
| Kernel API log file | Agent output (`trial_N.txt`) | Agent log viewer |

### Post-Trial Crash Collection Script

After a trial fails, run this inside the container to collect structured diagnostics:

```bash
#!/bin/bash
# collect_crash_diagnostics.sh
OUT="/workspace/.amdpilot/crash_diagnostics.json"

python3 -c "
import json, os, glob

diag = {
    'kernel_api_log': '',
    'last_kernel_boundary': '',
    'crash_dumps': [],
    'gpu_arch': '',
    'exit_code': ${EXIT_CODE:-0},
}

# Read kernel API log
log_path = '/workspace/.amdpilot/kernel_api.log'
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    diag['kernel_api_log'] = ''.join(lines[-50:])
    # Extract last kernel boundary
    for line in reversed(lines):
        if 'Kernel API Call:' in line:
            diag['last_kernel_boundary'] = line.strip()
            break

# List crash dumps
dump_dir = '/workspace/.amdpilot/crash_dumps'
if os.path.isdir(dump_dir):
    diag['crash_dumps'] = glob.glob(f'{dump_dir}/**/metadata.json', recursive=True)[-5:]

# GPU arch
import subprocess
try:
    out = subprocess.check_output(['rocminfo'], text=True, timeout=5)
    for line in out.splitlines():
        if 'Name:' in line and 'gfx' in line:
            diag['gpu_arch'] = line.strip().split()[-1]
            break
except Exception:
    pass

print(json.dumps(diag, indent=2))
" > "$OUT"
```

The orchestrator can read this file post-trial and inject into `failure_reason` and
`events.detail_json` for dashboard display.

## Step 8: Escalation Flow for amdpilot Supervisor

Recommended supervisor behavior when a trial fails:

1. **First failure (any exit code != 0)**:
   - Read kernel API log (level 1 always active)
   - Extract last kernel boundary → store in `failure_reason`
   - If signal 137: check container OOM, reduce batch size on retry
   - If signal 139: escalate to level 10, retry with crash dumps enabled

2. **Second failure (same error pattern)**:
   - Escalate to level 3 or 5
   - Retry with expanded logging
   - Collect full container diagnostics post-failure

3. **Third failure (still stuck)**:
   - Enable level 10 with targeted dump filters
   - Collect crash_diagnostics.json
   - Log structured event for data flywheel collection

This escalation flow produces progressively richer crash evidence without paying
the full level-10 cost on every trial.

## Integration with Other Skills

- **env-probe**: Run env-probe first to detect known ROCm gotchas (inductor defaults,
  hipBLASLt bugs) before they cause crashes
- **rocprofv3-profiler**: Use profiler skill for performance issues; use this skill
  for correctness/crash issues
- **amd-rocm-porting**: When porting CUDA code, enable level 3+ to catch HIP translation
  errors at kernel boundaries

## Validation

This skill has been validated against:
- SGLang sglang-20691 run: 3 trials with OOM kill (signal 137) — level 1 logging would
  have identified the last kernel boundary before memory exhaustion
- vllm#33123 run: 9+ trials at 0.00 metric — level 3 logging would show whether the
  test harness is reaching the kernel path at all
- General MI355X container crashes during development: level 10 dumps enabled post-mortem
  analysis of flash attention FP8 path failures

## Adding New Crash Patterns

When you encounter a new ROCm crash pattern:
1. Add the error signature to the "Common ROCm Crash Patterns" section
2. Add a check to `collect_crash_diagnostics.sh`
3. Document what level of logging is needed to diagnose it
4. File the crash dump as a data flywheel sample if the fix is non-trivial
