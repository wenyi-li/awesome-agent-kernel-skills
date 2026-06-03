---
name: rocprofv3-profiler
description: Profile AMD GPU kernels using rocprofv3 and analyze performance bottlenecks. Use when the user wants to profile HIP/ROCm kernels, identify GPU performance issues, analyze hardware counters, or understand why a kernel is slow on AMD GPUs (MI100, MI200, MI300 series). Provides wrapper scripts for rocprofv3 execution and automated parsing of profiler output into structured, agent-friendly JSON with bottleneck classification.
---

# rocprofv3-profiler

Profile AMD GPU applications and identify performance bottlenecks using rocprofv3.

## Quick Start

### 1. Run Profiler

```bash
python3 scripts/rocprof_wrapper.py --mode counters -- ./your_app [args]
```

**Modes:**
- `counters` (default): Collect key performance counters for bottleneck analysis
- `trace`: Collect kernel execution traces (timing only)
- `full`: Collect both counters and traces

**Options:**
- `--output-dir <dir>`: Output directory (default: `./rocprof_output`)
- `--counters <file>`: Custom counter input file (optional)
- `--kernel <name>`: Target specific kernel by name

### 2. Parse Results

```bash
python3 scripts/parse_profile.py <output_dir>
```

Returns structured JSON with:
- Per-kernel metrics summary
- Bottleneck classification (compute/memory/lds/latency bound)
- Optimization hints
- Path to raw data

## Example Workflow

```bash
# Profile application
python3 scripts/rocprof_wrapper.py --mode counters -- ./matrix_multiply 1024

# Parse and analyze
python3 scripts/parse_profile.py ./rocprof_output
```

**Sample output:**
```json
{
  "kernels": [{
    "name": "matmul_kernel",
    "metrics": {
      "duration_ns": 145230,
      "occupancy_pct": 45.2,
      "valu_busy_pct": 78.5,
      "lds_bank_conflict_rate": 0.12,
      "l2_hit_rate": 0.65
    },
    "bottleneck": {
      "type": "memory_bound",
      "confidence": "high",
      "detail": "Low L2 hit rate (65%) with high memory stall cycles"
    }
  }],
  "raw_data_path": "./rocprof_output/pmc_1/counter_collection.csv"
}
```

## Bottleneck Classification

The parser classifies kernels into these categories:

| Bottleneck | Indicators |
|------------|------------|
| **compute_bound** | High VALU/MFMA busy, low memory stalls |
| **memory_bound** | High memory latency, low cache hit rates |
| **lds_bound** | High LDS bank conflicts or LDS instruction stalls |
| **latency_bound** | Low occupancy with high instruction latency |
| **balanced** | No single dominant bottleneck |

## Reference Documentation

- **[hardware_counters.md](references/hardware_counters.md)**: Key AMD GPU counters and their meaning
- **[bottleneck_heuristics.md](references/bottleneck_heuristics.md)**: Detailed bottleneck classification rules

## Direct rocprofv3 Usage

For advanced use cases, invoke rocprofv3 directly:

```bash
# List available counters
rocprofv3 -L

# Trace kernel execution
rocprofv3 --kernel-trace --stats -- ./app

# Collect specific counters
rocprofv3 -i counters.txt -- ./app
```

Counter input file format (`counters.txt`):
```
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM
pmc: TCC_HIT TCC_MISS
```

## Troubleshooting

**"rocprofv3 not found"**: Ensure ROCm is installed and `/opt/rocm/bin` is in PATH.

**"No GPU detected"**: Check `rocm-smi` output and `HSA_VISIBLE_DEVICES` environment variable.

**Multi-pass collection**: If too many counters requested, rocprofv3 replays the kernel. Use fewer counters per `pmc` line.
