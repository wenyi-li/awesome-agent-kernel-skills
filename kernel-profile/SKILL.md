---
name: kernel-profile
description: Standalone kernel profiling skill for cuda-cpp, cute-dsl, cutlass, and triton implementations. Checks CUDA/PyTorch/Triton/CuTe DSL/CUTLASS/NCU/nsight-python readiness, optionally locks GPU clocks, validates correctness, collects Nsight Compute metrics with nsight-python, produces env_check.md, correctness.md, ncu_summary.md and ncu_details.md, and classifies GPU bottlenecks from NCU evidence. Use when the user wants to profile a CUDA/CUTLASS .cu kernel or CuTe DSL/Triton .py kernel, compare against a Python reference, inspect occupancy, memory, compute, scheduler, stall, or branch metrics, or diagnose Memory-Bound, Compute-Bound, Latency-Bound, Occupancy-Bound, or Mixed behavior.
---

# kernel-profile

Use this skill to check environment readiness, validate kernel correctness, and collect Nsight Compute profiling artifacts for bottleneck diagnosis.

## Workflow

1. Run environment readiness check.
   - `env/scripts/env_check.py` writes `env_check.md`.
   - Stop if required items fail.
2. Optionally lock GPU clocks for stable measurement with `env/scripts/enc_config.py`.
3. Prepare a reference implementation.
   - `ref.py` must define `reference(**kwargs)`.
   - Optional module-level `atol` and `rtol` override default tolerances.
4. Choose the implementation mode: `cuda-cpp`, `cute-dsl`, `cutlass`, or `triton`.
5. Run correctness before profiling.
   - `cuda-cpp` and `cutlass` kernels must expose `extern "C" void solve(...)` from a compiled shared library.
   - `cute-dsl` and `triton` modules must define `setup(**kwargs)` and `run_kernel(**kwargs)`.
   - See `scripts/scripts.md` for full usage and options.
6. Collect NCU metrics with `scripts/ncu_profile.py`.
   - Execution time is measured before profiling with CUDA event timing.
   - See `scripts/scripts.md` for full usage and options.
7. Read `ncu_summary.md` first, then consult `reference/NCU.md` for metric interpretation.
8. Classify the bottleneck from measured evidence.

Keep the same dimensions, seed, implementation, GPU, and pointer sizing across versions when comparing kernels.

## Supported Implementations

| Implementation | Input | Profiling ABI | Notes |
|---|---|---|---|
| `cuda-cpp` | `.cu` + compiled `.so` | `extern "C" void solve(...)` | Default for non-`.py` files |
| `cutlass` | `.cu` + compiled `.so` | `extern "C" void solve(...)` | CUTLASS code must be wrapped by `solve(...)`; compile with required CUTLASS include/library flags |
| `cute-dsl` | `.py` | `setup(**kwargs)` + `run_kernel(**kwargs)` | Pass `--implementation=cute-dsl` explicitly |
| `triton` | `.py` | `setup(**kwargs)` + `run_kernel(**kwargs)` | Default for `.py` files |

`--backend` is kept as a compatibility alias for `--implementation`. Prefer `--implementation` in new commands.

## Environment Check

Run before correctness and profiling:

```bash
python env/scripts/env_check.py -o profile_out/env_check.md --gpu 0
```

Base required checks include PyTorch import, CUDA runtime availability, selected GPU, `ncu`, and `nsight-python`.

At least one implementation backend must be ready before profiling:

| Implementation | Readiness requirement |
|---|---|
| `cuda-cpp` | `nvcc` executable |
| `cute-dsl` | importable `cutlass.cute` Python package |
| `cutlass` | `nvcc` executable plus CUTLASS C++ headers |
| `triton` | importable `triton` Python package |

For CUTLASS header detection, set `CUTLASS_PATH`, `CUTLASS_ROOT`, or `CUTLASS_HOME` to the CUTLASS root when it is not in a standard location.

For more stable performance data, lock GPU SM clocks when permitted by the system:

```bash
python env/scripts/enc_config.py --gpu 0
```

If either step fails, fix the environment before collecting NCU profiles. See `env/ENV.md` for the detailed command reference.

## Correctness Check

Validates kernel output against a Python reference. See `scripts/scripts.md` for detailed command examples and the full options table.

Script: `scripts/correctness_check.py`
Output: `<output-dir>/correctness.md`

## NCU Profiling

Collects Nsight Compute metrics. Run only after correctness passes. See `scripts/scripts.md` for detailed command examples and the full options table.

Script: `scripts/ncu_profile.py`
Outputs: `ncu_summary.md`, `ncu_details.md`

## Bottleneck Classification

Use `ncu_summary.md` as the primary evidence:

| Condition | Classification |
|---|---|
| Memory SOL > 60% and much higher than SM SOL | Memory-Bound |
| SM SOL > 60% and much higher than Memory SOL | Compute-Bound |
| Both SM SOL and Memory SOL < 40% | Latency-Bound |
| Achieved occupancy is far below theoretical occupancy with a clear resource limit | Occupancy-Bound |
| No single dominant symptom | Mixed |

Secondary signals:

| Symptom | Likely issue |
|---|---|
| Global Load/Store Efficiency < 100%, Sectors/Request > 1 | Uncoalesced or misaligned memory access |
| L1/L2 hit rate too low | Poor locality or working set too large |
| Shared memory efficiency low or bank conflicts high | Shared memory bank conflicts |
| Issue slot utilization < 50% | Scheduler/compute underutilization |
| Eligible warps per cycle low | Not enough schedulable work, ILP, or occupancy |
| Register spill > 0 | Register pressure causing local memory traffic |
| Stall Long Scoreboard high | Global memory latency |
| Stall Barrier high | Synchronization overhead |
| Branch efficiency < 100% or divergent branches high | Warp divergence |

For complete metric definitions and category-specific interpretation, read `reference/NCU.md` only when needed.
