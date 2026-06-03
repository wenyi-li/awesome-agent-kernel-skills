---
name: env
description: Environment readiness check and GPU configuration for cuda-cpp, cute-dsl, cutlass, and triton kernel profiling.
---

# env

## Directory Structure

```
env/
├── ENV.md
└── scripts/
    ├── enc_config.py
    └── env_check.py
```

Environment check and configuration is essential preparation before kernel profiling. **If any required item fails, stop correctness/profiling until the environment is fixed.**

## Scripts

| Script | Responsibility |
|---|---|
| `scripts/env_check.py` | Detect and validate CUDA/PyTorch/Triton/CuTe DSL/CUTLASS/NCU/nsight-python environment, output Markdown report |
| `scripts/enc_config.py` | Lock GPU clocks for more stable profiling measurements |

---

## scripts/env_check.py

### Usage

```bash
python env/scripts/env_check.py -o <output_dir>/env_check.md [--gpu 0]
```

Use `<output_dir>/env_check.md` as the environment baseline for kernel profiling.

The check validates these base required items: `PyTorch`, `CUDA runtime`, selected GPU, `ncu`, and `nsight-python`.

For implementation readiness, at least one of the following must pass:

| Implementation | Readiness requirement |
|---|---|
| `cuda-cpp` | `nvcc` executable |
| `cute-dsl` | importable `cutlass.cute` Python package |
| `cutlass` | `nvcc` executable plus CUTLASS C++ headers |
| `triton` | importable `triton` Python package |

For CUTLASS header detection, set one of `CUTLASS_PATH`, `CUTLASS_ROOT`, or `CUTLASS_HOME` to a CUTLASS source/install root containing `include/cutlass` and `include/cute`.

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-o / --out` | Yes | — | Markdown report output path |
| `--gpu` | No | `0` | GPU device index |

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Environment ready — all required items passed |
| `1` | Environment not ready — one or more required items failed |
| `2` | Parameter error |

---

## scripts/enc_config.py

- Call before kernel optimization to lock the target GPU's SM clocks to maximum frequency, eliminating frequency jitter from performance data.
- If the setting fails, further optimization is also not allowed.

### Usage

```bash
python env/scripts/enc_config.py --gpu [0,1,2...]
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Failure |
