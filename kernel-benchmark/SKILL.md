---
name: kernel-benchmark
description: Standalone kernel benchmarking skill for cuda-cpp, cutlass, cute-dsl, and triton implementations. Use when the user wants to compare a custom CUDA/CUTLASS .cu kernel or CuTe DSL/Triton .py kernel against selectable PyTorch eager, torch.compile, or FlashInfer baselines, validate correctness, measure execution time with KernelBench-style CUDA event timing, or generate benchmark.md for kernel optimization results.
---

# kernel-benchmark

Use this skill to benchmark a custom GPU kernel against a PyTorch reference implementation.

## What It Does

`scripts/benchmark.py` compares a solution against selectable baselines:

| Variant | Timing method |
|---|---|
| Custom CUDA-C++/CUTLASS/CuTe DSL/Triton solution | `cuda_event` by default; optional `host_time` |
| PyTorch eager reference | same timing method; default baseline |
| `torch.compile` reference | same timing method; opt in with `--baselines=torch-compile` |
| FlashInfer baseline | same timing method; opt in with `--baselines=flashinfer` |

The benchmark validates correctness against the PyTorch eager reference before timing. Selected baselines are also checked against the eager reference. The report is written to `<output-dir>/benchmark.md`.

Default timing uses KernelBench-style CUDA events with explicit warmup/trial counts and cold-cache L2 thrashing. Use `--timing-method=host_time` only when diagnosing Python/runtime overhead.

## Required Interfaces

### Reference And Baselines

The reference file must define:

```python
def reference(**kwargs):
    ...
```

The function receives the same tensor/scalar kwargs as the solution. It may update output tensors in place or return a tensor, tuple/list of tensors, or dict keyed by output name. Optional scalar and dimension parameters are passed through `kwargs`.

When `--baselines=flashinfer` is selected, the runner calls the same function with `baseline="flashinfer"` in `kwargs`. When no `baseline` kwarg is provided, `reference(**kwargs)` is the PyTorch eager reference.

### CUDA-C++ / CUTLASS Solution

CUDA-C++ and CUTLASS solutions are `.cu` files with a compiled shared library beside them. The script loads the shared library; it does not compile it.

The `.cu` file must expose:

```cpp
extern "C" void solve(...);
```

Compile first:

```bash
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu
```

CUTLASS code must be wrapped by `solve(...)`; compile with the required CUTLASS include/library flags.

### CuTe DSL / Triton Solution

CuTe DSL and Triton solutions are `.py` files and must define:

```python
def setup(**kwargs):
    return {"inputs": {...}, "outputs": ["out"]}

def run_kernel(**kwargs):
    ...
```

`setup()["inputs"]` must contain every tensor/scalar passed to the kernel and reference. `setup()["outputs"]` names the tensors compared for correctness.

## Supported Implementations

| Implementation | Input | Benchmark ABI | Notes |
|---|---|---|---|
| `cuda-cpp` | `.cu` + compiled `.so` | `extern "C" void solve(...)` | Default for non-`.py` files |
| `cutlass` | `.cu` + compiled `.so` | `extern "C" void solve(...)` | CUTLASS code must be wrapped by `solve(...)` |
| `cute-dsl` | `.py` | `setup(**kwargs)` + `run_kernel(**kwargs)` | Pass `--implementation=cute-dsl` explicitly |
| `triton` | `.py` | `setup(**kwargs)` + `run_kernel(**kwargs)` | Default for `.py` files |

`--backend` is kept as a compatibility alias for `--implementation`. Prefer `--implementation` in new commands.

## Usage

CUDA-C++:

```bash
python scripts/benchmark.py kernel.cu \
  --implementation=cuda-cpp \
  --baselines=pytorch-eager \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

CUTLASS:

```bash
python scripts/benchmark.py cutlass_kernel.cu \
  --implementation=cutlass \
  --baselines=pytorch-eager,torch-compile \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

Triton:

```bash
python scripts/benchmark.py kernel.py \
  --implementation=triton \
  --baselines=pytorch-eager,flashinfer \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

CuTe DSL:

```bash
python scripts/benchmark.py cute_kernel.py \
  --implementation=cute-dsl \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

`--implementation=auto` selects Triton for `.py` files and CUDA-C++ otherwise.

## Options

| Option | Default | Meaning |
|---|---:|---|
| `solution_file` | required | `.cu` or `.py` solution file |
| `--ref=<path>` | required | Python reference defining `reference(**kwargs)` |
| `--output-dir=<dir>` | required | Directory for `benchmark.md` |
| `--implementation=<auto/cuda-cpp/cute-dsl/cutlass/triton>` | `auto` | Solution implementation; `--backend` is an alias |
| `--baselines=<list>` | `pytorch-eager` | Comma-separated baselines from `pytorch-eager`, `torch-compile`, `flashinfer`, or `all`/`none` |
| `--timing-method=<cuda_event/host_time>` | `cuda_event` | Timing backend |
| `--num-warmup=<n>` | `5` | Warmup calls for fixed-trial timing |
| `--num-trials=<n>` | `100` | Measured trials for fixed-trial timing |
| `--discard-first=<n>` | `1` | First measured trials to discard for fixed-trial timing |
| `--prewarm-calls=<n>` | `1` | Untimed calls before correctness/timing to trigger lazy init/JIT |
| `--ptr-size=<n>` | `0` | Override CUDA pointer buffer element count |
| `--arch=<sm_XX>` | auto | GPU architecture, for messages and CUDA compile hint |
| `--gpu=<id>` | `0` | CUDA device index |
| `--atol`, `--rtol` | `ref.py` module-level if set, else `1e-4` / `1e-3` | Correctness tolerances. CLI flag takes highest precedence, then `ref.py` module-level `atol`/`rtol`, then internal defaults. Set `atol=1e-2, rtol=1e-2` in `ref.py` for bfloat16 FlashInfer baselines. |
| `--seed=<n>` | `42` | Random seed |
| `--NAME=VALUE` | as needed | Integer dimensions/scalars used by the kernel signature or Triton setup |

Keep dimensions, seed, pointer sizing, GPU, and tolerances fixed when comparing kernel versions.

## Output

| File | Purpose |
|---|---|
| `<output-dir>/benchmark.md` | Correctness status, GPU/arch/dimensions, timing config, mean/median/p20/p80/min/max/std/sample-count timing, and speedups versus selected baselines |

If correctness fails, inspect the output tensors before using the timing result for optimization decisions.

For GPU-kernel latency, use the default `cuda_event`. For end-to-end Python/runtime overhead diagnostics, use `host_time`.
