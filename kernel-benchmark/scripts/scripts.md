# Scripts Reference

## benchmark.py

Benchmarks a CUDA-C++, CUTLASS, CuTe DSL, or Triton solution against selectable PyTorch eager, torch.compile, and FlashInfer baselines.

The default timing method is KernelBench-style CUDA event timing with explicit trial control and L2 cache thrashing. Use `--timing-method=host_time` only when investigating Python, JIT, runtime, or launch overhead.

```bash
python scripts/benchmark.py <solution.{cu,py}> \
  --implementation=auto \
  --baselines=pytorch-eager \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

CUDA-C++ and CUTLASS `.cu` solutions require a compiled shared object beside the source:

```bash
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu
```

CUTLASS uses the same shared-library ABI:

```bash
python scripts/benchmark.py cutlass_kernel.cu \
  --implementation=cutlass \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

CuTe DSL and Triton `.py` solutions must define `setup(**kwargs)` and `run_kernel(**kwargs)`:

```bash
python scripts/benchmark.py cute_kernel.py \
  --implementation=cute-dsl \
  --ref=ref.py \
  --output-dir=bench_out \
  --M=1024 --N=1024
```

FlashInfer baselines use the same `reference(**kwargs)` function. When `--baselines=flashinfer` is selected, the runner passes `baseline="flashinfer"` in `kwargs`; otherwise no `baseline` kwarg is provided.

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--implementation=<auto/cuda-cpp/cute-dsl/cutlass/triton>` | `auto` | Implementation selection; `.py` defaults to Triton, otherwise CUDA-C++ |
| `--backend=<...>` | `auto` | Compatibility alias for `--implementation` |
| `--ref=<path>` | required | Python reference defining `reference(**kwargs)` |
| `--output-dir=<dir>` | required | Directory for `benchmark.md` |
| `--baselines=<list>` | `pytorch-eager` | Comma-separated baselines: `pytorch-eager`, `torch-compile`, `flashinfer`, `all`, or `none` |
| `--timing-method=<cuda_event/host_time>` | `cuda_event` | Timing backend |
| `--num-warmup=<n>` | `5` | Warmup calls for timing |
| `--num-trials=<n>` | `100` | Measured trials for timing |
| `--discard-first=<n>` | `1` | First measured trials to discard |
| `--prewarm-calls=<n>` | `1` | Untimed calls before correctness/timing to trigger lazy init/JIT |
| `--gpu=<id>` | `0` | CUDA device index |
| `--arch=<sm_XX>` | auto | GPU architecture |
| `--ptr-size=<n>` | auto | Override CUDA pointer buffer element count |
| `--atol`, `--rtol` | `ref.py` module-level if set, else `1e-4` / `1e-3` | Output comparison tolerance. CLI flag takes highest precedence, then ref.py module-level `atol`/`rtol`, then internal defaults. When using FlashInfer or other bfloat16 baselines, set `atol=1e-2, rtol=1e-2` in ref.py so the benchmark picks it up automatically. |
| `--seed=<n>` | `42` | Random seed for generated inputs |
| `--NAME=VALUE` | required as needed | Integer dimension/scalar args |

The script writes `benchmark.md` into the output directory.
