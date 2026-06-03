# Scripts Reference

## correctness_check.py

Validates kernel output against a Python reference implementation.

### CUDA-C++ / CUTLASS

CUDA requires a compiled shared library next to the `.cu` file. The checker loads the `.so`; it does not compile it.

```bash
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

python scripts/correctness_check.py kernel.cu \
  --implementation=cuda-cpp \
  --ref=ref.py \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

CUTLASS uses the same shared-library ABI:

```bash
python scripts/correctness_check.py cutlass_kernel.cu \
  --implementation=cutlass \
  --ref=ref.py \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

### CuTe DSL / Triton

No `.so` required:

```bash
python scripts/correctness_check.py kernel.py \
  --implementation=triton \
  --ref=ref.py \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--implementation=<auto/cuda-cpp/cute-dsl/cutlass/triton>` | `auto` | Implementation selection; `.py` defaults to Triton, otherwise CUDA-C++ |
| `--gpu=<id>` | `0` | CUDA device index |
| `--arch=<sm_XX>` | auto | GPU architecture for CUDA loading context |
| `--ptr-size=<n>` | auto | Override CUDA pointer buffer element count |
| `--atol`, `--rtol` | `1e-4`, `1e-3` | Output comparison tolerance |
| `--seed=<n>` | `42` | Random seed for generated inputs |
| `--NAME=VALUE` | required as needed | Integer dimension/scalar args from kernel signature |

### Output

`<output-dir>/correctness.md`

---

## ncu_profile.py

Collects Nsight Compute metrics for bottleneck diagnosis. Run only after correctness passes.

### CUDA-C++ / CUTLASS

```bash
python scripts/ncu_profile.py kernel.cu \
  --implementation=cuda-cpp \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

CUTLASS:

```bash
python scripts/ncu_profile.py cutlass_kernel.cu \
  --implementation=cutlass \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

### CuTe DSL / Triton

```bash
python scripts/ncu_profile.py kernel.py \
  --implementation=triton \
  --output-dir=profile_out \
  --M=10240 --N=10240
```

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--implementation=<auto/cuda-cpp/cute-dsl/cutlass/triton>` | `auto` | Implementation selection |
| `--warmup=<n>` | `20` | Warmup iterations before profiling |
| `--timing-warmup=<n>` | `5` | Warmup calls for CUDA event timing |
| `--timing-trials=<n>` | `100` | Measured trials for CUDA event timing |
| `--timing-discard-first=<n>` | `1` | First measured timing trials to discard |
| `--timing-cache-mode=<triton/torch/hot>` | `hot` | Cache behavior during execution-time measurement |
| `--prewarm-calls=<n>` | `1` | Untimed calls before timing to trigger lazy init/JIT |
| `--gpu=<id>` | `0` | CUDA device index |
| `--arch=<sm_XX>` | auto | GPU architecture |
| `--ptr-size=<n>` | auto | Override CUDA pointer buffer element count |
| `--seed=<n>` | `42` | Random seed for generated inputs |
| `--NAME=VALUE` | required as needed | Integer dimension/scalar args |

### Outputs

| File | Purpose |
|---|---|
| `ncu_summary.md` | Key metrics grouped by Speed of Light, memory, compute, occupancy, launch, scheduler, stall, and branch categories |
| `ncu_details.md` | Full metric table with averages, min/max, standard deviation, and stability flags |

The profiler uses `nsight-python` and manages the NCU subprocess internally. Do not wrap it in a manual `ncu` command unless debugging the profiler itself.

The execution time shown in `ncu_summary.md` is measured with CUDA events before NCU collection, outside the profiler injection process. NCU metrics are still collected from a separate annotated solve call after the timing pass.
