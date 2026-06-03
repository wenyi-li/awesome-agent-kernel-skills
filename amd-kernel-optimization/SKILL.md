---
name: amd-kernel-optimization
description: >
  Optimize inference latency and throughput of PyTorch models on AMD GPUs (MI250/MI300/MI350) with ROCm.
  Use when profiling and optimizing GEMM, attention, elementwise ops, torch.compile, CUDAGraphs,
  or Triton kernels on AMD hardware. Covers the full optimize cycle: benchmark → profile → analyze →
  implement → verify. Also covers benchmarking methodology and common pitfalls that waste time.
---

# AMD Kernel Optimization (ROCm)

## 3 Rules (read first)

1. **torch.compile FIRST.** `torch.compile(mode="default")` with correct inductor config gives 2-5x speedup. Get this working before any manual optimization. Any code change that breaks compile is a net regression.

2. **Profile before optimizing.** Never guess where time is spent. Run `torch.profiler`, classify GPU time into GEMM / attention / elementwise / launch overhead, then optimize the largest category.

3. **Measure after every change.** Benchmark with proper warmup and iterations (see below). Revert if performance regresses.

## Benchmarking (do this correctly or waste hours)

**NEVER reduce warmup/iterations to "save time" — you get garbage numbers.**

- **Minimum**: 3 warmup runs, 10 measurement iterations. Use the benchmark script's defaults.
- **Use GPU timing**, not wall-clock:
  ```python
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record(); result = model(input); end.record()
  torch.cuda.synchronize()
  ms = start.elapsed_time(end)
  ```
- **Report mean AND std.** If std > 10% of mean, something is wrong (graph breaks, recompilation).
- **First-run penalty is NORMAL on AMD** — torch.compile takes 2-15 min on first run. Set timeout ≥ 600s. Do NOT conclude "compile doesn't work" or kill jobs under 15 min.
- **Never report first-run latency as baseline.** Always use mean of post-warmup iterations.

## Optimization Ladder

Each level builds on the previous. **Do NOT skip to Level 3+ without Level 2.**

### Level 1: Environment (no code changes)
- Set env vars: `GPU_MAX_HW_QUEUES=2`, `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `AMD_LOG_LEVEL=0`
- Disable NUMA balancing: `sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'`
- Enable GEMM tuning: `PYTORCH_TUNABLEOP_ENABLED=1`, `TORCH_BLAS_PREFER_HIPBLASLT=1`
- Set `torch.set_float32_matmul_precision('high')`
- Audit env vars: `env | grep -iE 'TORCH|INDUCTOR|AUTOTUNE'` — unset `TORCHINDUCTOR_MAX_AUTOTUNE` if present

### Level 2: torch.compile (MANDATORY — do this first)
- Apply inductor config, compile with `mode="default"`. Details: [references/torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)
- Fix ALL graph breaks before Level 3: `TORCH_LOGS="graph_breaks" python3 ...`
- If kernel launch overhead is high after compile, try manual CUDAGraph capture (see reference)
- This alone typically gives 2-5x speedup

### Level 3: Model surgery (must preserve compile compatibility)
- **Profile first** → read [references/benchmarking-and-profiling.md](references/benchmarking-and-profiling.md)
- Fuse QKV projections (3 GEMMs → 1), Gate+Up projections (2 GEMMs → 1) → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- Replace manual attention with SDPA or aiter flash attention → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- Look for compute-reduction: skip masked/padding inputs, avoid `repeat_kv`, cache unchanged outputs
- **Test**: `TORCH_LOGS="graph_breaks"` — verify no new breaks after each change

### Level 4: Kernel & inductor tuning
- Write Triton kernels for elementwise fusions (RMSNorm, SiLU+Mul, Add+RMSNorm) → [references/triton-on-rocm.md](references/triton-on-rocm.md)
- Route fused GEMMs through aiter tuned GEMM with M-threshold gating → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- Inductor tuning flags: `coordinate_descent_tuning`, `benchmark_kernel`, `freezing` (increase compile time, improve steady-state)

### Level 5: Architecture-specific kernels
- aiter kernels for attention/GEMM — use `torch.ops.aiter.*` (compile-safe) not Python wrappers
- Weight preshuffling for asm paths (benchmark: may help or hurt per shape)
- If custom kernel breaks compile, wrap with `@torch.compiler.disable` as last resort

## AMD-Specific Alternatives Quick Reference

### GEMM / Linear
| Option | Notes |
|---|---|
| rocBLAS (default) | Vendor BLAS; generally well-tuned |
| hipBLASLt | Fused epilogues; may beat rocBLAS for some shapes |
| aiter tuned GEMM | Auto-dispatches best kernel per (M,N,K) from tuned configs |
| FP8 GEMM (MI300+) | `gemm_a8w8` via aiter; gfx942=`e4m3fnuz`, gfx950=`e4m3fn` |

### Attention
| Option | Notes |
|---|---|
| aiter flash attention | `torch.ops.aiter.mha_fwd.default(...)` — compile-friendly, GQA native |
| SDPA | `F.scaled_dot_product_attention(...)` — good for KV-cache decode |
| Manual bmm+softmax+bmm | Slowest; replace with SDPA |

### Compilation & Graphs
| Option | Notes |
|---|---|
| `torch.compile(mode="default")` | **Start here.** Stable on ROCm with correct inductor config |
| Manual CUDAGraph capture | Wrap full inference in one graph; needs Dynamo RNG patch |
| `reduce-overhead` / `max-autotune` | **Avoid on ROCm** unless you have verified stability |

## Common Pitfalls

- **Optimizing without profiling**: Classify kernels by category (GEMM/attention/elementwise/other) and compute percentages. "Top 10 kernels" is not enough.
- **Skipping torch.compile**: Manual fusion that saves 10% is worthless if you're missing the 3x from compile. Get compile working first.
- **Giving up on first failure**: When a technique causes regression, diagnose and adjust (e.g., M-threshold gating for aiter GEMM), don't abandon it entirely.
- **Treating blockers as dead ends**: "CUDAGraph doesn't support control flow" means "refactor the control flow." "Requires editing HuggingFace modeling file" means "go edit the modeling file" — that IS the work.
- **Not modifying inner model layers**: The hottest code is in attention and MLP modules, often in third-party libraries. Locate them: `python -c "import transformers; print(transformers.__file__)"` and edit directly.
- **Testing only in isolation**: Optimizations compose. A technique showing 0% alone may enable others. Build incrementally — each new technique applied ON TOP of all previous ones.
- **Reducing benchmark parameters**: Setting `WARMUP=0 ITERATIONS=1` gives meaningless numbers. Optimize the code, not the test.
- **aiter tuned GEMM with no tuned configs**: The default config CSV ships empty. Without it, `gemm_a16w16` silently falls back to plain `F.linear` — no error, no crash, no benefit. Diagnose with `AITER_LOG_TUNED_CONFIG=1`; if you see "using torch solution:0", generate configs via `AITER_TUNE_GEMM=1` + aiter's `GemmTuner`, or fall back to `PYTORCH_TUNABLEOP_ENABLED=1`. See [references/gemm-and-linear.md](references/gemm-and-linear.md) for the full workflow.

## Reference Files

Read as needed for implementation details:

- **[benchmarking-and-profiling.md](references/benchmarking-and-profiling.md)** — Proper measurement, GPU timing, profile interpretation, what to look for in traces
- **[torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)** — Inductor config, graph breaks and fixes, manual CUDAGraph capture, Dynamo RNG patch, env var audit
- **[gemm-and-linear.md](references/gemm-and-linear.md)** — GEMM backend APIs, aiter tuned GEMM, projection fusion, attention backends, weight preshuffling
- **[triton-on-rocm.md](references/triton-on-rocm.md)** — ROCm Triton gotchas, kernel templates for RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm
