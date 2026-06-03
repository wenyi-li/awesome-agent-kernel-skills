---
name: auto-benchmark-rocm
description: >
  Run AI-driven benchmark searches on AMD ROCm with tiered server-flag sweeps for vLLM/SGLang,
  canonical dataset preparation, SLA or fixed-QPS benchmarking, CSV export, and resume.
  Adapted from SGLang auto-benchmark for MI355X (gfx950) / MI300X (gfx942) on ROCm 7.x.
  Use when the user wants an automated benchmark workflow on AMD GPUs rather than a one-off
  bench_serving command. Integrates with amdpilot executor task_spec_json for resource-aware launch.
---

# Auto Benchmark — AMD ROCm / MI355X

This skill is for repeatable, AI-driven performance tuning of vLLM and SGLang on AMD Instinct GPUs.

Adapted from the upstream SGLang `sglang-auto-benchmark` skill. All CUDA-specific references have
been replaced with ROCm equivalents; attention backends, environment variables, resource classes,
and Docker image selection reflect the AMD MI355X (gfx950) and MI300X (gfx942) ecosystem.

## Preconditions

- **GPU architecture confirmed**: run `rocminfo | grep gfx` to verify gfx950 (MI355X) or gfx942 (MI300X).
- **ROCm version checked**: `cat /opt/rocm/.info/version` — this skill targets ROCm 7.x (7.0 or 7.2).
- **Docker image identified**: know which base image you are in (`rocm/sgl-dev` or `rocm/vllm-dev`).
  Use the `env-probe` skill first if unsure.
- **Server can launch**: vLLM or SGLang can already start and serve the target model in this container.
- **Model path exists**: local path under `/data/` preferred over NFS `/mnt/dcgpuval/` for I/O speed.
- **Goal is clear**:
  - benchmark a fixed QPS list, or
  - search the maximum QPS that satisfies `max_ttft_ms` / `max_tpot_ms`.

If any precondition is not met, fix it before running a large search.

## Most Important Rule

If the user wants the best command for a **real production or real workload scenario**, the benchmark
must use **their real request distribution** — real prompt lengths, output lengths, multi-turn patterns,
and sampling settings. `sharegpt`, `random`, and `generated-shared-prefix` are useful for sanity
checks, but they are not a substitute for real traffic.

## AMD-Specific Attention Backends

On ROCm, the available attention backends differ from NVIDIA:

| Backend | Description | Notes |
|---------|-------------|-------|
| `aiter` | ROCm-native unified attention kernel library | **Primary AMD backend**, best for MI355X/MI300X |
| `triton` | Cross-platform Triton kernels | Works on ROCm, good fallback |
| `torch_native` | PyTorch native SDPA | Baseline, no AMD-specific optimization |

**Do NOT use** NVIDIA-specific backends: `fa3`, `fa4`, `flashinfer`, `flashmla`, `trtllm_*`, `cutlass_*`.
These will fail or silently produce wrong results on ROCm.

### aiter-Specific Constraints

- MLA (Multi-Head Latent Attention): aiter ASM kernels require `heads_per_gpu % 16 == 0`.
  For models with 64 heads, TP must be ≤ 4 (giving 16 heads/GPU).
- FP8 prefill attention on gfx950: enable with `SGLANG_AITER_FP8_PREFILL_ATTN=1`.
- MLA persist design for FP8 KV cache: enable with `SGLANG_AITER_MLA_PERSIST=1`.

## Environment Variables

Replace CUDA environment variables with ROCm equivalents:

| CUDA (do not use) | ROCm equivalent | Purpose |
|--------------------|-----------------|---------|
| `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` | GPU device selection |
| `CUDA_LAUNCH_BLOCKING` | `HIP_LAUNCH_BLOCKING` | Synchronous kernel launch for debugging |
| — | `SGLANG_USE_AITER=1` | Explicitly enable aiter backend |
| — | `SGLANG_AITER_MLA_PERSIST=1` | Enable MLA persist design |
| — | `SGLANG_AITER_FP8_PREFILL_ATTN=1` | FP8 prefill on gfx950 |
| — | `HSA_FORCE_FINE_GRAIN_PCIE=1` | Fine-grain PCIe for host-device transfers |

## GPU Topology and Resource Classes

Our node has 8× AMD Instinct MI355X (gfx950), ROCm 7.2.0.

| Resource class | GPU count | Typical use case |
|----------------|-----------|------------------|
| `single-gpu` | 1 | Small models (≤8B), quick sanity checks |
| `multi-gpu` | 4 | Medium models (32B–70B), TP=4 |
| `full-node` | 8 | Large models (70B+), TP=8 or TP=4×DP=2 |

When `server.parallel` is used and `dp_size` is not set explicitly:

```
dp_size = visible_gpus / (tp_size * pp_size)
```

Visible GPU count is inferred from `HIP_VISIBLE_DEVICES`, or from `server.parallel.gpu_count`.

## Supported Dataset Kinds

Same as upstream:

- `sharegpt` — auto-download supported, converted to canonical JSONL.
- `custom` — old `bench_serving` format or canonical autobench JSONL.
- `random` — synthetic/random benchmark path.
- `generated-shared-prefix` — shared-prefix synthetic generator.

## Canonical Dataset Format

Identical to upstream. JSONL, one request per line:

```json
{"prompt": "Write a summary.", "output_len": 256}
{"prompt": [{"role": "user", "content": "Summarize."}], "output_len": 256}
```

## Search Tiers

- **Tier 1**: Fast sanity sweep. Baseline + small one-at-a-time scan.
- **Tier 2**: Good default. Small cartesian on high-priority keys + expansion for rest.
- **Tier 3**: Full cartesian product. Slowest, but thorough when space is bounded.

YAML key order matters. Put the most important search keys first.

## What Is Tunable (AMD/ROCm)

`server.base_flags` and `server.search_space` are passed to the server launcher. Any valid
vLLM/SGLang server CLI flag can be set or searched.

### Kernel / Backend (AMD-specific)

- `attention_backend` — search `[aiter, triton]`
- `prefill_attention_backend` — if split prefill/decode is supported
- `decode_attention_backend` — if split prefill/decode is supported
- `sampling_backend`

### Batching / Scheduling

- `max_running_requests`
- `chunked_prefill_size` — common values: `[4096, 8192, 16384, 131072]`
- `prefill_max_requests`
- `max_prefill_tokens`
- `schedule_policy` — `[lpm, fcfs]`
- `schedule_conservativeness`
- `num_continuous_decode_steps`

### Memory / Cache

- `mem_fraction_static` — critical for ROCm; MI355X HBM3e is larger than H100, so ranges differ.
  Typical search: `[0.80, 0.85, 0.88, 0.90]`
- `max_total_tokens`
- `page_size`
- `disable_radix_cache`
- `kv_cache_dtype` — `[auto, fp8_e4m3]` for FP8 KV cache on MI355X

### Parallel / Distributed

- `tp_size` — must respect aiter head constraints (heads_per_gpu % 16 == 0)
- `pp_size`
- `dp_size`
- `load_balance_method`
- `enable_dp_attention`
- `enable_aiter_allreduce_fusion` — AMD-specific distributed optimization

### Runtime / HIP Graph

- Keep HIP graph enabled by default for performance benchmarking (same concept as CUDA graph on ROCm).
- `cuda_graph_max_bs` — flag name is unchanged in vLLM/SGLang even on ROCm
- `disable_cuda_graph_padding`
- Do not put `disable_cuda_graph` into the default search space.

### Optional Speculative / EAGLE Stage

Speculative decoding support on ROCm may be limited. Verify availability before enabling:

- `speculative_num_steps`
- `speculative_eagle_topk`
- `speculative_num_draft_tokens`

**Order**: always tune the non-speculative base server first, then optionally add speculative search.

## Base Tuning Before EAGLE

Never start by tuning EAGLE first. Use this order:

1. Tune the non-speculative base server first.
2. Find the best normal config for the target dataset and SLA.
3. Only if the user explicitly asks and draft model assets exist, run speculative search.

## Running The Workflow

### Prepare dataset

```bash
python3 -m sglang.auto_benchmark convert \
  --kind sharegpt \
  --tokenizer /data/meta-llama/Meta-Llama-3.1-70B-Instruct \
  --num-prompts 1200 \
  --output /tmp/sharegpt.autobench.jsonl
```

### Run from config

```bash
python3 -m sglang.auto_benchmark run --config /path/to/config.yaml
```

### Outputs

- Prepared canonical dataset JSONL
- Per-run `results.jsonl`
- Summary `results.csv`
- Per-candidate server logs

## Integration with amdpilot

When running benchmarks through the amdpilot executor, the benchmark task spec maps to
`task_spec_json` in the queue DB:

```json
{
  "gpu_count": 8,
  "resource_class": "full-node",
  "base_image": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317",
  "gpu_free_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "disk_free_gb": 2100,
  "timeout_minutes": 120
}
```

The dashboard `GET /api/{job_name}/system_info` endpoint reads from this column to display
experiment runtime info (GPU arch, ROCm version, base image, resource class).

Benchmark results should be written as structured artifacts so the dashboard can render them
and feed them into the data flywheel for downstream LoRA/SFT training signal.

## Config Template

Use the reference configs in `references/`:

| Config | Model | GPUs | Notes |
|--------|-------|------|-------|
| `config-example-rocm.yaml` | Generic | 4 | Starting template |
| `llama3.1-70b-mi355x.yaml` | Llama 3.1 70B Instruct | 8 | Full-node TP=8 |
| `qwen3-32b-mi355x.yaml` | Qwen3 32B | 4 | TP=4, aiter search |
| `deepseek-r1-mi355x.yaml` | DeepSeek R1 671B (FP8/MXFP4) | 8 | FP8 KV cache, AllReduce fusion |

## What To Report Back

After a run, summarize:

- **Hardware**: GPU arch (gfx950/gfx942), ROCm version, Docker image tag
- **Search config**: which tier, dataset kind (synthetic vs real)
- **Best config found**: attention backend, TP/DP, key flags
- **Best QPS** that satisfied SLA (or fixed QPS results)
- **Whether speculative tuning was skipped or run**
- **Paths to artifacts**: dataset JSONL, `results.jsonl`, `results.csv`, server logs
- **Anomalies**: OOM events, HIP graph capture failures, aiter constraint violations

## Differences From Upstream (CUDA) Skill

| Aspect | CUDA / NVIDIA | ROCm / AMD |
|--------|---------------|------------|
| GPU visibility | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` |
| Attention backends | `fa3`, `flashinfer` | `aiter`, `triton` |
| Graph runtime | CUDA graph | HIP graph (**flag names unchanged** — `cuda_graph_max_bs`, `disable_cuda_graph` etc. still use "cuda" prefix in vLLM/SGLang CLI even on ROCm) |
| GPU query | `nvidia-smi` | `rocm-smi --showproductname` |
| Arch detection | N/A | `rocminfo \| grep gfx` |
| AllReduce fusion | N/A | `--enable-aiter-allreduce-fusion` |
| FP8 prefill | Built-in | `SGLANG_AITER_FP8_PREFILL_ATTN=1` |
| MLA persist | Built-in | `SGLANG_AITER_MLA_PERSIST=1` |
| Memory range | 0.85–0.92 typical | 0.80–0.90 typical (HBM3e larger) |
| Docker images | `nvcr.io/nvidia/*` | `rocm/sgl-dev`, `rocm/vllm-dev` |
