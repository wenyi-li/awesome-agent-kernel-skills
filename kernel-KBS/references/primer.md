---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# KBS Primer

This is the agent-facing quick topic map. Open it only when the user asks a broad question, provides only an alias/symptom, or you need to find a canonical page id quickly. Retrieval, ranking, filtering, and evidence lookup still go through SQLite:

```bash
python3 scripts/kbs.py query "<terms>" --limit 10
python3 scripts/kbs.py get <doc-id> --follow-sources --include-code
```

Do not treat this file as a full index, and do not recursively scan `store/docs/` as a replacement for SQLite. Current corpus counts come from `python3 scripts/kbs.py stats`.

## Use Pattern

1. Extract 1-3 key entities from the user request: hardware, kernel, repo, language, symptom, or data type.
2. Use the query seeds below with `kbs.py query` to get candidate doc ids.
3. Use `kbs.py get <id> --follow-sources --include-code` to read the body, source links, and artifact metadata.
4. In answers, cite `id`, `path`, and `confidence`; performance numbers must come from structured `performance_claims`.

## Query Seeds

| User question type | Preferred query | Common landing pages |
|---|---|---|
| Blackwell MMA / UMMA / tensor core gen05 | `python3 scripts/kbs.py query tcgen05 --architecture sm100` | `hw-tcgen05-mma`, `migration-wgmma-to-tcgen05`, CUTLASS/vLLM/FlashInfer PR |
| Tensor Memory / accumulator pressure | `python3 scripts/kbs.py query tmem --architecture sm100` | `hw-tmem`, `migration-register-to-tmem`, epilogue / FA4 pages |
| TMA / async bulk copy | `python3 scripts/kbs.py query tma --architecture sm100` | `hw-tma`, pipeline-stage pages, source PRs |
| CLC / persistent scheduling / tail effect | `python3 scripts/kbs.py query clc persistent --architecture sm100` | `hw-clc`, `technique-persistent-kernels`, `pattern-tail-effect` |
| NVFP4 / FP4 / block scale | `python3 scripts/kbs.py query nvfp4 --architecture sm100` | `hw-nvfp4`, `kernel-nvfp4-gemm`, `kernel-nvfp4-gemv` |
| GEMM on B200 | `python3 scripts/kbs.py query --kind wiki-kernel --tag gemm --architecture sm100` | DeepGEMM, FP8 block-scale GEMM, NVFP4 GEMM |
| Attention / MLA / FlashAttention | `python3 scripts/kbs.py query attention --architecture sm100 --limit 10` | `kernel-flash-attention-4`, `kernel-flashmla`, source PRs |
| MoE / grouped GEMM / fused MoE | `python3 scripts/kbs.py query moe --architecture sm100 --limit 10` | `kernel-fused-moe`, `kernel-grouped-gemm`, FlashInfer/vLLM/SGLang PRs |
| Memory-bound symptoms | `python3 scripts/kbs.py query --symptom memory-bound --limit 10` | `pattern-memory-bound`, `kernel-nvfp4-gemv`, vectorized/cache-policy pages |
| Low SM utilization | `python3 scripts/kbs.py query --symptom low-sm-utilization --limit 10` | `pattern-low-sm-utilization`, CLC, persistent kernels, tile scheduling |
| Register pressure | `python3 scripts/kbs.py query --symptom register-pressure --limit 10` | `pattern-register-pressure`, TMEM, register-budgeting |
| Triton on Blackwell | `python3 scripts/kbs.py query triton blackwell --language triton` | `lang-triton`, Triton-related vLLM/SGLang/PyTorch PRs |
| CuTe DSL / CUTLASS | `python3 scripts/kbs.py query cute sm100 --language cute-dsl --repo cutlass` | `lang-cute-dsl`, CUTLASS SM100 PRs |

## Stable Page IDs

These IDs are high-frequency entry points. Use `query` first to confirm current ranking, then use `get` to open the specific page.

| Area | IDs |
|---|---|
| Hardware | `hw-tcgen05-mma`, `hw-tmem`, `hw-tma`, `hw-clc`, `hw-2sm-cooperative`, `hw-nvfp4`, `hw-pdl-gdc`, `hw-mbarrier` |
| Techniques | `technique-warp-specialization`, `technique-persistent-kernels`, `technique-ping-pong-scheduling`, `technique-pipeline-stages`, `technique-epilogue-fusion`, `technique-swizzling`, `technique-vectorized-loads`, `technique-cache-policy`, `technique-register-budgeting`, `technique-fine-grained-quantization` |
| Kernels | `kernel-flash-attention-4`, `kernel-deepgemm`, `kernel-nvfp4-gemm`, `kernel-nvfp4-gemv`, `kernel-fp8-block-scale-gemm`, `kernel-fused-moe`, `kernel-grouped-gemm`, `kernel-flashmla`, `kernel-sparse-mla`, `kernel-nsa`, `kernel-gated-delta-net` |
| Patterns | `pattern-low-sm-utilization`, `pattern-memory-bound`, `pattern-register-pressure`, `pattern-compute-bound`, `pattern-tail-effect`, `pattern-pipeline-stalls`, `pattern-moe-load-imbalance` |
| Languages | `lang-cuda-cpp`, `lang-cute-dsl`, `lang-ptx`, `lang-triton` |
| Migration | `migration-wgmma-to-tcgen05`, `migration-register-to-tmem` |

## Repo Filters

Use repo filters when the user asks "where in X" or wants implementation provenance:

```bash
python3 scripts/kbs.py query tcgen05 --repo cutlass --limit 20
python3 scripts/kbs.py query fp8 moe --repo flashinfer --limit 20
python3 scripts/kbs.py query mla --repo vllm --architecture sm100 --limit 20
python3 scripts/kbs.py query triton --repo sglang --language triton --limit 20
```

Tracked repo names can be fuzzy (`cutlass`, `vllm`, `sglang`, `flashinfer`, `pytorch`, `deepgemm`). Use `stats` for current corpus counts instead of hard-coding them in answers.

## Alias Hints

SQLite query normalization uses `store/schemas/aliases.yaml`. These are the high-value user spellings to remember:

| User phrase | Canonical query/tag |
|---|---|
| UMMA, tensor core gen 05, `tcgen05.mma` | `tcgen05` |
| Tensor Memory, accumulator memory | `tmem` |
| Cluster Launch Control | `clc` |
| Tensor Memory Accelerator, `cp.async.bulk` | `tma` |
| 2-SM cooperative, 2CTA, `cta_group::2` | `2sm-cooperative` |
| NVFP4, E2M1, FP4 E2M1 | `nvfp4` |
| block scaling, UE8M0, MX | `block-scale` |
| B200, B100, GB200, Blackwell | `sm100` |
| H100, H200, H800, Hopper | `sm90` |
| MLA, multi-head latent attention | `mla` |
| NSA, native sparse attention | `sparse-attention` |
| GDN, GatedDeltaNet | `gated-delta-net` |

## Evidence Rules

- `verified`: requires an `evidence_basis` with official documentation and upstream code.
- `source-reported`: backed by at least one authoritative source page; quote with source id.
- `inferred`: synthesized; say it is inferred when using it as a claim.
- `experimental`: undocumented or version-sensitive; include the relevant tool/CUDA/version caveat.

For performance claims, prefer:

```bash
python3 scripts/kbs.py get <kernel-id> --json
```

Then report only fields present in `performance_claims`: `gpu`, `dtype`, `shape`, `metric`, `value`, `utilization`, `source_id`.

## Version-Sensitive Note

Triton Blackwell support is version-sensitive. The local claim tracked by `vs-triton-3.6-blackwell-tcgen05` says Triton 3.6+ has native SM100 lowering surfaces for tcgen05/TMEM-related paths. If the user asks for "latest Triton" behavior, verify current upstream state before answering.
