# FlyDSL GEMM Optimization Reference

FlyDSL's primary GEMM surface is the "preshuffle GEMM" pattern, which matches
the layout AMD MFMA instructions want natively. For large GEMMs in common
shapes you should NOT reach for FlyDSL — hipBLASLt, aiter tuned GEMM, and CK
GEMM are already well-tuned. Use FlyDSL GEMM ONLY when:

- You have a concrete shape/dtype where the library coverage is genuinely
  bad (verified by measurement, not by assumption), OR
- You need a kernel that fuses GEMM with a custom elementwise epilog that
  the libraries do not expose, OR
- You're writing a grouped GEMM for an unusual routing pattern.

## MFMA instructions available

```python
from flydsl.expr import rocdl

# fp16 / bf16 MFMA
result = rocdl.mfma_f32_16x16x16_f16(a, b, acc)           # gfx942 K=16
result = rocdl.mfma_f32_16x16x32_bf16(a, b, acc)           # gfx950 K=32

# fp8 MFMA (MI300+)
result = rocdl.mfma_f32_16x16x32_fp8(a, b, acc)

# int8 MFMA
result = rocdl.mfma_i32_16x16x32i8(a, b, acc)
```

## Preshuffle GEMM design (the canonical FlyDSL pattern)

Core steps:

1. B matrix is PRE-SHUFFLED to layout `(N/16, K/64, 4, 16, kpack_bytes)` at
   model init, once.
2. A tiles are loaded from global memory to LDS with XOR16 swizzle for
   bank-conflict avoidance.
3. K64-byte micro-step: each step issues 2 x K32 MFMA ops.
4. Ping-pong LDS (`lds_stage=2`) for overlapping loads with compute.
5. Epilogue: direct row-major store OR CShuffle via LDS for packing.

See `/opt/FlyDSL/kernels/preshuffle_gemm.py` (this is the reference
implementation).

## Key tunable parameters

| Parameter | Typical values | Meaning |
|---|---|---|
| `tile_m`, `tile_n`, `tile_k` | 128x128x64, 128x64x64, 256x128x64 | Block tile in elements |
| `num_warps` | 4, 8 | Warps per block |
| `lds_stage` | 1, 2 | LDS pipeline depth |
| `use_cshuffle_epilog` | `False`, `True` | CShuffle epilogue enables packed stores |
| `in_dtype` | `"bf16"`, `"fp16"`, `"fp8"`, `"int8"`, `"int4"`, `"int4_bf16"` | Input dtype |
| `out_dtype` | `"f16"`, `"bf16"` | Output dtype |
| `group_size` | 32 | For W4A16 groupwise scale |
| `k_batch` | 1, 2, 4 | Split-K factor |

Shape constraints:
- `tile_m * tile_k * elem_bytes` must be divisible by `256` (block threads).
- `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).
- Split-K requires `model_dim % k_batch == 0` AND `K_per_batch / tile_k >= 4` and even.

## LDS optimization

On gfx950 (MI350) LDS is 160 KB per CU with 64 banks. The classic XOR16
swizzle designed for the 32-bank gfx942 configuration may be suboptimal on
gfx950 — consider adjusting the swizzle mask. See the FlyDSL repo's
`.claude/skills/lds-optimization/SKILL.md` for the full bank-conflict analysis
if your kernel is LDS-bound.

`ds_read_b128` / `ds_write_b128` (16-byte LDS ops) are the fastest; always
prefer these over narrower LDS ops when alignment allows.

## Hot-loop scheduling and prefetch

Load for iteration K+1 BEFORE the compute for iteration K finishes — the load
latency hides behind MFMA. See the `scf.for` loop-carried values pattern in
`references/kernel-authoring.md` and `/opt/FlyDSL/kernels/mfma_preshuffle_pipeline.py`.

## MoE grouped GEMM (`/opt/FlyDSL/kernels/moe_gemm_2stage.py`)

`compile_moe_gemm1` compiles a two-stage MoE GEMM kernel. WARNING: in practice,
the Python-side routing construction and weight preshuffle cache can easily
dominate the kernel speedup. Concretely:

1. **Weight preshuffle cache keyed on `data_ptr()` is fragile** — optimizer
   in-place updates keep the pointer the same but change the values, giving
   you stale shuffled weights. Re-shuffle each call, or key on a version
   counter.
2. **`aiter::moe_sorting` routing** must be run fresh when `group_lens`
   changes. Cache routing only when `group_lens` is constant across calls.
3. **Output buffers cannot be cached + `zero_()`'d** — autograd rejects
   inplace modification of a previously returned tensor.

Given those constraints, benchmarked isolation speedup does not reliably
translate to e2e wins. Take the MoE grouped GEMM path ONLY when you have
measured evidence the library grouped GEMM leaves performance on the table
AND you can carefully handle the above correctness gotchas.

## Data type casts (gfx942 vs gfx950)

On gfx950, use the hardware `v_cvt_pk_bf16_f32` / `v_cvt_pk_f16_f32` for
f32 → bf16 pack — one instruction, no rounding bias:

```python
out_e = y.to(BFloat16)   # single-op cast on gfx950
```

On gfx942, you must emit the round-to-nearest-even pack manually (bias 0x7FFF
with LSB-tie handling). The existing `rmsnorm_kernel.py` has the reference
implementation for the manual path.

## Autotune

```python
from flydsl.autotune import autotune, Config

@autotune(
    configs=[
        Config(tile_m=128, tile_n=128, tile_k=64),
        Config(tile_m=128, tile_n=64,  tile_k=64),
        Config(tile_m=256, tile_n=128, tile_k=64),
    ],
    key=["M", "N", "K"],
    warmup=5, rep=25,
)
@flyc.jit
def gemm(A, B, C, M, N, K,
         tile_m: fx.Constexpr[int],
         tile_n: fx.Constexpr[int],
         tile_k: fx.Constexpr[int],
         stream: fx.Stream = fx.Stream(None)):
    ...
```

First call benchmarks all configs; subsequent calls with the same `key`
values use the cached best. Disk cache at `~/.flydsl/autotune/{func}.json`.

### `waves_per_eu` is NOT settable via `gpu-module-to-binary opts=`

Known limitation — set as an LLVM function attribute or through
`rocdl-attach-target` if you need it. Most kernels do not need this.
