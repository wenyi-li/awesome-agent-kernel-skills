# Reductions and Norms in FlyDSL

Reductions (per-row sum / max / variance) are the sweet spot for FlyDSL —
they combine layout-driven loading, LDS staging, and block-wide tree
reduction. The patterns below are battle-tested.

## AMD wavefront reduction (width 64)

XOR-shuffle reduction within a single wave. AMD wavefronts are ALWAYS 64
threads on gfx9xx — use shifts `[32, 16, 8, 4, 2, 1]`.

```python
def wave_reduce_add(x):
    width_i32 = fx.Int32(WARP_SIZE)    # WARP_SIZE = 64 on gfx9xx
    w = x
    for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
        off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
        peer = w.shuffle_xor(off, width_i32)
        w = w.addf(peer, fastmath=fm_fast)
    return w
```

## Block-wide reduction (two-step: wave reduce + LDS cross-wave)

1. Each wave reduces internally via XOR shuffle.
2. Lane 0 of each wave writes the per-wave partial to LDS.
3. `gpu.barrier()`.
4. Wave 0 reads all wave partials, reduces across them.
5. Result is in lane 0 of wave 0. Store to LDS slot 0 and barrier so all
   threads can read it.

```python
def block_reduce_add(val):
    dummy = fx.Float32(0.0)
    r0, _ = block_reduce_add2(val, dummy)   # one reduction cost
    return r0
```

## Dual reduction: `block_reduce_add2` (strictly better than two separate)

For norms that need **two** reductions (e.g. mean AND variance in the same
pass), combine them. Each wave reduces both in parallel, writes TWO slots to
LDS, barriers ONCE, then reduces both from LDS. This saves one barrier
round-trip vs calling `block_reduce_add` twice.

```python
def block_reduce_add2(val0, val1):
    if RED_SLOTS == 1:
        return wave_reduce_add(val0), wave_reduce_add(val1)

    lane = tid % WARP_SIZE
    wave = tid // WARP_SIZE
    w0 = wave_reduce_add(val0)
    w1 = wave_reduce_add(val1)

    if lane == fx.Int32(0):
        wave_idx = ArithValue(wave).index_cast(T.index)
        s_red.store(w0, [wave_idx])
        s_red2.store(w1, [wave_idx])
    gpu.barrier()

    if wave == fx.Int32(0):
        in_range = lane < RED_SLOTS
        lane_safe = in_range.select(lane, fx.Int32(0))
        lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
        v0 = s_red.load([lane_safe_idx])
        v1 = s_red2.load([lane_safe_idx])
        z = fx.Float32(0.0)
        ww0 = in_range.select(v0, z)
        ww1 = in_range.select(v1, z)
        ww0 = wave_reduce_add(ww0)
        ww1 = wave_reduce_add(ww1)
        if lane == fx.Int32(0):
            c0_idx = fx.Index(0)
            s_red.store(ww0, [c0_idx])
            s_red2.store(ww1, [c0_idx])
    gpu.barrier()

    c0_idx = fx.Index(0)
    return s_red.load([c0_idx]), s_red2.load([c0_idx])
```

## A concrete norm template (per-row, two-pass)

```python
# Pass 1: reduce sum_sq across N
for tile_i in range_constexpr(num_tiles):
    idx = tid + tile_i * BLOCK_THREADS
    vec = _load_vec(in_div, idx)
    in_local.append(vec)
    x = vec.to(Float32)
    x2 = x * x
    red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
    thread_sumsq = ArithValue(thread_sumsq) + red2

_, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
mean_sq = ArithValue(sum_sq) / n_float
ms_eps  = mean_sq + eps_c
rrms    = ms_eps.rsqrt(fastmath=fm_fast)

# Pass 2: normalize + scale + store (reuse cached inputs)
for tile_i in range_constexpr(num_tiles):
    idx = tid + tile_i * BLOCK_THREADS
    g = _load_vec(gamma_div, idx).to(Float32)
    x = in_local[tile_i].to(Float32)
    y = (x * rrms) * g
    # cast back to bf16 / fp16 / fp32 as needed, then store
```

## Two-kernel vs fused design for backward norms

A backward norm kernel needs both a per-row computation (`dx`) AND a
reduction across rows (`dw`). Two designs are common:

**Design A (two kernels, simple but bandwidth-wasteful):**
1. Main kernel reads `dy, x, gamma, rstd` and writes `dx`.
2. Finalize kernel re-reads `dy, x, rstd` and writes `dw`.

Memory traffic: reads `dy + x` TWICE (32 MB + 32 MB = 64 MB of redundant
reads per call at M=8192 / N=2048 / bf16). Total ~160 MB.

**Design B (fused, ~2x bandwidth win):**
- One main kernel: handles `TILE_M` rows per block; for each row it does the
  per-row reduction for `dx`, then accumulates `dy * x * rstd` into a
  per-thread register vector.
- After `TILE_M` rows, each thread writes its accumulated `dw` partial to
  `DwPartial[block_id, thread_col_start:end]` as f32.
- Small finalize kernel reduces `DwPartial[num_blocks, N] -> dw[N]` (typically
  2 MB of partials for M=8192, N=2048).

Memory traffic: reads `dy + x` ONCE plus a 2 MB partial tensor. Total
~100 MB — ~37% fewer bytes, ~2x faster in practice.

**Choose TILE_M carefully**. In a row-stripe fused design:
- `TILE_M` too small (e.g. 1, 2): low occupancy per block, many kernel launches.
- `TILE_M` too large (e.g. 64, 128): each block has too many sequential
  reductions; occupancy suffers.
- Sweet spot for M=8192, N=2048, bf16 on gfx950 is around `TILE_M=16`.
  Always autotune for your specific shape.

Keep `num_blocks = M / TILE_M` divisible by the finalize kernel's block
size (commonly 64 = one wave) so you don't need bounds checks in the
finalize kernel.

## Gotchas when writing reductions

- **Uncoalesced memory access patterns silently cost you bandwidth.** For
  cross-row reduction, threads within a warp should read CONSECUTIVE bytes
  of the same row, not bytes stride=N apart. The latter still works (AMD
  hardware gather/scatters), but each warp consumes many cachelines per
  cycle and effective bandwidth drops to ~40% of peak.
- **VEC width too large for buffer_load.** `buffer_load` is limited to 128
  bits per issue. For bf16 that's `VEC=8` (8x2 bytes). If you try `VEC=8`
  with default `dtype=f32`, the compiler sees a 32-byte load and fails with
  "Cannot select BUFFER_LOAD v8f32". Pass `dtype=T.bf16` for the right size.
- **Per-element vs vectorized reductions.** Use `Vec.reduce(ReductionOp.ADD)`
  for vectors of f32 — it is strictly faster than an unrolled loop of scalar
  adds.
- **The dw partial buffer is f32, not bf16.** Atomics on f32 are fast and
  accurate; on bf16 they are slow and can lose precision. Write partials as
  f32 and cast to bf16 only at the end (in the finalize kernel).
