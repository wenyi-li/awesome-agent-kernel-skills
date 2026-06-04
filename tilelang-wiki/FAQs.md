# Frequently Asked Questions

## [Question] `ParallelOpNode::RecordBufferAccess` fails on an interleaved complex tile

If you hit an error like:

```text
tvm.error.InternalError: Check failed: (StructuralEqual()(it->second.indices, indices)) is false: q_tile: (tid * 2 + 1,) and (tid * 2,)
```

or the 2D variant:

```text
tvm.error.InternalError: Check failed: (StructuralEqual()(it->second.indices, indices)) is false: q_tile: (tid, 1) and (tid, 0)
```

inside a `T.Parallel(...)` loop, a common cause is using one fragment buffer to
represent interleaved complex lanes and then reading sibling indices such as
real/imag from that same buffer in the parallel loop body.

This shows up naturally in RoPE-style kernels where the host data is laid out as
`[..., head_dim]`, but the kernel tries to treat it as complex pairs by loading
one tile and then accessing:

```python
q_real = q_tile[pair_idx, 0]
q_imag = q_tile[pair_idx, 1]
```

or:

```python
q_real = q_tile[real_idx]
q_imag = q_tile[imag_idx]
```

In current TileLang lowering, that access pattern can attach incompatible access
records to the same parallel buffer and fail during layout inference.

The robust workaround is to split the lanes into separate buffers before the
parallel loop:

```python
q_real_tile = T.alloc_fragment((tile_pairs,), dtype)
q_imag_tile = T.alloc_fragment((tile_pairs,), dtype)
freq_real_tile = T.alloc_fragment((tile_pairs,), freq_dtype)
freq_imag_tile = T.alloc_fragment((tile_pairs,), freq_dtype)

T.copy(q[..., 0], q_real_tile)
T.copy(q[..., 1], q_imag_tile)
T.copy(freqs[..., 0], freq_real_tile)
T.copy(freqs[..., 1], freq_imag_tile)
```

Then use one stable index per buffer inside `T.Parallel(...)`:

```python
for tid, lane in T.Parallel(threads, elements_per_thread):
    pair_idx = tid * elements_per_thread + lane
    q_real = q_real_tile[pair_idx]
    q_imag = q_imag_tile[pair_idx]
```

Rule of thumb: for complex-number or pairwise kernels, do not rely on
interleaved real/imag fragment accesses inside `T.Parallel(...)`. Either split
the lanes into separate buffers or move the pair extraction outside the parallel
region.

## [Question] A FullRow GEMM accumulator stores interleaved gate/up pairs, and the postprocess has to split them out manually

Another manifestation of the same limitation appears in GEMM-driven pairwise
epilogues such as RMSNorm + SwiGLU.

Suppose a GEMM writes one fragment tile `acc` whose columns are interleaved as:

```text
[gate_0, up_0, gate_1, up_1, ...]
```

The natural postprocess is to read sibling lanes from the same accumulator,
for example:

```python
gate = acc[i, j * 2]
up = acc[i, j * 2 + 1]
```

In current TileLang lowering, that sibling-fragment access pattern is the same
known issue as the minimal repro above. Even when the larger kernel is
restructured to extract `gate_frag` and `up_frag` in separate `T.Parallel(...)`
loops before applying RMSNorm and SwiGLU, that split is still a workaround for
the underlying layout-inference restriction.

This is why a larger fused kernel may end up written in the more awkward form:

```python
gate_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)
up_frag = T.alloc_fragment((block_M, paired_outputs_per_tile), accum_dtype)

for i, j in T.Parallel(block_M, paired_outputs_per_tile):
    gate_frag[i, j] = acc[i, j * 2]
for i, j in T.Parallel(block_M, paired_outputs_per_tile):
    up_frag[i, j] = acc[i, j * 2 + 1]
```

rather than reading both sibling lanes directly in one pairwise epilogue loop.

By contrast, Triton does not suffer from this specific layout-inference failure,
so the analogous fused GEMM + RMSNorm + SwiGLU implementation can keep the
pairwise logic in a more natural form, either by using separate accumulators for
gate and up or by reshaping/selecting lanes from an interleaved accumulator.

Practical workarounds in TileLang are:

1. Split the sibling lanes into separate buffers before the main pairwise
   `T.Parallel(...)` postprocess.
2. Materialize the interleaved result through shared memory before re-reading
   pair members.
3. Change the computation so gate and up are produced by separate GEMM paths or
   separate weight layouts instead of one interleaved fragment tile.

Rule of thumb: if a fragment produced by `T.gemm(...)` represents logical pairs
such as gate/up, real/imag, or even/odd lanes, treat direct sibling reads from
that fragment inside `T.Parallel(...)` as fragile. Split the lanes first, or
store them in a layout that avoids pair extraction from one fragment buffer.

## [Question] Layout infer conflict

If you hit an error like:

```text
tvm.error.InternalError: Layout infer conflict between acc_s and acc_s_cast in T.Parallel loop
```

the usual cause is that two `T.gemm(...)` calls expect different layouts for the
same intermediate buffer. In the report from issue
`tile-ai/tilelang#1165`, the first GEMM writes `acc_s` with
`policy=T.GemmWarpPolicy.FullCol`, while the later GEMM path expects a layout
compatible with `FullRow`.

Two known fixes are:

1. Change the casted buffer from fragment memory to shared memory:

```python
acc_s_cast = T.alloc_shared([block_N, block_M * heads], dtype)
```

This works because shared memory layout is more flexible.

2. Keep the casted buffer as a fragment, but align the GEMM layout policy:

```python
T.gemm(
    K_shared,
    Q_shared,
    acc_s,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
)
```

This is the preferred fix when valid for your kernel, because it avoids an
extra register-to-shared-memory copy and is typically faster.

Rule of thumb: if a fragment buffer is produced by one `T.gemm(...)` and later
consumed by another GEMM-related path, make sure both operations agree on the
fragment layout policy. If they do not, either align the policies or move the
intermediate through shared memory.

## [Question] `no available layout found` with two reductions

If you hit an error like:

```text
tvm.error.InternalError: Check failed: (min_reg_num < INT64_MAX) is false: no available layout found
```

and your kernel applies multiple reductions to the same fragment buffer, the
cause may be conflicting layout constraints from the reductions themselves. In
issue `tile-ai/tilelang#1714`, the kernel does:

```python
b = T.alloc_fragment([tilesize, nstr], dtype=dtype)
T.reduce_sum(R, b, dim=-1)
T.reduce_sum(R, b, dim=-2)
```

The problem is that `T.reduce_sum(...)` constrains both source and destination
layouts. Reducing into the same fragment buffer `b` with different reduction
dimensions can attach incompatible layout requirements, and layout inference
fails with `no available layout found`.

A simple workaround is to allocate the destination as shared memory instead of
a fragment:

```python
b = T.alloc_shared([tilesize, nstr], dtype=dtype)
```

Rule of thumb: if multiple reductions write into the same intermediate and the
reduction dimensions differ, avoid reusing a fragment buffer for all of them.
Use shared memory for the intermediate, or split the computation so each
reduction gets a layout-compatible destination.

## [Question] Observed `blockDim` does not match `threads=...`

If you write a kernel like:

```python
with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
```

but Nsight Compute shows `blockDim=(256, 1, 1)`, the usual reason is that TMA
was enabled and the compiler inserted an extra producer warp group.

In issue `tile-ai/tilelang#1523`, the TileLang maintainers explained that when
the compiler detects TMA copy usage, it may launch an extra warp group to issue
those TMA operations. That means the runtime CUDA block size can be larger than
the `threads=` value you passed in `T.Kernel(...)`.

So in this situation, the extra threads do not necessarily mean your
`threads=128` argument was ignored. Instead, TileLang augmented the launch
configuration to support TMA.

Rule of thumb: if profilers show more threads than expected, check whether the
kernel or pass configuration allowed TMA. When TMA is active, TileLang may add
producer warps on top of your requested worker threads.

## [Question] Autotune fails for every config on a metadata-driven kernel

If autotuning logs repeated validation failures or even reports that no
configuration succeeded, and your kernel takes structured metadata tensors such
as offsets, lengths, masks, or grouped-GEMM size tables, the usual cause is
that autotune is benchmarking the kernel with auto-generated inputs that do not
respect the metadata contract.

For example, a grouped kernel may expect inputs like:

```python
packed_lhs: T.Tensor((group_size, padded_M, padded_K), dtype)
packed_rhs: T.Tensor((group_size, padded_K, padded_N), dtype)
group_sizes: T.Tensor((group_size, 3), "int32")
```

where `group_sizes[g] = (M_g, N_g, K_g)` drives which output rows and columns
are valid for each group. If autotune generates arbitrary tensors for the data
inputs and metadata input independently, the reference program and the kernel
can disagree on what the valid region is, so every candidate config appears
wrong even when the kernel is fine.

The fix is to capture real, mutually consistent inputs with
`set_autotune_inputs(...)`:

```python
from tilelang.autotuner import AutoTuner, set_autotune_inputs

with set_autotune_inputs(packed_lhs, packed_rhs, group_sizes):
    result = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(ref_prog=packed_reference, skip_check=False)
        .run(warmup=3, rep=20)
    )
```

Two extra checks help a lot:

1. Ensure the reference program accepts the same input signature as the kernel's
   non-output inputs.
2. If you compare full packed outputs, define the padded or invalid region
   explicitly, for example by zero-filling skipped tiles, rather than leaving
   that region unspecified.

Rule of thumb: whenever kernel correctness depends on metadata tensors rather
than just shapes and dtypes, autotune with real captured inputs instead of
relying on automatic input generation.

## [Question] I changed the kernel source, but rerunning the profiler did not recompile it

If a profiling or autotuning rerun finishes suspiciously quickly, reuses an old
best config, or does not print the usual compile logs after you edited the
kernel, the usual cause is a cache hit rather than TileLang ignoring the edit.

TileLang has multiple cache layers:

- JIT caches compiled kernels per specialization.
- The autotuner caches tuning results in memory and on disk.

For autotuning, a subtle failure mode is that the cache key is built from the
source of the callable you pass into the autotuner. If that callable is a small
outer kernel factory like:

```python
def kernel(block_M=None, threads=None):
    return make_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        block_M=block_M,
        threads=threads,
        dtype=rope_dtype,
    )
```

then editing the internals of `make_kernel(...)` may not change the autotune
cache key if the outer `kernel(...)` closure source stays the same. In that
case, the autotuner can reload a previously tuned kernel and skip visible
recompilation.

Typical signs:

1. The rerun completes much faster than a fresh tune.
2. The previous best config appears immediately.
3. You changed helper or lowering code, but not the outer autotune closure.

Ways to force a fresh run:

1. Set `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` to disable autotune disk cache.
2. Set `TILELANG_DISABLE_CACHE=1` to disable TileLang caches globally.
3. Delete the relevant cache directory under `~/.tilelang/cache`.
4. Run the direct JIT path once without autotuning to confirm the new kernel
   itself recompiles.
5. For debugging only, make a trivial source change inside the autotune closure
   so its cache key changes too.

Rule of thumb: if a kernel edit does not appear to trigger recompilation, check
which callable owns the active cache key. Changing a nested helper is not always
enough to invalidate an autotune cache entry.
