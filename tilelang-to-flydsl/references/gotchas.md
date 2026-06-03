# Gotchas and review checklist

Run through every section before declaring a conversion complete. Each item is
a mistake the agent has made (or is highly likely to make) on a TileLang →
FlyDSL port. Items marked **(silent)** produce code that compiles and runs
without error but gives wrong results.

---

## A. Launch shape and indexing

- [ ] `with T.Kernel(gx, gy, gz, threads=N) as (pid_x, pid_y, pid_z):` —
      check the order. TileLang binds `(pid_x, pid_y, pid_z)` to
      `(blockIdx.x, blockIdx.y, blockIdx.z)`, *first* tuple element to `.x`.
      Make sure `bid_x = fx.block_idx.x` reads the first one. **(silent)**
- [ ] `T.Kernel(gx, threads=N) as pid:` (single grid dim, no tuple) is the
      same as `as (pid,)`. Don't accidentally drop the tuple.
- [ ] Multi-axis `threads=(tx, ty)` (rare) requires `block=(tx, ty, 1)` and
      reading both `tid_x` and `tid_y`. Easy to flatten by mistake.
- [ ] Grid expressions involving `T.ceildiv` must be re-derived on the host
      side — TileLang infers them from the kernel shape annotations. Compute
      `gx = (N + BLOCK - 1) // BLOCK` in the `@flyc.jit` body.
- [ ] Empty inputs: TileKernels guards `if num_tokens > 0: kernel(...)` in
      the Python wrapper. Preserve that guard — calling FlyDSL with grid=0
      may behave differently. Check the wrapper.

## B. Dtype mapping

- [ ] AMD historically uses **FNUZ** variants of FP8 on gfx94x:
      `Float8E4M3FNUZ`, `Float8E5M2FNUZ`. The TileLang spelling is
      `T.float8_e4m3` (OCP) — the corresponding FlyDSL type depends on the
      target arch. On gfx950 the OCP variants `Float8E4M3FN`/`Float8E5M2`
      apply. **Confirm the arch with `get_rocm_arch()` and pick accordingly.**
      Picking the wrong variant produces garbage output. **(silent)**
- [ ] `T.bfloat16` ↔ `fx.BFloat16`. On gfx950 there is a HW pack instruction
      (`USE_HW_CVT_PK_BF16_F32`) that the FlyDSL kernels guard with an
      `if const_expr(arch.startswith("gfx95")):` branch. If the TileLang
      kernel is bf16-output and you target gfx94x, you may need a manual
      round-and-pack — see `rmsnorm_kernel.py` for the pattern.
- [ ] `T.cast(x, dtype)` and `x.astype(dtype)` both become `value.to(dtype)`
      on a `Vector` or `Numeric`. For a single scalar use `fx.Float32(x)` /
      `fx.Int32(x)` constructors. Don't reach for `arith.fptosi` etc.
      directly.

## C. Memory allocation

- [ ] Every `T.alloc_shared` increases the LDS allocator pointer. Allocate
      *all* shared buffers up-front (outside the `@flyc.kernel` body) and
      then call `allocator.finalize()` *inside* the `@flyc.jit` body, in the
      `gpu_module_body` insertion-point context. Forgetting `finalize` makes
      the LDS layout invalid and the kernel will read undefined values.
      **(silent)** if FlyDSL's verifier doesn't catch it.
- [ ] LDS bank conflicts: TileLang authors often pad shared dimensions
      (`block_x + block_k` rather than `block_x`). Preserve the padding in the
      `SmemAllocator` allocation size *and* in the `SmemPtr` shape. Dropping
      the pad turns a 0-conflict access into a 4-way one — silently slower,
      not silently wrong, but easy to miss. Run `/lds-optimization` if
      profiling shows the issue post-port.
- [ ] `T.alloc_var(dtype, init=v)` that is *reassigned* inside an `scf.for`
      loop must be carried via `range(..., init=[v])`. A naive Python
      assignment captures the value once and the body becomes incorrect
      because MLIR's `scf.for` does not see the reassignment. **(silent)**
- [ ] `T.alloc_local` inside a loop: TileLang re-allocates per iteration
      (registers); FlyDSL `fx.memref_alloca` allocates *once* per scope. If
      the TileLang kernel relies on a fresh buffer per iteration, lift the
      allocation outside the loop — the body should overwrite it on each
      iteration, which is fine.
- [ ] `T.alloc_fragment` shape *must* match the operand shape that the
      consuming op expects. When converting to FlyDSL fragments, use
      `thr_mma.make_fragment_A(...)` rather than picking an arbitrary
      register layout — the layout is dictated by the MFMA atom, not by the
      kernel author.

## D. Loops and parallelism

- [ ] `T.Parallel(N)` is **not** the same as a Python `for i in range(N):`.
      It distributes iterations across all `threads=K` of the workgroup. A
      naive `range(N)` in FlyDSL replicates the loop on every thread, doing
      `K * N` work. **(silent, gives wrong output)** — every thread will
      stomp on the same data.
- [ ] `T.Pipelined(K, num_stages=k)` translates to a hand-written prefetch
      loop in FlyDSL. Don't drop the pipelining silently — the GEMM main
      loop becomes ~2× slower if the prefetch is missed. Use the
      `prefetch-data-load` skill.
- [ ] `T.unroll(n)` is fully unrolled: use `range_constexpr(n)`. `T.serial`
      is a runtime loop: use `range(n)`. Don't swap them (one becomes
      compile-time, the other doesn't, and constexpr-only ops in the body
      won't trace correctly).
- [ ] Loops over `fx.Int32` runtime values must use `range`, not
      `range_constexpr`. The latter expects a Python int.

## E. Data movement

- [ ] `fx.rocdl.make_buffer_tensor` is required *before* `BufferCopy*` atoms
      can act on a tensor. Forgetting it produces an MLIR verification error
      at compile time, but the error message is opaque. Always wrap the
      input `fx.Tensor` arg in `make_buffer_tensor` once at the top of the
      kernel body.
- [ ] `BufferCopy128b` requires the underlying memory to be 16-byte aligned
      and the access pattern to be contiguous in 16 bytes. If the leading
      dim is not divisible by `16 / elem_bytes`, fall back to
      `BufferCopy64b` / `BufferCopy32b` and adjust `VEC` accordingly. The
      host-side `mark_layout_dynamic(divisibility=d)` declares the
      divisibility guarantee.
- [ ] When `T.copy(global, shared)` writes contiguous bytes that *do not*
      match the natural shared-memory layout (e.g., transpose), the FlyDSL
      version needs **two** separate copies: global → register fragment, and
      register fragment → LDS, with `gpu.barrier()` between them. Skipping
      the barrier is undefined. **(silent, race condition)**
- [ ] `T.copy(buf, dst, disable_tma=True)` — drop the `disable_tma` flag.
      AMD has no TMA, the flag is a no-op there.

## F. Reductions

- [ ] `T.reduce_max` zero-initialises the output by default
      (`clear=True`); `clear=False` accumulates. Replicate the same:
      `frag_C.fill(neutral)` if clearing, otherwise carry the previous value
      through.
- [ ] Wave-butterfly: the loop `for _e in range_constexpr(int(math.log2(WARP_SIZE)))`
      assumes `WARP_SIZE` is a power of two (always true on AMD). Don't
      hardcode `64` — use `get_warp_size()` because RDNA / gfx1250 are
      wave-32. **(silent, gives wrong reduction on RDNA)**
- [ ] `T.alloc_reducer(replication='all')` means *every* thread sees the
      reduced value. Make sure the FlyDSL block-reduction broadcasts the
      result back via the LDS load at the end, not just leaves it in slot 0
      readable only by lane 0. The `block_reduce` helper in
      `idioms.md` §4 already does this.
- [ ] `T.finalize_reducer` is a no-op once the result is in the LDS slot —
      the load step is enough.

## G. Math

- [ ] TileLang's `T.exp` / `T.log` are precise; FlyDSL's `rocdl.exp2` is a
      single hardware instruction with lower precision. Default to
      `fmath.exp(...)` / `fmath.log(...)` unless the kernel was already
      using `T.exp2` (faster path). Check the test tolerance — if it is
      tight (`atol=1e-7`), the rocdl variants will fail.
- [ ] Mixed-precision: TileLang lets you write `T.float32(x_bf16) * y_f32`.
      FlyDSL requires explicit `.to(fx.Float32)` casts on the bf16 side.
      Forgetting one yields a type error at trace time (loud), but
      forgetting one in the *output* path silently truncates. **(silent)**
- [ ] `T.infinity(T.float32)` is positive infinity. For negative infinity
      use `-T.infinity(...)` — the TileLang expression. In FlyDSL the
      negative form is `fx.Float32(float("-inf"))`. Don't translate
      `-T.infinity(T.float32)` as `fx.Float32(-float("inf"))`; that is the
      same value, but use the canonical spelling.

## H. Strides and host integration

- [ ] If the test calls `tile_kernels.X.X(twice_stride(x))` (or any
      non-contiguous input), the FlyDSL wrapper must do
      `flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=k, divisibility=d)`
      before passing to the kernel. The `leading_dim` is the axis whose
      stride is dynamic; `divisibility` is the smallest divisor the test
      respects (typically 4 or 8). **(silent, reads from wrong rows)**
- [ ] Empty tensors: respect `if num_tokens > 0: kernel(...)`. Some FlyDSL
      kernels segfault on `grid=(0, ...)` due to ROCm runtime quirks.
- [ ] The Python wrapper's signature must remain *byte-identical* to the
      TileLang version — same arg names, same types, same return tuple
      (e.g., `(out, out_sf)`). The TileKernels tests import by name and
      destructure the return.

## I. Constexpr vs runtime

- [ ] Anything you `if`-branch on at trace time must be `const_expr(...)`.
      `if dtype_str == "f32":` evaluates at trace time only because
      `dtype_str` is a Python `str` (which is `Constexpr`-ish by default).
      `if N >= 1024:` evaluates at trace time only if `N` is a Python `int`,
      not an `fx.Int32`. Forgetting this drops you into a runtime branch
      that may not be valid (compile-time-only ops can't appear there).
- [ ] `range_constexpr(N)` requires `N` to be a Python int. `range_constexpr(fx.Int32(N))`
      crashes at trace time. Use `const_expr(N // tile)` to evaluate at
      trace time before passing.

## J. Caching and reuse

- [ ] Two `@flyc.jit` calls with the same Python source but different
      Constexpr arg values produce two cache entries, as expected. But two
      calls with the same source and same Constexprs but a *different
      closure value* (e.g., the JIT factory captures `arch`) also produce
      different cache entries. This is correct behaviour — but watch for it
      if the user wonders why their cache is "growing too fast".
- [ ] When debugging post-port, set `FLYDSL_RUNTIME_ENABLE_CACHE=0` to
      bypass the disk cache. Otherwise an old buggy compile shadows your
      fix.

## K. Diff with the FlyDSL "production" reference

After conversion, do an explicit comparison pass:

1. Open the closest matching FlyDSL reference kernel
   (`softmax_kernel.py` for soft-reductions, `rmsnorm_kernel.py` for
   norms, `preshuffle_gemm.py` for GEMM, `silu_and_mul_fq.py` for fused
   activation+quant).
2. Walk through the converted kernel side-by-side. Specifically check:
   - LDS allocator pattern matches (allocate-up-front, finalize-in-jit).
   - Buffer-tensor wrap (`make_buffer_tensor`) appears once at the top.
   - Tiled-copy partition pattern (`get_slice` → `partition_S/D` →
     `copy(...)`) is structurally similar.
   - For reductions, the `wave_reduce` / `block_reduce` helpers are present
     and use the same `RED_SLOTS = max(1, ...)` convention.
3. List every place where the converted kernel deviates from the reference,
   and justify each deviation either as "imposed by the TileLang source" or
   "intentional simplification". Record this in the task summary so the
   user can audit it.

If a deviation cannot be justified, the conversion is not done — return to
step 4 of the workflow and re-derive that block.
