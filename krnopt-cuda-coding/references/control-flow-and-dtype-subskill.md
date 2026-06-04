# Control Flow And Dtype Subskill

Use this integrated subskill when the likely problems involve divergence,
synchronization shape, or the wrong instruction path.

## Control-Flow Questions

Inspect:

- branches whose condition varies within a warp
- divergent branches before barriers
- loops with lane-varying trip counts
- reduction or staging phases that force uneven arrival at synchronization

The right question is not "is there a branch?" but "does warp structure stay
reasonably uniform through the hot path?"

## Synchronization Questions

Check:

- whether `__syncthreads()` or warp-sync is semantically required
- whether all producers and consumers are covered by the chosen sync scope
- whether the synchronization shape is frequent enough to dominate work

## Dtype And Instruction Path Questions

Inspect:

- whether dtypes actually enable the intended tensor-core or packed path
- whether literals or helper functions quietly promote the computation
- whether generic math is used where a specialized intrinsic or vector path is
  expected
- whether alignment and tile shape match the intended load or matrix path

## Common Warnings

- accidental FP64 from literals or intermediate expressions
- ornate branch structure in hot inner loops
- barriers that serialize work the code could stage more uniformly
- source that looks tensor-core-ready but misses alignment, shape, or dtype
  requirements

## Doctrine From CUDA Best Practices

The best practices guide places instruction-level tuning *after* hotspot
selection, memory behavior, and execution configuration. Micro-optimizing
math in a kernel dominated by uncoalesced global loads is wasted effort. Once
memory is under control, the instruction chapter reduces to three goals:
avoid low-throughput arithmetic, reduce unnecessary instructions, and prevent
control flow from destroying warp efficiency.

### Warp divergence is a first-class hazard

When threads within a warp follow different paths, those paths serialize. The
practical rule is not "avoid branches" but "write conditions so whole warps
take the same path." The canonical warp-aligned shape is:

```cuda
if ((threadIdx.x / warpSize) == 0) {
    // whole warp 0 takes this path; no divergence inside a warp
}
```

Divergence plus synchronization is a classic serialization pattern: if warps
diverge before a `__syncthreads()`, the barrier forces the slow side to
catch up while other lanes sit idle.

### Predication is the cheap escape hatch

For short branches, the compiler may replace the branch with predicated
instructions. All instructions are scheduled, but only threads with a true
predicate commit results. This avoids actual divergence and makes tiny
conditionals cheap. Help this path by:

- keeping conditional bodies short
- making branch structure simple (avoid nested branches in hot loops)
- using `#pragma unroll` where loop unrolling is profitable

On Volta and later, independent thread scheduling means a warp can stay
diverged beyond a conditional region. Insert `__syncwarp()` to guarantee
reconvergence before code that assumes lane-uniform state.

### Synchronization is not free

`__syncthreads()` has throughput cost and can idle the SM. That does not mean
never synchronize; it means remove unnecessary barriers and do not pretend
they are free. Kernels that synchronize frequently often benefit from
multiple smaller blocks per SM (so other blocks fill the gap) rather than
one giant synchronized block monopolizing residency.

Every barrier in the source should answer: which producers and which
consumers does it cover? A barrier without a clear data-sharing boundary
is either missing semantics or missing a reason to exist.

### Trade precision for speed only where safe

The guide recommends speed/precision tradeoffs when the end result permits:
single precision instead of double, specialized intrinsics instead of more
accurate standard functions, fast-math options only where behavior change is
acceptable. The aggressive `-use_fast_math` flag is a blunt instrument;
selective use of `__expf`, `__sinf`, `rsqrtf`, and similar intrinsics at
specific call sites is usually preferable.

### Expensive math has recognizable failure modes

Specific cases the best practices guide calls out:

- division and modulo are expensive; if the divisor is a power of two,
  shift/mask forms are much better
- `rsqrtf` should be expressed directly when the semantics allow, not via
  `1.0f / sqrtf(x)`
- trigonometric functions become dramatically slower for large arguments
  because argument reduction takes a slow path that also touches local
  memory
- `pow` for small integer exponents is much heavier than an explicit
  multiply; `exp2`, `cbrt`, or `sinpi`/`cospi` are often correct specialized
  alternatives
- signed loop counters enable better compiler optimization than unsigned
  loop counters in some patterns

These are instances of the same rule: prefer specialized operations with
simpler hardware or compiler paths over general ones.

### Type conversion rules that are easy to forget

- use `f` suffixes in single-precision code (`1.0f`, `2.5f`) to prevent
  implicit double-to-float conversions
- small integer powers should be written explicitly (`x * x * x` beats
  `pow(x, 3)`)
- keep literal types consistent with the dtype of the hot expression so a
  stray `1.0` does not promote an FP16 or FP32 pipeline to FP64

### Memory instructions are instruction-latency problems too

Loads and stores to uncached local/global memory carry hundreds of cycles
of latency. A kernel can look arithmetic-light while being instruction-
latency heavy. Preferring shared memory over global memory, when possible,
is not only a bandwidth point; it is also an instruction-latency point.

### Hardware path must match source path

When a kernel intends to use a tensor-core or vector path, several
preconditions all have to hold at once:

- dtypes must be compatible (for example, BF16 MMA requires BF16 operands
  with an FP32 accumulator)
- operand tiles must be aligned for the specific load primitive
- tile shapes must match the MMA `m*n*k` supported by the target arch
- shared-memory layout may need a skew to avoid bank conflicts while
  preserving alignment

Source that looks tensor-core-ready but misses one of these silently falls
back to a slower path or fails to compile into the intended instruction.
Check all four before blaming the MMA choice itself.

## Practical Control-Flow And Dtype Rules

- Fix higher-level issues (memory shape, occupancy) before instruction tuning.
- Prefer warp-aligned conditions (`threadIdx.x / warpSize`) over lane-varying
  branches in hot paths.
- Let short branches become predicated; keep their bodies small.
- Use fast-math intrinsics selectively, not reflexively.
- Avoid integer division/modulo when algebraic rewrites exist.
- Remove unnecessary barriers; one `__syncthreads()` per reuse boundary is
  usually enough.
- Pin literal types to the kernel's working dtype to prevent silent FP64
  promotion.
- When targeting tensor cores, validate dtype, alignment, tile shape, and
  SMEM layout together.
