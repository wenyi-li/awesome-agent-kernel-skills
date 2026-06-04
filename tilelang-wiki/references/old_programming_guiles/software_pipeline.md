# Software Pipeline

`T.Pipelined` marks a serial loop whose iterations can be overlapped. The
common use is a tiled producer/consumer loop: load the next tile into shared
memory while later stages compute on an earlier tile.

This guide assumes the surrounding kernel shape from `language_basics.md` and
the copy/TMA instruction surface from `instructions.md`.

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

This is the pattern used by the working GEMM examples. `num_stages=3` asks the
compiler to keep up to three pipeline versions of the producer buffers and to
emit the prologue, steady-state loop, and epilogue needed to overlap the copy
and compute stages.

## API

```python
for i in T.Pipelined(
    start_or_extent,
    stop=None,
    num_stages=0,
    order=None,
    stage=None,
    sync=None,
    group=None,
):
    ...
```

- `T.Pipelined(n)` iterates from `0` to `n`.
- `T.Pipelined(start, stop)` iterates from `start` to `stop`.
- `num_stages=0` keeps the loop valid but does not request compiler-inferred
  software pipelining.
- `num_stages=1` is accepted and useful for testing or tuning, but it does not
  provide meaningful multi-version buffering for ordinary shared/local buffers.
- `num_stages >= 2` enables real pipeline-depth behavior. Versioned buffers are
  indexed by the logical iteration modulo the pipeline depth.
- `order` and `stage` are manual scheduling annotations. Use them when the
  compiler-inferred schedule is not the one you want.
- `sync` and `group` are lower-level metadata for manual pipeline lowering.
  Most kernels should not need them.

For ordinary GEMM-like loops, start with `num_stages` only:

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

Autotuning examples usually include `num_stages` in the tuning config together
with block sizes and thread count. Values like `0`, `1`, `2`, and `3` are all
seen in examples; the best value depends on memory latency, shared-memory use,
register pressure, and target backend. See `autotuning.md` for the tuning flow.

## Manual Stage And Order

Manual annotations describe executable pipeline statements in source order.
Each scheduled statement has:

- `stage`: the logical producer/consumer stage.
- `order`: the emission order after the loop body has been classified.

```python
for ko in T.Pipelined(
    T.ceildiv(K, block_K),
    stage=[0, 0, 1],
    order=[0, 1, 2],
):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

Here the two copies are stage 0 producers and the GEMM is a stage 1 consumer.
The annotation entries correspond to the scheduled statements:

```text
copy A -> stage 0, order 0
copy B -> stage 0, order 1
gemm   -> stage 1, order 2
```

The compiler validates the final schedule:

- Every scheduled statement must have a unique `order`.
- If statement A writes a buffer that statement B reads, A cannot be in a later
  stage than B.
- If dependent statements are in the same stage, the producer must have a
  smaller `order` than the consumer.

When you provide `stage` and `order` without `num_stages`, the pipeline depth is
inferred as `max(stage) + 1`. In normal user code, prefer either:

```python
T.Pipelined(n, num_stages=3)
```

or:

```python
T.Pipelined(n, stage=[...], order=[...])
```

Avoid mixing both unless you intentionally need to override the inferred depth.

## What Counts As A Scheduled Statement

Count effectful statements, not every IR node in the loop body.

Count these:

- `T.copy`, including global-to-shared copies.
- `T.tma_copy`, `T.async_copy`, and explicit async-copy bookkeeping.
- `T.fill`, `T.clear`, reductions, stores, and atomics.
- `T.gemm` and other tile operations.
- Explicit waits, commits, arrives, and synchronization statements when they
  are part of the manual pipeline.
- Scalar `Bind` statements that read a buffer written inside the same pipeline
  body.

Do not count these:

- Buffer declarations or allocations at the beginning of the loop body.
- Replayable scalar aliases such as `base: T.int32 = ko * block_K`.
- Replayable scalar aliases that read only buffers not written by the pipeline
  body, such as `idx: T.int32 = Indices[ko]`.

The replayable-bind rule matters because the compiler may need the same scalar
alias at different logical pipeline iterations for different consumers.

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[0, 1],
    order=[1, 0],
):
    base: T.int32 = ko * block_K
    T.copy(A[base], A_shared)
    T.copy(A_shared, B[base])
```

The `stage` and `order` arrays contain two entries, not three. They annotate the
two copies. `base` is replayed before each use with that consumer's logical
pipeline index.

Replayable binds may depend on earlier replayable binds:

```python
base: T.int32 = ko * block_K
offset: T.int32 = base + tx
T.copy(A[offset], A_shared[tx])
```

The compiler replays `base`, then `offset`, then the consumer statement.

If a scalar bind reads a pipeline-written buffer, it is not replayable:

```python
T.copy(A[ko * block_K], A_shared)
value: T.float32 = A_shared[tx]
C[ko * block_K + tx] = value
```

Here `value` depends on `A_shared`, which is produced inside the pipeline. The
bind must be scheduled, and it must be in the same stage as every consumer that
uses it. If you need to load once and use the value across later stages,
materialize it in an explicit local or fragment buffer instead of relying on a
scalar alias.

Older code may include replayable scalar binds in `stage` and `order`. TileLang
accepts that form for compatibility and ignores those entries, but new code
should omit replayable binds from the annotation arrays.

## TMA And Async Copies

For ordinary `T.copy(global -> shared)`, software-pipelined loops are the path
that lets later passes use pipeline-managed async copy lowering where the target
supports it.

For explicit TMA copies, there are two patterns.

User-managed synchronization:

```python
mbar_A = T.alloc_barrier(128)

for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
    T.barrier_arrive(mbar_A)
    T.mbarrier_wait_parity(mbar_A, k % 2)
    T.gemm(A_shared, B_shared, C_local)
```

For TMA loads, `T.tma_copy` emits the transfer-side work. The user supplies the
arrive and wait. Pipeline lowering expands barrier buffers to the pipeline
depth and rewrites barrier indices and parity expressions.

Compiler-managed TMA pipeline barriers:

When pipeline planning classifies copy statements as TMA producers, the
software pipeline pass can rewrite them to share a pipeline barrier and insert
an `mbarrier_wait_parity` before the first consumer stage. This path is mainly
for compiler-planned TMA producer/consumer loops.

For `T.async_copy`, the operation lowers to cp.async-style copy and commit. It
does not by itself insert every wait a full algorithm may need, so keep waits
and synchronization explicit when writing manual async pipelines.

## Restrictions And Caveats

- A pipeline body must lower to a sequence of statements. If the whole body is
  wrapped in an `if`, the pipeline pass can handle an `if` without `else`; an
  `else` branch is rejected.
- `T.Pipelined` inside `T.Parallel` is rejected by TileLang's semantic checker.
- On ROCm targets other than `gfx950`, current pipeline planning strips
  `num_stages >= 1` and lowers the loop sequentially.
- Warp-specialized producer/consumer rewriting is a narrower path than ordinary
  `T.Pipelined`: current implementation is focused on pure TMA pipelines and
  has restrictions around mixed TMA/cp.async, conditionally guarded bodies, and
  multiple pipelined loops in the same block.
- The Python dtype and pipeline APIs are broader than backend support. Always
  validate low-precision and async pipelines on the target architecture.

## Checklist

- Use `T.Pipelined(..., num_stages=N)` first for regular tiled loops.
- Tune `num_stages`; do not assume larger is always faster.
- For manual `stage` and `order`, count only scheduled executable statements.
- Omit replayable scalar aliases from manual annotations.
- Keep every `order` unique.
- Keep buffer producers before consumers according to stage/order dependency
  rules.
- For explicit TMA, decide whether synchronization is user-managed or
  compiler-planned, then keep the barrier pattern consistent.
