# Advanced Allocation

This page covers allocation helpers used for explicit workspaces,
synchronization, reducers, descriptors, and target-specific tensor memory.

## Global Workspace

`T.alloc_global(shape, dtype, scope="global")` allocates a global workspace
through backend APIs.

```python
workspace = T.alloc_global((num_splits, block_M), T.float32)
```

Prefer passing a workspace tensor from the host when the framework should own
allocation and lifetime. `T.alloc_global` is mainly for specialized kernels and
testing, and it is not available on every backend.

## Barriers

`T.alloc_barrier` and `T.alloc_cluster_barrier` allocate shared-memory barrier
buffers.

```python
loaded = T.alloc_barrier([128, 128])
remote_done = T.alloc_cluster_barrier([1])
```

An integer creates one barrier. A list creates one barrier per element and
records each arrival count. `T.alloc_barrier` uses `shared.barrier` scope.
`T.alloc_cluster_barrier` uses `shared.cluster_barrier` scope.

Barrier buffers are for kernels that explicitly manage asynchronous TMA or
cluster synchronization. They are not needed for ordinary `T.copy` inside a
compiler-managed `T.Pipelined` loop.

## Tensor Memory

`T.alloc_tmem(shape, dtype)` allocates Blackwell tensor memory in
`shared.tmem` scope.

```python
C_tmem = T.alloc_tmem((block_M, block_N), T.float32)
```

The shape must be two-dimensional. Use tensor memory only with SM100/Blackwell
kernel patterns that also use the matching TCGEN5 instructions and lifetime
rules. Ordinary GEMM accumulators should start with `T.alloc_fragment`.

## Descriptor Buffers

Descriptor helpers allocate one-element local descriptor buffers for WGMMA and
TCGEN5 paths:

```python
wgmma_desc = T.alloc_wgmma_desc()
smem_desc = T.alloc_tcgen05_smem_desc()
instr_desc = T.alloc_tcgen05_instr_desc()
generic = T.alloc_descriptor("wgmma")
```

Public descriptor kinds are `"wgmma"`, `"tcgen05_smem"`, and
`"tcgen05_instr"`. These are low-level building blocks for kernels that already
manage descriptor construction explicitly.

## Reducers

`T.alloc_reducer(shape, dtype, op="sum", replication=None)` creates a
`local.fragment` reducer buffer annotated with reducer metadata.

```python
partial = T.alloc_reducer((block_M,), T.float32, op="sum", replication="all")
```

Valid operations are `"sum"`, `"max"`, and `"min"`. Valid replication values
are `"all"` and `"none"`; omitted replication currently defaults to `"none"`.

Update the reducer according to its operation: sum reducers use addition,
max reducers use `T.max`, and min reducers use `T.min`. Fill with the matching
identity value before reducing, then call `T.finalize_reducer` before reading
the finalized partial result.
