# GEMM Operations: Advanced

This page covers explicit GEMM variants and shape caveats. Start with
`T.gemm` or `T.gemm_sp` unless you need manual synchronization, sparse paths, or
Blackwell tensor-memory behavior.

## Shape Checks

Dense `T.gemm` checks these constraints before lowering:

- `C` is rank 2 and supplies `(M, N)`.
- `A` and `B` have rank at least 2. If either has leading dimensions, those
  leading extents must be `1`; the final two dimensions are the matrix tile.
- With `transpose_A=False`, `A` is `(M, K)`; with `transpose_A=True`, `A` is
  `(K, M)`.
- With `transpose_B=False`, `B` is `(K, N)`; with `transpose_B=True`, `B` is
  `(N, K)`.
- The first matrix-dimension offset of `A` and `B` must be `0`. The second
  matrix-dimension offset is forwarded to the backend.

Sparse `T.gemm_sp` follows the same rank and offset style, but its logical K is
`2 * compressed_K` from `A_sparse`.

## Explicit Hopper WGMMA

```python
T.wgmma_gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)

T.wgmma_gemm_sp(
    A_sparse,
    E,
    B,
    C,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)
```

These APIs request Hopper WGMMA lowering and do not emit the implicit
warp-group wait used by the high-level synchronous path. Use them only when the
surrounding schedule explicitly manages WGMMA completion. If the target or
operand pattern cannot use WGMMA, compilation fails instead of falling back.

## Explicit Blackwell TCGEN05

```python
T.tcgen05_gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    *,
    mbar,
    use_2cta=False,
)
```

`T.tcgen05_gemm` requests Blackwell TCGEN05 lowering and requires an mbarrier.
It issues the operation without the implicit mbarrier wait, so the schedule must
wait before consuming the tensor-memory result. With `use_2cta=True`, each CTA
provides half of `N` from `B`; the wrapper checks `N_B * 2 == N_C` and expects a
matching 2-CTA cluster launch.

The sparse form has the same explicit-asynchronous contract:

```python
T.tcgen05_gemm_sp(
    A_sparse,
    E,
    B,
    C,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
)
```

## Block-Scaled TCGEN05

```python
T.tcgen05_gemm_blockscaled(
    A,
    B,
    C,
    SFA_tmem,
    SFB_tmem,
    transpose_A=False,
    transpose_B=False,
    clear_accum=False,
    wg_wait=0,
    mbar=None,
    sf_a_id=0,
    sf_b_id=0,
    *,
    use_2cta=False,
)
```

This is the explicit Blackwell block-scaled path. `A` and `B` are FP8 shared
memory operands, `C` is the tensor-memory accumulator, and `SFA_tmem` /
`SFB_tmem` are E8M0 scale factors already in tensor memory. The wrapper requires
`mbar` and always uses `GemmWarpPolicy.Square`.

When copying a block-scaled accumulator out of tensor memory, annotate the
tensor-memory layout:

```python
layout = T.make_blockscaled_gemm_layout(C_tmem, A_shared, transpose_A=False)
T.annotate_layout({C_tmem: layout})
```

Use the block-scaled path only for kernels already structured around Blackwell
tensor memory, explicit barriers, and scale-factor movement. It is not a drop-in
replacement for ordinary `T.gemm`.
