# Magic Enums

Enums that matter in practice.

## `T.GemmWarpPolicy`

`T.GemmWarpPolicy` is the warp-partition policy for `T.gemm`,
`T.wgmma_gemm`, and `T.tcgen05_gemm`. It controls how warps in one CTA split a
GEMM output tile across the `M x N` axes. It does not change correctness; it
changes work partitioning after lowering and therefore affects performance.

The public values are:

- `T.GemmWarpPolicy.Square`
- `T.GemmWarpPolicy.FullRow`
- `T.GemmWarpPolicy.FullCol`

`T.gemm(...)` defaults to `T.GemmWarpPolicy.Square`.

## Values

### `T.GemmWarpPolicy.Square`

Default policy. It keeps warp allocation relatively balanced across `M` and
`N`. Use it for square or near-square tiles.

```python
block_M = 128
block_N = 128
T.gemm(a_shared, w_shared, acc, policy=T.GemmWarpPolicy.Square)
```

### `T.GemmWarpPolicy.FullRow`

Bias warp allocation toward the `M` axis:

```python
m_warp = num_warps
n_warp = 1
```

Use it for tall tiles or row-oriented kernels.

```python
block_M = 256
block_N = 64
T.gemm(a_shared, w_shared, acc, policy=T.GemmWarpPolicy.FullRow)
```

### `T.GemmWarpPolicy.FullCol`

Bias warp allocation toward the `N` axis:

```python
m_warp = 1
n_warp = num_warps
```

Use it for wide tiles.

```python
block_M = 64
block_N = 256
T.gemm(a_shared, w_shared, acc, policy=T.GemmWarpPolicy.FullCol)
```

## Usage

Pass it as the `policy` argument:

```python
T.gemm(
    a_shared,
    w_shared,
    acc,
    policy=T.GemmWarpPolicy.Square,
)
```

The same enum is also used by `T.wgmma_gemm(...)` and `T.tcgen05_gemm(...)`.

## Heuristic

Use `Square` when `block_M` and `block_N` are similar, `FullRow` when `block_M`
is much larger than `block_N`, and `FullCol` when `block_N` is much larger than
`block_M`. This is only a starting point: the best choice also depends on block
sizes, warp count, dtype, target GPU, lowering path, and epilogue cost.

## Fused Kernels

For fused kernels such as `gemm_residual_rmsnorm_gemm_swiglu`, include
`policy` in autotuning instead of hard-coding it:

```python
policy_options = [
    T.GemmWarpPolicy.Square,
    T.GemmWarpPolicy.FullRow,
    T.GemmWarpPolicy.FullCol,
]
```

Once the epilogue becomes significant, GEMM shape alone is not enough to pick
the best policy.

## Default Choice

For a square tile such as:

```python
block_M = 128
block_N = 128
threads = 128
```

the default is usually a reasonable starting point:

```python
T.gemm(a_shared, w_shared, acc, policy=T.GemmWarpPolicy.Square)
```
