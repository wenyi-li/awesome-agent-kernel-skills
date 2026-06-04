# Type System

TileLang dtypes are TVM `DataType` objects exposed through
`tilelang.language` as `T.float32`, `T.int32`, `T.bfloat16`, and similar names.
Use them to annotate kernel arguments, allocate buffers, cast scalar
expressions, and choose accumulation precision.

## Specifying Dtypes

Most dtype-taking APIs accept any dtype-like value that TileLang can normalize:

```python
dtype = T.float16
accum_dtype = T.float32

@T.prim_func
def main(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), torch.float16),
):
    ...
```

Accepted forms include:

- TileLang/TVM dtype objects: `T.float32`, `T.int8`, `T.bfloat16`.
- Canonical strings: `"float32"`, `"int8"`, `"bfloat16"`.
- Python scalar types: `bool`, `int`, `float`.
- NumPy dtypes: `np.float32`, `np.int32`, `np.dtype("float16")`.
- Torch dtypes: `torch.float32`, `torch.int8`, `torch.bfloat16`, and supported
  newer Torch FP8/FP4 dtypes when present in the installed Torch version.

Common aliases normalize to canonical names:

```text
T.float  -> float32
T.half   -> float16
T.double -> float64
T.int    -> int32
T.uint   -> uint32
T.long   -> int64
T.short  -> int16
```

For test inputs and integration code, convert a TileLang dtype back to Torch
with:

```python
torch_dtype = T.dtype(dtype).as_torch()
```

`T.float8_e4m3.as_torch()` is backend-sensitive: CUDA maps to the `fn` Torch
variant, while HIP maps to the `fnuz` variant when available.

## Tensor Annotations

Use `T.Tensor` for kernel parameters:

```python
@T.prim_func
def matmul(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    C: T.Tensor((M, N), T.float16),
):
    ...
```

`T.Tensor` defaults to global memory and row-major contiguous strides. A scalar
shape is promoted to a one-dimensional shape. The subscript form is also
supported and appears in examples:

```python
A: T.Tensor[[M, K], T.float16]
B: T.Tensor[(K, N), T.float16]
```

`T.Buffer` still exists for compatibility, but new TileLang code should prefer
`T.Tensor`.

Eager-style kernels can declare outputs with `T.empty`:

```python
C = T.empty((M, N), dtype)
```

## Buffer Allocation

Allocation helpers are typed wrappers around TileLang's block allocation
builder. This section keeps only the dtype-relevant parts; see
`language_basics.md` for the broader memory-scope model and `instructions.md`
for the fuller helper inventory.

Use the helper that matches the memory space:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
tmp = T.alloc_local((128,), T.float32)
flag = T.alloc_var(T.int32, init=0)
```

Common dtype-sensitive notes:

- `T.alloc_shared(shape, dtype, scope="shared.dyn")` keeps the requested dtype,
  but `bool` uses static `"shared"` scope because the shared-memory merge pass
  does not currently handle boolean dynamic shared buffers.
- `T.alloc_fragment(shape, dtype, scope="local.fragment")` is the common typed
  fragment/register allocation for GEMM accumulators and reductions.
- `T.alloc_var(dtype, init=None, scope="local.var")` casts numeric initializers
  to the requested dtype.
- Other helpers such as `T.alloc_local`, `T.alloc_global`, `T.alloc_barrier`,
  `T.alloc_cluster_barrier`, `T.alloc_tmem`, and `T.alloc_reducer` follow the
  same dtype normalization rules described on this page.

## Casts And Scalar Construction

Dtype objects are callable:

```python
x = T.float32(1)
y = T.int32(i)
v = T.float16x2(a, b)
```

Calling a dtype with one argument constructs or converts a scalar expression.
Calling a vector dtype with multiple arguments packs values through a TIR
shuffle, for example `T.bfloat16x2(a, b)`.

Use `T.cast` for numeric conversion:

```python
acc[i] = T.cast(src[i], T.float32)
```

Use `T.reinterpret(value, dtype)` for bit reinterpretation rather than numeric
conversion.

## Dtype Families

Common scalar types:

- Boolean: `bool`.
- Signed integers: `int4`, `int8`, `int16`, `int32`, `int64`.
- Unsigned integers: `uint8`, `uint16`, `uint32`, `uint64`.
- Floating point: `float16`, `bfloat16`, `float32`, `float64`.
- TensorFloat32: `tfloat32`.

Low-precision floating point:

- FP8: `float8_e3m4`, `float8_e4m3`, `float8_e4m3b11fnuz`,
  `float8_e4m3fn`, `float8_e4m3fnuz`, `float8_e5m2`,
  `float8_e5m2fnuz`, `float8_e8m0fnu`.
- FP6: `float6_e2m3fn`, `float6_e3m2fn`.
- FP4: `float4_e2m1fn`.

Many scalar families also expose vector-lane forms:

```text
int8x2, int8x4, ..., int8x64
uint32x2, uint32x4, ..., uint32x64
float16x2, float16x4, ..., float16x64
float8_e4m3fnx2, float8_e4m3fnx4, ...
float6_e2m3fnx8
float4_e2m1fnx16
bfloat16x2
tfloat32x2, tfloat32x4, ..., tfloat32x64
```

The Python dtype table is intentionally broader than what every backend can
lower. Treat low-precision and wide-vector names as frontend availability, then
check backend and architecture support.

## Backend Notes

- CUDA templates include half, bfloat16, and FP8 support through CUDA/Cutlass
  helpers. E8M0 FP8 support depends on newer CUDA headers.
- CUDA FP4 support is guarded by architecture support, and vectorized FP4 tests
  require recent compute capability.
- HIP supports FP8 E4/E5 `fn`/`fnuz` variants, but E8M0 is marked unsupported
  in the HIP template.
- HIP FP4 support is guarded to `gfx950`.
- Examples that want portable FP8 selection use
  `tilelang.utils.target.determine_fp8_type()` rather than hard-coding one FP8
  spelling.

## Mixed Precision

Choose input, output, and accumulation dtypes independently:

```python
dtype = T.float16
accum_dtype = T.float32

A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.gemm(A_shared, B_shared, C_local)
```

For FP8 GEMM examples, inputs are commonly FP8, scale tensors are `T.float32`,
accumulators are `T.float32`, and output may be `T.bfloat16` or `T.float32`.
Keep accumulation dtype explicit; do not rely on an operator default when
numerical behavior matters.
