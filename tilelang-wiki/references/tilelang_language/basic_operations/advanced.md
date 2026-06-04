# Basic Operations: Advanced

This page covers lower-level buffer construction helpers and explicit math
intrinsics. Most kernels only need `T.Tensor`, `T.StridedTensor`, `T.fill`, and
`T.clear`.

## Buffer Proxy Helpers

`T.Buffer` is still exported for compatibility, but `T.Tensor` is the preferred
kernel argument annotation.

```python
T.Buffer(shape, dtype=T.float32, data=None, strides=None, scope="global")
T.Buffer[...]
T.Buffer.from_ptr(pointer_var, shape, dtype="float32", strides=None)
```

Use `from_ptr` when a pointer variable should be matched to a buffer view with a
known shape, dtype, and optional strides.

The scoped proxy classes use the same style but different default scopes:

```python
T.FragmentBuffer(shape, dtype="float32")  # local.fragment
T.SharedBuffer(shape, dtype="float32")    # shared.dyn
T.LocalBuffer(shape, dtype="float32")     # local
```

Most user kernels allocate these memory spaces with `T.alloc_fragment`,
`T.alloc_shared`, and `T.alloc_local`; the proxy forms are mainly useful for
explicit annotations and low-level helpers.

## Pointer Views

```python
T.ptr(dtype=None, storage_scope="global", *, is_size_var=False)
T.make_tensor(ptr, shape, dtype="float32", strides=None)
T.make_tensor_from_addr(addr, shape, dtype="float32", strides=None, storage_scope="global")
```

`T.ptr` creates a handle-typed TIR variable with optional pointed-to dtype and
storage scope. `T.make_tensor` creates a buffer view from a pointer variable or
pointer-valued expression. `T.make_tensor_from_addr` reinterprets an integer or
handle address as a typed pointer first, then creates the view.

Both `make_tensor` helpers require an active TileLang builder context, so use
them inside `@tilelang.jit` or `@T.prim_func` code.

## Fast-Math Intrinsics

```python
T.__log(x)
T.__log2(x)
T.__log10(x)
T.__tan(x)
T.__cos(x)
T.__sin(x)
T.__exp10(x)
T.__exp(x)
```

These convert `x` to a TIR expression and emit the corresponding `tl.__*`
intrinsic with the same dtype. Prefer public math functions such as `T.exp`,
`T.exp2`, `T.log`, and `T.sqrt` unless you specifically need these fast-math
intrinsic names.

## IEEE Rounded Intrinsics

```python
T.ieee_add(x, y, rounding_mode="rn")
T.ieee_sub(x, y, rounding_mode="rn")
T.ieee_mul(x, y, rounding_mode="rn")
T.ieee_fmaf(x, y, z, rounding_mode="rn")
T.ieee_frcp(x, rounding_mode="rn")
T.ieee_fsqrt(x, rounding_mode="rn")
T.ieee_frsqrt(x)
T.ieee_fdiv(x, y, rounding_mode="rn")
```

Valid rounding modes are `"rn"`, `"rz"`, `"ru"`, and `"rd"`. Invalid strings
raise `ValueError` in the Python wrapper. The result dtype follows the first
converted operand.

## Packed X2 Intrinsics

```python
T.add2(x, y)
T.sub2(x, y)
T.mul2(x, y)
T.fma2(x, y, z)
T.max2(x, y)
T.min2(x, y)
T.abs2(x)
```

Packed x2 intrinsics require `PrimExpr` inputs with dtype `float32x2`,
`bfloat16x2`, or `float16x2`. The wrapper raises `TypeError` for non-expression
inputs and `ValueError` for unsupported packed dtypes.
