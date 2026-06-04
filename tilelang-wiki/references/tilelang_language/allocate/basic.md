# Allocation Basics

TileLang kernels allocate buffers in explicit memory scopes. The common helpers
are `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_local`, `T.alloc_var`, and
`T.empty`.

## Tile Buffers

Use the helper that matches where the data should live:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_frag = T.alloc_fragment((block_M, block_N), T.float32)
tmp = T.alloc_local((4,), dtype)
```

`T.alloc_shared(shape, dtype, scope="shared.dyn")` allocates block-visible
shared memory. The default scope is dynamic shared memory. If `dtype == "bool"`,
TileLang uses static shared memory scope internally because bool dynamic shared
buffers are not handled by the shared-memory merge path.

`T.alloc_fragment(shape, dtype, scope="local.fragment")` is the usual
accumulator or register-style storage for tile operations. GEMM and attention
examples use fragments for accumulators and row-wise softmax state.

`T.alloc_local(shape, dtype, scope="local")` allocates thread-local storage.
It appears in lower-level kernels that explicitly manage per-thread data rather
than using tile operators throughout.

## Scalar Variables

`T.alloc_var` allocates a one-element local variable buffer.

```python
counter = T.alloc_var("int32", init=0)
scale = T.alloc_var(T.float32, 1.0)
legacy = T.alloc_var("int32", "local.var")
explicit = T.alloc_var("int32", 1, "local.var")
```

Accepted forms include:

- `T.alloc_var(dtype)`
- `T.alloc_var(dtype, init)`
- `T.alloc_var(dtype, scope_string)`
- `T.alloc_var(dtype, init, scope_string)`
- `T.alloc_var(dtype, scope=..., init=...)`

If both a positional initializer and `init=` are supplied, TileLang raises a
`TypeError`. If a scope is supplied positionally, it must be a string.

## Eager Outputs

Use `T.empty(...)` to declare an output tensor in eager-style `@tilelang.jit`
kernels, then return it from the JIT function.

```python
C = T.empty((M, N), dtype)
D = T.empty(M, N, dtype=dtype)
```

The shape can be a tuple/list or variadic dimensions. The compatibility form
`T.empty((M, N), "float16")` is also accepted.
