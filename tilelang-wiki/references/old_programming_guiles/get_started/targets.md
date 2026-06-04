# Understanding Targets

TileLang uses TVM targets to describe the device code it should generate. The target
selects the code generator, carries architecture options such as CUDA `arch` or ROCm
`mcpu`, and constrains which Python execution backend can wrap the compiled kernel.

There are two separate questions:

- What target can TileLang parse or lower source for?
- Which target/backend pairs are runnable through `tilelang.compile` or `@tilelang.jit`
  in the current JIT adapters?

Keep those separate when debugging. `llvm` and `webgpu`, for example, appear in the
accepted target list, but the normal runnable CPU path in the current examples and
tests is the C target (`target="c"`) with a JIT execution backend.

## Common Target Names

TileLang exposes `tilelang.utils.target.SUPPORTED_TARGETS` through
`describe_supported_targets()`. These are the accepted base names:

| Name | What it means in TileLang |
| --- | --- |
| `auto` | Detect a target. First respects an active `tvm.target.Target.current()`. On ROCm PyTorch builds it prefers HIP when ROCm is available; otherwise it checks CUDA, then HIP, then Metal. |
| `cuda` | NVIDIA CUDA target. Use a dict for architecture options, for example `{"kind": "cuda", "arch": "sm_90"}`. |
| `hip` | AMD ROCm/HIP target. Use a dict for architecture options, for example `{"kind": "hip", "mcpu": "gfx942"}`. TileLang adds ROCm attributes such as `mtriple` and warp size when it can infer them. |
| `metal` | Apple Metal target for arm64 Macs. |
| `c` | C source backend. This is the practical CPU-style JIT path used by TileLang tests. |
| `cutedsl` | CuTe DSL variant of CUDA. TileLang normalizes this to a CUDA TVM target with a `cutedsl` key; it is not a separate low-level TVM target kind. |
| `llvm` | Accepted by target parsing and some lower/codegen paths, but not the usual runnable TileLang JIT CPU path. Prefer `target="c"` for runnable CPU-oriented examples. |
| `webgpu` | Accepted by target parsing and source/codegen paths, but not a primary runnable JIT path in the current adapters. |

You can inspect the list at runtime:

```python
from tilelang.utils.target import describe_supported_targets

for name, doc in describe_supported_targets().items():
    print(f"{name:>7}: {doc}")
```

## Choosing A Target

Most CUDA examples can omit the target and let `auto` detect the current machine:

```python
kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
```

Pass a target when you need a specific architecture or backend:

```python
target = {"kind": "cuda", "arch": "sm_90"}

kernel = matmul.compile(
    M=1024,
    N=1024,
    K=1024,
    block_M=128,
    block_N=128,
    block_K=32,
    target=target,
    execution_backend="tvm_ffi",
)
```

The same options can be attached to `@tilelang.jit`:

```python
@tilelang.jit(target={"kind": "hip", "mcpu": "gfx942"}, execution_backend="cython")
def kernel(A, B, block_M, block_N, block_K):
    ...
```

For CuTe DSL, use `target="cutedsl"` or `{"kind": "cutedsl", "arch": "sm_90"}`.
TileLang converts that request to a CUDA target with the `cutedsl` key and requires
the CuTe DSL package to be available.

## Runnable Backend Matrix

`target` controls code generation. `execution_backend` controls how the generated
kernel is compiled, loaded, and called from Python. The resolver currently allows:

| Target kind | Allowed execution backends |
| --- | --- |
| CUDA | `tvm_ffi`, `nvrtc`, `cython` |
| HIP | `tvm_ffi`, `cython` |
| Metal | `tvm_ffi`, `torch` |
| C | `cython`, `tvm_ffi` |
| CuTe DSL CUDA target | `cutedsl` only |

If `execution_backend` is omitted or set to `"auto"`, TileLang chooses a default:
`tvm_ffi` for CUDA, HIP, and Metal; `cython` for C and fallback targets; `cutedsl`
for CuTe DSL targets.

The public `tilelang.compile` API accepts a TileLang `PrimFunc`, merges function-level
attributes such as output indices, pass configs, and compile flags, then delegates to
the cache/JIT path. `@tilelang.jit` wraps a Python function, infers lazy or eager mode,
and caches compiled kernels per argument specialization.

## Environment Defaults

The cache layer reads these environment variables when explicit arguments are not
provided:

| Variable | Meaning |
| --- | --- |
| `TILELANG_TARGET` | Default target, usually `"auto"`. |
| `TILELANG_EXECUTION_BACKEND` | Default execution backend, usually `"auto"`. |
| `TILELANG_VERBOSE` | Enables verbose compilation logging when truthy. |

The persistent kernel cache key includes the target, target host, execution backend,
pass configs, compile flags, TileLang version, platform information, and native
library stamp. If any of those change, TileLang should compile a fresh kernel.

## Architecture Options

For CUDA, pass a compute capability as `arch`:

```python
target = {"kind": "cuda", "arch": "sm_80"}   # A100-class target
target = {"kind": "cuda", "arch": "sm_90"}   # H100-class target
```

For ROCm, pass the GPU architecture as `mcpu`:

```python
target = {"kind": "hip", "mcpu": "gfx942"}
target = {"kind": "hip", "mcpu": "gfx950"}
```

When `auto` can query the current device through PyTorch, TileLang tries to fill these
architecture options automatically. Supplying them explicitly is still useful for
offline compilation, reproducible benchmarking, or autotuning.

## Troubleshooting

- If target construction fails, prefer a target config dict over a command-line-style
  string. For example, use `{"kind": "cuda", "arch": "sm_80"}`.
- If `execution_backend` fails validation, check the backend matrix above and try
  `execution_backend="auto"`.
- If CUDA reports that no kernel image is available, the selected `arch` probably
  does not match the GPU running the kernel. Drop the architecture option or use the
  correct compute capability.
- If a target appears in `describe_supported_targets()` but does not run through
  `tilelang.compile`, check whether a JIT adapter exists for that target/backend pair.
  Accepted target parsing and runnable Python execution are related but not identical.
