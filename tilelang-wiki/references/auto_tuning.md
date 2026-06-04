# Auto Tuning

Use autotuning when a TileLang kernel has explicit schedule parameters such as
tile sizes, pipeline stages, thread count, rasterization, or warp policy, and
you want TileLang to compile and benchmark a candidate set.

For exact API behavior, use:

- `tilelang/autotune/basic.md` for decorator and `AutoTuner.from_kernel(...)`
  usage.
- `tilelang/autotune/advanced.md` for validation, captured inputs, grouped
  compile, timeouts, multi-GPU benchmarking, and cache behavior.
- `jit_autotune.md` for the short compile/tuning workflow.

## Basic Flow

1. Write the kernel factory with tunable keyword parameters.
2. Build a list of config dictionaries whose keys match those parameters.
3. Compile and benchmark the configs through `@tilelang.autotune(...)` or
   `AutoTuner.from_kernel(...)`.
4. Validate candidates against a reference unless you deliberately set
   `skip_check=True`.
5. Use the returned best `JITKernel` and record the best config.

## Minimal Pattern

```python
import itertools
import tilelang
import tilelang.language as T


def get_configs():
    keys = ["block_M", "block_N", "block_K", "num_stages", "threads"]
    values = [
        [64, 128],
        [64, 128],
        [32, 64],
        [2, 3],
        [128, 256],
    ]
    return [dict(zip(keys, cfg)) for cfg in itertools.product(*values)]


@tilelang.autotune(configs=get_configs(), warmup=3, rep=20)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=2, threads=128):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


best_kernel = matmul.compile(4096, 4096, 4096)
```

The config keys must match factory parameters. If the caller supplies every
tunable value explicitly, the decorator has nothing to search and compiles that
fixed configuration.

## Programmatic Pattern

Use `AutoTuner.from_kernel(...)` when you need explicit compile/profile
arguments or the full result object.

```python
import tilelang as tl
from tilelang.autotuner import AutoTuner


result = (
    AutoTuner.from_kernel(kernel=kernel, configs=configs)
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=reference,
        skip_check=False,
        backend="event",
    )
    .run(warmup=3, rep=20, timeout=30)
)

best_kernel = result.kernel
best_config = result.config
best_latency = result.latency
```

Use `set_autotune_inputs(...)` for kernels whose correctness depends on real
metadata tensors, masks, packed offsets, grouped-GEMM descriptors, or varlen
inputs. Generated random inputs are enough only when the kernel shape and data
contract are simple.

## Config Design

Keep the first search space small and meaningful:

- Tune tile sizes that affect memory reuse: `block_M`, `block_N`, `block_K`.
- Tune pipeline depth: `num_stages`.
- Tune launch width: `threads` or `thread_num`.
- Tune architecture-specific flags only after the baseline is correct.
- Avoid combining many weak knobs into a huge Cartesian product.

For GEMM-like kernels, Carver/Roller hints can generate candidate tile shapes,
but treat them as a starting point. Filter hints against the target GPU,
available shared memory, legal thread count, and the kernel's dtype path.

## Debugging Notes

- Disable autotune result caching with `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`
  when debugging config search.
- Also disable the regular JIT cache with `TILELANG_DISABLE_CACHE=1` when you
  need to force recompilation of source changes.
- Keep `skip_check=False` until a trusted reference path has validated the
  kernel family.
- Use `timeout` to avoid spending the whole run on broken or slow candidates.
