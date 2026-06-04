"""Reusable torch.profiler template for TileLang kernels.

Copy this file next to a kernel script and replace the marked kernel section.
Outputs:
  - trace.json for chrome://tracing or https://ui.perfetto.dev/
  - a key_averages summary table on stdout
"""

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 1. Replace this example kernel with the kernel under study.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def lookup_gpu_us_per_call(prof, name):
    """Return microseconds per call for the GPU-attributed event row."""
    for evt in prof.key_averages():
        if evt.key == name and evt.self_device_time_total > 0 and evt.count:
            return evt.self_device_time_total / evt.count
    return None


def main():
    M, N, K = 4096, 4096, 4096

    kernel = matmul(M, N, K, 128, 128, 32)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Warm up so JIT compile, module loading, and allocator first-touch do not
    # contaminate the measured trace.
    for _ in range(10):
        _ = kernel(a, b)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(20):
            with record_function("tilelang_matmul"):
                c = kernel(a, b)
            with record_function("torch_matmul"):
                ref = a @ b
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    prof.export_chrome_trace("trace.json")
    print("Wrote trace.json")

    for label, flops in (
        ("tilelang_matmul", 2 * M * N * K),
        ("torch_matmul", 2 * M * N * K),
    ):
        us = lookup_gpu_us_per_call(prof, label)
        if us:
            tflops = flops / (us * 1e-6) / 1e12
            print(f"{label}: {us / 1e3:.3f} ms/call, {tflops:.1f} TFLOPS")

    _ = c, ref


def main_scheduled():
    """Stub for loop-based profiling.

    For training or multi-step inference, replace the simple profile block above
    with a scheduled profiler so only a stable window is captured.
    """
    raise NotImplementedError("Adapt this template for your loop-based workload.")


if __name__ == "__main__":
    main()
