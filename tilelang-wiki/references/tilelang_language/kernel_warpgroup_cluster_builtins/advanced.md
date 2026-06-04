# Kernel, Warpgroup, And Cluster Builtins: Advanced

This page collects lower-level launch, warpgroup, cluster, and architecture
helpers. Use these APIs when the high-level tile operators do not expose the
control you need.

## Launch Frame Details

`T.Kernel` has the full form:

```python
T.Kernel(
    *blocks,
    threads=None,
    cluster_dims=None,
    is_cpu=False,
    prelude=None,
)
```

`is_cpu=True` creates a CPU-style frame and skips thread bindings. `prelude=`
injects C/CUDA source into the generated kernel through an import-C pragma.
Keep prelude text small and local to declarations that must be visible to the
generated device code.

`T.import_source(source)` attaches import-C source directly. Passing `None` is a
no-op.

## External CUDA Source Kernels

`T.CUDASourceCodeKernel` launches an external CUDA `__global__` function from
inline source text or a source file path:

```python
T.CUDASourceCodeKernel(
    blocks,
    threads=128,
    source_code_or_path=cuda_src,
    entry_name="main_kernel",
)
```

If `source_code_or_path` names an existing file, TileLang reads that file.
Otherwise, strings containing markers such as `__global__`, `extern "C"`,
`#include`, or a newline are treated as inline CUDA source. Path-like strings
ending in CUDA/C/C++ source suffixes fail if the file is missing.

Use this as an interop escape hatch. The CUDA body is outside normal TileLang
DSL analysis.

## Warp Specialization And WGMMA Ordering

`T.ws(*warp_group_idx)` enters one or more 128-thread warp-specialization
groups:

```python
with T.Kernel(num_blocks, threads=256) as bid:
    with T.ws(0):
        ...
    with T.ws(1):
        ...
```

At least one group id is required. The warpgroup size is fixed at 128, so this
is NVIDIA-oriented and requires enough launched threads for the selected
groups.

Low-level WGMMA ordering helpers are:

```python
T.warpgroup_arrive()
T.warpgroup_commit_batch()
T.warpgroup_wait(num_mma)
T.wait_wgmma(id)
T.warpgroup_fence_operand(buffer_or_ptr, offset=0, num_regs=None, dtype=None)
```

`T.warpgroup_fence_operand` can infer register count from a fully static buffer
shape. For a `BufferLoad`, symbolic region, or raw pointer, provide
`num_regs`; for raw pointers, provide `dtype` unless TileLang can infer it from
the pointer expression.

Register allocation helpers emit target-sensitive hints:

```python
T.set_max_nreg(reg_count, is_inc)
T.inc_max_nreg(reg_count)
T.dec_max_nreg(reg_count)
T.no_set_max_nreg()
T.disable_warp_group_reg_alloc()
T.annotate_producer_reg_dealloc(reg_count=24)
T.annotate_consumer_reg_alloc(reg_count=240)
```

Validate these on the exact target. They can change occupancy, spilling, and
warp-specialized producer/consumer balance.

## Cluster Synchronization And Launch Control

The individual cluster barrier pieces are:

```python
T.cluster_arrive_relaxed()
T.cluster_arrive()
T.cluster_wait()
T.cluster_sync()
```

Cluster mbarrier arrival can target a peer CTA:

```python
T.mbarrier_arrive(mbarrier, cta_id)
T.ptx_arrive_cluster_barrier(mbarrier, cta_id)
```

Passing `cta_id` requires a barrier allocated in `shared.cluster_barrier`
scope. Without `cta_id`, `T.mbarrier_arrive` arrives on the current CTA's
barrier.

Cluster launch control helpers are specialized primitives:

```python
T.clc_try_cancel(result, mbarrier)
T.clc_try_cancel_multicast(result, mbarrier)
T.clc_is_canceled(result)
T.clc_get_first_ctaid_x(result)
T.clc_get_first_ctaid_y(result)
T.clc_get_first_ctaid_z(result)
```

`result` is written by the try-cancel operation and read by the query helpers.
`mbarrier` is used read-write. Treat these as advanced persistent-kernel or
programmatic scheduling tools, not as ordinary cluster-copy APIs.

## Blackwell And TCGEN05 Helpers

High-level SM100 GEMM operators should be the default interface. The low-level
TCGEN05 helpers exist for kernels that need explicit UMMA/TMEM control:

```python
T.initialize_tcgen05_descriptor(...)
T.initialize_wgmma_descriptor(...)
T.increase_descriptor_offset(descriptor, offset)
T.tcgen05_mma_arrive(mbar, arrive_2cta=False)
T.tcgen05_before_thread_sync()
T.tcgen05_after_thread_sync()
T.tcgen05_cp_warpx4(smem_src, tmem_dst, tmem_col_offset=0, use_2cta=False)
T.tcgen05_sf_warp_transpose(smem_src)
T.deallocate_tmem(tmem)
```

Descriptor arguments must be descriptor buffers or descriptor buffer loads.
`T.deallocate_tmem` accepts only buffers allocated in `shared.tmem` scope and
makes the buffer lifetime user-managed from that point onward. Follow the
hardware rule that the same warp that allocated TMEM should deallocate it.

`T.ptx_mma_sm70(...)` is a legacy low-level SM70 tensor-core intrinsic wrapper.
Prefer current tile GEMM operators unless maintaining a specialized path.

