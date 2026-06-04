# Code Review Checklist Subskill

Use this integrated subskill when the user wants a practical checklist or a
reusable review shape for CUDA source.

## Recommended Review Template

```text
Target Code Region:
Intent:
Correctness Posture:
- Indexing / bounds:
- Synchronization semantics:
- Numeric / dtype semantics:
- Validation path:

Thread-Data Mapping:
Memory Access Shape:
Reuse Structure:
Shared-Memory Layout:
Register / Residency Risk:
Control-Flow / Synchronization Risk:
Instruction / Datatype Path:
Architecture Build Target:

Likely Strengths:
Likely Risks:
Top Hypotheses:
Suggested Next Checks:
Need Profiling Next?:
```

## Checklist Discipline

- Separate correctness risks from performance risks.
- Keep hypotheses concrete and source-grounded.
- Prefer one or two likely issues over a laundry list.
- If runtime dominance is uncertain, say profiling is the next owner.
- If the code already looks structurally sound, say what is good instead of only
  listing dangers.

## When To Escalate

Escalate to `krnopt-cuda-profiling` when:

- several source-level explanations remain plausible
- end-to-end importance is unclear
- a suspected issue depends on measured stall or throughput evidence

## Quick Rules (Do / Don't)

A compact mnemonic layer for one-pass source review. Each rule maps to a
deeper argument in a sibling subskill; these are the checks to actually run
line by line.

### Memory access shape

- Do map neighboring lanes to nearby bytes.
- Don't let a warp touch far-apart 32-byte segments unless the algorithm
  truly forces it.
- Do use shared memory when it repairs global-memory access shape.
- Don't assume shared memory is automatically fast if the bank pattern is
  bad; check for stride-32 patterns on column-major reads.
- Do treat local memory as off-chip: arrays with dynamic indexing, large
  structs, or spills all behave like global memory from a performance
  point of view.

### Residency and occupancy

- Do keep occupancy high enough to hide latency.
- Don't chase occupancy if the tradeoff is register spilling or broken reuse.
- Do declare `__launch_bounds__` when a specific launch envelope is intended.
- Don't clamp registers via `-maxrregcount` blindly; it often just converts
  register pressure into local-memory traffic.

### Synchronization and streams

- Do use pinned host buffers for async overlap (`cudaMallocHost`).
- Don't expect pageable-host `cudaMemcpyAsync()` to overlap cleanly.
- Do use events and stream-scoped synchronization.
- Don't leave `cudaDeviceSynchronize()` in a tuned pipeline unless a global
  fence is genuinely needed.
- Do check both launch-time (`cudaGetLastError()`) and completion-time
  (`cudaStreamSynchronize`) errors.
- Don't treat `cudaSuccess` from the launch path as proof that kernel
  execution succeeded.

### Tensor-core and async-staging idioms

- Do create cooperative-group handles early and partition collectively.
- Don't create group handles conditionally.
- Do elect one thread for TMA-style bulk launches.
- Don't rely on a casual `threadIdx.x == 0` pattern when compiler visibility
  of the elected lane matters.
- Do pass tensor maps as `const __grid_constant__` parameters when possible.
- Don't re-plumb them through global memory unless necessary.
- Do store WMMA fragments through shared memory before crossing external
  interfaces.
- Don't pass opaque fragment types across mixed-architecture compilation
  boundaries.

### Control flow and dtype

- Do prefer warp-aligned conditions (`threadIdx.x / warpSize`) over
  lane-varying branches.
- Don't nest conditionals in hot inner loops; keep branch bodies short so
  predication can win.
- Do use `f` suffixes on literals in single-precision code.
- Don't let a stray `1.0` quietly promote part of the expression to FP64.
- Do use specialized intrinsics at specific call sites.
- Don't reach for `-use_fast_math` as a default; it is a blunt instrument.
- Do use architecture-specialized build targets such as `sm_90a` or `sm_100a`
  when the kernel depends on target-specific instructions or features.
- Don't compile those kernels only for general targets such as `sm_90` or
  `sm_100` and assume the specialized path will still be available.

### Legacy folklore

- Do ignore texture/surface-memory performance folklore for new code on
  current GPUs.
- Don't cargo-cult legacy advice into modern kernels; re-validate each
  rule against the current generation before propagating it.

## Canned Snippets For Reviews

When the review suggests a fix, point at the shape of the fix rather than
prescribing it blind. These snippets are useful reference shapes.

### Stream-scoped dependency instead of device-wide sync

```cpp
cudaStream_t s_copy, s_compute;
cudaEvent_t copied;
cudaStreamCreate(&s_copy);
cudaStreamCreate(&s_compute);
cudaEventCreate(&copied);

cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, s_copy);
cudaEventRecord(copied, s_copy);
cudaStreamWaitEvent(s_compute, copied);
kernel<<<grid, block, 0, s_compute>>>(...);
```

### Shared-memory tile with bank-conflict padding

```cpp
template <int TILE>
__global__ void tiled_copy(float* out, const float* in, int ld) {
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = in[y * ld + x];
  __syncthreads();
  out[y * ld + x] = tile[threadIdx.y][threadIdx.x];
}
```

### Elected-thread helper for TMA-style launches

```cpp
__device__ inline bool elected_copy_thread() {
  namespace ptx = cuda::ptx;
  unsigned warp_id = threadIdx.x / 32;
  unsigned uniform_warp_id = __shfl_sync(0xffffffff, warp_id, 0);
  return uniform_warp_id == 0 && ptx::elect_sync(0xffffffff);
}

if (elected_copy_thread()) {
  // launch one bulk copy
}
```

The snippets are shape references, not drop-in code. A review should cite
them when it is useful to show what the corrected structure looks like, not
as a substitute for understanding why the original shape was wrong.
