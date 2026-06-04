# Redesign Family Map

Use this reference when the input already suggests the problem is structural and
the remaining question is which redesign family best matches it.

## Mismatch To Redesign Family

| Structural mismatch | Likely redesign family |
| --- | --- |
| Boundary mismatch | fusion, boundary elimination, epilogue or prologue fusion, library handoff |
| Scheduler mismatch | persistent scheduling, queues, static batching, retiled ownership |
| Primitive mismatch | grouped GEMM, block-sparse reformulation, tuned-library substitution |
| Pipeline mismatch | producer-consumer overlap, warp specialization, staged buffers, pipelined epilogue |
| Metadata mismatch | index-driven execution, prefix maps, descriptor pools, compact schedule metadata |

## Symptom To Redesign Family

| Symptom cluster | Likely redesign family |
| --- | --- |
| Many tiny launches, host gaps, or sync-heavy phase boundaries | fusion, batching, CUDA Graphs, persistent execution |
| Irregular work leaves the machine underfilled | persistent scheduling, queues, grouped execution, different tile ownership |
| Padding, permute, scatter, gather, or combine buffers dominate bytes moved | layout redesign, fused producer-consumer path, index-driven execution |
| Coalescing is fixed but the kernel still streams too much global data | reuse and staging redesign, cache-aware ordering, decomposition |
| Shared-memory sweep patterns and barriers dominate | warp-level reductions, shorter barrier domains, different reduction tree, pipelined schedule |
| GEMM-like code misses tensor hardware or spends effort reimplementing dense math | library handoff, grouped GEMM reformulation, tile and dtype recast |
| The kernel is near a real memory roofline after earlier fixes | decomposition, algorithm change, or tuned-library substitution |
| Descriptor churn or setup metadata dominates irregular grouped work | compact metadata, descriptor pooling, different staging strategy |
| Extra postprocess kernels exist mainly to scale, gate, dequant, or combine already-hot data | epilogue or prologue fusion |

## Selection Rules

- Prefer the redesign family that removes the measured boundary, not the one
  that sounds most advanced.
- Prefer a redesign that changes one structural cause at a time.
- Prefer tuned-library handoff when the custom code is reproducing a commodity
  primitive with worse math, scheduling, or instruction selection.
- Prefer fusion only when it removes real movement or launch overhead rather
  than just combining source files.
- Prefer persistent scheduling when the work is irregular enough that static
  launch geometry keeps underfilling the device.
- Prefer metadata compaction when full payload copies exist mainly to encode
  schedule information.
- Prefer decomposition when the remaining kernel is already close to a real
  roofline and local tuning headroom is thin.

## Stop Conditions

Do not push a structural redesign when:

- the problem is still just hotspot discovery
- one small local fix has not been tried yet and clearly matches the evidence
- the proposed redesign lacks a measurable success signal
- the redesign is really a hardware-feature choice rather than a structural one

Do push toward structural redesign when:

- the same local class of fix has already moved the bottleneck sideways twice
- the hot path spends more on boundaries than on useful math
- the scheduler clearly does not match workload irregularity
- the current primitive is obviously the wrong abstraction for the work
