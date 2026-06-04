# Hardware-Aware Plan Subskill

Use this integrated subskill when the user wants a practical output instead of
free-form architecture discussion.

## Recommended Plan Template

```text
Target Hardware:
Compute Capability:
Required Build Target:
Kernel / Workload Type:
Current Question:
Utilization Target:
Useful-Work Waste Source:

Candidate Hardware Features:
- Feature:
  Why it may matter:

Generation Branch:
- SM90 / Hopper:
- SM100 / B200:
- Other / unresolved:

Feature-To-Code Mapping:
- Feature:
  Required code shape:
  Required contracts:

Architecture-Specific Contract To Verify Next:
Hardware-Utilization Confirmation Metric:
Hardware-Utilization Rejection Metric:
Likely Wins:
Likely Risks / Traps:
Fallback Path:
Likely CUTLASS / Source Layer To Inspect Next:
Need Profiling Next?:
Need Generic CUDA Source Review Next?:
Suggested Next Step:
```

## Plan Discipline

- Keep feature choice tied to workload shape.
- Separate likely wins from proven wins.
- Call out contracts explicitly: layouts, clusters, barriers, descriptor rules,
  occupancy tradeoffs, or dtype requirements.
- When utilization is the goal, name the limiting hardware path and useful-work
  waste source. Use occupancy only as supporting evidence, not as the objective.
- Call out the architecture-specialized build target when the feature depends
  on it, such as `sm_90a` or `sm_100a`; do not let a generic `sm_90` or
  `sm_100` build stand in for the specialized path.
- Make the generation branch explicit when SM90/Hopper and SM100/B200
  datacenter Blackwell would suggest different kernel structures. Never
  collapse them into "modern GPU."
- Do not recommend TMEM-, CLC-, or block-scaled-specific rewrites until the
  compute capability is resolved; consumer Blackwell (CC 12.x) is not the
  same surface as datacenter SM100.
- Be explicit about instruction names where it matters:
  `wgmma` vs `tcgen05.mma.kind::f8f6f4` vs `mxf8f6f4.block_scale`; and
  specify `cta_group::1` vs `cta_group::2` if 2-CTA mode is on the table.
- Be explicit about scale-layout contracts when a block-scaled path is
  proposed: block size, scale dtype, packed `UE8M0`, GEMM-swizzled vs
  contiguous, and whether the contest format matches the library path.
- Be explicit about scheduler mode: static, static persistent, CLC (SM100
  only), Stream-K, or PDL across kernels.
- Tie every utilization claim to a measurable signal: tensor/MMA issue or
  active cycles, memory bytes per useful output, L2 locality, SMEM/MIO stalls,
  register spills, TMEM readback pressure, eligible warps, launch gaps, or
  end-to-end wall time.
- If a feature is uncertain, say what needs to be verified.
- Say when the next owner should be `krnopt-cuda-profiling` or
  `krnopt-cuda-coding`.

## Worked Example Sections To Include

When the plan covers an SM100 rewrite, include at minimum:

- the proposed `tcgen05.mma.kind::*` variant and whether it is 1SM or 2SM
- whether accumulators move to TMEM and the approximate column budget
- the chosen staging path (TMA, 2SM TMA, `tma_gather4`, tensormap
  replacement, or a `cp.async` residual)
- the scheduler mode and whether it is CLC-backed
- the precision / scale-layout contract and which backend is expected to
  satisfy it
- the fallback path if the architecture-specific route is a poor fit
