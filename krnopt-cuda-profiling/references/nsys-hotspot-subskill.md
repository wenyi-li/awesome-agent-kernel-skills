# nsys Hotspot Subskill

Use this integrated subskill when the question is "where is time going?" or
"is the problem above the kernel body?"

## nsys Owns The Broad View

Use `nsys` first for:

- top kernels by total GPU time
- launch counts
- host-side gaps between launches
- CUDA API overhead and synchronization
- memory transfer timing
- NVTX phase summaries

Representative command shapes:

```bash
nsys profile --trace=cuda,nvtx,osrt -o report ./program
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
nsys stats report.nsys-rep --report cuda_api_sum
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum
nsys stats report.nsys-rep --report nvtx_sum
```

For this repository, when the profiled program is `mlsys-cli eval timing`, use
the persistent runner:

```bash
nsys profile --trace=cuda,nvtx,osrt -o report \
  pixi run mlsys-cli eval timing --workload-set moe-medium --runner persistent
```

Do not profile the default isolated runner for this command surface. It may
hang, miss child-process CUDA work, or produce inconclusive reports.

## Questions nsys Should Answer

- Which phases dominate runtime?
- Which kernels dominate total GPU time?
- Are there many tiny kernels?
- Are there visible host gaps or synchronization stalls?
- Is transfer overhead relevant?
- Which named NVTX ranges dominate?

## Hotspot Selection Rules

- Pick the top one to three kernels by total GPU time for `ncu`.
- If many tiny kernels dominate, classify that as orchestration or launch
  overhead before diving into one micro-kernel.
- If NVTX ranges expose one dominant phase, keep later `ncu` work scoped to
  kernels from that phase.

## What nsys Does Not Give You

`nsys` does not tell you exactly why one kernel is slow. It narrows the target
set and tells you whether the next question belongs to:

- orchestration-level work
- kernel-level diagnosis

Use `ncu-bottleneck-subskill.md` for the latter.

## The Five-Step nsys-to-ncu Workflow

Treat Nsight metrics as a decision tree, not as isolated KPIs. `nsys` answers
"where time is going" (launch, sync, transfer, overlap). `ncu` answers "what
resource is limiting a chosen kernel." The disciplined sequence is:

### Step 1: Mark phases with NVTX

Use NVTX ranges or `cudaProfilerStart/Stop` to separate phases such as
prefill/decode, forward/backward, or one MoE layer. NVTX CPU ranges project
onto the GPU timeline and make later filtering possible.

### Step 2: nsys to pick the battle

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=nvtx --stats=true -o run ./app
nsys stats --report nvtx_kern_sum --report cuda_gpu_kern_sum run.nsys-rep
```

Use `cuda_gpu_kern_sum` and NVTX-projected range summaries to find the
dominant phase and the top kernels. For multi-stream code, the timeline shows
overlap but not SM occupancy, which is why step 3 and onward are needed for
concurrency judgment.

### Step 3: ncu on selected kernels only

```bash
ncu --set basic --kernel-name regex:<top_kernel> ./app
```

Do not run `--set full` across the whole application. Section sets require
replay passes, and multi-pass replay compounds overhead.

### Step 4: Classify via SOL and roofline before individual counters

```bash
ncu --set detailed \
  --section SpeedOfLight_RooflineChart \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  --section InstructionStats \
  --kernel-name regex:<top_kernel> ./app
```

SOL and roofline give the first useful split: compute pressure versus memory
pressure versus latency/underfill. Only then move to scheduler and warp-state
stats.

### Step 5: Drill into stall reasons and source lines

Only focus on stall reasons when schedulers are not issuing every cycle.
Build with `-lineinfo`, not `-G`. The `-G` flag disables device optimizations
and distorts the profile.

## Nsys-Level Symptom Mapping

Use the timeline view to classify the orchestration problem before diving into
kernel bodies:

| Symptom on the nsys timeline | Classification |
|---|---|
| Big host gaps, syncs, expensive memcpy | Orchestration-limited. Fix overlap and transfer first. NVTX-isolate the phase before touching any kernel body. |
| Launches packed but each kernel tiny | Launch overhead or kernel fragmentation. Consider fusion, CUDA Graphs, or persistent kernels before tuning one kernel. |
| Dominant NVTX range already identified | Keep later `ncu` work scoped to kernels launched inside that range. |
| Multi-stream overlap present | Timeline cannot show SM occupancy; need GPU metrics from `ncu` for concurrency judgment. |

## Stopping Criteria At The nsys Layer

Decide, before moving to `ncu`, whether the question is genuinely kernel-body
sized:

- if a small number of kernels dominate total GPU time, select them for
  focused `ncu` diagnosis
- if many tiny kernels dominate, the question is orchestration, not kernel
  body, and the next owner is a fusion or graph discussion rather than
  `ncu --set full`
- if transfer time dominates, the question is about keeping data resident or
  overlapping copies, not about the math inside a kernel
- if host gaps dominate, the question is about CPU-side launch cadence and
  synchronization, not kernel internals
