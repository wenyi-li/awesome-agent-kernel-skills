# NCU Metric Interpretation Reference

## 1. Bottleneck Classification (SpeedOfLight)

| Metric | Meaning |
|---|---|
| `SM SOL %` | Compute unit utilization as a percentage of peak |
| `Memory SOL %` | Memory bandwidth utilization as a percentage of peak |

**Decision rules:**

| Condition | Conclusion | Next Step |
|---|---|---|
| Memory SOL > 60% and much higher than SM SOL | **Memory-Bound** | Check MemoryWorkloadAnalysis |
| SM SOL > 60% and much higher than Memory SOL | **Compute-Bound** | Check ComputeWorkloadAnalysis |
| Both < 40% | **Latency-Bound** | Check Occupancy + WarpStateStatistics |
| Achieved Occ << Theoretical with clear limiting factor | **Occupancy-Bound** | Check LaunchStatistics |
| No single dominant symptom | **Mixed** | Address the most obvious category first |

---

## 2. Memory-Bound Breakdown (MemoryWorkloadAnalysis)

| NCU Metric | Problem Signal | Meaning |
|---|---|---|
| `Global Load/Store Efficiency` | < 100% | Uncoalesced access, wasted bandwidth |
| `Sectors/Request` | > 1 (ideal: 1) | Multiple sectors per request — misaligned or uncoalesced |
| `L1 Hit Rate` | too low | Poor data locality, L1 cannot reuse |
| `L2 Hit Rate` | too low | Working set exceeds L2, frequent DRAM access |
| `Shared Memory Efficiency` | < 100% | Bank conflicts, accesses serialized |
| `DRAM Throughput` | near peak but kernel still slow | Bandwidth limit reached; algorithm must reduce memory access volume |

---

## 3. Compute-Bound Breakdown (ComputeWorkloadAnalysis)

| NCU Metric | Problem Signal | Meaning |
|---|---|---|
| `FP32/FP16/Tensor Pipe Utilization` | imbalanced or low | Not using the right compute pipeline (e.g., Tensor Core not used when it should be) |
| `Issue Slot Utilization` | < 50% | Idle instruction slots, unsaturated scheduling |
| `Warp Execution Efficiency` | < 100% | Warp divergence, some lanes idle |
| `Eligible Warps Per Cycle` | too low | Insufficient schedulable warps — lack of ILP or occupancy too low |
| `Register Spill (Local Memory)` | > 0 | Register overflow, degraded to global memory access |

---

## 4. Occupancy (Occupancy + LaunchStatistics)

| NCU Metric | Meaning |
|---|---|
| `Achieved Occupancy` | Actual resident warps / SM theoretical maximum |
| `Theoretical Occupancy` | Theoretical upper limit by resource constraints |
| `Registers Per Thread` | Per-thread register usage (primary limiting factor) |
| `Shared Memory Per Block` | Per-block shared memory usage (primary limiting factor) |

**Interpretation**: Achieved << Theoretical indicates resource constraints; see the `LaunchStatistics` section for the specific limiting factor. Note that occupancy is not always better when higher — use measured latency as the final criterion.

---

## 5. Warp Scheduling & Stalls (SchedulerStatistics + WarpStateStatistics)

| Stall Type | Meaning | Common Cause |
|---|---|---|
| `Stall Barrier` | Waiting for `__syncthreads()` to complete | Too many sync points |
| `Stall Long Scoreboard` | Waiting for global memory data | Global memory latency not hidden |
| `Stall Short Scoreboard` | Waiting for Shared Memory / L1 data | Bank conflicts or uncoalesced access |
| `Stall MIO Throttle` | Memory instruction queue full | Too high density of memory instructions |
| `Stall No Instructions` | Instruction cache miss | Kernel code too large |
| Low `Eligible Warps Per Cycle` | Insufficient schedulable warps per cycle | Insufficient ILP / occupancy too low |

---

## 6. Branch Divergence (SourceCounters + InstructionStatistics)

| NCU Metric | Meaning |
|---|---|
| `Branch Efficiency` | Undivergent branches as a percentage of all branches; < 100% means warp divergence exists |
| `Divergent Branches` | Number of divergent branches |

**Interpretation**: Low Branch Efficiency means threads in the same warp took different branch paths; the GPU must execute both paths serially, halving effective throughput.

---

Use the classification above to choose the next kernel investigation or optimization direction.
