# AMD GPU Hardware Counters Reference

This document describes key hardware performance counters available on AMD Instinct GPUs (MI100, MI200, MI300) via rocprofv3.

## Contents
1. [Wavefront Counters](#wavefront-counters)
2. [Instruction Counters](#instruction-counters)
3. [Compute Unit (CU) Utilization](#compute-unit-utilization)
4. [Memory System Counters](#memory-system-counters)
5. [LDS Counters](#lds-counters)
6. [Derived Metrics](#derived-metrics)

---

## Wavefront Counters

| Counter | Description |
|---------|-------------|
| `SQ_WAVES` | Total number of wavefronts launched |
| `SQ_BUSY_CYCLES` | Number of cycles the shader sequencer (SQ) is busy |
| `SQ_WAIT_ANY` | Cycles wavefronts are waiting (memory, LDS, etc.) |
| `SQ_ACTIVE_INST_VALU` | Cycles with active VALU instructions |

**Interpretation:**
- High `SQ_WAIT_ANY` / `SQ_BUSY_CYCLES` ratio indicates memory or dependency stalls
- Compare `SQ_WAVES` to theoretical maximum for occupancy estimation

---

## Instruction Counters

| Counter | Description |
|---------|-------------|
| `SQ_INSTS_VALU` | Vector ALU instructions executed |
| `SQ_INSTS_VMEM` | Vector memory instructions (global/scratch) |
| `SQ_INSTS_SALU` | Scalar ALU instructions |
| `SQ_INSTS_SMEM` | Scalar memory instructions |
| `SQ_INSTS_LDS` | LDS (Local Data Share) instructions |
| `SQ_INSTS_FLAT` | Flat memory instructions |
| `SQ_INSTS_FLAT_LDS_ONLY` | Flat instructions resolved to LDS |

**Instruction Mix Analysis:**
- **Compute-intensive**: High `SQ_INSTS_VALU` / total instructions
- **Memory-intensive**: High `SQ_INSTS_VMEM` / total instructions
- **LDS-heavy**: High `SQ_INSTS_LDS` / total instructions

---

## Compute Unit Utilization

| Counter | Description |
|---------|-------------|
| `GRBM_GUI_ACTIVE` | GPU active cycles (graphics/compute) |
| `GRBM_COUNT` | Total clock cycles |
| `SQ_BUSY_CU_CYCLES` | CU busy cycles |
| `SQ_VALU_MFMA_BUSY_CYCLES` | MFMA (matrix) unit busy cycles |

**Key Ratios:**
- **GPU Utilization**: `SQ_BUSY_CYCLES` / `GRBM_GUI_ACTIVE`
- **MFMA Utilization**: `SQ_VALU_MFMA_BUSY_CYCLES` / `SQ_BUSY_CYCLES`

---

## Memory System Counters

### L2 Cache (TCC - Texture Cache per Channel)

| Counter | Description |
|---------|-------------|
| `TCC_HIT_sum` | L2 cache hits (summed across channels) |
| `TCC_MISS_sum` | L2 cache misses (summed across channels) |
| `TCC_EA_RDREQ_32B_sum` | 32-byte read requests to HBM |
| `TCC_EA_WRREQ_sum` | Write requests to HBM |
| `TCC_TAG_STALL[n]` | L2 tag stall cycles per channel |

**L2 Hit Rate**: `TCC_HIT_sum` / (`TCC_HIT_sum` + `TCC_MISS_sum`)

### Vector L1 Cache (TCP)

| Counter | Description |
|---------|-------------|
| `TCP_TOTAL_READ[n]` | Total L1 read requests |
| `TCP_TOTAL_WRITE[n]` | Total L1 write requests |
| `TCP_TCC_READ_REQ_sum` | L1 to L2 read requests |
| `TCP_TCC_WRITE_REQ_sum` | L1 to L2 write requests |

### Texture Units

| Counter | Description |
|---------|-------------|
| `TA_BUSY_sum` | Texture Addressing unit busy cycles |
| `TD_BUSY_sum` | Texture Data unit busy cycles |

---

## LDS Counters

| Counter | Description |
|---------|-------------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict cycles |
| `SQ_LDS_ADDR_CONFLICT` | LDS address conflicts |
| `SQ_LDS_MEM_VIOLATIONS` | LDS memory violations |
| `SQ_INSTS_LDS` | Total LDS instructions |

**Bank Conflict Rate**: `SQ_LDS_BANK_CONFLICT` / `SQ_WAVES`

High bank conflicts indicate suboptimal LDS access patterns. Each LDS bank can serve one request per cycle; conflicts cause serialization.

---

## Derived Metrics

rocprofv3 supports derived metrics calculated from basic counters. Common ones:

| Metric | Formula | Description |
|--------|---------|-------------|
| `VALUUtilization` | VALU_busy_cycles / total_cycles | VALU efficiency |
| `VALUBusy` | Percentage of cycles VALU active | Compute intensity |
| `SALUBusy` | Percentage of cycles SALU active | Scalar utilization |
| `FetchSize` | Total bytes fetched from memory | Memory read bandwidth |
| `WriteSize` | Total bytes written to memory | Memory write bandwidth |
| `L2CacheHit` | TCC_HIT / (TCC_HIT + TCC_MISS) | L2 efficiency |
| `MemUnitBusy` | Memory unit busy percentage | Memory pressure |
| `LDSBankConflict` | Percentage cycles with conflicts | LDS efficiency |

---

## Listing Available Counters

To list all available counters on your GPU:

```bash
# List basic counters
rocprofv3 -L --list-basic

# List derived metrics  
rocprofv3 -L --list-derived
```

Counter availability varies by GPU architecture (gfx908 for MI100, gfx90a for MI200, gfx942 for MI300).

---

## Counter Collection Limitations

1. **Hardware limits**: Only ~10-16 counters can be collected per pass
2. **Multi-pass**: Exceeding limits causes kernel replay
3. **Kernel perturbation**: Profiling adds overhead; use representative workloads
4. **Counter groups**: Some counters conflict; organize into compatible groups

Example multi-pass input file:
```
# Pass 1: Instruction counters
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS

# Pass 2: Memory counters  
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum

# Pass 3: Utilization counters
pmc: SQ_BUSY_CYCLES SQ_WAIT_ANY GRBM_GUI_ACTIVE
```

---

## Further Reading

- [MI300/MI200 Performance Counters](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html)
- [rocprofv3 Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [ROCm Compute Profiler (omniperf)](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
