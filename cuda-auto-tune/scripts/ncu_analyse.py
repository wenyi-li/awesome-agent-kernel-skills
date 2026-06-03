#!/usr/bin/env python3
"""
NCU deep performance analysis tool — multi-dimensional diagnosis from CSV raw metrics.

Supports automatic detection and specialized analysis for Native CUDA / CUTLASS / Triton / CuTe DSL kernels.
Compatible with Python 3.6+

Usage:
    python3 ncu_analyse.py <report.csv>
    python3 ncu_analyse.py <report.ncu-rep>          # auto-export CSV then analyze
    python3 ncu_analyse.py <report.csv> -o report.md
    python3 ncu_analyse.py <report.csv> --diff <baseline.csv>
    python3 ncu_analyse.py <report.csv> --json
    python3 ncu_analyse.py <report.csv> --type cutedsl   # override kernel type (CuTe DSL)
    python3 ncu_analyse.py <report.csv> --type cutedsl --kernel "kernel"
"""

import argparse
import csv
import json
import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Metric key constants (must match NCU CSV column headers exactly)
# ---------------------------------------------------------------------------

FUNCTION_NAME = "Function Name"
DEVICE_NAME = "Device Name"
GRID_SIZE = "Grid Size"
BLOCK_SIZE = "Block Size [block]"
GPU_TIME_DURATION = "gpu__time_duration.sum [us]"

SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed [%]"
MEM_THROUGHPUT = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed [%]"
DRAM_THROUGHPUT = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed [%]"

L1_SECTORS_GLOBAL_LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum [sector]"
L1_REQUESTS_GLOBAL_LD = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
L1_HIT_RATE = "l1tex__t_sector_hit_rate.pct [%]"
L2_HIT_RATE = "lts__t_sector_hit_rate.pct [%]"
L1_SECTORS_GLOBAL_ST = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum [sector]"
L1_REQUESTS_GLOBAL_ST = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"

SHMEM_BANK_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
LOCAL_MEM_STORE_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum [sector]"

DRAM_BYTES_READ = "dram__bytes_read.sum [Gbyte]"
DRAM_BYTES_WRITE = "dram__bytes_write.sum [Gbyte]"

WARPS_ACTIVE_PCT = "sm__warps_active.avg.pct_of_peak_sustained_active [%]"
REGISTERS_PER_THREAD = "launch__registers_per_thread [register/thread]"
SHARED_MEM_PER_BLOCK = "launch__shared_mem_per_block [Kbyte/block]"
OCCUPANCY_LIMIT_REGISTERS = "launch__occupancy_limit_registers [block]"
OCCUPANCY_LIMIT_SHARED_MEM = "launch__occupancy_limit_shared_mem [block]"
OCCUPANCY_LIMIT_WARPS = "launch__occupancy_limit_warps [block]"
OCCUPANCY_LIMIT_BLOCKS = "launch__occupancy_limit_blocks [block]"
THEORETICAL_OCCUPANCY = "launch__occupancy_cluster_pct [%]"

WARPS_ELIGIBLE_PER_CYCLE = "smsp__warps_eligible.avg.per_cycle_active [warp]"

STALL_LONG_SCOREBOARD = "smsp__pcsamp_warps_issue_stalled_long_scoreboard [warp]"
STALL_SHORT_SCOREBOARD = "smsp__pcsamp_warps_issue_stalled_short_scoreboard [warp]"
STALL_WAIT = "smsp__pcsamp_warps_issue_stalled_wait [warp]"
STALL_SLEEPING = "smsp__pcsamp_warps_issue_stalled_sleeping [warp]"
STALL_BARRIER = "smsp__pcsamp_warps_issue_stalled_barrier [warp]"
STALL_MIO_THROTTLE = "smsp__pcsamp_warps_issue_stalled_mio_throttle [warp]"
STALL_LG_THROTTLE = "smsp__pcsamp_warps_issue_stalled_lg_throttle [warp]"
STALL_MATH_PIPE_THROTTLE = "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle [warp]"
STALL_DRAIN = "smsp__pcsamp_warps_issue_stalled_drain [warp]"
STALL_NOT_SELECTED = "smsp__pcsamp_warps_issue_stalled_not_selected [warp]"
STALL_SELECTED = "smsp__pcsamp_warps_issue_stalled_selected [warp]"

PIPE_FMA = "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active [%]"
PIPE_ALU = "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active [%]"
PIPE_LSU = "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active [%]"
PIPE_TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active [%]"
PIPE_TENSOR_HMMA = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active [%]"
PIPE_FMA_FP16 = "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active [%]"

AVG_THREAD_EXECUTED = "derived__avg_thread_executed [thread]"
AVG_THREAD_EXECUTED_TRUE = "derived__avg_thread_executed_true [thread]"

DEVICE_ARCH = "device__attribute_architecture"

ALL_STALL_KEYS = [
    STALL_LONG_SCOREBOARD, STALL_SHORT_SCOREBOARD, STALL_WAIT,
    STALL_SLEEPING, STALL_BARRIER, STALL_MIO_THROTTLE,
    STALL_LG_THROTTLE, STALL_MATH_PIPE_THROTTLE, STALL_DRAIN,
    STALL_NOT_SELECTED, STALL_SELECTED,
]

STALL_DISPLAY_NAMES = {
    STALL_LONG_SCOREBOARD: "Long Scoreboard",
    STALL_SHORT_SCOREBOARD: "Short Scoreboard",
    STALL_WAIT: "Wait",
    STALL_SLEEPING: "Sleeping",
    STALL_BARRIER: "Barrier",
    STALL_MIO_THROTTLE: "MIO Throttle",
    STALL_LG_THROTTLE: "LG Throttle",
    STALL_MATH_PIPE_THROTTLE: "Math Pipe Throttle",
    STALL_DRAIN: "Drain",
    STALL_NOT_SELECTED: "Not Selected",
    STALL_SELECTED: "Selected",
}

# Severity levels
SEV_INFO = 0
SEV_WARNING = 1
SEV_CRITICAL = 2

SEV_NAMES = {SEV_INFO: "INFO",
    SEV_WARNING: "WARNING", SEV_CRITICAL: "CRITICAL"}


# ---------------------------------------------------------------------------
# Data model (Python 3.6 compatible — no dataclasses)
# ---------------------------------------------------------------------------

class KernelData(object):
    __slots__ = [
        "kernel_name", "device_name", "grid_size", "block_size", "duration_us",
        "sm_throughput_pct", "mem_throughput_pct", "dram_throughput_pct",
        "l1_sectors_global_ld", "l1_requests_global_ld", "l1_hit_rate_pct",
        "l2_hit_rate_pct", "l1_sectors_global_st", "l1_requests_global_st",
        "shared_mem_bank_conflicts", "local_mem_store_sectors",
        "dram_read_gbytes", "dram_write_gbytes",
        "warps_active_pct", "registers_per_thread", "shared_mem_per_block_kb",
        "occupancy_limit_registers", "occupancy_limit_shared_mem",
        "occupancy_limit_warps", "occupancy_limit_blocks",
        "theoretical_occupancy_pct", "warps_eligible_per_cycle",
        "stall_long_scoreboard", "stall_short_scoreboard", "stall_wait",
        "stall_sleeping", "stall_barrier", "stall_mio_throttle",
        "stall_lg_throttle", "stall_math_pipe_throttle", "stall_drain",
        "stall_not_selected", "stall_selected",
        "pipe_fma_pct", "pipe_alu_pct", "pipe_lsu_pct",
        "pipe_tensor_pct", "pipe_tensor_hmma_pct", "pipe_fma_fp16_pct",
        "avg_thread_executed", "avg_thread_executed_true",
        "arch_sm", "raw",
    ]

    def __init__(self):
        self.kernel_name = ""
        self.device_name = ""
        self.grid_size = ""
        self.block_size = ""
        self.duration_us = 0.0
        self.sm_throughput_pct = 0.0
        self.mem_throughput_pct = 0.0
        self.dram_throughput_pct = 0.0
        self.l1_sectors_global_ld = 0.0
        self.l1_requests_global_ld = 0.0
        self.l1_hit_rate_pct = 0.0
        self.l2_hit_rate_pct = 0.0
        self.l1_sectors_global_st = 0.0
        self.l1_requests_global_st = 0.0
        self.shared_mem_bank_conflicts = 0.0
        self.local_mem_store_sectors = 0.0
        self.dram_read_gbytes = 0.0
        self.dram_write_gbytes = 0.0
        self.warps_active_pct = 0.0
        self.registers_per_thread = 0.0
        self.shared_mem_per_block_kb = 0.0
        self.occupancy_limit_registers = 0.0
        self.occupancy_limit_shared_mem = 0.0
        self.occupancy_limit_warps = 0.0
        self.occupancy_limit_blocks = 0.0
        self.theoretical_occupancy_pct = 0.0
        self.warps_eligible_per_cycle = 0.0
        self.stall_long_scoreboard = 0.0
        self.stall_short_scoreboard = 0.0
        self.stall_wait = 0.0
        self.stall_sleeping = 0.0
        self.stall_barrier = 0.0
        self.stall_mio_throttle = 0.0
        self.stall_lg_throttle = 0.0
        self.stall_math_pipe_throttle = 0.0
        self.stall_drain = 0.0
        self.stall_not_selected = 0.0
        self.stall_selected = 0.0
        self.pipe_fma_pct = 0.0
        self.pipe_alu_pct = 0.0
        self.pipe_lsu_pct = 0.0
        self.pipe_tensor_pct = 0.0
        self.pipe_tensor_hmma_pct = 0.0
        self.pipe_fma_fp16_pct = 0.0
        self.avg_thread_executed = 0.0
        self.avg_thread_executed_true = 0.0
        self.arch_sm = 0
        self.raw = {}

    def total_stall_samples(self):
        return (
            self.stall_long_scoreboard + self.stall_short_scoreboard
            + self.stall_wait + self.stall_sleeping + self.stall_barrier
            + self.stall_mio_throttle + self.stall_lg_throttle
            + self.stall_math_pipe_throttle + self.stall_drain
            + self.stall_not_selected + self.stall_selected
        )

    def stall_breakdown(self):
        """Returns list of (name, count, pct) sorted by count desc."""
        total = self.total_stall_samples()
        if total == 0:
            return []
        reasons = [
            ("Long Scoreboard", self.stall_long_scoreboard),
            ("Short Scoreboard", self.stall_short_scoreboard),
            ("Wait", self.stall_wait),
            ("Sleeping", self.stall_sleeping),
            ("Barrier", self.stall_barrier),
            ("MIO Throttle", self.stall_mio_throttle),
            ("LG Throttle", self.stall_lg_throttle),
            ("Math Pipe Throttle", self.stall_math_pipe_throttle),
            ("Drain", self.stall_drain),
            ("Not Selected", self.stall_not_selected),
            ("Selected", self.stall_selected),
        ]
        reasons.sort(key=lambda x: -x[1])
        return [(name, count, count / total * 100) for name, count in reasons]

    def divergence_pct(self):
        if self.avg_thread_executed == 0:
            return 0.0
        return (1.0 - self.avg_thread_executed_true / self.avg_thread_executed) * 100.0

    def load_coalescing_ratio(self):
        if self.l1_requests_global_ld == 0:
            return 0.0
        return self.l1_sectors_global_ld / self.l1_requests_global_ld

    def store_coalescing_ratio(self):
        if self.l1_requests_global_st == 0:
            return 0.0
        return self.l1_sectors_global_st / self.l1_requests_global_st

    def occupancy_limiter(self):
        """Returns (limiter_name, value) for the tightest occupancy limiter."""
        limiters = [
            ("Registers", self.occupancy_limit_registers),
            ("Shared Memory", self.occupancy_limit_shared_mem),
            ("Warps", self.occupancy_limit_warps),
            ("Blocks", self.occupancy_limit_blocks),
        ]
        valid = [(n, v) for n, v in limiters if v > 0]
        if not valid:
            return ("Unknown", 0.0)
        return min(valid, key=lambda x: x[1])


class Finding(object):
    __slots__ = ["severity", "title", "detail", "action", "source"]

    def __init__(self, severity, title, detail, action, source=""):
        self.severity = severity
        self.title = title
        self.detail = detail
        self.action = action
        self.source = source


# ---------------------------------------------------------------------------
# CSV Parser
# ---------------------------------------------------------------------------

def _fval(row, key):
    v = row.get(key, "").strip().strip('"').replace(",", "")
    if not v or v == "n/a":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def parse_csv(csv_path):
    """Parse NCU raw CSV export into per-kernel KernelData objects."""
    kernels = []

    with open(csv_path, newline="") as f:
        content = f.read()

    # Handle BOM
    if content.startswith("\ufeff"):
        content = content[1:]

    import io
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        return []

    for row in reader:
        kd = KernelData()
        kd.raw = dict(row)
        kd.kernel_name = row.get(FUNCTION_NAME, "").strip('"').strip()
        kd.device_name = row.get(DEVICE_NAME, "").strip('"').strip()
        kd.grid_size = row.get(GRID_SIZE, "").strip('"').strip()
        kd.block_size = row.get(BLOCK_SIZE, "").strip('"').strip()

        kd.duration_us = _fval(row, GPU_TIME_DURATION)
        kd.sm_throughput_pct = _fval(row, SM_THROUGHPUT)
        kd.mem_throughput_pct = _fval(row, MEM_THROUGHPUT)
        kd.dram_throughput_pct = _fval(row, DRAM_THROUGHPUT)

        kd.l1_sectors_global_ld = _fval(row, L1_SECTORS_GLOBAL_LD)
        kd.l1_requests_global_ld = _fval(row, L1_REQUESTS_GLOBAL_LD)
        kd.l1_hit_rate_pct = _fval(row, L1_HIT_RATE)
        kd.l2_hit_rate_pct = _fval(row, L2_HIT_RATE)
        kd.l1_sectors_global_st = _fval(row, L1_SECTORS_GLOBAL_ST)
        kd.l1_requests_global_st = _fval(row, L1_REQUESTS_GLOBAL_ST)

        kd.shared_mem_bank_conflicts = _fval(row, SHMEM_BANK_CONFLICTS)
        kd.local_mem_store_sectors = _fval(row, LOCAL_MEM_STORE_SECTORS)

        kd.dram_read_gbytes = _fval(row, DRAM_BYTES_READ)
        kd.dram_write_gbytes = _fval(row, DRAM_BYTES_WRITE)

        kd.warps_active_pct = _fval(row, WARPS_ACTIVE_PCT)
        kd.registers_per_thread = _fval(row, REGISTERS_PER_THREAD)
        kd.shared_mem_per_block_kb = _fval(row, SHARED_MEM_PER_BLOCK)
        kd.occupancy_limit_registers = _fval(row, OCCUPANCY_LIMIT_REGISTERS)
        kd.occupancy_limit_shared_mem = _fval(row, OCCUPANCY_LIMIT_SHARED_MEM)
        kd.occupancy_limit_warps = _fval(row, OCCUPANCY_LIMIT_WARPS)
        kd.occupancy_limit_blocks = _fval(row, OCCUPANCY_LIMIT_BLOCKS)
        kd.theoretical_occupancy_pct = _fval(row, THEORETICAL_OCCUPANCY)

        kd.warps_eligible_per_cycle = _fval(row, WARPS_ELIGIBLE_PER_CYCLE)

        kd.stall_long_scoreboard = _fval(row, STALL_LONG_SCOREBOARD)
        kd.stall_short_scoreboard = _fval(row, STALL_SHORT_SCOREBOARD)
        kd.stall_wait = _fval(row, STALL_WAIT)
        kd.stall_sleeping = _fval(row, STALL_SLEEPING)
        kd.stall_barrier = _fval(row, STALL_BARRIER)
        kd.stall_mio_throttle = _fval(row, STALL_MIO_THROTTLE)
        kd.stall_lg_throttle = _fval(row, STALL_LG_THROTTLE)
        kd.stall_math_pipe_throttle = _fval(row, STALL_MATH_PIPE_THROTTLE)
        kd.stall_drain = _fval(row, STALL_DRAIN)
        kd.stall_not_selected = _fval(row, STALL_NOT_SELECTED)
        kd.stall_selected = _fval(row, STALL_SELECTED)

        kd.pipe_fma_pct = _fval(row, PIPE_FMA)
        kd.pipe_alu_pct = _fval(row, PIPE_ALU)
        kd.pipe_lsu_pct = _fval(row, PIPE_LSU)
        kd.pipe_tensor_pct = _fval(row, PIPE_TENSOR)
        kd.pipe_tensor_hmma_pct = _fval(row, PIPE_TENSOR_HMMA)
        kd.pipe_fma_fp16_pct = _fval(row, PIPE_FMA_FP16)

        kd.avg_thread_executed = _fval(row, AVG_THREAD_EXECUTED)
        kd.avg_thread_executed_true = _fval(row, AVG_THREAD_EXECUTED_TRUE)

        arch_str = row.get(DEVICE_ARCH, "").strip().strip('"')
        if arch_str:
            try:
                kd.arch_sm = int(arch_str)
            except ValueError:
                pass

        if kd.kernel_name:
            kernels.append(kd)

    return kernels


def ncu_rep_to_csv(ncu_rep_path):
    """Convert .ncu-rep to CSV via ncu CLI, returns CSV path."""
    base = os.path.splitext(ncu_rep_path)[0]
    csv_path = base + ".csv"
    cmd = ["ncu", "--import", ncu_rep_path, "--page", "raw", "--csv"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            sys.stderr.write("ncu export failed: {}\n".format(result.stderr))
            sys.exit(1)
        with open(csv_path, "w") as f:
            f.write(result.stdout)
        return csv_path
    except FileNotFoundError:
        sys.stderr.write("Error: ncu not found in PATH\n")
        sys.exit(1)


def is_triton_kernel(kernel_name):
    """Detect if a kernel was generated by Triton compiler."""
    name = kernel_name.lower()
    if name.startswith("triton_"):
        return True
    if "triton" in name:
        return True
    import re
    if re.search(r'_\d+d\d+d\d+d.*e\b', name):
        return True
    if re.search(r'_kernel_\d+d', name):
        return True
    return False


def classify_triton_kernel_type(kernel_name):
    """Classify Triton kernel subtype based on naming conventions."""
    name = kernel_name.lower()
    if "triton_poi_" in name:
        return "inductor_pointwise"
    if "triton_red_" in name:
        return "inductor_reduction"
    if "triton_per_" in name:
        return "inductor_persistent_reduction"
    if "triton_hel_" in name:
        return "inductor_helper"
    if any(kw in name for kw in ("fwd_kernel", "bwd_kernel", "chunk_fwd", "chunk_bwd")):
        return "custom_fwd_bwd"
    if any(kw in name for kw in ("flash_attention", "flash_attn", "_attn_fwd", "_attn_bwd")):
        return "attention"
    if any(kw in name for kw in ("matmul", "gemm", "linear")):
        return "matmul"
    return "custom"


def is_cutlass_kernel(kernel_name):
    """Detect if a kernel was generated by CUTLASS library."""
    name = kernel_name.lower()
    if name.startswith("cutlass_"):
        return True
    if "cutlass" in name:
        return True
    import re
    if re.match(r'sm\d+_xmma_', name):
        return True
    if re.match(r'(ampere|hopper|turing)_.*tensorop', name):
        return True
    if re.match(r'sm\d+_.*tensorop', name):
        return True
    if re.search(r'(s1688|h1688|h16816|d884|i8816)gemm', name):
        return True
    return False


def is_cutedsl_kernel(kernel_name):
    """Detect if a kernel was generated by CuTe DSL (cutlass.cute Python API).

    CuTe DSL kernel names come from @cute.kernel decorated methods, making them
    harder to auto-detect from NCU alone. This uses heuristics; for reliable
    detection, use --type cutedsl flag.
    """
    name = kernel_name.lower()
    if "cute_dsl" in name or "cutedsl" in name:
        return True
    if "cute_" in name and "cutlass" not in name:
        return True
    if name.startswith("tvm_") or "tvm_ffi" in name:
        return True
    return False


def is_library_kernel(kernel_name):
    """Detect if a kernel is from cuBLAS/cuDNN (not user-written)."""
    name = kernel_name.lower()
    lib_prefixes = ("cublas", "cublaslt", "cudnn")
    return any(name.startswith(p) for p in lib_prefixes)


# Global override for kernel type (set via --type flag)
_kernel_type_override = None


def classify_kernel_type(kernel_name):
    """Classify kernel into implementation type."""
    if _kernel_type_override:
        return _kernel_type_override
    if is_cutlass_kernel(kernel_name):
        return "cutlass"
    if is_triton_kernel(kernel_name):
        return "triton"
    if is_cutedsl_kernel(kernel_name):
        return "cutedsl"
    if is_library_kernel(kernel_name):
        return "library"
    return "native_cuda"


class CutlassConfig(object):
    """Parsed CUTLASS configuration from kernel name."""
    __slots__ = [
        "version", "arch_sm", "compute_type", "instruction_shape",
        "threadblock_m", "threadblock_n", "threadblock_k", "stages",
        "layout_a", "layout_b", "alignment", "is_tensorop", "is_simt",
        "schedule",
    ]

    def __init__(self):
        self.version = ""
        self.arch_sm = 0
        self.compute_type = ""
        self.instruction_shape = ""
        self.threadblock_m = 0
        self.threadblock_n = 0
        self.threadblock_k = 0
        self.stages = 0
        self.layout_a = ""
        self.layout_b = ""
        self.alignment = 0
        self.is_tensorop = False
        self.is_simt = False
        self.schedule = ""


def parse_cutlass_kernel_name(kernel_name):
    """Extract CUTLASS configuration from kernel name."""
    import re
    cfg = CutlassConfig()
    name = kernel_name.lower()

    sm_match = re.search(r'sm(\d+)', name) or re.search(r'cutlass_(\d+)_', name)
    if sm_match:
        cfg.arch_sm = int(sm_match.group(1))

    if re.search(r'ampere', name):
        cfg.arch_sm = cfg.arch_sm or 80
    elif re.search(r'hopper', name):
        cfg.arch_sm = cfg.arch_sm or 90

    if "sm90_xmma" in name or cfg.arch_sm >= 90:
        cfg.version = "3.x"
    else:
        cfg.version = "2.x"

    cfg.is_tensorop = "tensorop" in name or "xmma" in name
    cfg.is_simt = "simt" in name

    instr_match = re.search(r'([shdi]\d{3,5})(gemm|conv)', name)
    if instr_match:
        cfg.instruction_shape = instr_match.group(1)
        dtype_char = instr_match.group(1)[0]
        cfg.compute_type = {
            's': 'fp32', 'h': 'fp16', 'd': 'fp64', 'i': 'int8'
        }.get(dtype_char, 'unknown')

    tile_match = re.search(r'(\d+)x(\d+)[x_](\d+)(?:x(\d+))?', name)
    if tile_match:
        cfg.threadblock_m = int(tile_match.group(1))
        cfg.threadblock_n = int(tile_match.group(2))
        cfg.threadblock_k = int(tile_match.group(3))
        if tile_match.group(4):
            cfg.stages = int(tile_match.group(4))

    tilesize_match = re.search(r'tilesize(\d+)x(\d+)x(\d+)', name)
    if tilesize_match:
        cfg.threadblock_m = int(tilesize_match.group(1))
        cfg.threadblock_n = int(tilesize_match.group(2))
        cfg.threadblock_k = int(tilesize_match.group(3))

    layout_match = re.search(r'_([nt])([nt])(?:_|$)', name)
    if layout_match:
        layout_map = {'n': 'col-major', 't': 'row-major'}
        cfg.layout_a = layout_map.get(layout_match.group(1), '')
        cfg.layout_b = layout_map.get(layout_match.group(2), '')

    align_match = re.search(r'align(\d+)', name)
    if align_match:
        cfg.alignment = int(align_match.group(1))

    if "warpspecialized" in name.replace("_", ""):
        if "cooperative" in name:
            cfg.schedule = "WarpSpecializedCooperative"
        elif "pingpong" in name:
            cfg.schedule = "WarpSpecializedPingpong"
        else:
            cfg.schedule = "WarpSpecialized"

    return cfg


def select_user_kernel(kernels):
    """Pick the most likely user kernel; prefers user-written over library kernels."""
    lib_prefixes = ("cublas", "cublaslt", "cudnn")
    user = [k for k in kernels if not any(
        k.kernel_name.lower().startswith(p) for p in lib_prefixes)]
    if not user:
        user = kernels
    return max(user, key=lambda k: k.duration_us) if user else kernels[0]


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------

def analyze_roofline(data):
    findings = []
    sm = data.sm_throughput_pct
    mem = data.mem_throughput_pct

    if sm < 40 and mem < 40:
        findings.append(Finding(
            SEV_CRITICAL, "Latency Bound",
            "SM throughput ({:.1f}%) and memory throughput ({:.1f}%) are both < 40%. "
            "The GPU is neither computing nor moving data efficiently.".format(
                sm, mem),
            "Analyze warp stall reasons to find root cause. "
            "Consider increasing occupancy, reducing synchronization, or restructuring launch config.",
        ))
    elif sm > mem + 20:
        findings.append(Finding(
            SEV_WARNING, "Compute Bound",
            "SM throughput ({:.1f}%) exceeds memory throughput ({:.1f}%) by > 20pp.".format(
                sm, mem),
            "Check for Tensor Core opportunities, operator fusion, or reduce redundant computation.",
        ))
    elif mem > sm + 20:
        sub = _classify_memory_sublevel(data)
        findings.append(Finding(
            SEV_WARNING, "Memory Bound ({})".format(sub),
            "Memory throughput ({:.1f}%) exceeds SM throughput ({:.1f}%) by > 20pp. "
            "Sub-level: {}. L1 hit: {:.1f}%, L2 hit: {:.1f}%, DRAM throughput: {:.1f}%.".format(
                mem, sm, sub, data.l1_hit_rate_pct, data.l2_hit_rate_pct, data.dram_throughput_pct),
            _memory_sublevel_action(sub),
        ))
    elif sm > 60 and mem > 60 and abs(sm - mem) < 20:
        findings.append(Finding(
            SEV_INFO, "Balanced Utilization",
            "SM throughput ({:.1f}%) and memory throughput ({:.1f}%) are both high and balanced.".format(
                sm, mem),
            "Near peak utilization. Focus on micro-optimizations or algorithmic changes.",
        ))
    else:
        label = "Compute Bound" if sm > mem else "Memory Bound"
        findings.append(Finding(
            SEV_INFO, "Moderate {}".format(label),
            "SM throughput ({:.1f}%), memory throughput ({:.1f}%).".format(
                sm, mem),
            "Consider profiling individual sections for deeper insight.",
        ))
    return findings


def _classify_memory_sublevel(data):
    if data.dram_throughput_pct > 70:
        return "DRAM-Bound"
    if data.l2_hit_rate_pct < 50 and data.dram_throughput_pct > 40:
        return "DRAM-Bound"
    if data.l1_hit_rate_pct < 20 and data.l2_hit_rate_pct >= 50:
        return "L2-Bound"
    if data.l1_hit_rate_pct < 20:
        return "L1-Bound"
    if data.dram_throughput_pct > 30:
        return "DRAM-Bound"
    return "L2-Bound"


def _memory_sublevel_action(sub):
    actions = {
        "DRAM-Bound": (
            "DRAM bandwidth is the bottleneck. Reduce data movement via mixed precision, "
            "compression, or algorithmic changes. Use L2 persistence hints (cudaAccessPolicyWindow) "
            "to cache hot data. Apply block tiling with shared memory."
        ),
        "L2-Bound": (
            "L2 cache misses are driving DRAM traffic. Improve data reuse with tiling, "
            "restructure access patterns for temporal locality. "
            "Use L2 persistence policies on Ampere+."
        ),
        "L1-Bound": (
            "L1 cache is the bottleneck. Use shared memory for frequently accessed data, "
            "apply tiling strategies, improve spatial locality of global memory accesses."
        ),
    }
    return actions.get(sub, "Analyze memory access patterns.")


def analyze_memory(data):
    findings = []

    ld_ratio = data.load_coalescing_ratio()
    if data.l1_requests_global_ld > 0 and ld_ratio > 8:
        sev = SEV_WARNING if ld_ratio < 16 else SEV_CRITICAL
        findings.append(Finding(
            sev,
            "Uncoalesced Global Load (ratio={:.1f})".format(ld_ratio),
            "L1 load sectors/requests = {:.1f} (ideal <= 4 for 32-bit). "
            "Threads load from non-contiguous addresses.".format(ld_ratio),
            "Switch to SoA layout, ensure warp threads access consecutive addresses, "
            "or pad data for alignment.",
        ))

    st_ratio = data.store_coalescing_ratio()
    if data.l1_requests_global_st > 0 and st_ratio > 8:
        sev = SEV_WARNING if st_ratio < 16 else SEV_CRITICAL
        findings.append(Finding(
            sev,
            "Uncoalesced Global Store (ratio={:.1f})".format(st_ratio),
            "L1 store sectors/requests = {:.1f} (ideal <= 4 for 32-bit).".format(
                st_ratio),
            "Ensure store addresses are contiguous within a warp. "
            "Use shared memory as a staging buffer for scatter writes.",
        ))

    if 0 < data.l1_hit_rate_pct < 20:
        findings.append(Finding(
            SEV_WARNING,
            "Low L1 Cache Hit Rate ({:.1f}%)".format(data.l1_hit_rate_pct),
            "L1 texture cache hit rate is only {:.1f}%.".format(
                data.l1_hit_rate_pct),
            "Use __shared__ memory for frequently accessed data, apply tiling.",
        ))

    if 0 < data.l2_hit_rate_pct < 50:
        findings.append(Finding(
            SEV_WARNING,
            "Low L2 Cache Hit Rate ({:.1f}%)".format(data.l2_hit_rate_pct),
            "L2 cache hit rate is only {:.1f}%.".format(data.l2_hit_rate_pct),
            "Use cudaAccessPolicyWindow (Ampere+) to pin hot data, "
            "restructure access patterns for temporal locality.",
        ))

    bc = data.shared_mem_bank_conflicts
    if bc > 100000:
        sev = SEV_CRITICAL if bc > 1000000 else SEV_WARNING
        findings.append(Finding(
            sev,
            "Shared Memory Bank Conflicts ({:,.0f})".format(bc),
            "Detected {:,.0f} bank conflicts. "
            "Bank conflicts serialize shared memory accesses within a warp.".format(
                bc),
            "Pad shared memory arrays (+1 column), rearrange access patterns "
            "so threads access different banks, or use swizzle.",
        ))

    if data.local_mem_store_sectors > 0:
        findings.append(Finding(
            SEV_WARNING,
            "Register Spills ({:,.0f} local store sectors)".format(
                data.local_mem_store_sectors),
            "Registers spill to local memory (slow L1/L2/DRAM path). "
            "This adds significant latency.",
            "Use __launch_bounds__ to guide register allocation, "
            "reduce TM/TN per thread, simplify per-thread state.",
        ))

    if data.dram_throughput_pct > 80:
        total_gb = data.dram_read_gbytes + data.dram_write_gbytes
        findings.append(Finding(
            SEV_INFO,
            "High DRAM Bandwidth ({:.1f}%)".format(data.dram_throughput_pct),
            "DRAM at {:.1f}% of peak. Total: {:.2f} GB (R: {:.2f}, W: {:.2f}).".format(
                data.dram_throughput_pct, total_gb, data.dram_read_gbytes, data.dram_write_gbytes),
            "Near DRAM ceiling. Reduce data movement via compression, "
            "mixed precision, or increase arithmetic intensity.",
        ))

    return findings


def analyze_warp_stalls(data):
    findings = []
    breakdown = data.stall_breakdown()
    if not breakdown:
        return findings

    if 0 < data.warps_eligible_per_cycle < 1.0:
        findings.append(Finding(
            SEV_WARNING,
            "Low Warp Scheduling Efficiency ({:.2f}/cycle)".format(
                data.warps_eligible_per_cycle),
            "Only {:.2f} eligible warps per cycle (ideal >= 2). "
            "The scheduler frequently has no work to issue.".format(
                data.warps_eligible_per_cycle),
            "Increase occupancy or reduce per-warp latency.",
        ))

    stall_actions = {
        "Long Scoreboard": (
            "Warps wait for global/L2 memory. Use cp.async / TMA for async data movement, "
            "prefetch data, improve L2 locality, or restructure access patterns."
        ),
        "Short Scoreboard": (
            "Warps wait for shared memory / L1 results. Reduce bank conflicts (padding), "
            "reorder shared memory accesses, reduce dependency chains."
        ),
        "Wait": (
            "Warps stalled on explicit wait (cp.async.wait, named barriers). "
            "Overlap more computation with async ops, reduce wait granularity."
        ),
        "Sleeping": "Warps explicitly sleeping. Check for unnecessary nanosleep/yield calls.",
        "Barrier": (
            "Warps stalled at __syncthreads(). Reduce barrier frequency, "
            "use warp-level primitives (__shfl_sync, cooperative_groups)."
        ),
        "MIO Throttle": (
            "MIO pipeline saturated. Reduce shared memory / SFU operation rate, "
            "interleave with other instructions."
        ),
        "LG Throttle": (
            "Local/global memory pipeline throttled. Reduce outstanding memory requests, "
            "improve access patterns to reduce L1 tag pressure."
        ),
        "Math Pipe Throttle": (
            "Math pipeline fully utilized (positive for compute-bound). "
            "Consider Tensor Cores or reduce instruction count."
        ),
        "Drain": "Warp draining at kernel/block end. Balance work across warps.",
        "Not Selected": "Warps eligible but not selected (scheduler contention). Usually not actionable.",
        "Selected": "Productive work (not a concern).",
    }

    for reason, count, pct in breakdown[:5]:
        if pct < 10:
                continue
        sev = SEV_CRITICAL if pct >= 25 else SEV_WARNING
        action = stall_actions.get(reason, "Consult NCU documentation.")
        findings.append(Finding(
            sev,
            "Warp Stall: {} ({:.1f}%)".format(reason, pct),
            "{} accounts for {:.1f}% of all stall samples ({:,.0f} warps).".format(
                reason, pct, count),
            action,
        ))

    return findings


def analyze_instruction_mix(data):
    findings = []

    if data.pipe_lsu_pct > 0 and data.pipe_fma_pct > 0:
        if data.pipe_lsu_pct > 2 * data.pipe_fma_pct:
            findings.append(Finding(
                SEV_WARNING,
                "LSU-Dominated Instruction Mix (LSU={:.1f}%, FMA={:.1f}%)".format(
                    data.pipe_lsu_pct, data.pipe_fma_pct),
                "Load/Store instructions dominate the instruction mix. "
                "The kernel spends most time on memory operations.",
                "Increase compute reuse per loaded element, "
                "use tiling to amortize loads over more FMAs.",
            ))

    if data.pipe_fma_pct > 60 and data.pipe_tensor_pct < 5 and data.pipe_fma_fp16_pct > 10:
        findings.append(Finding(
            SEV_WARNING,
            "FP16 Kernel Without Tensor Core Usage",
            "FMA FP16 utilization is {:.1f}% but Tensor Core "
            "utilization is only {:.1f}%.".format(
                data.pipe_fma_fp16_pct, data.pipe_tensor_pct),
            "Use WMMA / mma.sync / CUTLASS to leverage Tensor Cores for FP16 computation.",
        ))

    if data.pipe_tensor_pct > 50:
        findings.append(Finding(
            SEV_INFO,
            "Good Tensor Core Utilization ({:.1f}%)".format(
                data.pipe_tensor_pct),
            "Tensor Core pipeline is well-utilized.",
            "",
        ))

    return findings


def analyze_occupancy(data):
    findings = []

    if 0 < data.warps_active_pct < 50:
        limiter_name, limiter_val = data.occupancy_limiter()
        findings.append(Finding(
            SEV_WARNING,
            "Low Occupancy ({:.1f}%), Limited by {}".format(
                data.warps_active_pct, limiter_name),
            "Achieved occupancy is {:.1f}% (theoretical: {:.1f}%). "
            "Primary limiter: {} ({:.0f} blocks/SM).".format(
                data.warps_active_pct, data.theoretical_occupancy_pct,
                limiter_name, limiter_val),
            _occupancy_action(limiter_name),
        ))

    rpt = data.registers_per_thread
    if rpt >= 128:
        findings.append(Finding(
            SEV_CRITICAL,
            "Very High Register Usage ({:.0f} regs/thread)".format(rpt),
            "Extremely high register count severely limits occupancy.",
            "Use __launch_bounds__(threads, minBlocks) to cap registers, "
            "reduce per-thread state (TM/TN), or split into sub-kernels.",
        ))
    elif rpt >= 64:
        findings.append(Finding(
            SEV_WARNING,
            "High Register Usage ({:.0f} regs/thread)".format(rpt),
            "High register count may limit occupancy.",
            "Consider __launch_bounds__ or reducing per-thread computation.",
        ))

    return findings


def _occupancy_action(limiter):
    actions = {
        "Registers": (
            "Registers are the occupancy bottleneck. Use __launch_bounds__(threads, minBlocks), "
            "reduce TM/TN, or use volatile to encourage register reuse."
        ),
        "Shared Memory": (
            "Shared memory is the occupancy bottleneck. Use extern __shared__ for dynamic allocation, "
            "reduce tile size, or share smem across warps."
        ),
        "Warps": (
            "Warp count per block limits occupancy. Increase block size "
            "(more threads per block = more warps)."
        ),
        "Blocks": "Block count limit reached. Decrease block size to fit more blocks per SM.",
    }
    return actions.get(limiter, "Adjust launch configuration.")


def analyze_divergence(data):
    findings = []
    div = data.divergence_pct()
    if div > 20:
        findings.append(Finding(
            SEV_WARNING,
            "Significant Thread Divergence ({:.1f}%)".format(div),
            "Avg threads executed: {:.1f}, avg active (true): {:.1f}. "
            "Divergence: {:.1f}%.".format(
                data.avg_thread_executed, data.avg_thread_executed_true, div),
            "Restructure branch logic for warp-uniform conditions, "
            "use predication, or reorganize data to reduce divergence.",
        ))
    elif div > 10:
        findings.append(Finding(
            SEV_INFO,
            "Moderate Thread Divergence ({:.1f}%)".format(div),
            "Divergence: {:.1f}%. May or may not impact performance.".format(
                div),
            "Monitor if this correlates with performance issues.",
        ))
    return findings


def analyze_triton(data):
    """Triton-specific analysis: detects Triton kernels and provides targeted suggestions."""
    findings = []
    ktype = classify_kernel_type(data.kernel_name)
    if ktype != "triton":
        return findings

    triton_type = classify_triton_kernel_type(data.kernel_name)
    block_size_val = 0
    try:
        block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
    except (ValueError, IndexError):
        pass

    num_warps = block_size_val // 32 if block_size_val > 0 else 0

    findings.append(Finding(
        SEV_INFO,
        "Triton Kernel Detected ({})".format(triton_type),
        "Kernel '{}' identified as Triton-compiled (type: {}). "
        "Inferred num_warps={} from block_size={}.".format(
            data.kernel_name[:60], triton_type, num_warps, data.block_size),
        "",
        "Triton Detection",
    ))

    if triton_type in ("inductor_pointwise", "inductor_reduction",
                       "inductor_persistent_reduction", "inductor_helper"):
        findings.append(Finding(
            SEV_INFO,
            "Torch Inductor Generated Kernel",
            "This kernel was auto-generated by torch.compile / Inductor. "
            "Direct Triton source modification is not recommended.",
            "Optimize at PyTorch level: adjust fusion strategy via "
            "torch._inductor.config, or rewrite as custom @triton.jit kernel "
            "if this is a hot path.",
            "Triton Detection",
        ))

    if num_warps > 0:
        if data.registers_per_thread >= 128 and num_warps >= 8:
            findings.append(Finding(
                SEV_CRITICAL,
                "Triton: Excessive num_warps ({}) with High Register Pressure ({:.0f} regs)".format(
                    num_warps, data.registers_per_thread),
                "num_warps={} combined with {:.0f} registers/thread severely limits occupancy. "
                "Each warp requires its own register file allocation.".format(
                    num_warps, data.registers_per_thread),
                "Reduce num_warps (try 4 or 2). Alternatively, reduce BLOCK_* sizes "
                "to lower per-thread register demand.",
                "Triton Occupancy",
            ))
        elif data.registers_per_thread >= 64 and num_warps >= 8:
            findings.append(Finding(
                SEV_WARNING,
                "Triton: High num_warps ({}) with Elevated Register Usage ({:.0f} regs)".format(
                    num_warps, data.registers_per_thread),
                "num_warps={} with {:.0f} registers/thread may limit occupancy.".format(
                    num_warps, data.registers_per_thread),
                "Consider reducing num_warps to 4, or reducing tile dimensions.",
                "Triton Occupancy",
            ))

    smem_kb = data.shared_mem_per_block_kb
    if smem_kb > 0:
        if smem_kb > 164 and data.arch_sm >= 80:
            findings.append(Finding(
                SEV_WARNING,
                "Triton: Very High Shared Memory ({:.1f} KB/block)".format(smem_kb),
                "Shared memory usage ({:.1f} KB) is very high, likely from large tile sizes "
                "or high num_stages. This limits blocks/SM.".format(smem_kb),
                "Reduce BLOCK_* tile dimensions or decrease num_stages. "
                "On Ampere+, max dynamic smem is ~164 KB; on Hopper, ~228 KB.",
                "Triton Pipeline",
            ))
        limiter_name, _ = data.occupancy_limiter()
        if limiter_name == "Shared Memory" and data.warps_active_pct < 40:
            findings.append(Finding(
                SEV_CRITICAL,
                "Triton: Shared Memory Limits Occupancy ({:.1f}%)".format(
                    data.warps_active_pct),
                "Shared memory ({:.1f} KB/block) is the occupancy bottleneck. "
                "Achieved occupancy: {:.1f}%.".format(smem_kb, data.warps_active_pct),
                "Reduce num_stages (each stage doubles smem buffer), "
                "or reduce BLOCK_M/BLOCK_N/BLOCK_K tile sizes.",
                "Triton Pipeline",
            ))

    has_dot_like = triton_type in ("matmul", "attention", "custom_fwd_bwd", "custom")
    if has_dot_like and data.pipe_tensor_pct < 5 and data.pipe_fma_pct > 20:
        findings.append(Finding(
            SEV_WARNING,
            "Triton: Potential Missing Tensor Core Usage (tensor={:.1f}%, FMA={:.1f}%)".format(
                data.pipe_tensor_pct, data.pipe_fma_pct),
            "Kernel type '{}' likely contains tl.dot but Tensor Core utilization "
            "is only {:.1f}%. FMA at {:.1f}% suggests scalar path is used.".format(
                triton_type, data.pipe_tensor_pct, data.pipe_fma_pct),
            "1) Ensure tl.dot uses allow_tf32=True for fp32 inputs. "
            "2) Ensure BLOCK_K is a multiple of 16. "
            "3) Verify input dtypes are fp16/bf16/tf32/fp8 compatible. "
            "4) Check that tl.dot operand shapes are [M,K] x [K,N].",
            "Triton Tensor Core",
        ))

    if has_dot_like and data.pipe_tensor_pct > 50:
        findings.append(Finding(
            SEV_INFO,
            "Triton: Good Tensor Core Utilization ({:.1f}%)".format(
                data.pipe_tensor_pct),
            "Tensor Core pipeline is well-utilized at {:.1f}%.".format(
                data.pipe_tensor_pct),
            "",
            "Triton Tensor Core",
        ))

    total_stalls = data.total_stall_samples()
    if total_stalls > 0:
        long_sb_pct = data.stall_long_scoreboard / total_stalls * 100
        wait_pct = data.stall_wait / total_stalls * 100

        if long_sb_pct > 30:
            findings.append(Finding(
                SEV_WARNING,
                "Triton: Long Scoreboard Stall ({:.1f}%) — Increase num_stages".format(
                    long_sb_pct),
                "Global memory latency dominates ({:.1f}% of stalls). "
                "Triton's software pipelining (num_stages) can hide this.".format(
                    long_sb_pct),
                "Increase num_stages (2→3→4 on Ampere, 2→3 on Hopper). "
                "On Hopper, also consider tl.make_block_ptr() for TMA-based loads.",
                "Triton Pipeline",
            ))
        elif wait_pct > 30 and long_sb_pct < 15:
            findings.append(Finding(
                SEV_WARNING,
                "Triton: Wait Stall ({:.1f}%) — Reduce num_stages or Increase Tile".format(
                    wait_pct),
                "Async copy wait dominates ({:.1f}%), but long scoreboard is low ({:.1f}%). "
                "Pipeline is over-buffered or tile computation is too fast.".format(
                    wait_pct, long_sb_pct),
                "Reduce num_stages (saves smem, may improve occupancy), "
                "or increase BLOCK_M/BLOCK_N to add more compute per stage.",
                "Triton Pipeline",
            ))

    ld_ratio = data.load_coalescing_ratio()
    if ld_ratio > 8:
        findings.append(Finding(
            SEV_WARNING,
            "Triton: Uncoalesced Loads ({:.1f} sectors/req) — Check tl.load Offsets".format(
                ld_ratio),
            "Load coalescing ratio is {:.1f}x (ideal <= 4). Triton may have generated "
            "non-contiguous memory access patterns.".format(ld_ratio),
            "1) Add tl.multiple_of() / tl.max_contiguous() hints to offset calculations. "
            "2) Ensure innermost dimension stride is 1. "
            "3) Consider tl.make_block_ptr() for structured access patterns. "
            "4) Check if transposed loads need explicit tl.trans().",
            "Triton Memory",
        ))

    return findings


def analyze_cutlass(data):
    """CUTLASS-specific analysis: detects CUTLASS kernels and provides targeted suggestions."""
    findings = []
    ktype = classify_kernel_type(data.kernel_name)
    if ktype != "cutlass":
        return findings

    cfg = parse_cutlass_kernel_name(data.kernel_name)

    type_desc = "CUTLASS {ver} ({comp})".format(
        ver=cfg.version,
        comp="Tensor Core" if cfg.is_tensorop else ("SIMT" if cfg.is_simt else "Unknown"),
    )
    tile_desc = ""
    if cfg.threadblock_m > 0:
        tile_desc = "ThreadblockShape={m}x{n}x{k}".format(
            m=cfg.threadblock_m, n=cfg.threadblock_n, k=cfg.threadblock_k)
        if cfg.stages > 0:
            tile_desc += ", stages={}".format(cfg.stages)

    findings.append(Finding(
        SEV_INFO,
        "CUTLASS Kernel Detected ({})".format(type_desc),
        "Kernel '{}' identified as CUTLASS ({})."
        " Parsed config: {}. Alignment={}.".format(
            data.kernel_name[:60], cfg.version,
            tile_desc if tile_desc else "could not parse tile",
            cfg.alignment if cfg.alignment > 0 else "N/A"),
        "",
        "CUTLASS Detection",
    ))

    if cfg.is_simt and not cfg.is_tensorop:
        if data.pipe_tensor_pct < 1 and data.pipe_fma_pct > 20:
            findings.append(Finding(
                SEV_WARNING,
                "CUTLASS SIMT Configuration (No Tensor Core)",
                "This CUTLASS kernel uses SIMT (CUDA Core FMA) path. "
                "Tensor Core utilization is {:.1f}%.".format(data.pipe_tensor_pct),
                "If the data type supports Tensor Core (FP16/BF16/TF32/FP8/INT8), "
                "switch to a tensorop CUTLASS configuration for 2-8x speedup.",
                "CUTLASS Compute",
            ))

    if cfg.is_tensorop and data.pipe_tensor_pct < 5 and data.pipe_fma_pct > 10:
        findings.append(Finding(
            SEV_WARNING,
            "CUTLASS TensorOp Config But Low Tensor Core Usage ({:.1f}%)".format(
                data.pipe_tensor_pct),
            "Kernel is configured for tensorop but Tensor Core utilization is only {:.1f}%. "
            "FMA utilization is {:.1f}%, suggesting scalar fallback.".format(
                data.pipe_tensor_pct, data.pipe_fma_pct),
            "Check alignment requirements (LD must be multiple of alignment). "
            "Verify InstructionShape matches data type. "
            "Ensure matrix dimensions are multiples of the instruction tile.",
            "CUTLASS Compute",
        ))

    if cfg.is_tensorop and data.pipe_tensor_pct > 50:
        findings.append(Finding(
            SEV_INFO,
            "CUTLASS: Good Tensor Core Utilization ({:.1f}%)".format(
                data.pipe_tensor_pct),
            "Tensor Core pipeline is well-utilized at {:.1f}%.".format(
                data.pipe_tensor_pct),
            "",
            "CUTLASS Compute",
        ))

    if cfg.alignment > 0:
        ld_ratio = data.load_coalescing_ratio()
        if ld_ratio > 4 and cfg.alignment < 8:
            findings.append(Finding(
                SEV_WARNING,
                "CUTLASS: Low Alignment ({}) with Poor Coalescing ({:.1f})".format(
                    cfg.alignment, ld_ratio),
                "CUTLASS alignment={} may cause sub-optimal vectorized loads. "
                "Load coalescing ratio is {:.1f} sectors/req.".format(
                    cfg.alignment, ld_ratio),
                "Increase alignment to 8 (128 bytes). Ensure matrix leading dimensions "
                "are multiples of alignment. Pad if necessary.",
                "CUTLASS Memory",
            ))

    smem_kb = data.shared_mem_per_block_kb
    if smem_kb > 0 and cfg.threadblock_m > 0:
        limiter_name, _ = data.occupancy_limiter()
        if limiter_name == "Shared Memory" and data.warps_active_pct < 40:
            findings.append(Finding(
                SEV_CRITICAL,
                "CUTLASS: Shared Memory Limits Occupancy ({:.1f}%)".format(
                    data.warps_active_pct),
                "Shared memory ({:.1f} KB/block) is the occupancy bottleneck. "
                "ThreadblockShape={}x{}x{}, stages={}. Occupancy: {:.1f}%.".format(
                    smem_kb, cfg.threadblock_m, cfg.threadblock_n,
                    cfg.threadblock_k, cfg.stages, data.warps_active_pct),
                "1) Reduce stages (e.g. 5→3) to free smem. "
                "2) Reduce ThreadblockShape (e.g. 256x128→128x128). "
                "3) On Hopper, use TMA schedule (lower smem requirement).",
                "CUTLASS Occupancy",
            ))
        elif limiter_name == "Registers" and data.warps_active_pct < 40:
            findings.append(Finding(
                SEV_WARNING,
                "CUTLASS: Registers Limit Occupancy ({:.1f}%)".format(
                    data.warps_active_pct),
                "Registers ({:.0f}/thread) are the occupancy bottleneck. "
                "This is common with large WarpShape in CUTLASS.".format(
                    data.registers_per_thread),
                "Use a smaller ThreadblockShape/WarpShape configuration "
                "to reduce per-thread register demand.",
                "CUTLASS Occupancy",
            ))

    total_stalls = data.total_stall_samples()
    if total_stalls > 0:
        long_sb_pct = data.stall_long_scoreboard / total_stalls * 100
        wait_pct = data.stall_wait / total_stalls * 100

        if long_sb_pct > 30:
            stages_hint = ""
            if cfg.stages > 0:
                stages_hint = " Current stages={}. ".format(cfg.stages)
            findings.append(Finding(
                SEV_WARNING,
                "CUTLASS: Long Scoreboard Stall ({:.1f}%) — Increase Stages".format(
                    long_sb_pct),
                "Global memory latency dominates ({:.1f}% of stalls).{}".format(
                    long_sb_pct, stages_hint),
                "Increase pipeline stages (Ampere: 3→5, Hopper: 2→3). "
                "On Hopper, switch to WarpSpecialized schedule with TMA for better latency hiding.",
                "CUTLASS Pipeline",
            ))
        elif wait_pct > 30 and long_sb_pct < 15:
            findings.append(Finding(
                SEV_WARNING,
                "CUTLASS: Wait Stall ({:.1f}%) — Pipeline Over-buffered".format(
                    wait_pct),
                "Async wait dominates ({:.1f}%), long scoreboard is low ({:.1f}%). "
                "Pipeline is deeper than needed.".format(wait_pct, long_sb_pct),
                "Reduce stages to save shared memory and improve occupancy, "
                "or increase ThreadblockShape to add more compute per stage.",
                "CUTLASS Pipeline",
            ))

    if data.l2_hit_rate_pct > 0 and data.l2_hit_rate_pct < 50:
        findings.append(Finding(
            SEV_WARNING,
            "CUTLASS: Low L2 Hit Rate ({:.1f}%) — Consider Swizzle".format(
                data.l2_hit_rate_pct),
            "L2 cache hit rate is {:.1f}%. Default CTA ordering may cause L2 thrashing "
            "for large GEMM.".format(data.l2_hit_rate_pct),
            "Use ThreadblockSwizzle (CUTLASS 2.x: GemmIdentityThreadblockSwizzle<N>, "
            "CUTLASS 3.x: StreamK or tile swizzle) to improve L2 locality.",
            "CUTLASS Memory",
        ))

    if cfg.version == "2.x" and cfg.arch_sm >= 90:
        findings.append(Finding(
            SEV_INFO,
            "CUTLASS 2.x on Hopper — Consider Upgrading to 3.x",
            "Kernel appears to be CUTLASS 2.x running on SM_90 (Hopper). "
            "CUTLASS 3.x provides TMA, warp specialization, and cluster support.",
            "Upgrade to CUTLASS 3.x with WarpSpecialized schedule and TMA loads "
            "for potentially 1.2-1.5x improvement.",
            "CUTLASS Architecture",
        ))

    return findings


def analyze_native_cuda(data):
    """Native CUDA-specific analysis for hand-written kernels."""
    findings = []
    ktype = classify_kernel_type(data.kernel_name)
    if ktype != "native_cuda":
        return findings

    block_size_val = 0
    try:
        block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
    except (ValueError, IndexError):
        pass

    if 0 < block_size_val < 128 and data.warps_active_pct < 50:
        findings.append(Finding(
            SEV_WARNING,
            "CUDA: Small Block Size ({}) May Limit Latency Hiding".format(
                block_size_val),
            "Block size {} provides only {} warps/block. "
            "With low occupancy ({:.1f}%), scheduler may lack eligible warps.".format(
                block_size_val, block_size_val // 32, data.warps_active_pct),
            "Increase block size to 128-256 threads. "
            "Use __launch_bounds__(256, minBlocks) to guide register allocation.",
            "CUDA Launch Config",
        ))

    if data.registers_per_thread >= 128:
        findings.append(Finding(
            SEV_CRITICAL,
            "CUDA: Extreme Register Pressure ({:.0f} regs/thread)".format(
                data.registers_per_thread),
            "Very high register count limits occupancy and may cause spills.",
            "1) Add __launch_bounds__(maxThreads, minBlocks) to control registers. "
            "2) Reduce per-thread work (smaller TM/TN). "
            "3) Move intermediate state to shared memory. "
            "4) Split into multiple simpler kernels.",
            "CUDA Registers",
        ))

    total_stalls = data.total_stall_samples()
    if total_stalls > 0:
        long_sb_pct = data.stall_long_scoreboard / total_stalls * 100
        if long_sb_pct > 30 and data.arch_sm >= 80:
            findings.append(Finding(
                SEV_WARNING,
                "CUDA: High Global Memory Latency ({:.1f}%) — Use Async Copy".format(
                    long_sb_pct),
                "Long scoreboard stalls dominate at {:.1f}%. "
                "On SM_{}, async copy is available.".format(long_sb_pct, data.arch_sm),
                "Replace global→register→shared path with cp.async: "
                "__pipeline_memcpy_async(&smem[i], &gmem[i], size). "
                "Implement double buffering for pipelined execution.",
                "CUDA Async",
            ))

    if data.pipe_fma_pct > 60 and data.pipe_tensor_pct < 5:
        has_fp16 = data.pipe_fma_fp16_pct > 10
        if has_fp16:
            findings.append(Finding(
                SEV_WARNING,
                "CUDA: FP16 Compute Without Tensor Core",
                "FP16 FMA at {:.1f}% but Tensor Core at {:.1f}%. "
                "Missing significant acceleration opportunity.".format(
                    data.pipe_fma_fp16_pct, data.pipe_tensor_pct),
                "Use WMMA API (wmma::mma_sync) or inline PTX (mma.sync) "
                "for 2-8x speedup. Or use CUTLASS/Triton for automatic Tensor Core usage.",
                "CUDA Tensor Core",
            ))

    return findings


def analyze_cutedsl(data):
    """CuTe DSL-specific analysis: detects CuTe DSL kernels and provides targeted suggestions."""
    findings = []
    ktype = classify_kernel_type(data.kernel_name)
    if ktype != "cutedsl":
        return findings

    block_size_val = 0
    try:
        block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
    except (ValueError, IndexError):
        pass

    warps_per_cta = block_size_val // 32 if block_size_val > 0 else 0

    findings.append(Finding(
        SEV_INFO,
        "CuTe DSL Kernel Detected",
        "Kernel '{}' identified as CuTe DSL (cutlass.cute compiled). "
        "threads_per_cta={}, warps_per_cta={}.".format(
            data.kernel_name[:60], block_size_val, warps_per_cta),
        "CuTe DSL kernels are optimized by adjusting threads_per_cta, "
        "elems_per_thread, CopyAtom (num_bits_per_copy), and shared memory usage.",
        "CuTe DSL Detection",
    ))

    if warps_per_cta > 0:
        if data.registers_per_thread >= 128 and warps_per_cta >= 8:
            findings.append(Finding(
                SEV_CRITICAL,
                "CuTe DSL: High Register Pressure ({:.0f} regs) with {} warps".format(
                    data.registers_per_thread, warps_per_cta),
                "threads_per_cta={} ({} warps) combined with {:.0f} registers/thread "
                "severely limits occupancy.".format(
                    block_size_val, warps_per_cta, data.registers_per_thread),
                "Reduce threads_per_cta (try 128 or 256), reduce elems_per_thread, "
                "or add '--maxrregcount=128' to cute.compile() options.",
                "CuTe DSL Occupancy",
            ))
        elif data.registers_per_thread >= 64 and warps_per_cta >= 8:
            findings.append(Finding(
                SEV_WARNING,
                "CuTe DSL: Elevated Register Usage ({:.0f} regs) with {} warps".format(
                    data.registers_per_thread, warps_per_cta),
                "threads_per_cta={} ({} warps) with {:.0f} registers/thread "
                "may limit occupancy.".format(
                    block_size_val, warps_per_cta, data.registers_per_thread),
                "Consider reducing threads_per_cta or elems_per_thread.",
                "CuTe DSL Occupancy",
            ))

    smem_kb = data.shared_mem_per_block_kb
    if smem_kb > 0:
        limiter_name, _ = data.occupancy_limiter()
        if limiter_name == "Shared Memory" and data.warps_active_pct < 40:
            findings.append(Finding(
                SEV_CRITICAL,
                "CuTe DSL: Shared Memory Limits Occupancy ({:.1f}%)".format(
                    data.warps_active_pct),
                "Shared memory ({:.1f} KB/block) is the occupancy bottleneck. "
                "CuTe DSL smem comes from smem.allocate_tensor() and TiledCopy buffers. "
                "Occupancy: {:.1f}%.".format(smem_kb, data.warps_active_pct),
                "Reduce threads_per_cta (fewer warps → smaller reduce buffer), "
                "or reduce elems_per_thread to lower smem requirements.",
                "CuTe DSL Occupancy",
            ))
        elif limiter_name == "Registers" and data.warps_active_pct < 40:
            findings.append(Finding(
                SEV_WARNING,
                "CuTe DSL: Registers Limit Occupancy ({:.1f}%)".format(
                    data.warps_active_pct),
                "Registers ({:.0f}/thread) are the occupancy bottleneck.".format(
                    data.registers_per_thread),
                "Reduce elems_per_thread, reduce threads_per_cta, "
                "or add '--maxrregcount=N' to cute.compile() options.",
                "CuTe DSL Occupancy",
            ))

    total_stalls = data.total_stall_samples()
    if total_stalls > 0:
        long_sb_pct = data.stall_long_scoreboard / total_stalls * 100
        wait_pct = data.stall_wait / total_stalls * 100
        barrier_pct = data.stall_barrier / total_stalls * 100

        if long_sb_pct > 30:
            findings.append(Finding(
                SEV_WARNING,
                "CuTe DSL: Long Scoreboard Stall ({:.1f}%) — Increase Vectorization".format(
                    long_sb_pct),
                "Global memory latency dominates ({:.1f}% of stalls). "
                "CuTe DSL loads may not be fully hiding memory latency.".format(
                    long_sb_pct),
                "1) Increase num_bits_per_copy to 128 in make_copy_atom(). "
                "2) Increase elems_per_thread for more data reuse per thread. "
                "3) On SM_80+, switch CopyAtom to CpAsyncOp for async copies. "
                "4) Consider double-buffering with pipeline commit/wait.",
                "CuTe DSL Memory",
            ))

        if barrier_pct > 25:
            findings.append(Finding(
                SEV_WARNING,
                "CuTe DSL: Barrier Stall ({:.1f}%) — Optimize cta_reduce".format(
                    barrier_pct),
                "Barrier stalls at {:.1f}%. CuTe DSL kernels using cta_reduce() "
                "have sync_threads() overhead proportional to warps_per_cta.".format(
                    barrier_pct),
                "1) Reduce threads_per_cta to reduce warps_per_cta (fewer warps at barrier). "
                "2) If warps_per_cta <= 32, replace second sync_threads with shuffle broadcast. "
                "3) Merge multiple reduce calls into a single cta_reduce. "
                "4) For small reductions, consider warp-only reduce without smem.",
                "CuTe DSL Reduce",
            ))

        if wait_pct > 30 and long_sb_pct < 15:
            findings.append(Finding(
                SEV_WARNING,
                "CuTe DSL: Wait Stall ({:.1f}%) — Pipeline Over-buffered".format(
                    wait_pct),
                "Async wait dominates ({:.1f}%), long scoreboard is low ({:.1f}%). "
                "If using CpAsyncOp, pipeline depth may be excessive.".format(
                    wait_pct, long_sb_pct),
                "Increase elems_per_thread to add more compute per async stage, "
                "or reduce pipeline depth.",
                "CuTe DSL Memory",
            ))

    ld_ratio = data.load_coalescing_ratio()
    if ld_ratio > 8:
        findings.append(Finding(
            SEV_WARNING,
            "CuTe DSL: Uncoalesced Loads ({:.1f} sectors/req) — Check TiledCopy".format(
                ld_ratio),
            "Load coalescing ratio is {:.1f}x (ideal <= 4). CuTe DSL TiledCopy "
            "configuration may produce non-contiguous access.".format(ld_ratio),
            "1) Increase num_bits_per_copy to 128 for wider vectorized loads. "
            "2) Verify t_layout (thread layout) distributes threads along contiguous addresses. "
            "3) Ensure from_dlpack() uses assumed_align=16 for aligned base pointers. "
            "4) Check that v_layout maps elements to contiguous memory.",
            "CuTe DSL Memory",
        ))

    div = data.divergence_pct()
    if div > 20:
        findings.append(Finding(
            SEV_WARNING,
            "CuTe DSL: Thread Divergence ({:.1f}%) — Check Predication".format(div),
            "Divergence is {:.1f}%. CuTe DSL predicated copies (if pred[i]: autovec_copy) "
            "may cause divergence when N is not divisible by threads_per_cta * elems_per_thread.".format(
                div),
            "Adjust threads_per_cta so that threads_per_cta * elems_per_thread "
            "closely matches the problem dimension N to minimize predicated-off threads.",
            "CuTe DSL Divergence",
        ))

    if data.pipe_fma_pct > 60 and data.pipe_tensor_pct < 5 and data.pipe_fma_fp16_pct > 10:
        findings.append(Finding(
            SEV_WARNING,
            "CuTe DSL: FP16 Compute Without Tensor Core (tensor={:.1f}%)".format(
                data.pipe_tensor_pct),
            "FP16 FMA at {:.1f}% but Tensor Core at {:.1f}%. "
            "CuTe DSL is not using MMA atoms.".format(
                data.pipe_fma_fp16_pct, data.pipe_tensor_pct),
            "For GEMM/MatMul operations, use cute.make_mma_atom() with MmaOp to "
            "leverage Tensor Cores. For reduction-only kernels (RMSNorm, LayerNorm), "
            "Tensor Core is typically not applicable.",
            "CuTe DSL Compute",
        ))

    if data.pipe_tensor_pct > 50:
        findings.append(Finding(
            SEV_INFO,
            "CuTe DSL: Good Tensor Core Utilization ({:.1f}%)".format(
                data.pipe_tensor_pct),
            "Tensor Core pipeline is well-utilized at {:.1f}%.".format(
                data.pipe_tensor_pct),
            "",
            "CuTe DSL Compute",
        ))

    return findings


ALL_ANALYZERS = [
    ("Roofline", analyze_roofline),
    ("Memory Hierarchy", analyze_memory),
    ("Warp Stalls", analyze_warp_stalls),
    ("Instruction Mix", analyze_instruction_mix),
    ("Occupancy", analyze_occupancy),
    ("Thread Divergence", analyze_divergence),
    ("CUTLASS", analyze_cutlass),
    ("Triton", analyze_triton),
    ("CuTe DSL", analyze_cutedsl),
    ("Native CUDA", analyze_native_cuda),
]


def run_all_analyzers(data):
    findings = []
    for name, func in ALL_ANALYZERS:
        for f in func(data):
            f.source = name
            findings.append(f)
    findings.sort(key=lambda f: -f.severity)
    return findings


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def status_icon(val, good, bad, higher_is_better=True):
    if higher_is_better:
        if val >= good:
            return "OK"
        elif val >= bad:
            return "Low"
        else:
            return "Very Low"
    else:
        if val <= good:
            return "OK"
        elif val <= bad:
            return "Moderate"
        else:
            return "High"


def _classify_overall(data):
    """Classify overall bottleneck type from roofline metrics."""
    sm = data.sm_throughput_pct
    mem = data.mem_throughput_pct
    if sm > mem + 20:
        return "COMPUTE_BOUND"
    if mem > sm + 20:
        return "MEMORY_BOUND"
    if sm < 40 and mem < 40:
        return "LATENCY_BOUND"
    if sm > 60 and mem > 60:
        return "BALANCED"
    return "COMPUTE_BOUND" if sm > mem else "MEMORY_BOUND"


def _playbook_reference(ktype):
    """Return type-specific playbook guidance for the report."""
    playbooks = {
        "native_cuda": (
            "### Applicable Playbook: Native CUDA\n\n"
            "Focus areas (see SKILL.md Section 3.1 for full details):\n"
            "- **Launch config**: block size (128-256), `__launch_bounds__`, grid coverage\n"
            "- **Memory access**: coalescing (AoS->SoA), vectorized loads (float4), shared memory tiling\n"
            "- **Latency hiding**: `cp.async` + double buffering (SM>=80), TMA (SM>=90)\n"
            "- **Tensor Core**: WMMA / mma.sync for FP16/BF16/TF32 GEMM-like patterns\n"
            "- **Hazards**: shared memory padding (+1) for bank conflicts, `__launch_bounds__` for spills\n"
        ),
        "cutlass": (
            "### Applicable Playbook: CUTLASS\n\n"
            "Focus areas (see SKILL.md Section 3.2 for full details):\n"
            "- **Tile shape**: ThreadblockShape / WarpShape for occupancy vs reuse tradeoff\n"
            "- **Pipeline stages**: increase for long_scoreboard, decrease for wait stalls or smem pressure\n"
            "- **Alignment**: increase to 8 (128 bytes), pad matrix leading dims\n"
            "- **Schedule**: WarpSpecialized + TMA on Hopper, CUTLASS 2.x->3.x upgrade path\n"
            "- **Epilogue fusion**: fuse GEMM + bias + activation to reduce DRAM traffic\n"
            "- **CTA swizzle**: ThreadblockSwizzle / StreamK for L2 locality on large GEMM\n"
        ),
        "triton": (
            "### Applicable Playbook: Triton\n\n"
            "Focus areas (see SKILL.md Section 3.3 for full details):\n"
            "- **num_warps**: reduce if register pressure high (>=64 regs + >=8 warps)\n"
            "- **num_stages**: increase for long_scoreboard stalls, decrease if wait-dominated\n"
            "- **BLOCK_* sizes**: tune BLOCK_M/N/K for register pressure vs compute intensity\n"
            "- **Memory hints**: `tl.multiple_of()`, `tl.max_contiguous()`, `tl.make_block_ptr()`\n"
            "- **Tensor Core**: `allow_tf32=True`, BLOCK_K multiple of 16, correct dtypes\n"
            "- **Autotune**: narrow config space based on NCU bottleneck evidence\n"
        ),
        "cutedsl": (
            "### Applicable Playbook: CuTe DSL\n\n"
            "Focus areas (see SKILL.md Section 3.4 for full details):\n"
            "- **threads_per_cta**: reduce for register/barrier pressure (try 128-256)\n"
            "- **elems_per_thread**: reduce for register pressure, increase for data reuse\n"
            "- **CopyAtom**: `num_bits_per_copy=128` for vectorized loads, CpAsyncOp on SM>=80\n"
            "- **TiledCopy**: verify t_layout for contiguous addressing, assumed_align=16\n"
            "- **cta_reduce**: reduce barrier cost via fewer warps, shuffle broadcast, merged reduces\n"
            "- **Cache**: clear `__pycache__/`, `.cache/`, `/tmp/cutlass_cute_cache/` before re-profiling\n"
        ),
    }
    return playbooks.get(ktype, "")


def generate_report(data, findings):
    lines = []
    w = lines.append

    w("# NCU Deep Performance Analysis Report\n")

    ktype = classify_kernel_type(data.kernel_name)
    ktype_display = {
        "cutlass": "CUTLASS",
        "triton": "Triton",
        "cutedsl": "CuTe DSL",
        "native_cuda": "Native CUDA",
        "library": "Library (cuBLAS/cuDNN)",
    }.get(ktype, "Unknown")

    # Workflow enforcement header
    w("> **Mandatory Workflow**: Profile (Step 1) -> Analyze (Step 2) -> "
      "Change (Step 3) -> Re-profile (Step 4)")
    w("> ")
    w("> This report is the output of **Step 2 (Analysis)**. "
      "Do NOT modify kernel code until you have reviewed findings below.")
    w("> Every code change MUST cite specific NCU metric values from this report as evidence.")
    w("")

    w("## Report Info")
    w("- **Kernel**: `{}`".format(data.kernel_name))
    w("- **Kernel Type**: **{}**".format(ktype_display))
    w("- **Device**: `{}`".format(data.device_name))
    if data.arch_sm:
        w("- **Architecture**: SM_{}".format(data.arch_sm))
    w("- **Duration**: {:.2f} us".format(data.duration_us))
    w("- **Grid**: {}, **Block**: {}".format(data.grid_size, data.block_size))
    if data.registers_per_thread > 0:
        w("- **Registers/Thread**: {:.0f}".format(data.registers_per_thread))
    if data.shared_mem_per_block_kb > 0:
        w("- **Shared Mem/Block**: {:.1f} KB".format(data.shared_mem_per_block_kb))
    w("")

    # Overall classification using decision tree
    overall = _classify_overall(data)
    main = findings[0] if findings else None
    if main:
        w("## Overall Classification: {} ({})\n".format(overall, main.title))

    w("## Roofline Overview\n")
    w("| Metric | Value | Status |")
    w("|------|------|------|")
    w("| SM Throughput | {:.1f}% | {} |".format(
        data.sm_throughput_pct, status_icon(data.sm_throughput_pct, 60, 40)))
    w("| Memory Throughput | {:.1f}% | {} |".format(
        data.mem_throughput_pct, status_icon(data.mem_throughput_pct, 60, 40)))
    w("| DRAM Throughput | {:.1f}% | {} |".format(
        data.dram_throughput_pct, status_icon(data.dram_throughput_pct, 60, 40)))
    w("| Occupancy (Active Warps) | {:.1f}% | {} |".format(
        data.warps_active_pct, status_icon(data.warps_active_pct, 50, 25)))
    if data.warps_eligible_per_cycle > 0:
        elig_status = "OK" if data.warps_eligible_per_cycle >= 2 else (
            "Low" if data.warps_eligible_per_cycle >= 1 else "Very Low")
        w("| Eligible Warps/Cycle | {:.2f} | {} |".format(
            data.warps_eligible_per_cycle, elig_status))
    w("")

    w("## Memory Hierarchy\n")
    w("| Metric | Value | Status |")
    w("|------|------|------|")
    if data.l1_hit_rate_pct > 0:
        w("| L1 Cache Hit Rate | {:.1f}% | {} |".format(
            data.l1_hit_rate_pct, status_icon(data.l1_hit_rate_pct, 50, 20)))
    if data.l2_hit_rate_pct > 0:
        w("| L2 Cache Hit Rate | {:.1f}% | {} |".format(
            data.l2_hit_rate_pct, status_icon(data.l2_hit_rate_pct, 70, 50)))

    ld_ratio = data.load_coalescing_ratio()
    if ld_ratio > 0:
        ld_status = "OK" if ld_ratio <= 4 else (
            "Warning" if ld_ratio <= 8 else "Critical")
        w("| Global Load Coalescing | {:.1f} sectors/req | {} |".format(
            ld_ratio, ld_status))

    st_ratio = data.store_coalescing_ratio()
    if st_ratio > 0:
        st_status = "OK" if st_ratio <= 4 else (
            "Warning" if st_ratio <= 8 else "Critical")
        w("| Global Store Coalescing | {:.1f} sectors/req | {} |".format(
            st_ratio, st_status))

    bc = data.shared_mem_bank_conflicts
    bc_status = "OK" if bc < 100000 else (
        "Warning" if bc < 1000000 else "Critical")
    w("| Bank Conflicts | {:,.0f} | {} |".format(bc, bc_status))

    if data.local_mem_store_sectors > 0:
        w("| Register Spills (local store) | {:,.0f} sectors | Warning |".format(
            data.local_mem_store_sectors))
    if data.dram_read_gbytes > 0 or data.dram_write_gbytes > 0:
        w("| DRAM Read | {:.3f} GB | -- |".format(data.dram_read_gbytes))
        w("| DRAM Write | {:.3f} GB | -- |".format(data.dram_write_gbytes))
    w("")

    breakdown = data.stall_breakdown()
    if breakdown:
        w("## Top Warp Stall Reasons\n")
        w("| Stall Reason | Samples | Percentage |")
        w("|-------------|---------|-----------|")
        for reason, count, pct in breakdown[:7]:
            if pct < 1:
                continue
            w("| {} | {:,.0f} | {:.1f}% |".format(reason, count, pct))
        w("")

    has_pipe = any(v > 0 for v in [
                   data.pipe_fma_pct, data.pipe_alu_pct, data.pipe_lsu_pct, data.pipe_tensor_pct])
    if has_pipe:
        w("## Instruction Mix (Pipeline Utilization)\n")
        w("| Pipeline | Utilization |")
        w("|---------|------------|")
        if data.pipe_fma_pct > 0:
            w("| FMA | {:.1f}% |".format(data.pipe_fma_pct))
        if data.pipe_alu_pct > 0:
            w("| ALU | {:.1f}% |".format(data.pipe_alu_pct))
        if data.pipe_lsu_pct > 0:
            w("| LSU (Load/Store) | {:.1f}% |".format(data.pipe_lsu_pct))
        if data.pipe_tensor_pct > 0:
            w("| Tensor Core | {:.1f}% |".format(data.pipe_tensor_pct))
        if data.pipe_tensor_hmma_pct > 0:
            w("| Tensor Core HMMA | {:.1f}% |".format(
                data.pipe_tensor_hmma_pct))
        if data.pipe_fma_fp16_pct > 0:
            w("| FMA FP16 | {:.1f}% |".format(data.pipe_fma_fp16_pct))
        w("")

    has_limits = any(v > 0 for v in [
        data.occupancy_limit_registers, data.occupancy_limit_shared_mem,
        data.occupancy_limit_warps, data.occupancy_limit_blocks,
    ])
    if has_limits:
        limiter_name, _ = data.occupancy_limiter()
        w("## Occupancy Limiters\n")
        w("| Limiter | blocks/SM | Bottleneck |")
        w("|---------|----------|---------|")
        for name, val in [
            ("Registers", data.occupancy_limit_registers),
            ("Shared Memory", data.occupancy_limit_shared_mem),
            ("Warps", data.occupancy_limit_warps),
            ("Blocks", data.occupancy_limit_blocks),
        ]:
            if val > 0:
                is_bn = "**<-- bottleneck**" if name == limiter_name else ""
                w("| {} | {:.0f} | {} |".format(name, val, is_bn))
        if data.theoretical_occupancy_pct > 0:
            w("\nTheoretical Occupancy: {:.1f}%, Achieved: {:.1f}%".format(
                data.theoretical_occupancy_pct, data.warps_active_pct))
        w("")

    div = data.divergence_pct()
    if data.avg_thread_executed > 0:
        w("## Thread Divergence\n")
        w("| Metric | Value |")
        w("|------|------|")
        w("| Avg Threads Executed | {:.1f} |".format(data.avg_thread_executed))
        w("| Avg Threads Active (True) | {:.1f} |".format(
            data.avg_thread_executed_true))
        w("| Divergence | {:.1f}% |".format(div))
        w("")

    if ktype == "cutlass":
        cfg = parse_cutlass_kernel_name(data.kernel_name)
        w("## CUTLASS Kernel Details\n")
        w("| Attribute | Value |")
        w("|------|------|")
        w("| CUTLASS Version | {} |".format(cfg.version))
        w("| Compute Type | {} |".format(
            "Tensor Core" if cfg.is_tensorop else ("SIMT" if cfg.is_simt else "Unknown")))
        if cfg.instruction_shape:
            w("| Instruction Shape | {} |".format(cfg.instruction_shape))
        if cfg.threadblock_m > 0:
            w("| ThreadblockShape | {}x{}x{} |".format(
                cfg.threadblock_m, cfg.threadblock_n, cfg.threadblock_k))
        if cfg.stages > 0:
            w("| Pipeline Stages | {} |".format(cfg.stages))
        if cfg.layout_a:
            w("| Layout (A/B) | {}/{} |".format(cfg.layout_a, cfg.layout_b))
        if cfg.alignment > 0:
            w("| Alignment | {} |".format(cfg.alignment))
        if cfg.schedule:
            w("| Schedule | {} |".format(cfg.schedule))
        w("| Registers/Thread | {:.0f} |".format(data.registers_per_thread))
        w("| Shared Mem/Block | {:.1f} KB |".format(data.shared_mem_per_block_kb))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        w("| Tensor Core | {} |".format(tc_status))
        w("")

        cutlass_findings = [f for f in findings if f.source and "CUTLASS" in f.source]
        if cutlass_findings:
            w("### CUTLASS Recommendations\n")
            for f in cutlass_findings:
                if f.action:
                    w("- **{}**: {}".format(f.title, f.action))
            w("")

    if ktype == "triton":
        triton_type = classify_triton_kernel_type(data.kernel_name)
        block_size_val = 0
        try:
            block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
        except (ValueError, IndexError):
            pass
        inferred_warps = block_size_val // 32 if block_size_val > 0 else 0

        w("## Triton Kernel Details\n")
        w("| Attribute | Value |")
        w("|------|------|")
        w("| Kernel Type | {} |".format(triton_type))
        w("| Inferred num_warps | {} |".format(inferred_warps if inferred_warps > 0 else "N/A"))
        w("| Registers/Thread | {:.0f} |".format(data.registers_per_thread))
        w("| Shared Mem/Block | {:.1f} KB |".format(data.shared_mem_per_block_kb))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        w("| Tensor Core | {} |".format(tc_status))
        w("")

        triton_findings = [f for f in findings if f.source and "Triton" in f.source]
        if triton_findings:
            w("### Triton Recommendations\n")
            for f in triton_findings:
                if f.action:
                    w("- **{}**: {}".format(f.title, f.action))
            w("")

    if ktype == "cutedsl":
        block_size_val = 0
        try:
            block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
        except (ValueError, IndexError):
            pass
        warps_per_cta = block_size_val // 32 if block_size_val > 0 else 0

        w("## CuTe DSL Kernel Details\n")
        w("| Attribute | Value |")
        w("|------|------|")
        w("| threads_per_cta | {} |".format(block_size_val if block_size_val > 0 else "N/A"))
        w("| warps_per_cta | {} |".format(warps_per_cta if warps_per_cta > 0 else "N/A"))
        w("| Registers/Thread | {:.0f} |".format(data.registers_per_thread))
        w("| Shared Mem/Block | {:.1f} KB |".format(data.shared_mem_per_block_kb))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        w("| Tensor Core | {} |".format(tc_status))
        div = data.divergence_pct()
        if div > 0:
            w("| Thread Divergence | {:.1f}% |".format(div))
        w("")

        cutedsl_findings = [f for f in findings if f.source and "CuTe DSL" in f.source]
        if cutedsl_findings:
            w("### CuTe DSL Recommendations\n")
            for f in cutedsl_findings:
                if f.action:
                    w("- **{}**: {}".format(f.title, f.action))
            w("")

    if ktype == "native_cuda":
        cuda_findings = [f for f in findings if f.source and "CUDA" in f.source
                         and "CUTLASS" not in f.source and "CuTe DSL" not in f.source]
        if cuda_findings:
            w("## Native CUDA Details\n")
            w("### Native CUDA Recommendations\n")
            for f in cuda_findings:
                if f.action:
                    w("- **{}**: {}".format(f.title, f.action))
            w("")

    critical = [f for f in findings if f.severity == SEV_CRITICAL]
    warnings = [f for f in findings if f.severity == SEV_WARNING]
    if critical or warnings:
        w("## Optimization Priorities\n")
        rank = 1
        for f in critical[:3]:
            w("{}. **[CRITICAL]** {}".format(rank, f.title))
            rank += 1
        for f in warnings[:max(0, 5 - len(critical))]:
            w("{}. **[WARNING]** {}".format(rank, f.title))
            rank += 1
        w("")

    w("## Detailed Findings and Actions\n")
    for i, f in enumerate(findings, 1):
        w("### {}. [{}] {}".format(i, SEV_NAMES[f.severity], f.title))
        w("- **Source**: {}".format(f.source))
        w("- **Detail**: {}".format(f.detail))
        if f.action:
            w("- **Action**: {}".format(f.action))
        w("")

    w("## Next Steps\n")
    w("- [ ] Implement the highest-priority optimization")
    w("- [ ] Re-profile: `ncu --set full -o <name>_v2 --target-processes all ./kernel_v2`")
    w("- [ ] Export: `ncu --import <name>_v2.ncu-rep --page raw --csv > <name>_v2.csv`")
    w("- [ ] Compare: `python3 ncu_analyse.py <name>_v2.csv --diff <name>.csv`")
    w("")

    # Structured conclusion matching SKILL.md template
    limiter_name, limiter_val = data.occupancy_limiter()
    overall = _classify_overall(data)
    w("## Conclusion (SKILL.md Step 2.5 Template)\n")
    w("```")
    w("=== Conclusion ===")
    w("Kernel:    {}".format(data.kernel_name[:80]))
    w("Type:      {}".format(ktype_display))
    if data.arch_sm:
        w("Arch:      SM_{}".format(data.arch_sm))
    w("Overall:   {}".format(overall))
    w("Duration:  {:.2f} us".format(data.duration_us))
    w("Roofline:  SM {:.1f}%, MEM {:.1f}%, DRAM {:.1f}%".format(
        data.sm_throughput_pct, data.mem_throughput_pct, data.dram_throughput_pct))
    w("Occupancy: {:.1f}% (theoretical: {:.1f}%), limited by {}".format(
        data.warps_active_pct, data.theoretical_occupancy_pct, limiter_name))
    w("Regs/Thread: {:.0f}, Smem/Block: {:.1f} KB".format(
        data.registers_per_thread, data.shared_mem_per_block_kb))
    w("")
    w("Findings (sorted by severity):")
    for f in findings[:10]:
        w("  [{}] {}: {} -> {}".format(
            SEV_NAMES[f.severity], f.title,
            f.detail[:80], f.action[:80] if f.action else ""))
    w("")
    w("Optimization priorities:")
    rank = 1
    for f in findings:
        if f.severity >= SEV_WARNING and f.action and rank <= 3:
            w("  {}. {} (evidence: {})".format(rank, f.action[:80], f.detail[:60]))
            rank += 1
    w("```")
    w("")

    # Type-specific playbook reference
    playbook_ref = _playbook_reference(ktype)
    if playbook_ref:
        w(playbook_ref)
        w("")

    w("---\n")
    w("> **Reminder**: After implementing changes (Step 3), you MUST re-profile (Step 4) ")
    w("> and compare with `--diff` to verify improvement. Do NOT skip verification.")
    w("")
    w("---\n*Report generated by ncu_analyse.py — profile -> analyze -> change -> verify*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff_kernels(before, after):
    lines = []
    w = lines.append

    w("# NCU Performance Comparison Report (Step 4: Verify)\n")
    w("> This is Step 4 of the mandatory workflow: Profile -> Analyze -> Change -> **Verify**")
    w("")
    w("- **Kernel**: `{}` vs `{}`".format(before.kernel_name, after.kernel_name))
    delta_pct = (after.duration_us - before.duration_us) / \
                 max(before.duration_us, 0.001) * 100
    w("- **Duration**: {:.2f} us -> {:.2f} us (**{:+.1f}%**)".format(
        before.duration_us, after.duration_us, delta_pct))

    before_ktype = classify_kernel_type(before.kernel_name)
    after_ktype = classify_kernel_type(after.kernel_name)
    ktype_display = {
        "cutlass": "CUTLASS", "triton": "Triton", "cutedsl": "CuTe DSL",
        "native_cuda": "Native CUDA", "library": "Library",
    }
    w("- **Kernel Type**: {} -> {}".format(
        ktype_display.get(before_ktype, "Unknown"),
        ktype_display.get(after_ktype, "Unknown")))
    w("")

    w("| Metric | Before | After | Change |")
    w("|--------|--------|-------|--------|")

    def row(name, bv, av, unit="%", invert=False):
        if bv == 0 and av == 0:
            return
        delta = av - bv
        pct = delta / max(abs(bv), 0.001) * 100
        arrow = "+" if delta > 0 else ""
        good = (delta < 0) if invert else (delta > 0)
        icon = "Improved" if good else (
            "Regressed" if not good and abs(pct) > 5 else "--")
        w("| {} | {:.1f}{} | {:.1f}{} | {}{:.1f} ({}{:.0f}%) {} |".format(
            name, bv, unit, av, unit, arrow, delta, arrow, pct, icon))

    row("SM Throughput", before.sm_throughput_pct, after.sm_throughput_pct)
    row("Memory Throughput", before.mem_throughput_pct, after.mem_throughput_pct)
    row("DRAM Throughput", before.dram_throughput_pct,
        after.dram_throughput_pct, invert=True)
    row("Occupancy", before.warps_active_pct, after.warps_active_pct)
    row("L1 Hit Rate", before.l1_hit_rate_pct, after.l1_hit_rate_pct)
    row("L2 Hit Rate", before.l2_hit_rate_pct, after.l2_hit_rate_pct)
    row("Eligible Warps/Cycle", before.warps_eligible_per_cycle,
        after.warps_eligible_per_cycle, unit="")
    row("Bank Conflicts", before.shared_mem_bank_conflicts,
        after.shared_mem_bank_conflicts, unit="", invert=True)
    row("Registers/Thread", before.registers_per_thread,
        after.registers_per_thread, unit="", invert=True)
    row("Smem/Block KB", before.shared_mem_per_block_kb,
        after.shared_mem_per_block_kb, unit=" KB", invert=True)
    row("Duration", before.duration_us, after.duration_us, unit=" us", invert=True)
    w("")

    speedup = before.duration_us / max(after.duration_us, 0.001)
    w("**Speedup: {:.2f}x**".format(speedup))
    w("")

    # Verification checklist
    improved = after.duration_us < before.duration_us
    w("## Verification Checklist\n")
    w("- [{}] Duration improved: {:.2f} us -> {:.2f} us".format(
        "x" if improved else " ", before.duration_us, after.duration_us))

    before_overall = _classify_overall(before)
    after_overall = _classify_overall(after)
    w("- Bottleneck classification: {} -> {}".format(before_overall, after_overall))

    at_ceiling = (after.sm_throughput_pct > 80 or after.dram_throughput_pct > 85)
    w("- [{}] At hardware ceiling (SM>{:.0f}% or DRAM>{:.0f}%)".format(
        "x" if at_ceiling else " ",
        after.sm_throughput_pct, after.dram_throughput_pct))
    w("")

    # Iteration log template (pre-filled)
    w("## Iteration Log (fill in details)\n")
    w("```")
    w("=== Iteration N ===")
    w("Change:       <what was changed and why>")
    w("NCU evidence: <metric>=<before_value> -> <finding>")
    w("")
    w("Result:")
    w("  Duration:      {:.2f} us -> {:.2f} us ({:+.1f}%)".format(
        before.duration_us, after.duration_us, delta_pct))
    w("  SM Throughput:  {:.1f}% -> {:.1f}%".format(
        before.sm_throughput_pct, after.sm_throughput_pct))
    w("  MEM Throughput: {:.1f}% -> {:.1f}%".format(
        before.mem_throughput_pct, after.mem_throughput_pct))
    w("  Occupancy:     {:.1f}% -> {:.1f}%".format(
        before.warps_active_pct, after.warps_active_pct))
    w("")
    decision = "CONTINUE" if improved and not at_ceiling else (
        "STOP -- at ceiling" if at_ceiling else "ROLLBACK -- regression")
    w("Decision: {}".format(decision))
    w("```")
    w("")

    if improved and not at_ceiling:
        w("> **Next**: Continue to Step 1 (re-profile) for the next optimization iteration.")
    elif at_ceiling:
        w("> **Done**: Hardware ceiling reached. Further gains require algorithmic changes.")
    else:
        w("> **Action**: Regression detected. Rollback the change and try a different approach.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(data, findings):
    ktype = classify_kernel_type(data.kernel_name)
    ktype_display = {
        "cutlass": "CUTLASS",
        "triton": "Triton",
        "cutedsl": "CuTe DSL",
        "native_cuda": "Native CUDA",
        "library": "Library (cuBLAS/cuDNN)",
    }.get(ktype, "Unknown")

    print("=" * 72)
    print("  NCU Deep Performance Analysis")
    print("=" * 72)
    print("  Kernel:   {}".format(data.kernel_name[:60]))
    print("  Type:     {}".format(ktype_display))
    print("  Device:   {}".format(data.device_name))
    print("  Duration: {:.2f} us".format(data.duration_us))
    print("  Grid: {}  Block: {}".format(data.grid_size, data.block_size))
    if data.registers_per_thread > 0:
        print("  Regs/Thread: {:.0f}".format(data.registers_per_thread))
    print()

    print("  [Metrics Overview]")
    metrics = [
        ("SM Throughput", data.sm_throughput_pct, "%"),
        ("Memory Throughput", data.mem_throughput_pct, "%"),
        ("DRAM Throughput", data.dram_throughput_pct, "%"),
        ("Occupancy (Active)", data.warps_active_pct, "%"),
        ("L1 Hit Rate", data.l1_hit_rate_pct, "%"),
        ("L2 Hit Rate", data.l2_hit_rate_pct, "%"),
        ("Eligible Warps/Cycle", data.warps_eligible_per_cycle, ""),
        ("Bank Conflicts", data.shared_mem_bank_conflicts, ""),
    ]
    for name, val, unit in metrics:
        if val > 0:
            print("    {:<30s}: {:>10.2f}{}".format(name, val, unit))
    print()

    ld = data.load_coalescing_ratio()
    st = data.store_coalescing_ratio()
    if ld > 0:
        print(
            "    {:<30s}: {:>10.1f} sectors/req".format("Load Coalescing Ratio", ld))
    if st > 0:
        print(
            "    {:<30s}: {:>10.1f} sectors/req".format("Store Coalescing Ratio", st))

    breakdown = data.stall_breakdown()
    if breakdown:
        print()
        print("  [Top Warp Stall Reasons]")
        for reason, count, pct in breakdown[:5]:
            if pct < 1:
                continue
            print("    {:<30s}: {:>6.1f}%  ({:>10,.0f})".format(
                reason, pct, count))

    pipe_data = [
        ("FMA", data.pipe_fma_pct), ("ALU", data.pipe_alu_pct),
        ("LSU", data.pipe_lsu_pct), ("Tensor", data.pipe_tensor_pct),
    ]
    if any(v > 0 for _, v in pipe_data):
        print()
        print("  [Pipeline Utilization]")
        for name, val in pipe_data:
            if val > 0:
                print("    {:<30s}: {:>6.1f}%".format(name, val))

    div = data.divergence_pct()
    if data.avg_thread_executed > 0:
        print()
        print("  [Thread Divergence]: {:.1f}%".format(div))

    if ktype == "cutlass":
        cfg = parse_cutlass_kernel_name(data.kernel_name)
        print()
        print("  [CUTLASS Kernel]")
        print("    {:<30s}: {}".format("Version", cfg.version))
        print("    {:<30s}: {}".format("Compute",
              "Tensor Core" if cfg.is_tensorop else ("SIMT" if cfg.is_simt else "Unknown")))
        if cfg.threadblock_m > 0:
            print("    {:<30s}: {}x{}x{}".format("ThreadblockShape",
                  cfg.threadblock_m, cfg.threadblock_n, cfg.threadblock_k))
        if cfg.stages > 0:
            print("    {:<30s}: {}".format("Stages", cfg.stages))
        if cfg.alignment > 0:
            print("    {:<30s}: {}".format("Alignment", cfg.alignment))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        print("    {:<30s}: {}".format("Tensor Core", tc_status))

    if ktype == "triton":
        triton_type = classify_triton_kernel_type(data.kernel_name)
        block_size_val = 0
        try:
            block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
        except (ValueError, IndexError):
            pass
        inferred_warps = block_size_val // 32 if block_size_val > 0 else 0
        print()
        print("  [Triton Kernel]")
        print("    {:<30s}: {}".format("Type", triton_type))
        print("    {:<30s}: {}".format("Inferred num_warps", inferred_warps if inferred_warps > 0 else "N/A"))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        print("    {:<30s}: {}".format("Tensor Core", tc_status))

    if ktype == "cutedsl":
        block_size_val = 0
        try:
            block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
        except (ValueError, IndexError):
            pass
        warps_per_cta = block_size_val // 32 if block_size_val > 0 else 0
        print()
        print("  [CuTe DSL Kernel]")
        print("    {:<30s}: {}".format("threads_per_cta", block_size_val if block_size_val > 0 else "N/A"))
        print("    {:<30s}: {}".format("warps_per_cta", warps_per_cta if warps_per_cta > 0 else "N/A"))
        print("    {:<30s}: {:.0f}".format("Registers/Thread", data.registers_per_thread))
        print("    {:<30s}: {:.1f} KB".format("Shared Mem/Block", data.shared_mem_per_block_kb))
        tc_status = "Active ({:.1f}%)".format(data.pipe_tensor_pct) if data.pipe_tensor_pct > 5 else "Inactive ({:.1f}%)".format(data.pipe_tensor_pct)
        print("    {:<30s}: {}".format("Tensor Core", tc_status))
        div = data.divergence_pct()
        if div > 10:
            print("    {:<30s}: {:.1f}% (check predication)".format("Thread Divergence", div))

    print()
    print("  [Findings]")
    for i, f in enumerate(findings, 1):
        icons = {SEV_CRITICAL: "!!!", SEV_WARNING: " >>", SEV_INFO: "   "}
        icon = icons.get(f.severity, "   ")
        print("    {}. {} [{}] {}".format(
            i, icon, SEV_NAMES[f.severity], f.title))
        print("         {}".format(f.detail[:100]))
        if f.action:
            print("         -> {}".format(f.action[:100]))
        print()

    # Structured conclusion
    limiter_name, _ = data.occupancy_limiter()
    overall = _classify_overall(data)
    print("  === Conclusion ===")
    print("  Type:      {}".format(ktype_display))
    print("  Overall:   {}".format(overall))
    print("  Duration:  {:.2f} us".format(data.duration_us))
    print("  Roofline:  SM {:.1f}%, MEM {:.1f}%, DRAM {:.1f}%".format(
        data.sm_throughput_pct, data.mem_throughput_pct, data.dram_throughput_pct))
    print("  Occupancy: {:.1f}% (theo: {:.1f}%), limited by {}".format(
        data.warps_active_pct, data.theoretical_occupancy_pct, limiter_name))
    print()

    # Workflow reminder
    print("  *** MANDATORY WORKFLOW ***")
    print("  This is Step 2 (Analysis). Do NOT modify code until findings are reviewed.")
    print("  Step 3: Apply {} playbook based on findings above.".format(ktype_display))
    print("  Step 4: Re-profile and compare with --diff to verify improvement.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NCU Deep Performance Analysis (CSV-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 ncu_analyse.py profile.csv
    python3 ncu_analyse.py profile.ncu-rep
    python3 ncu_analyse.py profile.csv -o analysis.md
    python3 ncu_analyse.py after.csv --diff before.csv
    python3 ncu_analyse.py profile.csv --json
        """,
    )
    parser.add_argument("input", help=".csv or .ncu-rep file")
    parser.add_argument("-o", "--output", help="Save markdown report to file")
    parser.add_argument("--diff", help="Baseline CSV for comparison")
    parser.add_argument("--json", action="store_true",
                        help="Output key metrics as JSON")
    parser.add_argument(
        "--kernel", help="Select specific kernel by name substring")
    parser.add_argument(
        "--type",
        choices=["native_cuda", "cutlass", "triton", "cutedsl"],
        help="Override auto-detected kernel type (useful for CuTe DSL whose "
             "kernel names are generic and hard to auto-detect)")

    args = parser.parse_args()
    input_path = args.input

    global _kernel_type_override
    if args.type:
        _kernel_type_override = args.type

    if input_path.endswith(".ncu-rep"):
        sys.stderr.write("Converting {} to CSV...\n".format(input_path))
        input_path = ncu_rep_to_csv(input_path)
        sys.stderr.write("CSV saved: {}\n".format(input_path))

    if not os.path.exists(input_path):
        sys.stderr.write("Error: {} not found\n".format(input_path))
        sys.exit(1)

    kernels = parse_csv(input_path)
    if not kernels:
        sys.stderr.write("Error: no kernels found in CSV\n")
        sys.exit(1)

    sys.stderr.write("Found {} kernel(s)\n".format(len(kernels)))

    if args.kernel:
        matched = [k for k in kernels if args.kernel in k.kernel_name]
        if not matched:
            sys.stderr.write(
                "Error: no kernel matching '{}'\n".format(args.kernel))
            sys.exit(1)
        data = matched[0]
    else:
        data = select_user_kernel(kernels)

    findings = run_all_analyzers(data)

    if args.json:
        out = {
            "kernel_name": data.kernel_name,
            "device": data.device_name,
            "duration_us": data.duration_us,
            "sm_throughput_pct": data.sm_throughput_pct,
            "mem_throughput_pct": data.mem_throughput_pct,
            "dram_throughput_pct": data.dram_throughput_pct,
            "occupancy_pct": data.warps_active_pct,
            "l1_hit_rate_pct": data.l1_hit_rate_pct,
            "l2_hit_rate_pct": data.l2_hit_rate_pct,
            "load_coalescing_ratio": data.load_coalescing_ratio(),
            "store_coalescing_ratio": data.store_coalescing_ratio(),
            "bank_conflicts": data.shared_mem_bank_conflicts,
            "register_spills": data.local_mem_store_sectors,
            "registers_per_thread": data.registers_per_thread,
            "warps_eligible_per_cycle": data.warps_eligible_per_cycle,
            "divergence_pct": data.divergence_pct(),
            "stall_breakdown": [
                {"reason": r, "count": c, "pct": p}
                for r, c, p in data.stall_breakdown()[:7]
            ],
            "pipe_fma_pct": data.pipe_fma_pct,
            "pipe_lsu_pct": data.pipe_lsu_pct,
            "pipe_tensor_pct": data.pipe_tensor_pct,
            "kernel_type": classify_kernel_type(data.kernel_name),
            "is_triton_kernel": is_triton_kernel(data.kernel_name),
            "is_cutlass_kernel": is_cutlass_kernel(data.kernel_name),
            "is_cutedsl_kernel": classify_kernel_type(data.kernel_name) == "cutedsl",
            "findings": [
                {"severity": SEV_NAMES[f.severity],
                    "title": f.title, "action": f.action}
                for f in findings
            ],
        }
        json_ktype = classify_kernel_type(data.kernel_name)
        if json_ktype == "cutlass":
            cfg = parse_cutlass_kernel_name(data.kernel_name)
            out["cutlass"] = {
                "version": cfg.version,
                "is_tensorop": cfg.is_tensorop,
                "instruction_shape": cfg.instruction_shape,
                "threadblock_shape": "{}x{}x{}".format(
                    cfg.threadblock_m, cfg.threadblock_n, cfg.threadblock_k
                ) if cfg.threadblock_m > 0 else None,
                "stages": cfg.stages if cfg.stages > 0 else None,
                "alignment": cfg.alignment if cfg.alignment > 0 else None,
                "layout": "{}/{}".format(cfg.layout_a, cfg.layout_b) if cfg.layout_a else None,
                "tensor_core_active": data.pipe_tensor_pct > 5,
            }
        if json_ktype == "triton":
            block_size_val = 0
            try:
                block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
            except (ValueError, IndexError):
                pass
            out["triton"] = {
                "kernel_type": classify_triton_kernel_type(data.kernel_name),
                "inferred_num_warps": block_size_val // 32 if block_size_val > 0 else None,
                "shared_mem_kb": data.shared_mem_per_block_kb,
                "tensor_core_active": data.pipe_tensor_pct > 5,
            }
        if classify_kernel_type(data.kernel_name) == "cutedsl":
            block_size_val = 0
            try:
                block_size_val = int(str(data.block_size).strip().split(",")[0].strip("()[] "))
            except (ValueError, IndexError):
                pass
            warps_per_cta = block_size_val // 32 if block_size_val > 0 else 0
            out["cutedsl"] = {
                "threads_per_cta": block_size_val if block_size_val > 0 else None,
                "warps_per_cta": warps_per_cta if warps_per_cta > 0 else None,
                "shared_mem_kb": data.shared_mem_per_block_kb,
                "registers_per_thread": data.registers_per_thread,
                "tensor_core_active": data.pipe_tensor_pct > 5,
                "divergence_pct": data.divergence_pct(),
            }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.diff:
        diff_path = args.diff
        if diff_path.endswith(".ncu-rep"):
            diff_path = ncu_rep_to_csv(diff_path)
        baseline_kernels = parse_csv(diff_path)
        if not baseline_kernels:
            sys.stderr.write(
                "Error: no kernels in baseline {}\n".format(diff_path))
            sys.exit(1)
        baseline = select_user_kernel(baseline_kernels)
        report = diff_kernels(baseline, data)
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            sys.stderr.write("Diff report saved: {}\n".format(args.output))
        else:
            print(report)
        return

    print_summary(data, findings)
    report = generate_report(data, findings)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        sys.stderr.write("Report saved: {}\n".format(args.output))
    else:
        print()
        print(report)


if __name__ == "__main__":
    main()
