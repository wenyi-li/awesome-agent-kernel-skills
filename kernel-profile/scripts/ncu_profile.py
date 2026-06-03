#!/usr/bin/env python3
"""NCU profiling via nsight-python API.

Replaces manual `ncu` CLI invocation. Uses @nsight.analyze.kernel decorator
and nsight.annotate() context manager to profile kernels programmatically.

Usage:
    python ncu_profile.py kernel.cu --implementation=cuda-cpp --output-dir=./out --M=1024 --N=1024
"""

import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from correctness_check import (
    SUPPORTED_IMPLEMENTATIONS,
    detect_arch,
    _parse_dim_values,
    _setup_backend,
)

import nsight


@dataclass
class TimingConfig:
    num_warmup: int
    num_trials: int
    discard_first: int
    cache_mode: str
    prewarm_calls: int
    device: int

# NCU metrics (replace --section / --set flags)
CORE_METRICS = []

SOL_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]

MEMORY_METRICS = [
    "dram__bytes.sum.per_second",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second",
    "lts__t_bytes.sum.per_second",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
]

COMPUTE_METRICS = [
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg.per_cycle_active",
]

OCCUPANCY_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
]

LAUNCH_METRICS = [
    "launch__block_size",
    "launch__grid_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__waves_per_multiprocessor",
]

SCHEDULER_METRICS = [
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
]

WARP_STALL_METRICS = [
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
]

BRANCH_METRICS = [
    "smsp__sass_branch_targets.sum",
    "smsp__sass_branch_targets_threads_divergent.sum",
]

EXTRA_FULL_METRICS = [
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "l1tex__data_bank_conflicts_pipe_lsu.sum",
    "sm__sass_data_bytes_mem_shared.sum.per_second",
]

FULL_METRICS = (
    CORE_METRICS + SOL_METRICS + MEMORY_METRICS + COMPUTE_METRICS
    + OCCUPANCY_METRICS + LAUNCH_METRICS + SCHEDULER_METRICS
    + WARP_STALL_METRICS + BRANCH_METRICS + EXTRA_FULL_METRICS
)

METRIC_CATEGORIES = [
    ("Speed of Light", SOL_METRICS),
    ("Memory Workload Analysis", MEMORY_METRICS),
    ("Compute Workload Analysis", COMPUTE_METRICS),
    ("Occupancy", OCCUPANCY_METRICS),
    ("Launch Statistics", LAUNCH_METRICS),
    ("Scheduler Statistics", SCHEDULER_METRICS),
    ("Warp State / Stall Reasons", WARP_STALL_METRICS),
    ("Branch Divergence", BRANCH_METRICS),
    ("Additional Pipe Utilization", EXTRA_FULL_METRICS),
    ("Kernel Runtime", CORE_METRICS),
]

METRIC_LABELS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput (% of peak)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "Memory Throughput (% of peak)",
    "dram__bytes.sum.per_second": "DRAM Total Bandwidth (bytes/s)",
    "dram__bytes_read.sum.per_second": "DRAM Read Bandwidth (bytes/s)",
    "dram__bytes_write.sum.per_second": "DRAM Write Bandwidth (bytes/s)",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second": "L1 Global Load Bandwidth (bytes/s)",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second": "L1 Global Store Bandwidth (bytes/s)",
    "lts__t_bytes.sum.per_second": "L2 Total Bandwidth (bytes/s)",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "Global Load Efficiency (%)",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct": "Global Store Efficiency (%)",
    "l1tex__t_sector_hit_rate.pct": "L1 Hit Rate (%)",
    "lts__t_sector_hit_rate.pct": "L2 Hit Rate (%)",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "FMA Pipe Utilization (% of peak)",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "Tensor Core Utilization (% of peak)",
    "smsp__inst_executed.avg.per_cycle_active": "IPC (instructions per cycle)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy (%)",
    "sm__maximum_warps_per_active_cycle_pct": "Theoretical Occupancy (%)",
    "launch__block_size": "Block Size",
    "launch__grid_size": "Grid Size",
    "launch__registers_per_thread": "Registers / Thread",
    "launch__shared_mem_per_block_static": "Static Shared Memory (bytes)",
    "launch__shared_mem_per_block_dynamic": "Dynamic Shared Memory (bytes)",
    "launch__waves_per_multiprocessor": "Waves / SM",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "Issue Slot Utilization (% of peak)",
    "smsp__warps_eligible.avg.per_cycle_active": "Eligible Warps / Cycle",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "Stall: Barrier",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "Stall: Long Scoreboard",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "Stall: Short Scoreboard",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "Stall: Math Pipe Throttle",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "Stall: Wait",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio": "Stall: No Instruction",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": "Stall: Not Selected",
    "smsp__sass_branch_targets.sum": "Branch Targets (total)",
    "smsp__sass_branch_targets_threads_divergent.sum": "Divergent Branch Targets (total)",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed": "FADD Throughput (per cycle)",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed": "FMUL Throughput (per cycle)",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed": "FFMA Throughput (per cycle)",
    "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "LSU Pipe Utilization (% of peak)",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "Warp Execution Efficiency",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "L1 Bank Conflicts (total)",
    "sm__sass_data_bytes_mem_shared.sum.per_second": "Shared Memory Bandwidth (bytes/s)",
}


def combine_metrics(x, y):
    """Combine metrics from multiple kernels: take element-wise maximum."""
    return np.maximum(x, y)


# shared between parent and ncu child process
_kernel_state = None


def _get_kernel_state(solution_file, backend, dim_values, ptr_size, arch, seed):
    global _kernel_state
    if _kernel_state is None:
        _kernel_state = _setup_backend(
            solution_file=solution_file,
            backend_hint=backend,
            dim_values=dim_values,
            ptr_size_override=ptr_size,
            arch=arch,
            seed=seed,
        )
    return _kernel_state


def _summarize_times(times):
    if not times:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p20": 0.0,
            "p80": 0.0,
            "num_trials": 0,
        }
    vals = sorted(float(v) for v in times)

    def pct(p):
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p20": pct(0.20),
        "p80": pct(0.80),
        "num_trials": len(vals),
    }


def _clear_l2_cache_torch(device):
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def _get_triton_cache():
    from triton import runtime as triton_runtime
    return triton_runtime.driver.active.get_empty_cache_for_benchmark()


def _clear_l2_cache_triton(cache):
    from triton import runtime as triton_runtime
    triton_runtime.driver.active.clear_cache(cache)


def _clear_cache(cache_mode, device, triton_cache=None):
    if cache_mode == "hot":
        return
    if cache_mode == "torch":
        _clear_l2_cache_torch(device)
        return
    if cache_mode == "triton":
        if triton_cache is None:
            triton_cache = _get_triton_cache()
        _clear_l2_cache_triton(triton_cache)
        return
    raise ValueError(f"Unsupported cache mode: {cache_mode}")


def _prewarm(fn, calls, device):
    if calls <= 0:
        return
    print(f"[prewarm] running {calls} untimed call(s) to trigger lazy init/JIT...")
    with torch.cuda.device(device):
        for _ in range(calls):
            fn()
        torch.cuda.synchronize(device=device)


def _time_cuda_event(fn, cfg):
    device = torch.device(f"cuda:{cfg.device}")
    times = []
    with torch.cuda.device(device):
        for _ in range(cfg.num_warmup):
            fn()
            torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        triton_cache = _get_triton_cache() if cfg.cache_mode == "triton" else None
        for trial in range(cfg.num_trials + cfg.discard_first):
            torch.cuda.synchronize(device=device)
            _clear_cache(cfg.cache_mode, device, triton_cache)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device=device)
            if trial >= cfg.discard_first:
                times.append(float(start.elapsed_time(end)))
    return times


def time_kernel(solution_file, backend, dim_values, ptr_size, arch, seed, timing_config):
    """Measure kernel latency with CUDA events. Returns timing stats dict."""
    state = _get_kernel_state(solution_file, backend, dim_values, ptr_size, arch, seed)
    fn = state.callable

    _prewarm(fn, timing_config.prewarm_calls, timing_config.device)
    detail = (
        f"method=cuda_event, warmup_calls={timing_config.num_warmup}, "
        f"trials={timing_config.num_trials}, discard_first={timing_config.discard_first}, "
        f"cache={timing_config.cache_mode}"
    )
    print(f"[timing] measuring kernel ({detail})...")

    times = _time_cuda_event(fn, timing_config)
    stats = _summarize_times(times)
    print(
        f"[timing] {stats['mean']:.4f} ms ± {stats['std']:.4f} ms, "
        f"median {stats['median']:.4f} ms, min {stats['min']:.4f} ms, "
        f"max {stats['max']:.4f} ms, n={stats['num_trials']}"
    )
    if stats["num_trials"] < 3:
        print(
            f"[warn] timing produced only {stats['num_trials']} sample(s); "
            "increase --timing-trials for more stable statistics.",
            file=sys.stderr,
        )
    return stats


def run_profile(solution_file, backend, dim_values, ptr_size, arch,
                seed, warmup, output_dir):
    metrics = list(dict.fromkeys(FULL_METRICS))

    @nsight.analyze.kernel(
        metrics=metrics,
        runs=1,
        output="quiet",
        output_csv=False,
        clock_control="none",  # clocks locked externally by enc_config.py
        cache_control="all",
        combine_kernel_metrics=combine_metrics,
    )
    def profile_solve(warmup_count):
        state = _get_kernel_state(solution_file, backend, dim_values, ptr_size, arch, seed)
        fn = state.callable
        for _ in range(warmup_count):
            fn()
        torch.cuda.synchronize()
        with nsight.annotate("solve"):
            fn()

    result = profile_solve(warmup)
    df = result.to_dataframe()
    return df


def format_summary(df, solution_file, dim_values, arch, timing_stats, timing_config):
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    active_metrics = set(FULL_METRICS)
    timing_stats = timing_stats or _summarize_times([])

    lines = [
        "# NCU Profile Summary",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Kernel** | {os.path.basename(solution_file)} |",
        f"| **GPU** | {gpu_name} |",
        f"| **Arch** | {arch} |",
        f"| **Dims** | {dim_values} |",
        f"| **Execution Time** | {timing_stats['mean']:.4f} ms ± {timing_stats['std']:.4f} ms |",
        "",
    ]

    metric_values = {}
    for _, row in df.iterrows():
        metric_name = row.get("Metric", "")
        avg_val = row.get("AvgValue")
        if metric_name and avg_val is not None:
            metric_values[metric_name] = {
                "avg": avg_val,
                "std": row.get("StdDev"),
                "min": row.get("MinValue"),
                "max": row.get("MaxValue"),
                "kernel": row.get("Kernel", ""),
                "stable": row.get("StableMeasurement"),
            }

    for category_name, category_metrics in METRIC_CATEGORIES:
        section_metrics = [m for m in category_metrics if m in active_metrics and m in metric_values]
        if not section_metrics:
            continue

        lines.append(f"## {category_name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for metric in section_metrics:
            label = METRIC_LABELS.get(metric, metric)
            avg = metric_values[metric]["avg"]
            if isinstance(avg, float):
                val_str = f"{avg:.2e}" if avg > 1e6 else f"{avg:.4f}"
            else:
                val_str = str(avg)
            lines.append(f"| {label} | {val_str} |")
        lines.append("")

    if metric_values:
        kernel_name = next(iter(metric_values.values())).get("kernel", "unknown")
        lines.append(f"**Kernel name:** `{kernel_name}`")

    return "\n".join(lines)


def format_details(df):
    lines = [
        "# NCU Profile Details — All Metrics",
        "",
        "| Metric | AvgValue | StdDev | Min | Max | Stable |",
        "|--------|----------:|--------:|----:|----:|:------:|",
    ]

    for _, row in df.iterrows():
        metric = str(row.get("Metric", ""))
        label = METRIC_LABELS.get(metric, metric)
        avg = row.get("AvgValue")
        std = row.get("StdDev")
        mn = row.get("MinValue")
        mx = row.get("MaxValue")
        stable = row.get("StableMeasurement")

        def fmt(v):
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2e}"
            return str(v)

        stable_str = "yes" if stable else ("no" if stable is not None else "N/A")
        lines.append(f"| {label} | {fmt(avg)} | {fmt(std)} | {fmt(mn)} | {fmt(mx)} | {stable_str} |")

    lines.append("")

    extra_cols = ["Annotation", "Kernel", "GPU", "Host", "ComputeClock", "MemoryClock"]
    extra_lines = []
    for col in extra_cols:
        if col in df.columns and not df[col].isna().all():
            val = df[col].iloc[0] if len(df) > 0 else "N/A"
            extra_lines.append(f"| **{col}** | {val} |")
    if extra_lines:
        lines.append("## Context")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.extend(extra_lines)

    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser(
        description="NCU profiling via nsight-python API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to solution file (.cu or .py)")
    parser.add_argument("--implementation", "--impl", "--backend", dest="backend", type=str,
                        default="auto",
                        choices=SUPPORTED_IMPLEMENTATIONS + ("cuda", "cute", "cutedsl", "cpp"),
                        help="Kernel implementation: auto/cuda-cpp/cute-dsl/cutlass/triton (default: auto)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output files")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations before profiling (default: 20)")
    parser.add_argument("--timing-warmup", type=int, default=5,
                        help="Warmup calls for CUDA event timing (default: 5)")
    parser.add_argument("--timing-trials", type=int, default=100,
                        help="Measured trials for CUDA event timing (default: 100)")
    parser.add_argument("--timing-discard-first", type=int, default=1,
                        help="Discard this many first measured timing trials (default: 1)")
    parser.add_argument("--timing-cache-mode", type=str, default="hot",
                        choices=["triton", "torch", "hot"],
                        help="Cache behavior during execution-time measurement: triton clear_cache, torch L2 thrash, or hot")
    parser.add_argument("--prewarm-calls", type=int, default=1,
                        help="Untimed calls before execution-time measurement to trigger lazy init/JIT (default: 1)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args, unknown = parser.parse_known_args()

    dim_values = _parse_dim_values(unknown)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch(args.gpu)
    solution_file = str(Path(args.solution_file).resolve())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_config = TimingConfig(
        num_warmup=args.timing_warmup,
        num_trials=args.timing_trials,
        discard_first=args.timing_discard_first,
        cache_mode=args.timing_cache_mode,
        prewarm_calls=args.prewarm_calls,
        device=args.gpu,
    )

    under_nsight = (
        "CUDA_INJECTION64_PATH" in os.environ
        or "NV_NSIGHT_INJECTION64_PATH" in os.environ
        or any(k.startswith("NV_NSIGHT_") for k in os.environ)
    )

    timing_stats = _summarize_times([])
    if not under_nsight:
        timing_stats = time_kernel(
            solution_file=solution_file,
            backend=args.backend,
            dim_values=dim_values,
            ptr_size=args.ptr_size,
            arch=arch,
            seed=args.seed,
            timing_config=timing_config,
        )

    try:
        df = run_profile(
            solution_file=solution_file,
            backend=args.backend,
            dim_values=dim_values,
            ptr_size=args.ptr_size,
            arch=arch,
            seed=args.seed,
            warmup=args.warmup,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        print(f"[ncu_profile] profiling failed: {exc}", file=sys.stderr)
        return 1

    if under_nsight:
        return 0

    summary_txt = format_summary(df, solution_file, dim_values, arch, timing_stats, timing_config)
    details_txt = format_details(df)

    summary_path = output_dir / "ncu_summary.md"
    details_path = output_dir / "ncu_details.md"

    summary_path.write_text(summary_txt, encoding="utf-8")
    details_path.write_text(details_txt, encoding="utf-8")

    print(f"\n[ncu_profile] summary  -> {summary_path}")
    print(f"[ncu_profile] details  -> {details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
