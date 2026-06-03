#!/usr/bin/env python3
"""
Parse rocprofv3 output and produce structured JSON with bottleneck analysis.

This script reads rocprofv3 CSV output files and produces:
- Per-kernel metrics summary
- Bottleneck classification (compute/memory/lds/latency bound)
- Optimization hints
- Path to raw data for detailed inspection

Usage:
    python3 parse_profile.py <rocprof_output_dir>
    python3 parse_profile.py ./rocprof_output --format json
    python3 parse_profile.py ./rocprof_output --format summary
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# Bottleneck classification thresholds
THRESHOLDS = {
    "valu_busy_high": 70.0,       # VALU busy % indicating compute-bound
    "memory_stall_high": 40.0,    # Memory wait % indicating memory-bound
    "lds_conflict_high": 5.0,     # LDS bank conflict % indicating LDS-bound
    "occupancy_low": 30.0,        # Occupancy % indicating latency-bound
    "l2_hit_low": 50.0,           # L2 hit rate % below which is memory-bound
}


def find_csv_files(output_dir: Path) -> dict:
    """Find all relevant CSV files in the output directory."""
    files = {}
    
    # Counter collection files (in pmc_N directories)
    for pmc_dir in sorted(output_dir.glob("pmc_*")):
        if pmc_dir.is_dir():
            for csv_file in pmc_dir.glob("*counter_collection.csv"):
                pass_num = pmc_dir.name.split("_")[-1]
                files[f"counters_pass_{pass_num}"] = csv_file
    
    # Also check root directory for counter files
    for csv_file in output_dir.glob("*counter_collection.csv"):
        if "counters_pass_1" not in files:
            files["counters_pass_1"] = csv_file
    
    # Kernel trace file
    for csv_file in output_dir.glob("*kernel_trace.csv"):
        files["kernel_trace"] = csv_file
    
    # Stats file
    for csv_file in output_dir.glob("*stats.csv"):
        files["stats"] = csv_file
    
    # HIP trace
    for csv_file in output_dir.glob("*hip_api_trace.csv"):
        files["hip_trace"] = csv_file
    
    return files


def parse_counter_csv(filepath: Path) -> list:
    """Parse counter collection CSV file."""
    kernels = []
    
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel = {
                    "name": row.get("Kernel_Name", "unknown"),
                    "grid_size": int(row.get("Grid_Size", 0)),
                    "workgroup_size": int(row.get("Workgroup_Size", 0)),
                    "lds_size": int(row.get("LDS_Block_Size", 0)),
                    "vgpr_count": int(row.get("VGPR_Count", 0)),
                    "sgpr_count": int(row.get("SGPR_Count", 0)),
                    "dispatch_id": int(row.get("Dispatch_Id", 0)),
                    "counters": {}
                }
                
                # Extract counter values
                counter_name = row.get("Counter_Name", "")
                counter_value = row.get("Counter_Value", "0")
                
                try:
                    kernel["counters"][counter_name] = float(counter_value)
                except ValueError:
                    kernel["counters"][counter_name] = counter_value
                
                kernels.append(kernel)
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}", file=sys.stderr)
    
    return kernels


def parse_kernel_trace(filepath: Path) -> list:
    """Parse kernel trace CSV file."""
    kernels = []
    
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = int(row.get("Start_Timestamp", 0))
                end = int(row.get("End_Timestamp", 0))
                
                kernel = {
                    "name": row.get("Kernel_Name", "unknown"),
                    "duration_ns": end - start,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "grid_size": {
                        "x": int(row.get("Grid_Size_X", 0)),
                        "y": int(row.get("Grid_Size_Y", 0)),
                        "z": int(row.get("Grid_Size_Z", 0)),
                    },
                    "workgroup_size": {
                        "x": int(row.get("Workgroup_Size_X", 0)),
                        "y": int(row.get("Workgroup_Size_Y", 0)),
                        "z": int(row.get("Workgroup_Size_Z", 0)),
                    },
                }
                kernels.append(kernel)
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}", file=sys.stderr)
    
    return kernels


def aggregate_kernel_counters(kernels: list) -> dict:
    """Aggregate counters by kernel name."""
    aggregated = defaultdict(lambda: {
        "name": "",
        "dispatch_count": 0,
        "counters": defaultdict(float),
        "grid_size": 0,
        "workgroup_size": 0,
        "lds_size": 0,
        "vgpr_count": 0,
        "sgpr_count": 0,
    })
    
    for kernel in kernels:
        name = kernel["name"]
        agg = aggregated[name]
        agg["name"] = name
        agg["dispatch_count"] += 1
        agg["grid_size"] = max(agg["grid_size"], kernel.get("grid_size", 0))
        agg["workgroup_size"] = max(agg["workgroup_size"], kernel.get("workgroup_size", 0))
        agg["lds_size"] = max(agg["lds_size"], kernel.get("lds_size", 0))
        agg["vgpr_count"] = max(agg["vgpr_count"], kernel.get("vgpr_count", 0))
        agg["sgpr_count"] = max(agg["sgpr_count"], kernel.get("sgpr_count", 0))
        
        for counter_name, value in kernel.get("counters", {}).items():
            if isinstance(value, (int, float)):
                agg["counters"][counter_name] += value
    
    return dict(aggregated)


def compute_derived_metrics(counters: dict) -> dict:
    """Compute derived metrics from raw counters."""
    metrics = {}
    
    # Get raw counter values with defaults
    sq_waves = counters.get("SQ_WAVES", 0)
    sq_busy = counters.get("SQ_BUSY_CYCLES", 0)
    sq_wait = counters.get("SQ_WAIT_ANY", 0)
    grbm_active = counters.get("GRBM_GUI_ACTIVE", 1)  # Avoid div by zero
    
    valu_insts = counters.get("SQ_INSTS_VALU", 0)
    vmem_insts = counters.get("SQ_INSTS_VMEM", 0)
    salu_insts = counters.get("SQ_INSTS_SALU", 0)
    smem_insts = counters.get("SQ_INSTS_SMEM", 0)
    lds_insts = counters.get("SQ_INSTS_LDS", 0)
    
    tcc_hit = counters.get("TCC_HIT_sum", counters.get("TCC_HIT", 0))
    tcc_miss = counters.get("TCC_MISS_sum", counters.get("TCC_MISS", 0))
    
    lds_conflict = counters.get("SQ_LDS_BANK_CONFLICT", 0)
    
    # Compute derived metrics
    total_insts = valu_insts + vmem_insts + salu_insts + smem_insts + lds_insts
    if total_insts > 0:
        metrics["valu_inst_pct"] = round(100 * valu_insts / total_insts, 2)
        metrics["vmem_inst_pct"] = round(100 * vmem_insts / total_insts, 2)
        metrics["lds_inst_pct"] = round(100 * lds_insts / total_insts, 2)
    
    if grbm_active > 0:
        metrics["gpu_busy_pct"] = round(100 * sq_busy / grbm_active, 2)
    
    if sq_busy > 0:
        metrics["memory_stall_pct"] = round(100 * sq_wait / sq_busy, 2)
    
    tcc_total = tcc_hit + tcc_miss
    if tcc_total > 0:
        metrics["l2_hit_rate_pct"] = round(100 * tcc_hit / tcc_total, 2)
    
    if sq_waves > 0:
        metrics["lds_conflict_per_wave"] = round(lds_conflict / sq_waves, 4)
    
    metrics["total_waves"] = sq_waves
    metrics["total_instructions"] = total_insts
    
    return metrics


def classify_bottleneck(metrics: dict, counters: dict) -> dict:
    """Classify kernel bottleneck based on metrics."""
    scores = {
        "compute_bound": 0,
        "memory_bound": 0,
        "lds_bound": 0,
        "latency_bound": 0,
    }
    reasons = []
    
    # Check compute-bound indicators
    valu_pct = metrics.get("valu_inst_pct", 0)
    if valu_pct > 60:
        scores["compute_bound"] += 2
        reasons.append(f"High VALU instruction ratio ({valu_pct:.1f}%)")
    
    gpu_busy = metrics.get("gpu_busy_pct", 0)
    if gpu_busy > THRESHOLDS["valu_busy_high"]:
        scores["compute_bound"] += 1
        reasons.append(f"High GPU busy cycles ({gpu_busy:.1f}%)")
    
    # Check memory-bound indicators
    vmem_pct = metrics.get("vmem_inst_pct", 0)
    if vmem_pct > 30:
        scores["memory_bound"] += 2
        reasons.append(f"High VMEM instruction ratio ({vmem_pct:.1f}%)")
    
    memory_stall = metrics.get("memory_stall_pct", 0)
    if memory_stall > THRESHOLDS["memory_stall_high"]:
        scores["memory_bound"] += 2
        reasons.append(f"High memory stall cycles ({memory_stall:.1f}%)")
    
    l2_hit = metrics.get("l2_hit_rate_pct", 100)
    if l2_hit < THRESHOLDS["l2_hit_low"]:
        scores["memory_bound"] += 1
        reasons.append(f"Low L2 cache hit rate ({l2_hit:.1f}%)")
    
    # Check LDS-bound indicators
    lds_conflict = metrics.get("lds_conflict_per_wave", 0)
    if lds_conflict > 0.1:
        scores["lds_bound"] += 2
        reasons.append(f"High LDS bank conflicts ({lds_conflict:.3f} per wave)")
    
    lds_pct = metrics.get("lds_inst_pct", 0)
    if lds_pct > 20:
        scores["lds_bound"] += 1
        reasons.append(f"High LDS instruction ratio ({lds_pct:.1f}%)")
    
    # Determine primary bottleneck
    max_score = max(scores.values())
    if max_score == 0:
        bottleneck_type = "balanced"
        confidence = "low"
    else:
        bottleneck_type = max(scores, key=scores.get)
        confidence = "high" if max_score >= 3 else "medium" if max_score >= 2 else "low"
    
    return {
        "type": bottleneck_type,
        "confidence": confidence,
        "scores": scores,
        "reasons": reasons[:3],  # Top 3 reasons
    }


def get_optimization_hints(bottleneck: dict, metrics: dict) -> list:
    """Generate optimization hints based on bottleneck type."""
    hints = []
    btype = bottleneck["type"]
    
    if btype == "compute_bound":
        hints.extend([
            "Consider using matrix instructions (MFMA) if applicable",
            "Optimize ALU operations and reduce register pressure",
            "Consider loop unrolling for better instruction-level parallelism",
        ])
    elif btype == "memory_bound":
        hints.extend([
            "Improve data locality and cache utilization",
            "Use coalesced memory access patterns",
            "Consider using LDS for frequently accessed data",
            "Prefetch data to hide memory latency",
        ])
    elif btype == "lds_bound":
        hints.extend([
            "Reduce LDS bank conflicts by padding shared memory arrays",
            "Reorganize LDS access patterns to avoid conflicts",
            "Consider reducing LDS usage per workgroup",
        ])
    elif btype == "latency_bound":
        hints.extend([
            "Increase occupancy by reducing register/LDS usage",
            "Use more wavefronts to hide latency",
            "Consider smaller workgroup sizes for better scheduling",
        ])
    
    return hints[:3]


def process_results(output_dir: Path) -> dict:
    """Process all profiling results and generate analysis."""
    files = find_csv_files(output_dir)
    
    if not files:
        return {
            "error": f"No profiling output found in {output_dir}",
            "hint": "Ensure rocprofv3 completed successfully and generated CSV files",
        }
    
    result = {
        "output_dir": str(output_dir),
        "files_found": {k: str(v) for k, v in files.items()},
        "kernels": [],
    }
    
    # Parse counter files
    all_counter_kernels = []
    for key, filepath in files.items():
        if key.startswith("counters_pass_"):
            all_counter_kernels.extend(parse_counter_csv(filepath))
    
    # Parse kernel trace
    trace_kernels = []
    if "kernel_trace" in files:
        trace_kernels = parse_kernel_trace(files["kernel_trace"])
    
    # Aggregate counter data by kernel name
    aggregated = aggregate_kernel_counters(all_counter_kernels)
    
    # Process each kernel
    for name, kernel_data in aggregated.items():
        counters = dict(kernel_data["counters"])
        metrics = compute_derived_metrics(counters)
        bottleneck = classify_bottleneck(metrics, counters)
        hints = get_optimization_hints(bottleneck, metrics)
        
        # Find timing from trace if available
        duration_ns = None
        for trace in trace_kernels:
            if trace["name"] == name:
                duration_ns = trace["duration_ns"]
                break
        
        kernel_result = {
            "name": name,
            "dispatch_count": kernel_data["dispatch_count"],
            "config": {
                "grid_size": kernel_data["grid_size"],
                "workgroup_size": kernel_data["workgroup_size"],
                "lds_bytes": kernel_data["lds_size"],
                "vgpr_count": kernel_data["vgpr_count"],
                "sgpr_count": kernel_data["sgpr_count"],
            },
            "metrics": metrics,
            "bottleneck": bottleneck,
            "optimization_hints": hints,
        }
        
        if duration_ns is not None:
            kernel_result["duration_ns"] = duration_ns
            kernel_result["duration_ms"] = round(duration_ns / 1_000_000, 3)
        
        result["kernels"].append(kernel_result)
    
    # If no counter data but trace data exists, use trace only
    if not aggregated and trace_kernels:
        for trace in trace_kernels:
            result["kernels"].append({
                "name": trace["name"],
                "duration_ns": trace["duration_ns"],
                "duration_ms": round(trace["duration_ns"] / 1_000_000, 3),
                "grid_size": trace["grid_size"],
                "workgroup_size": trace["workgroup_size"],
                "bottleneck": {"type": "unknown", "confidence": "none"},
                "note": "Trace-only data; run with --mode counters for bottleneck analysis",
            })
    
    # Add raw data paths
    if files:
        result["raw_data_paths"] = {k: str(v) for k, v in files.items()}
    
    return result


def format_summary(result: dict) -> str:
    """Format result as human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("ROCPROFV3 PROFILING SUMMARY")
    lines.append("=" * 60)
    
    if "error" in result:
        lines.append(f"\nError: {result['error']}")
        if "hint" in result:
            lines.append(f"Hint: {result['hint']}")
        return "\n".join(lines)
    
    lines.append(f"\nOutput directory: {result['output_dir']}")
    lines.append(f"Files found: {len(result.get('files_found', {}))}")
    
    for kernel in result.get("kernels", []):
        lines.append("\n" + "-" * 60)
        lines.append(f"KERNEL: {kernel['name']}")
        lines.append("-" * 60)
        
        if "duration_ms" in kernel:
            lines.append(f"  Duration: {kernel['duration_ms']} ms")
        
        if "config" in kernel:
            cfg = kernel["config"]
            lines.append(f"  Grid size: {cfg.get('grid_size', 'N/A')}")
            lines.append(f"  Workgroup size: {cfg.get('workgroup_size', 'N/A')}")
            lines.append(f"  Registers: VGPR={cfg.get('vgpr_count', 'N/A')}, SGPR={cfg.get('sgpr_count', 'N/A')}")
        
        if kernel.get("metrics"):
            lines.append("\n  KEY METRICS:")
            metrics = kernel["metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.2f}")
                else:
                    lines.append(f"    {key}: {value}")
        
        bottleneck = kernel.get("bottleneck", {})
        lines.append(f"\n  BOTTLENECK: {bottleneck.get('type', 'unknown')} (confidence: {bottleneck.get('confidence', 'N/A')})")
        for reason in bottleneck.get("reasons", []):
            lines.append(f"    - {reason}")
        
        hints = kernel.get("optimization_hints", [])
        if hints:
            lines.append("\n  OPTIMIZATION HINTS:")
            for hint in hints:
                lines.append(f"    \u2022 {hint}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse rocprofv3 output and analyze performance bottlenecks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory containing rocprofv3 output files"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    result = process_results(output_dir)
    
    if args.format == "json":
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
    else:
        print(format_summary(result))


if __name__ == "__main__":
    main()
