#!/usr/bin/env python3
"""
rocprofv3 wrapper script for AMD GPU kernel profiling.

This script wraps rocprofv3 with sensible defaults for common profiling scenarios:
- counters: Collect key performance counters for bottleneck analysis
- trace: Collect kernel execution traces
- full: Collect both counters and traces

Usage:
    python3 rocprof_wrapper.py --mode counters -- ./your_app [args]
    python3 rocprof_wrapper.py --mode trace --output-dir ./results -- ./app
    python3 rocprof_wrapper.py --counters custom.txt -- ./app
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Default counters for bottleneck analysis (organized by pass to respect HW limits)
DEFAULT_COUNTERS = """# Pass 1: Wavefront and instruction counters
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_LDS

# Pass 2: Busy/utilization counters
pmc: SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_ACTIVE_INST_VALU GRBM_GUI_ACTIVE

# Pass 3: Cache counters (L2/TCC)
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum

# Pass 4: LDS and memory counters
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_FLAT_LDS_ONLY TA_BUSY_sum TD_BUSY_sum
"""


def find_rocprofv3():
    """Find rocprofv3 executable."""
    # Check common locations
    candidates = [
        shutil.which("rocprofv3"),
        "/opt/rocm/bin/rocprofv3",
        os.path.expanduser("~/rocm/bin/rocprofv3"),
    ]
    
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    
    # Check if rocprof (legacy) is available as fallback hint
    if shutil.which("rocprof"):
        print("Warning: rocprofv3 not found, but legacy rocprof is available.", file=sys.stderr)
        print("This skill is designed for rocprofv3 (ROCm 6.0+).", file=sys.stderr)
    
    return None


def create_counter_file(output_dir: Path, custom_file: str = None) -> Path:
    """Create counter input file for rocprofv3."""
    counter_file = output_dir / "counters_input.txt"
    
    if custom_file and os.path.exists(custom_file):
        shutil.copy(custom_file, counter_file)
    else:
        counter_file.write_text(DEFAULT_COUNTERS)
    
    return counter_file


def run_profiler(args, app_cmd: list, rocprofv3_path: str) -> int:
    """Run rocprofv3 with specified mode."""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build rocprofv3 command
    cmd = [rocprofv3_path]
    
    # Output options
    cmd.extend(["-d", str(output_dir)])
    cmd.extend(["-o", "profile"])
    cmd.extend(["--output-format", "csv"])
    
    # Mode-specific options
    if args.mode in ("trace", "full"):
        cmd.append("--kernel-trace")
        cmd.append("--stats")
        if args.mode == "full":
            cmd.append("--hip-trace")
    
    if args.mode in ("counters", "full"):
        counter_file = create_counter_file(output_dir, args.counters)
        cmd.extend(["-i", str(counter_file)])
    
    # Kernel filter
    if args.kernel:
        cmd.extend(["--kernel-names", args.kernel])
    
    # Truncate kernel names for readability
    cmd.append("-T")
    
    # Application command
    cmd.append("--")
    cmd.extend(app_cmd)
    
    # Print command for debugging
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    
    # Execute
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Could not execute rocprofv3 at {rocprofv3_path}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="rocprofv3 wrapper for AMD GPU profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode counters -- ./matrix_mult 1024
  %(prog)s --mode trace --output-dir ./results -- ./app
  %(prog)s --counters my_counters.txt -- ./app --size 2048

Modes:
  counters  Collect performance counters for bottleneck analysis (default)
  trace     Collect kernel execution traces (timing only)
  full      Collect both counters and traces
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["counters", "trace", "full"],
        default="counters",
        help="Profiling mode (default: counters)"
    )
    parser.add_argument(
        "--output-dir", "-d",
        default="./rocprof_output",
        help="Output directory (default: ./rocprof_output)"
    )
    parser.add_argument(
        "--counters", "-c",
        help="Custom counter input file (overrides defaults)"
    )
    parser.add_argument(
        "--kernel", "-k",
        help="Target specific kernel by name"
    )
    parser.add_argument(
        "--rocprofv3-path",
        help="Path to rocprofv3 executable (auto-detected if not specified)"
    )
    parser.add_argument(
        "app_cmd",
        nargs=argparse.REMAINDER,
        help="Application command (after --)"
    )
    
    args = parser.parse_args()
    
    # Handle the -- separator
    app_cmd = args.app_cmd
    if app_cmd and app_cmd[0] == "--":
        app_cmd = app_cmd[1:]
    
    if not app_cmd:
        parser.error("No application command specified. Use: script.py [options] -- ./app [args]")
    
    # Find rocprofv3
    rocprofv3_path = args.rocprofv3_path or find_rocprofv3()
    if not rocprofv3_path:
        print("Error: rocprofv3 not found.", file=sys.stderr)
        print("Ensure ROCm is installed and /opt/rocm/bin is in PATH:", file=sys.stderr)
        print("  export PATH=$PATH:/opt/rocm/bin", file=sys.stderr)
        sys.exit(1)
    
    # Run profiler
    ret = run_profiler(args, app_cmd, rocprofv3_path)
    
    if ret == 0:
        print(f"\nProfiling complete. Results in: {args.output_dir}", file=sys.stderr)
        print(f"Parse results with: python3 parse_profile.py {args.output_dir}", file=sys.stderr)
    
    sys.exit(ret)


if __name__ == "__main__":
    main()
