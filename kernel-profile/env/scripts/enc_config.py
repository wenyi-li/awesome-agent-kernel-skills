#!/usr/bin/env python3
"""GPU clock locking for stable CUDA kernel benchmarking.

Usage:
    python enc_config.py [--gpu 0]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess


def try_lock_gpu_clocks(gpu_index: int) -> tuple[bool, str]:
    """Lock GPU SM clocks to maximum for stable benchmarking.
    Returns (success, message). Silently skips if nvidia-smi is unavailable."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found; skipping clock lock"
    query = subprocess.run(
        [nvidia_smi, f"--id={gpu_index}", "--query-gpu=clocks.max.sm", "--format=csv,noheader"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    if query.returncode != 0:
        return False, f"nvidia-smi clock query failed (rc={query.returncode})"
    max_clock = query.stdout.strip().replace(" MHz", "").strip()
    if not max_clock.isdigit():
        return False, f"unexpected clock query output: {query.stdout.strip()!r}"
    lock = subprocess.run(
        [nvidia_smi, f"--id={gpu_index}", f"--lock-gpu-clocks={max_clock}"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    if lock.returncode != 0:
        return False, f"clock lock failed (rc={lock.returncode}): {lock.stdout.strip() or lock.stderr.strip()}"
    return True, f"GPU {gpu_index} SM clocks locked to {max_clock} MHz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock GPU SM clocks for stable CUDA kernel benchmarking")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    args = parser.parse_args()

    success, message = try_lock_gpu_clocks(args.gpu)
    print(message)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
