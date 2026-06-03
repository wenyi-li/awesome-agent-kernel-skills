#!/usr/bin/env python3
"""Environment checks for standalone kernel profiling.

Usage:
    python env_check.py [-o report.md] [--gpu 0]
"""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _trim_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"command": _shell_join(cmd), "returncode": 127, "stdout": "", "stderr": str(exc)}
    return {"command": _shell_join(cmd), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


# ---------------------------------------------------------------------------
# search-path helpers
# ---------------------------------------------------------------------------

def _find_cuda_roots() -> list[Path]:
    roots: list[Path] = []
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    return roots


def _find_ncu_roots() -> list[Path]:
    roots: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        nvidia_dir = Path(program_files) / "NVIDIA Corporation"
        if nvidia_dir.exists():
            roots.extend(sorted(nvidia_dir.glob("Nsight Compute*")))
    return roots


def _find_cutlass_roots() -> list[Path]:
    roots: list[Path] = []
    for var in ("CUTLASS_PATH", "CUTLASS_ROOT", "CUTLASS_HOME"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend([Path.cwd(), Path("/usr/local/cutlass"), Path("/opt/cutlass")])
    # dedup preserving order
    seen: set[str] = set()
    deduped: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _ensure_nvidia_cutlass_syspath() -> None:
    """Import nvidia_cutlass_dsl to locate its python_packages directory, then add
    it to sys.path so that ``cutlass`` and ``cutlass.cute`` become importable."""
    try:
        pkg = importlib.import_module("nvidia_cutlass_dsl")
        path = Path(pkg.__path__[0]) / "python_packages"
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------

def _probe_python_module(module_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"importable": False, "version": "", "path": "", "error": ""}
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["importable"] = True
    info["version"] = getattr(mod, "__version__", "unknown")
    info["path"] = getattr(mod, "__file__", "") or ""
    return info


def _resolve_executable(candidate: str, tool_name: str) -> str:
    candidate = candidate.strip().strip('"')
    direct = Path(candidate).expanduser()
    if direct.exists():
        return str(direct.resolve())
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    if any(sep in candidate for sep in ("\\", "/")):
        return ""
    extra_names = [candidate]
    if os.name == "nt" and not Path(candidate).suffix:
        extra_names.extend([f"{candidate}.exe", f"{candidate}.bat", f"{candidate}.cmd"])
    search_roots: list[Path] = []
    if tool_name == "nvcc":
        search_roots.extend(root / "bin" for root in _find_cuda_roots())
    elif tool_name == "ncu":
        search_roots.extend(_find_ncu_roots())
    for root in search_roots:
        for name in extra_names:
            probe = root / name
            if probe.exists():
                return str(probe.resolve())
    return ""


def _probe_executable(candidate: str, tool_name: str, version_args: list[str]) -> dict[str, Any]:
    resolved = _resolve_executable(candidate, tool_name)
    info: dict[str, Any] = {
        "requested": candidate,
        "resolved": resolved,
        "exists": bool(resolved),
        "version_command": "",
        "version_returncode": None,
        "version_output": "",
    }
    if not resolved:
        return info
    probe = _run([resolved, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    info["version_command"] = probe["command"]
    info["version_returncode"] = probe["returncode"]
    info["version_output"] = _trim_output(output)
    return info


def _probe_nvidia_smi() -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    info: dict[str, Any] = {
        "exists": bool(resolved), "resolved": resolved or "",
        "query_command": "", "returncode": None, "query_output": "", "gpus": [],
    }
    if not resolved:
        return info
    primary = _run([resolved, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    probe = primary
    if primary["returncode"] != 0 or not primary["stdout"].strip():
        fallback = _run([resolved, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if fallback["returncode"] == 0 and fallback["stdout"].strip():
            probe = fallback
    info["query_command"] = probe["command"]
    info["returncode"] = probe["returncode"]
    info["query_output"] = _trim_output((probe["stdout"] or probe["stderr"]).strip())
    if probe["returncode"] == 0:
        for line in probe["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append({"name": parts[0], "compute_capability": parts[1], "driver_version": parts[2]})
            elif len(parts) >= 2:
                info["gpus"].append({"name": parts[0], "compute_capability": "", "driver_version": parts[1]})
    return info


def _probe_torch_cuda(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False, "version": "", "cuda_version": "", "cuda_available": False,
        "device_count": 0, "selected_gpu_index": gpu_index, "selected_gpu_name": "",
        "selected_gpu_compute_capability": "", "selected_sm": "", "error": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["importable"] = True
    info["version"] = getattr(torch, "__version__", "")
    info["cuda_version"] = getattr(torch.version, "cuda", "") or ""
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            if 0 <= gpu_index < info["device_count"]:
                info["selected_gpu_name"] = torch.cuda.get_device_name(gpu_index)
                major, minor = torch.cuda.get_device_capability(gpu_index)
                info["selected_gpu_compute_capability"] = f"{major}.{minor}"
                info["selected_sm"] = f"sm_{major}{minor}"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _probe_cutlass_headers() -> dict[str, Any]:
    headers = [
        Path("include/cutlass/cutlass.h"),
        Path("include/cutlass/numeric_types.h"),
        Path("include/cute/tensor.hpp"),
    ]
    searched: list[str] = []
    for root in _find_cutlass_roots():
        searched.append(str(root))
        if all((root / h).exists() for h in headers):
            return {
                "exists": True,
                "root": str(root.resolve()),
                "detail": "found include/cutlass and include/cute headers",
                "searched": searched,
            }
    return {
        "exists": False,
        "root": "",
        "detail": "set CUTLASS_PATH/CUTLASS_ROOT/CUTLASS_HOME to a CUTLASS source/install root "
                  "containing include/cutlass and include/cute",
        "searched": searched,
    }


# ---------------------------------------------------------------------------
# requirements / checklist tracker
# ---------------------------------------------------------------------------

class _Checklist:
    """Tracks requirement items, base errors, and warnings during collection."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self._base_error_count = 0

    def add_base(self, name: str, ok: bool, detail: str) -> None:
        self.items.append({"name": name, "ok": ok, "detail": detail, "required": True})
        if not ok:
            self._base_error_count += 1
            self.errors.append(f"{name}: {detail}")

    def add_impl(self, name: str, ok: bool, detail: str) -> None:
        self.items.append({"name": name, "ok": ok, "detail": detail, "required": False})

    def has_base_errors(self) -> bool:
        return self._base_error_count > 0


# ---------------------------------------------------------------------------
# derived info builders
# ---------------------------------------------------------------------------

def _build_gpu_info(torch_info: dict[str, Any], smi_info: dict[str, Any], gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": torch_info.get("selected_gpu_name", ""),
        "compute_capability": torch_info.get("selected_gpu_compute_capability", ""),
        "sm": torch_info.get("selected_sm", ""),
        "driver_version": "",
        "source": "torch" if torch_info.get("selected_gpu_name") else "",
    }
    gpus = smi_info.get("gpus") or []
    if gpu_index < len(gpus):
        smi_gpu = gpus[gpu_index]
        if smi_gpu.get("name"):
            info["name"] = smi_gpu["name"]
            info["source"] = "nvidia-smi"
        if smi_gpu.get("compute_capability"):
            info["compute_capability"] = smi_gpu["compute_capability"]
            if not info["sm"] and "." in smi_gpu["compute_capability"]:
                major, minor = smi_gpu["compute_capability"].split(".", 1)
                info["sm"] = f"sm_{major}{minor}"
        if smi_gpu.get("driver_version"):
            info["driver_version"] = smi_gpu["driver_version"]
    return info


def _collect_env_vars() -> dict[str, str]:
    return {
        "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CUDA_ROOT": os.environ.get("CUDA_ROOT", ""),
        "CUTLASS_PATH": os.environ.get("CUTLASS_PATH", ""),
        "CUTLASS_ROOT": os.environ.get("CUTLASS_ROOT", ""),
        "CUTLASS_HOME": os.environ.get("CUTLASS_HOME", ""),
    }


def _build_impl_status(
    nvcc_info: dict[str, Any],
    cutlass_headers: dict[str, Any],
    cute_dsl_info: dict[str, Any],
    triton_info: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    nvcc_ok = bool(nvcc_info.get("exists"))
    headers_ok = bool(cutlass_headers.get("exists"))
    cute_ok = bool(cute_dsl_info.get("importable"))
    triton_ok = bool(triton_info.get("importable"))

    def _detail(info: dict[str, Any], ok: bool, ok_key: str, fallback: str) -> str:
        if ok:
            return info.get(ok_key, "") or ""
        return info.get("error") or fallback

    return {
        "cuda-cpp": {
            "ready": nvcc_ok,
            "requirements": ["nvcc executable"],
            "detail": nvcc_info.get("resolved") or "missing nvcc",
        },
        "cutlass": {
            "ready": nvcc_ok and headers_ok,
            "requirements": ["nvcc executable", "CUTLASS headers"],
            "detail": cutlass_headers["root"] if headers_ok else cutlass_headers["detail"],
        },
        "cute-dsl": {
            "ready": cute_ok,
            "requirements": ["cutlass.cute python package"],
            "detail": _detail(cute_dsl_info, cute_ok, "path", "import cutlass.cute failed"),
        },
        "triton": {
            "ready": triton_ok,
            "requirements": ["triton package"],
            "detail": f"triton {triton_info['version']}" if triton_ok else _detail(triton_info, triton_ok, "", "import triton failed"),
        },
    }


# ---------------------------------------------------------------------------
# main collector
# ---------------------------------------------------------------------------

def collect_env_check(gpu_index: int) -> dict[str, Any]:
    cl = _Checklist()

    # -- probe: torch / cuda -------------------------------------------------
    torch_info = _probe_torch_cuda(gpu_index)
    cl.add_base("PyTorch import", torch_info["importable"],
                torch_info["version"] if torch_info["importable"] else (torch_info["error"] or "torch import failed"))
    cuda_ok = torch_info["cuda_available"]
    cl.add_base("CUDA runtime", cuda_ok,
                f"torch CUDA {torch_info['cuda_version']}" if cuda_ok else (torch_info["error"] or "torch.cuda.is_available() returned false"))
    gpu_ok = cuda_ok and 0 <= gpu_index < int(torch_info["device_count"])
    cl.add_base(f"GPU index {gpu_index}", gpu_ok,
                f"{torch_info['selected_gpu_name']} ({torch_info['selected_sm']})" if gpu_ok else f"available device count: {torch_info['device_count']}")

    # -- probe: nvidia-smi ---------------------------------------------------
    smi_info = _probe_nvidia_smi()
    if not smi_info["exists"]:
        cl.warnings.append("nvidia-smi not found; GPU model falls back to PyTorch detection.")
    elif smi_info.get("returncode") not in (None, 0):
        cl.warnings.append("nvidia-smi is present but GPU query failed.")

    # -- derived: gpu info ---------------------------------------------------
    gpu_info = _build_gpu_info(torch_info, smi_info, gpu_index)

    # -- probe: nvcc ---------------------------------------------------------
    nvcc_info = _probe_executable("nvcc", "nvcc", ["--version"])
    cl.add_base("nvcc executable", nvcc_info["exists"], nvcc_info["resolved"] or "cannot resolve nvcc")
    if nvcc_info["exists"] and nvcc_info.get("version_returncode") not in (None, 0):
        cl.warnings.append("nvcc exists but `--version` did not exit cleanly.")

    # -- probe: ncu ----------------------------------------------------------
    ncu_info = _probe_executable("ncu", "ncu", ["--version"])
    cl.add_base("ncu executable", ncu_info["exists"], ncu_info["resolved"] or "cannot resolve ncu")
    if ncu_info["exists"] and ncu_info.get("version_returncode") not in (None, 0):
        cl.warnings.append("ncu exists but `--version` did not exit cleanly.")

    # -- probe: python packages ----------------------------------------------
    nsight_info = _probe_python_module("nsight")
    cl.add_base("nsight-python package", nsight_info["importable"],
                f"nsight {nsight_info['version']}" if nsight_info["importable"] else (nsight_info["error"] or "import nsight failed"))

    triton_info = _probe_python_module("triton")
    cl.add_impl("triton package", triton_info["importable"],
                f"triton {triton_info['version']}" if triton_info["importable"] else (triton_info["error"] or "import triton failed"))

    _ensure_nvidia_cutlass_syspath()

    cutlass_info = _probe_python_module("cutlass")
    cl.add_impl("cutlass python package", cutlass_info["importable"],
                f"cutlass {cutlass_info['version']} ({cutlass_info['path'] or 'path unknown'})"
                if cutlass_info["importable"] else (cutlass_info["error"] or "import cutlass failed"))

    cute_dsl_info = _probe_python_module("cutlass.cute")
    cl.add_impl("cute-dsl python package", cute_dsl_info["importable"],
                f"cutlass.cute {cute_dsl_info['version']} ({cute_dsl_info['path'] or 'path unknown'})"
                if cute_dsl_info["importable"] else (cute_dsl_info["error"] or "import cutlass.cute failed"))

    # -- probe: cutlass headers ----------------------------------------------
    cutlass_headers = _probe_cutlass_headers()
    cl.add_impl("CUTLASS headers", cutlass_headers["exists"],
                cutlass_headers["root"] if cutlass_headers["exists"] else cutlass_headers["detail"])

    # -- build implementation status -----------------------------------------
    impl_status = _build_impl_status(nvcc_info, cutlass_headers, cute_dsl_info, triton_info)
    impl_ready = any(s.get("ready") for s in impl_status.values())
    if not impl_ready:
        cl.errors.append("implementation backend: at least one of cuda-cpp, cute-dsl, cutlass, or triton must be ready")

    return {
        "checked_at": _now_iso(),
        "ready": not cl.has_base_errors() and impl_ready,
        "implementation_ready": impl_ready,
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "selected_gpu_index": gpu_index,
        "requirements": cl.items,
        "implementation_status": impl_status,
        "warnings": cl.warnings,
        "errors": cl.errors,
        "env_vars": _collect_env_vars(),
        "torch": torch_info,
        "nvidia_smi": smi_info,
        "gpu": gpu_info,
        "nvcc": nvcc_info,
        "ncu": ncu_info,
        "nsight_python": nsight_info,
        "triton_python": triton_info,
        "cutlass_python": cutlass_info,
        "cute_dsl_python": cute_dsl_info,
        "cutlass_headers": cutlass_headers,
    }


# ---------------------------------------------------------------------------
# markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []

    # status
    lines.extend([
        "# Environment Check",
        "",
        "## Status",
        f"- ready: {'yes' if result.get('ready') else 'no'}",
        f"- implementation backend ready: {'yes' if result.get('implementation_ready') else 'no'}",
        f"- checked at: {result.get('checked_at', '')}",
        f"- python: {result.get('python_executable', '')}",
        f"- python version: {result.get('python_version', '')}",
        f"- selected gpu index: {result.get('selected_gpu_index')}",
        "",
        "## Requirements",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ])

    # requirements table
    for item in result.get("requirements", []):
        status = "ok" if item.get("ok") else ("missing" if item.get("required") else "unavailable")
        tag = "yes" if item.get("required") else "implementation"
        detail = str(item.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {item.get('name')} ({tag}) | {status} | {detail} |")

    # implementation backends table
    impl_status: dict[str, Any] = result.get("implementation_status") or {}
    lines.extend([
        "",
        "## Implementation Backends",
        "",
        "| Implementation | Ready | Requirements | Detail |",
        "| --- | --- | --- | --- |",
    ])
    for name in ("cuda-cpp", "cute-dsl", "cutlass", "triton"):
        s = impl_status.get(name, {})
        ready = "yes" if s.get("ready") else "no"
        reqs = ", ".join(s.get("requirements", []))
        detail = str(s.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {name} | {ready} | {reqs} | {detail} |")

    # gpu section
    gpu = result.get("gpu") or {}
    torch_info = result.get("torch") or {}
    smi = result.get("nvidia_smi") or {}
    lines.extend([
        "",
        "## GPU",
        f"- model: {gpu.get('name') or 'unknown'}",
        f"- compute capability: {gpu.get('compute_capability') or 'unknown'}",
        f"- sm: {gpu.get('sm') or 'unknown'}",
        f"- driver version: {gpu.get('driver_version') or 'unknown'}",
        f"- torch: {torch_info.get('version') or 'not importable'}",
        f"- torch cuda: {torch_info.get('cuda_version') or 'unknown'}",
        f"- device count: {torch_info.get('device_count')}",
        f"- nvidia-smi: {smi.get('resolved') or 'not found'}",
    ])

    # tools section
    nvcc = result.get("nvcc") or {}
    ncu = result.get("ncu") or {}
    nsight_py = result.get("nsight_python") or {}
    triton_py = result.get("triton_python") or {}
    cutlass_py = result.get("cutlass_python") or {}
    cute_dsl_py = result.get("cute_dsl_python") or {}
    cutlass_h = result.get("cutlass_headers") or {}

    lines.extend([
        "",
        "## Tools",
        f"- nvcc: {nvcc.get('resolved') or 'not found'}",
        f"- nvcc version: {nvcc.get('version_output') or 'n/a'}",
        f"- ncu: {ncu.get('resolved') or 'not found'}",
        f"- ncu version: {ncu.get('version_output') or 'n/a'}",
        f"- nsight-python: {nsight_py.get('version') or 'not importable'}",
        f"- triton: {triton_py.get('version') or 'not importable'}",
        f"- cutlass python: {cutlass_py.get('version') or 'not importable'}",
        f"- cute-dsl python: {cute_dsl_py.get('version') if cute_dsl_py.get('importable') else 'not importable'}",
        f"- CUTLASS headers: {cutlass_h.get('root') or 'not found'}",
    ])

    # env vars
    env_vars: dict[str, str] = result.get("env_vars") or {}
    lines.extend([
        "",
        "## Environment variables",
    ])
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUTLASS_PATH", "CUTLASS_ROOT", "CUTLASS_HOME"):
        lines.append(f"- {var}: {env_vars.get(var) or '(unset)'}")

    # errors & warnings
    lines.extend(["", "## Errors"])
    errs = result.get("errors")
    if errs:
        lines.extend(f"- {e}" for e in errs)
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    warns = result.get("warnings")
    if warns:
        lines.extend(f"- {w}" for w in warns)
    else:
        lines.append("- none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Check environment readiness for CUDA kernel optimization")
    parser.add_argument("-o", "--out", default="", help="Write markdown report to this path (default: print to stdout)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    args = parser.parse_args()

    result = collect_env_check(args.gpu)
    md = render_markdown(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"Env check report written to: {out_path}")
    else:
        print(md)

    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
