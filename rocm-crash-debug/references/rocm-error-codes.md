# ROCm/HIP Error Codes Reference for MI300X/MI325X/MI355X

## HIP Runtime Errors

| Error Code | Name | Common Cause | Debug Level |
|------------|------|-------------|-------------|
| 1 | `hipErrorInvalidValue` | Invalid argument to HIP API | Level 3 (check shapes) |
| 2 | `hipErrorOutOfMemory` | GPU VRAM exhausted | Level 1 + `rocm-smi` |
| 98 | `hipErrorIllegalAddress` | Out-of-bounds memory access | Level 10 (dump inputs) |
| 710 | `hipErrorAssert` | Device-side assert triggered | Level 5 (check NaN/Inf) |
| 719 | `hipErrorLaunchFailure` | Kernel launch config invalid | Level 3 (check dims) |
| 209 | `hipErrorNoBinaryForGpu` | Wrong GPU arch target | `rocminfo` check |

## Linux Signal Codes in Containers

| Signal | Meaning | Action |
|--------|---------|--------|
| 137 (SIGKILL) | OOM killed by host kernel | Check container memory limit, model size, KV cache |
| 139 (SIGSEGV) | Segmentation fault | Enable level 10, check pointer validity |
| 134 (SIGABRT) | Abort (usually assert) | Check kernel API log for last boundary |
| 136 (SIGFPE) | Floating point exception | Enable level 5 for NaN/Inf detection |

## RCCL (ROCm Collective Communication) Errors

| Error Pattern | Meaning | Debug Approach |
|---------------|---------|----------------|
| `NCCL WARN Timeout` | Collective op timed out | Check `rocm-smi --showtopo`, set `NCCL_TIMEOUT` |
| `NCCL WARN Broken pipe` | GPU link failure | Check PCIe/XGMI fabric, `dmesg` for amdgpu errors |
| `unhandled system error` | RCCL internal error | Update ROCm/RCCL version, check GPU health |

## MI355X (gfx950) Specific Issues

| Issue | Symptom | Resolution |
|-------|---------|------------|
| SDMA timeout | Hang during memory copy | Set `HSA_ENABLE_SDMA=0` |
| XGMI link error | Multi-GPU collective fail | Check `rocm-smi --showtopo`, verify link training |
| New ISA instructions | Crash in JIT kernels | Ensure Triton/compiler targets gfx950 explicitly |
| HBM3e ECC errors | Sporadic wrong results | Check `rocm-smi --showras` for correctable errors |

## Diagnostic Command Quick Reference

```bash
# GPU health
rocm-smi --showuse --showmeminfo vram --showtemp --showras

# Topology (XGMI/PCIe links)
rocm-smi --showtopo

# Kernel driver errors
dmesg | grep -i "amdgpu\|drm\|gpu\|fault" | tail -30

# Process GPU memory usage
rocm-smi --showpids --showpidgpus

# Container exit analysis
docker inspect --format='ExitCode={{.State.ExitCode}} OOM={{.State.OOMKilled}}' CONTAINER

# ROCm version
cat /opt/rocm/.info/version
apt list --installed 2>/dev/null | grep rocm
```
