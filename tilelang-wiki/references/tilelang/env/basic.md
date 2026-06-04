# Environment Basics

TileLang exposes a process-wide environment object as `tilelang.env`. Most
users interact with it indirectly through top-level helpers and environment
variables.

```python
import tilelang

tilelang.disable_cache()
assert not tilelang.is_cache_enabled()
tilelang.enable_cache()

print(tilelang.env.get_default_target())
print(tilelang.env.get_default_execution_backend())
```

## Common APIs

| API | Purpose |
| --- | --- |
| `tilelang.env` | Global `Environment` object. Many fields read from `os.environ`. |
| `tilelang.enable_cache()` | Re-enable TileLang kernel caching in the current process. |
| `tilelang.disable_cache()` | Disable TileLang kernel caching in the current process. |
| `tilelang.is_cache_enabled()` | Return whether kernel caching is currently enabled. |
| `tilelang.clear_cache()` | Present at top level, but intentionally disabled. It raises instead of deleting the cache directory. |

`tilelang.disable_cache()` is a runtime switch. It is useful when debugging
whether a compile path is hitting cache.

The environment variable `TILELANG_DISABLE_CACHE=1` is stronger than
`tilelang.enable_cache()`. `is_cache_enabled()` returns `False` when either the
runtime switch is disabled or `TILELANG_DISABLE_CACHE` is truthy.

Truthy values are case-insensitive:

```text
1, true, yes, on
```

## Cache Directories

`TILELANG_CACHE_DIR` is the root for kernel cache files. The default is:

```text
~/.tilelang/cache
```

`TILELANG_TMP_DIR` stores temporary files during compile/cache writes. The
default is a `tmp` subdirectory under the cache root.

The compiled-kernel cache is namespaced by TileLang version and platform. Cache
keys include the lowered function script, output indices, specialization
arguments, target, host target, execution backend, pass configs, compile flags,
TileLang version, platform, and native-library stamp when available.

Practical implication: changing the kernel body, target, backend, pass configs,
or compile flags should produce a different kernel-cache key. Editing unrelated
Python scaffolding may not.

## Clearing Or Bypassing Cache

Do not call `tilelang.clear_cache()` expecting it to delete files. The current
helper raises a `RuntimeError` because deleting the whole cache directory is
considered dangerous.

For a fresh run:

1. Set `TILELANG_DISABLE_CACHE=1` to bypass kernel cache globally.
2. Call `tilelang.disable_cache()` before compiling in the current process.
3. Manually remove a specific cache directory only when you accept the risk.

Autotune has a separate disk-cache switch:

```text
TILELANG_AUTO_TUNING_DISABLE_CACHE=1
```

For a full fresh autotune debug run, use both autotune and kernel cache
switches.

## Target, Backend, And Verbose Defaults

If callers do not pass explicit compile arguments, TileLang reads defaults from
the environment:

| Variable | Default | Used for |
| --- | --- | --- |
| `TILELANG_TARGET` | `auto` | Default compilation target. |
| `TILELANG_EXECUTION_BACKEND` | `auto` | Default runtime execution backend. |
| `TILELANG_VERBOSE` | `0` | Verbose compilation and backend-resolution logs. |

Explicit arguments to `@tilelang.jit`, `tilelang.compile(...)`, or autotuner
compile settings override these defaults.

```bash
TILELANG_TARGET=cuda TILELANG_VERBOSE=1 python run_kernel.py
```

Use `TILELANG_TARGET` carefully in tests because it can change backend
availability and expected behavior.

## Compile Logging And Temporary Files

`TILELANG_PRINT_ON_COMPILATION` controls high-level compile start/end messages.
The default is enabled:

```text
TILELANG_PRINT_ON_COMPILATION=1
```

`TILELANG_VERBOSE` controls lower-level compile/cache/backend logs.

Temporary compiler files are cleaned by default:

```text
TILELANG_CLEANUP_TEMP_FILES=1
```

Set it to `0` when debugging temporary compiler inputs. HIP builds also expose:

```text
TILELANG_HIP_SAVE_TEMP_FILES=0
```

## CUDA And ROCm Discovery

`tilelang.env.CUDA_HOME` and `tilelang.env.ROCM_HOME` are detected during
environment initialization:

```python
import tilelang

print(tilelang.env.CUDA_HOME)
print(tilelang.env.ROCM_HOME)
```

CUDA discovery checks `CUDA_HOME`/`CUDA_PATH`, `nvcc` on `PATH`, the
`nvidia-cuda-nvcc` package when it contains `nvcc`, and standard install
locations. ROCm discovery checks `ROCM_PATH`/`ROCM_HOME`, `hipcc` on `PATH`,
and the standard ROCm install location.

If no location is found, the value is an empty string.

## Import-Time Behavior

Normal `import tilelang` does more than import Python symbols. It initializes
logging, configures library paths, imports TVM/native dependencies, loads the
TileLang native library unless `SKIP_LOADING_TILELANG_SO=1`, and exposes JIT,
profiler, language, autotune, lowering, layout, backend, math, and tool APIs.

Set environment variables that affect import or library discovery before
`import tilelang`.
