# TileLang Top-Level API Guides

This directory documents the public APIs exported from `import tilelang`.
It complements the `tilelang_language/` guides, which cover
`tilelang.language as T`.

Each topic is split into:

- `basic.md`: the API path to read first.
- `advanced.md`: lower-level behavior, debugging knobs, and target-specific
  details.

## Topics

- `../jit_autotune.md`: short TLDR for using `tilelang.jit` to compile a
  program and `tilelang.autotune` to scan a small config list.
- `jit/basic.md`: first-read guide for `tilelang.jit`, eager calls, lazy
  factory style, `.compile(...)`, `out_idx`, source inspection, and profiling.
- `jit/advanced.md`: advanced JIT details for `tilelang.compile(...)`,
  `tilelang.par_compile(...)`, mode inference, cache keys, pass configs,
  target/backend defaults, and debug artifacts.
- `autotune/basic.md`: first-read guide for `tilelang.autotune`,
  `AutoTuner.from_kernel(...)`, compile/profile arguments, captured inputs, and
  common tuning pitfalls.
- `autotune/advanced.md`: advanced autotune details for config binding,
  explicit caller values, result objects, validation paths, tuning caches,
  worker controls, timeouts, and grouped or multi-GPU benchmarking.
- `env/basic.md`: first-read guide for `tilelang.env`, cache helpers,
  environment defaults, target/backend settings, and import-time behavior.
- `env/advanced.md`: advanced environment details for cache directories,
  target/backend resolution, environment-variable precedence, and debugging
  runtime setup.
- `autodd_tools/basic.md`: first-read guide for AutoDD and the practical debug
  tooling exposed through `tilelang.tools`.
- `autodd_tools/advanced.md`: advanced AutoDD/debug-tool details such as layout
  plotting, generated-source callbacks, and lower-level instrumentation.
