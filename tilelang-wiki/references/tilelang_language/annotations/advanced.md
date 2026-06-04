# Annotations: Advanced

This page covers less common annotations and function-level metadata. These
APIs are useful for targeted compiler behavior, but they should not replace
clear data movement and layout choices in the kernel body.

## Safe Values

`T.annotate_safe_value(safe_value_map)` maps buffer data variables to fallback
values used by later passes:

```python
T.annotate_safe_value({tmp: 0.0})
```

Use this only when a transformation needs a known value for masked or
out-of-bounds behavior. The keys are buffers; TileLang stores the mapping on
their underlying data variables.

## L2 Hit-Ratio Hints

`T.annotate_l2_hit_ratio(l2_hit_ratio_map)` attaches persistent-L2 style hints
to global buffers:

```python
T.annotate_l2_hit_ratio({A: 0.75})
```

Every key must be a global-scope buffer. Non-global buffers assert. The values
are converted to `float32` immediates.

## Compile Flags

`T.annotate_compile_flags(flags)` attaches function-level compile flags from
inside a JIT factory or kernel definition:

```python
T.annotate_compile_flags([
    "-O3",
    "--use_fast_math",
])
```

`flags` may be a string or a list of strings. Externally supplied compile flags
are merged with function-level flags.

Use this when a kernel requires specific CUDA compilation flags. Prefer
decorator-level `compile_flags=` when the flags are part of the public kernel
configuration.

## Pass Configs

`T.annotate_pass_configs(configs)` attaches function-level pass metadata:

```python
T.annotate_pass_configs({
    "tir.UnrollLoop": {"auto_max_step": 128},
})
```

`configs` must be a dictionary. Externally supplied pass configs take priority
over function-level configs. These helpers require a TileLang builder context
and are ignored during the first eager JIT phase.

Use pass configs sparingly. A pass-level workaround should explain the specific
lowering behavior it is controlling.

