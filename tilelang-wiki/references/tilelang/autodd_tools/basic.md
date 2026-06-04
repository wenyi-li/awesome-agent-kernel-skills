# AutoDD And Debug Tools Basics

Use these APIs when a TileLang program fails to compile, produces wrong
results, or needs layout/debug visualization.

## Public Entry Points

`tilelang.tools` is imported as a top-level module:

```python
from tilelang.tools import plot_layout

# Also available after import tilelang:
# tilelang.tools.plot_layout(...)
```

AutoDD is a module command and direct import path, not a symbol exported as
`tilelang.autodd` by the top-level package:

```bash
python -m tilelang.autodd source.py --err-msg "error substring" -o minimized.py
```

```python
from tilelang.autodd import __freeze__
```

Generated-source post-processing callbacks are top-level functions:

```python
import tilelang

tilelang.register_cuda_postproc(callback)
tilelang.register_hip_postproc(callback)
tilelang.register_c_postproc(callback)
```

## First Debugging Steps

Start with the cheapest signal:

1. Print or inspect the TileLang function before compiling.
2. Compile and inspect generated source with `kernel.get_kernel_source()`.
3. Use `T.print(...)` inside small kernels when generated code runs but values
   are wrong.
4. Use AutoDD when a large program has a reproducible command-line failure.
5. Use layout visualization or `plot_layout` for fragment, swizzle, and index
   mapping questions.

## AutoDD: Minimize A Repro

AutoDD rewrites a Python source file and re-runs it. A candidate is kept when a
selected error substring still appears in stdout or stderr.

```bash
python -m tilelang.autodd tilelang_buggy.py \
  --err-msg "Dimension mismatch" \
  -o minimized.py \
  -j 4
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `source` | Python file to minimize. |
| `--err-msg` | Required substring that marks a candidate as still reproducing the failure. |
| `-o`, `--output` | Required output file. AutoDD writes accepted smaller programs here as it runs. |
| `--backend` | `runner` by default; `subproc` is more isolated but slower. |
| `--timeout` | Per-candidate timeout in seconds. Default `60`. |
| `-j`, `--jobs` | Number of parallel reducer workers. Default `1`. |

Choose an error substring that is stable and specific. AutoDD checks text
presence; it does not compare exception types or understand TileLang semantics.

## Freeze Code That Must Stay

Use `__freeze__` for code that AutoDD should not remove:

```python
from tilelang.autodd import __freeze__

with __freeze__:
    import tilelang
    import tilelang.language as T

shape = __freeze__((128, 128, 128))
```

Comment annotations are also converted before reduction:

```python
# autodd: freeze-start
import tilelang
import tilelang.language as T
# autodd: end-freeze

kernel = make_kernel()  # autodd: freeze
```

Use explicit `with __freeze__:` blocks for multi-line statements or string
literals. The single-line comment form is safest for physically single-line
statements.

## Layout Visualization From JIT

TileLang can print and save inferred fragment layouts during compilation by
using layout visualization pass configs:

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "svg",
    }
)
def kernel(...):
    ...
```

Supported format values include:

| Value | Output |
| --- | --- |
| `"txt"` | Textual layout output only. |
| `"png"` | PNG image output. |
| `"pdf"` | PDF image output. |
| `"svg"` | SVG image output. |
| `"all"` | PDF, PNG, and SVG. |
| `"txt,svg"` | Multiple comma-separated choices. |

Keep visualization disabled for normal builds because it is compile-time debug
output.

## Plot A Layout Directly

Use `tilelang.tools.plot_layout` for standalone visualization of `T.Layout` and
`T.Fragment` objects:

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(transpose, name="transpose_4x4", formats="png")
plot_layout(transpose, name="transpose_output", view="output", formats="svg")
```

For `T.Layout`, the default view is the input space. `view="output"` inverts the
view and shows which input coordinate maps into each output cell. For
`T.Fragment`, plots label source elements by thread id and local id.

## Basic Limits

- `plot_layout` expects a `T.Layout` or `T.Fragment`; other objects raise
  `TypeError`.
- Higher-dimensional `T.Layout` objects are flattened for visualization.
- Fragment plots assume a two-dimensional fragment input shape.
- Plotting depends on plotting packages at runtime.
- AutoDD executes generated Python candidates repeatedly; run it only on
  sources you are willing to execute.
