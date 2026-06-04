# Debug Tools For TileLang

Use this page as a short debugging workflow. For exact APIs, use:

- `tilelang/autodd_tools/basic.md` for AutoDD, `T.print`, generated-source
  callbacks, and `plot_layout(...)`.
- `tilelang/autodd_tools/advanced.md` for lower-level instrumentation and
  layout plotting details.
- `god_blessing/pass_config.md` for `pass_configs`, IR dumps, layout
  visualization, fast math, and compiler switch behavior.

## Choose The Debug Path

- Compile failure: inspect generated source and IR dumps, then minimize with
  AutoDD if the source is large.
- Wrong result: compare against a reference, add small `T.print(...)` probes,
  and inspect indexing/copy boundaries.
- Layout or swizzle confusion: enable layout visualization or call
  `tilelang.tools.plot_layout(...)`.
- Pass or lowering question: check `pass_configs` first; many debug outputs are
  controlled by compiler switches.
- Performance issue: first prove correctness, then use profiler tooling outside
  this page.

## Inspect Generated Source

For a compiled `JITKernel`, inspect the generated device source before assuming
the bug is in low-level code:

```python
kernel = my_kernel.compile(...)
print(kernel.get_kernel_source())
```

Generated-source callbacks can intercept source text during compilation:

```python
import tilelang


def show_cuda_source(code, target):
    print(code)
    return code


tilelang.register_cuda_postproc(show_cuda_source)
```

Use callbacks sparingly. They affect compilation globally in the current
process.

## Dump IR And Layouts

Pass configs are the main switchboard for compiler debugging:

```python
import tilelang


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
        tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir",
    }
)
def kernel(...):
    ...
```

For fragment layout debugging:

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

Keep these switches off in normal builds. They are compile-time diagnostics.

## Runtime Prints

Use `T.print(...)` only in small reproductions or narrow conditional branches.
GPU programs have many concurrent lanes, so unguarded prints become noisy.

```python
if bx == 0 and by == 0:
    T.print("value", C_local[0, 0])
```

Prefer printing scalar indices, predicates, and one or two fragment elements.
If the print changes timing-sensitive behavior, switch back to generated source
or IR inspection.

## Minimize With AutoDD

AutoDD repeatedly rewrites and executes a Python source file, keeping smaller
candidates that still emit a selected error substring.

```bash
python -m tilelang.autodd tilelang_buggy.py \
  --err-msg "Dimension mismatch" \
  -o minimized.py \
  -j 4
```

Key options:

| Option | Meaning |
| --- | --- |
| `--err-msg` | Stable substring that identifies the failure. |
| `-o`, `--output` | Minimized output file. |
| `--backend runner` | Faster default execution backend. |
| `--backend subproc` | More isolated backend for unstable cases. |
| `--timeout` | Per-candidate timeout in seconds. |
| `-j` | Parallel reducer workers. |

Freeze setup code that must survive reduction:

```python
from tilelang.autodd import __freeze__

with __freeze__:
    import tilelang
    import tilelang.language as T

shape = __freeze__((128, 128, 128))
```

Comment forms are also supported:

```python
# autodd: freeze-start
import tilelang
import tilelang.language as T
# autodd: end-freeze

kernel = make_kernel()  # autodd: freeze
```

Use a precise error substring. AutoDD checks text in stdout/stderr; it does not
understand exception classes or TileLang semantics.

## Plot Layouts Directly

For standalone layout questions, use `tilelang.tools.plot_layout(...)`:

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(transpose, name="transpose_input", formats="png")
plot_layout(transpose, name="transpose_output", view="output", formats="svg")
```

`plot_layout` is useful for checking logical-to-physical mapping without
compiling a full kernel.

## Practical Checklist

1. Reproduce with cache disabled if stale compilation is possible.
2. Validate against a small reference case.
3. Inspect generated source or dump IR.
4. Add guarded `T.print(...)` only when runtime values matter.
5. Use AutoDD once the failure has a stable command and error substring.
6. Record the minimized source and the exact error in `FAQs.md` if the failure
   is likely to recur.
