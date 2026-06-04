# Pass Config

`pass_configs` carries TileLang compiler and lowering switches.

```python
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
```

The source of truth is `tilelang/transform/pass_config.py`. In this workspace,
the installed enum contains 49 keys.

## Usage

Use `pass_configs` with `@tilelang.jit(...)` or `tilelang.compile(...)`.

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def kernel(...):
    ...
```

```python
jit_kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
```

## Frequent Keys

These keys appear repeatedly in the local examples:

- `TL_ENABLE_FAST_MATH`
- `TL_DISABLE_WARP_SPECIALIZED`
- `TL_DISABLE_TMA_LOWER`

## TileLang Simplification

- `TL_SIMPLIFY`: Default: uses the built-in simplify config dictionary rather than a single scalar default. This is the top-level TileLang simplification config and holds the sub-options listed below.
- `TL_SIMPLIFY_TRANSITIVELY_PROVE_INEQUALITIES`: Default: `False`. This enables transitive inequality proving during simplification.
- `TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS`: Default: `False`. This rewrites boolean expressions into AND-of-ORs form.
- `TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES`: Default: `False`. This uses known constraints to simplify boolean branches.
- `TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_PROVE_CONDITIONAL`: Default: `False`. This propagates known values to prove conditional expressions.
- `TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS`: Default: `False`. This propagates known values to simplify expressions more aggressively.
- `TL_SIMPLIFY_ENABLE_LET_INLINE`: Default: `True`. This enables let-statement inlining inside the simplification pass.

## TileLang Safety And Semantic Checks

- `TL_DISABLE_DATA_RACE_CHECK`: Default: `False`. This disables TileLang's data-race checking.
- `TL_DISABLE_PRELOWER_SEMANTIC_CHECK`: Default: `False`. This disables Python-side semantic checks before lowering.
- `TL_DISABLE_SAFE_MEMORY_ACCESS`: Default: `False`. This disables safe-memory legalization.
- `TL_DISABLE_OUT_OF_BOUND_WARNING`: Default: `True`. This disables out-of-bound warnings emitted by safe-memory legalization.

## TileLang Lowering And Performance

- `TL_DISABLE_WARP_SPECIALIZED` `(frequent)`: Default: `False`. This disables warp-specialization optimization.
- `TL_ENABLE_FAST_MATH` `(frequent)`: Default: `False`. This enables fast math and passes `--use_fast_math` to `nvcc`.
- `TL_DEVICE_COMPILE_FLAGS`: Default: `None`. This appends extra device-compiler flags and accepts either a shell-style string or a list of strings.
- `TL_CONFIG_INDEX_BITWIDTH`: Default: `32`. This sets the bitwidth used for configuration indices.
- `TL_DISABLE_TMA_LOWER` `(frequent, deprecated)`: Default: not explicitly documented in the installed file. This is a legacy flag that prevents plain `T.copy(...)` from auto-lowering to TMA store, and new code should prefer `T.copy(..., disable_tma=True)`.
- `TL_DISABLE_VECTORIZE_256`: Default: `False`. This disables use of LDG/STG 256 vectorization.
- `TL_ENABLE_ASYNC_COPY`: Default: `True`. This enables the cp.async lowering path for eligible copies, especially inside software-pipelined loops.
- `TL_ENABLE_LOWER_LDGSTG`: Default: `False`. This enables non-predicated LDG/STG lowering for global-memory access.
- `TL_ENABLE_LOWER_LDGSTG_PREDICATED`: Default: `False`. This enables predicated LDG/STG lowering for predicated loads and stores.
- `TL_ENABLE_VECTORIZE_PLANNER_VERBOSE`: Default: `False`. This prints verbose vectorization-planner information for debugging vectorization issues.
- `TL_DISABLE_WGMMA`: Default: `False`. This disables Hopper WGMMA lowering.
- `TL_DISABLE_SHUFFLE_ELECT`: Default: `False`. This disables shuffle-election optimization.
- `TL_DISABLE_LOOP_UNSWITCHING`: Default: `False`. This disables loop-unswitching optimization.
- `TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE`: Default: `False`. This allows more aggressive loop unswitching even when the else-path has side effects.
- `TL_DISABLE_THREAD_STORAGE_SYNC`: Default: `False`. This disables automatic synchronization insertion for thread-storage and shared-memory coordination.
- `TL_PTXAS_REGISTER_USAGE_LEVEL`: Default: `None`. This sets PTXAS register-usage aggressiveness on a scale documented as `[0, 10]`.
- `TL_ENABLE_PTXAS_VERBOSE_OUTPUT`: Default: `False`. This enables verbose PTXAS compile output.

## TileLang Memory Planning

- `TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS`: Default: `False`. This enables debug output for shared-memory allocation merging.
- `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`: Default: `False`. This enables a more aggressive shared-memory merge strategy.
- `TL_DISABLE_SHARED_MEMORY_REUSE`: Default: `False`. This keeps merged shared-memory allocations in separate dedicated regions instead of reusing them by lifetime.
- `TL_STORAGE_REWRITE_DETECT_INPLACE`: Default: `False`. This allows StorageRewrite to reuse a read buffer as a write buffer when it can prove the update is safely inplace.

## TileLang Debugging And IR Inspection

- `TL_FORCE_LET_INLINE`: Default: `False`. This forces eager let-inlining before the standard legalization pipeline.
- `TL_AST_PRINT_ENABLE`: Default: `False`. This enables TIR AST printing for debugging.
- `TL_LAYOUT_VISUALIZATION_ENABLE`: Default: `False`. This enables layout-inference visualization output.
- `TL_LAYOUT_VISUALIZATION_FORMATS`: Default: not explicitly documented in the installed file. This selects the visualization format and accepts `"pdf"`, `"png"`, `"svg"`, or `"all"`.
- `TL_ENABLE_DUMP_IR`: Default: `False`. This enables IR dumping between lowering passes.
- `TL_DUMP_IR_DIR`: Default: `./dump_ir/`. This sets the output directory used when IR dumping is enabled.

## TIR Pass Controls

- `TIR_ENABLE_EQUIV_TERMS_IN_CSE`: Default: `True`. This enables equivalent-term matching in TIR common-subexpression elimination.
- `TIR_DISABLE_CSE`: Default: `False`. This disables TIR common-subexpression elimination.
- `TIR_SIMPLIFY`: Default: `True`. This enables the TIR simplification passes.
- `TIR_DISABLE_STORAGE_REWRITE`: Default: `False`. This disables TIR storage-rewrite optimization.
- `TIR_DISABLE_VECTORIZE`: Default: `False`. This disables TIR vectorization.
- `TIR_USE_ASYNC_COPY`: Default: `True`. This enables asynchronous memory-copy operations in TIR lowering.
- `TIR_ENABLE_DEBUG`: Default: `False`. This enables debug information in generated code.
- `TIR_MERGE_STATIC_SMEM`: Default: `True`. This merges static shared-memory allocations.
- `TIR_ADD_LOWER_PASS`: Default: `None`. This injects additional lower passes into the TIR pipeline.
- `TIR_NOALIAS`: Default: `True`. This enables pointer no-alias assumptions.

## Output

- `CUDA_KERNELS_OUTPUT_DIR`: Default: empty string. This sets the output directory for generated CUDA kernels.

## Version Note

This workspace contains one usage of `TL_DISABLE_FAST_MATH`, but that key does
not appear in the installed `PassConfigKey` enum in this environment. Treat it
as stale, version-specific, or incorrect unless your TileLang build defines it.

## Practical Rule

When reading an example, check `pass_configs` early. In this workspace, the
first keys to notice are `TL_ENABLE_FAST_MATH`,
`TL_DISABLE_WARP_SPECIALIZED`, and `TL_DISABLE_TMA_LOWER`.
