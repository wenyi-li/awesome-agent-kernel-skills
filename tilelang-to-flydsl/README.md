# tilelang-to-flydsl skill

A Claude Code skill that teaches an agent how to port kernels written in
**TileLang** (the `@T.prim_func` DSL used by tile-ai/tilelang and
deepseek-ai/TileKernels) into equivalent **FlyDSL** (Python DSL on top of
the MLIR `fly` dialect, targeting AMD ROCm) kernels — while keeping the
existing pytest test suite as the correctness gate.

Designed for the case where:

- The target codebase has both a TileLang reference kernel and a FlyDSL
  installation available, but
- The agent cannot necessarily build or run either project from its
  environment, so validation is review-based.

## Files

- `SKILL.md` — entry point with the frontmatter Claude Code uses to surface
  the skill. Describes when to invoke and the high-level workflow.
- `references/api_mapping.md` — symbol-by-symbol translation table.
- `references/idioms.md` — side-by-side patterns for the five archetypes.
- `references/gotchas.md` — review checklist; **read before declaring a
  conversion done**.
- `references/workflow.md` — step-by-step procedure per kernel.
- `references/worked_examples/` — full conversions:
  - `normalize_weight.md` — small, scalar reduction.
  - `batched_transpose.md` — medium, shared-memory rearrange + strided input.
  - `gemm_skeleton.md` — annotated minimal MFMA GEMM (skeleton only).

## Installing in another project

The skill is a plain directory. To use it in a different repository, drop
it into the target repo's `.claude/skills/` tree so Claude Code picks it
up automatically:

```sh
git clone https://github.com/<your-org>/tilelang-to-flydsl-skills.git /tmp/tk2fly
mkdir -p <your-repo>/.claude/skills
cp -r /tmp/tk2fly/.claude/skills/tilelang-to-flydsl <your-repo>/.claude/skills/
```

Then in a Claude Code session inside that repo the skill will be listed
under available skills and triggered by descriptions matching its
frontmatter (e.g. "port this TileLang kernel to FlyDSL").

## Out of scope

This skill does not:

- Tune performance — see `gemm-optimization`, `lds-optimization`,
  `prefetch-data-load`, `kernel-trace-analysis`.
- Build / configure the FlyDSL toolchain — see `build-flydsl`.
- Debug a converted kernel that runs but is wrong — see
  `debug-flydsl-kernel`.

After a port: use this skill to produce a port → run the user's pytest
suite → if a test fails, hand off to `debug-flydsl-kernel`.
