# tle-developer-flagos: TLE Kernel and Feature Development Skill

[中文版本](README_zh.md)

## Overview

`tle-developer-flagos` is a self-contained skill for end-to-end TLE work, including kernel optimization, compiler feature implementation, and correctness/performance debugging with reproducible validation.

This skill is designed to keep TLE development disciplined and auditable by enforcing a fixed workflow:

`intake -> implementation -> validation -> artifacts -> merge decision`

## When to Use

Use this skill when you need to:

- Write or optimize TLE kernels.
- Implement TLE API / verifier / lowering / pipeline features.
- Debug TLE correctness, performance, or regression issues.

Typical trigger phrases:

- `write a TLE kernel`
- `optimize TLE operator`
- `debug TLE local_ptr`

## Usage

```bash
/tle-developer-flagos
```

Recommended input format:

```text
Goal:
Non-goal:
Acceptance:
Impact scope (optional):
```

## Working Contract

- Treat `references/tle-sources.md` as the technical source of truth.
- Treat `references/workflow-templates.md` as the workflow/template source of truth.
- Do not rely on docs outside this skill folder.
- Do not assume fixed Python environment names or build script names.

## Directory Structure

```text
skills/tle-developer-flagos/
├── SKILL.md
├── README.md
├── LICENSE.txt
├── agents/
│   └── openai.yaml
└── references/
    ├── tle-sources.md
    └── workflow-templates.md
```

## Files

- `SKILL.md`: skill entry, trigger conditions, guardrails, required outputs, completion checklist.
- `agents/openai.yaml`: assistant-facing interface metadata and default prompt.
- `references/tle-sources.md`: environment setup, development map, debugging and optimization references.
- `references/workflow-templates.md`: templates for intake, validation matrix, fix summary, lessons, and merge package.

## License

This skill is distributed under Apache License 2.0. See `LICENSE.txt`.
