# Contributing

Thanks for helping improve Awesome Agent Kernel Skills.

## What Belongs Here

Add skills that help AI agents write, test, debug, profile, or optimize kernels and accelerator code. Good candidates include skills for CUDA, Triton, TileLang, HIP, ROCm, MLIR, profiling tools, kernel verification, and framework integration.

Avoid adding general blog posts, papers, tutorials, or benchmark repositories unless they are packaged as agent skills or directly include reusable skill instructions.

## Entry Format

Each entry should include:

- Skill name.
- Source URL.
- One concise sentence describing what the skill helps an agent do.
- Category.
- Commit, tag, or release when available.
- Install path when the skill is inside a larger repository.

Example:

```markdown
- [skill-name](https://github.com/org/repo/tree/main/path/to/skill) - Short description. Target path: `skill-name/`. Commit: `abcdef...`.
```

## Categories

Use an existing category when possible:

- CUDA Kernel Development
- TileLang Kernel Development
- Profiling and Performance Analysis
- Debugging and Correctness
- Testing

If a new category is needed, keep it specific and add it to the README contents.

## Review Checklist

Before opening a PR:

- The source URL is public and points directly to the skill or skill repository.
- The description explains the agent workflow, not just the underlying tool.
- The entry is placed in one primary category.
- The README table of contents still matches the headings.
- New local skill folders, if added, contain a `SKILL.md`.
