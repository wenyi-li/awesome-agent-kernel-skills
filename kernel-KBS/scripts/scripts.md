# kernel_KBS Script Guide

This directory keeps one operational path: query, read, write, update, and delete through SQLite. For day-to-day Codex / Claude Code usage, call `scripts/kbs.py` first. Do not recreate recursive Markdown search scripts.

## Unified Entry Point

| Script | Role | Use |
|---|---|---|
| `kbs.py` | Public CLI | Dispatches SQLite CRUD/FTS commands and exposes `check ...` plus `update-prs`. Normal agents only need this entry point. |
| `kbs_db.py` | SQLite data layer | Defines schema v3, contentless FTS5, CRUD, schema docs, health checks, optimization, and Markdown import/export. Do not bypass `kbs.py` except for debugging. |
| `kbs_checks.py` | Lightweight governance checks | Merges the old freshness, fixtures, and size checks into one offline check entry. |

Common commands:

```bash
python3 scripts/kbs.py query tcgen05 --architecture sm100
python3 scripts/kbs.py get hw-tcgen05-mma --follow-sources
python3 scripts/kbs.py update hw-tcgen05-mma --append tags=mma --reason "add retrieval tag"
python3 scripts/kbs.py delete pr-cutlass-2472
python3 scripts/kbs.py restore pr-cutlass-2472
python3 scripts/kbs.py doctor
python3 scripts/kbs.py optimize
python3 scripts/kbs.py check all
```

`kbs.py check ...` subcommands:

```bash
python3 scripts/kbs.py check size
python3 scripts/kbs.py check freshness
python3 scripts/kbs.py check fixtures
python3 scripts/kbs.py check all
```

Removed behavior: there is no recursive Markdown search entry point. Retrieval must use `kbs.py query` / `kbs.py get`, backed by SQLite FTS.

## Internal Shared Module

| Script | Role | Use |
|---|---|---|
| `_wiki_root.py` | Layout resolver | Resolves shared paths such as `store/docs/sources`, `store/docs/wiki`, `store/docs/ledgers`, `store/corpus/artifacts`, `store/indexes`, and `store/state`. |

## Corpus Refresh And Collection

These scripts write Markdown or artifacts. After running them, rebuild/check SQLite:

```bash
python3 scripts/kbs.py build-db --reset
python3 scripts/kbs.py doctor
python3 scripts/kbs.py optimize
```

| Script | Role | Use |
|---|---|---|
| `update_pr_corpus.py` | Current PR refresh path | Uses the GitHub REST API to fetch merged PR metadata, writes `store/docs/sources/prs`, optionally fetches artifacts, and can rebuild SQLite with `--rebuild-db`. |
| `refresh_candidate_ledger.py` | Candidate ledger refresh | Uses GitHub Search to update candidate ledgers and refresh state. This is a network maintenance tool, not a daily retrieval tool. |
| `compute_core_prs.py` | Core PR derivation | Deterministically generates `core-prs.yaml`, `cute-dsl-universe.yaml`, and `triton-universe.yaml`. |
| `fetch_pr_diff.py` | PR artifact fetcher | Fetches a selected PR's diff/key files into `store/corpus/artifacts/prs`. |
| `extract_blog_code.py` | Blog code extractor | Extracts fenced code blocks from `store/docs/sources/blogs` into `store/corpus/artifacts/blogs`. |
| `collect_contest_code.py` | Contest code collector | Collects public contest submission code into `store/corpus/artifacts/contests` and updates contest source frontmatter. |

## Validation And Governance

| Script | Role | Use |
|---|---|---|
| `validate.py` | Corpus structure validator | Validates frontmatter, controlled vocabulary, links, artifact bundles, provenance, and layout invariants. |
| `verify_core_prs.py` | Core PR reproducibility check | Recomputes core PR manifests in memory and compares them with committed ledgers; `--strict` checks upstream state through `gh`. |
| `verify_verbatim.py` | Artifact byte verifier | Uses `gh` to verify whether `verbatim` / `upstream-patch` artifacts match pinned upstream content. |
| `kbs_checks.py` | Offline governance checks | `freshness` checks local version/ledger freshness, `fixtures` checks DoD fixtures, and `size` checks size budgets. |

`validate.py` defaults to SQLite-importable corpus checks. Manual refresh-round checks are available only when needed:

```bash
python3 scripts/validate.py --include-refresh-governance
```
