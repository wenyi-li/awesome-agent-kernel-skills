# kernel_KBS Store Layout

`store/` is the persistent data root for kernel_KBS. Agents should understand these functional zones before deciding whether to query SQLite, read Markdown, or inspect provenance/artifacts.

The daily read/write entry point is `scripts/kbs.py`. Do not add or restore recursive Markdown retrieval entry points. Query, read, write, update, and delete operations are centered on `store/indexes/kernel_kbs.sqlite` and exposed through `scripts/kbs.py`.

## Top-Level Directories

| Directory | Functional zone | Agent use |
|---|---|---|
| `config/` | KBS layout and refresh configuration. | Read this when you need directory rules, corpus scope, or PR refresh sources. Do not place content pages here. |
| `schemas/` | Frontmatter schemas, controlled tags, and alias vocabulary. | Check this before adding/updating document metadata to avoid invalid tags, architectures, languages, and related fields. |
| `docs/` | Markdown document layer: source notes, curated wiki pages, and ledgers. | This is the canonical full-body Markdown store. Retrieval still starts with SQLite, then follows the returned `path` to a specific file. |
| `corpus/` | Non-Markdown corpus assets, currently mostly artifacts. | Stores diffs, code snippets, provenance, and reference assets. Do not copy large artifacts into SQLite. |
| `indexes/` | SQLite database, manifest, and FTS index data. | Main operational surface for query, CRUD, schema docs, health checks, and SQLite maintenance. |
| `state/` | Unified maintenance state root. | Records refresh state, version state, and audit evidence in functional subdirectories. It is not a normal knowledge-body retrieval entry point. |
| `store.md` | This file. | Helps agents understand the functional zones under `store/`; it does not describe individual document contents. |

## `docs/`

| Directory | Purpose |
|---|---|
| `docs/sources/` | Raw or semi-structured source notes. Stores Markdown summaries/frontmatter for PRs, blogs, docs, and contests. |
| `docs/sources/prs/` | Merged/open/closed PR pages grouped by repo. Records repo, PR number, merge SHA, changed paths, kernel relevance, and artifact pointers. |
| `docs/sources/blogs/` | Community blog and technical article notes. Prose remains in Markdown; extracted code goes to `store/corpus/artifacts/blogs/`. |
| `docs/sources/docs/` | Official documentation, release notes, papers, and spec-like source notes. |
| `docs/sources/contests/` | Contest problem pages, track notes, leaderboard notes, and implementation summaries. |
| `docs/wiki/` | Curated synthesis pages organized by hardware, technique, kernel, language, pattern, and migration topics. |
| `docs/ledgers/` | Maintenance ledgers and policy inputs, such as candidate PRs, core PRs, inclusion policy, and contest source lists. |
| `docs/ledgers/candidates/` | Candidate PR ledgers per tracked repo. Maintained by refresh scripts; not normal answer prose. |

## `corpus/`

| Directory | Purpose |
|---|---|
| `corpus/artifacts/` | Root for traceable artifacts. Files are linked to documents through Markdown frontmatter or the SQLite `artifacts` table. |
| `corpus/artifacts/prs/` | PR diffs, key files, and PROVENANCE bundles, usually maintained by `fetch_pr_diff.py` or `update_pr_corpus.py --fetch-artifacts`. |
| `corpus/artifacts/blogs/` | Fenced code blocks extracted from blog Markdown plus their provenance. Canonical full path: `store/corpus/artifacts/blogs/`. |
| `corpus/artifacts/contests/` | Contest submission code, curated implementation assets, and matching provenance. |
| `corpus/artifacts/kernels/` | High-value kernel reference implementations and variants. Use for implementation details only after checking provenance strength. |

## `indexes/`

| File | Purpose |
|---|---|
| `indexes/kernel_kbs.sqlite` | Primary SQLite store. Holds normalized metadata, relationships, artifact pointers, performance claims, revisions, schema docs, and a contentless FTS5 index. |
| `indexes/kernel_kbs_manifest.json` | Manifest from the latest build/reindex, used to inspect import counts, paths, and build state. |

SQLite does not store a second full readable copy of every Markdown body. Full prose remains in `docs/`; SQLite stores `body_excerpt`, body hash, metadata, paths, and FTS terms. Prefer this read path:

```bash
python3 scripts/kbs.py query <terms>
python3 scripts/kbs.py get <doc-id> --follow-sources --include-code
python3 scripts/kbs.py schema
python3 scripts/kbs.py doctor
```

## `config/` And `schemas/`

| Path | Purpose |
|---|---|
| `config/layout.yaml` | Canonical path map. Use it for directory migrations, script path resolution, and storage placement decisions. |
| `config/corpus.yaml` | Corpus scope, logical layers, unified interface, and answer rules. |
| `config/pr-update.yaml` | Tracked repos, filters, and settings for PR refresh. |
| `schemas/page-schemas.yaml` | Schemas for Markdown/frontmatter, ledgers, and audit files. |
| `schemas/tags.yaml` | Controlled vocabulary. Check before adding tags, techniques, hardware features, kernel types, languages, and similar fields. |
| `schemas/aliases.yaml` | Alias map for query and tag normalization. |

## `state/`

`state/` is organized by maintenance responsibility. It should stay small and structured; do not place knowledge pages or artifact bodies here.

| Path | Purpose |
|---|---|
| `state/refresh/` | Refresh-run state. Written by PR/candidate refresh tools. |
| `state/refresh/pr-update-state.yaml` | Most recent `update_pr_corpus.py` / `kbs.py update-prs` run state. |
| `state/refresh/refresh-cutoff.yaml` | Candidate-ledger refresh cutoff and historical page list. |
| `state/refresh/refresh-search-results.yaml` | Snapshot of GitHub Search refresh results. |
| `state/versions/` | Version-sensitive state used by freshness checks and claim governance. |
| `state/versions/tool-versions.yaml` | Local version-of-record snapshot. |
| `state/versions/version-claims.yaml` | Registry for version-sensitive claims. |
| `state/audits/content/` | Human-approved curated content update audits. |
| `state/audits/refresh/` | Refresh governance audits, such as skipped PR-page records. |
| `state/audits/validation/` | Validation fixtures, size budgets, skill-load notes, and provenance audit outputs. |

Scripts should import the explicit directory constants from `scripts/_wiki_root.py` and use the canonical paths above. Active logic uses only the SQLite-centered store layout.

## Maintenance Rules

- Query, read, and CRUD: use `python3 scripts/kbs.py query|get|put|update|delete|restore|export`.
- After large Markdown moves or edits: run `python3 scripts/kbs.py build-db --reset`.
- After editing existing Markdown content only: run `python3 scripts/kbs.py reindex`.
- After maintenance: run `python3 scripts/kbs.py doctor`.
- After large import/delete batches: run `python3 scripts/kbs.py optimize`.
- Offline governance checks: run `python3 scripts/kbs.py check all`.

## Boundaries

- Document prose lives only under `docs/`.
- Code, diffs, provenance, and kernel reference assets live only under `corpus/artifacts/`.
- Schemas, controlled tags, and aliases live only under `schemas/`.
- Refresh configuration lives only under `config/`.
- Runtime state, audits, budgets, and version claims live only under the structured `state/` subdirectories.
- Retrieval indexes and SQLite operational data live only under `indexes/`.
- If retrieval capability changes are needed, modify `scripts/kbs_db.py` and expose the behavior through `scripts/kbs.py`.
