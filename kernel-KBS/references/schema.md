# Schema Guide

This file explains the KBS data boundaries and metadata rules for agents. It is not a duplicate of the live SQLite schema. To inspect current tables, columns, and workflow documentation, run:

```bash
python3 scripts/kbs.py schema
python3 scripts/kbs.py schema --table docs
python3 scripts/kbs.py schema --table tags
```

## Source Of Truth

| Need | Source |
|---|---|
| live SQLite tables, columns, LLM notes | `python3 scripts/kbs.py schema` |
| page/frontmatter schema | `store/schemas/page-schemas.yaml` |
| controlled tags/languages/architectures | `store/schemas/tags.yaml` |
| user spelling normalization | `store/schemas/aliases.yaml` |
| directory boundaries | `store/store.md` |
| script responsibilities | `scripts/scripts.md` |

SQLite is the operational surface for query, CRUD metadata, FTS ranking, relationships, artifacts, performance claims, and soft delete. Markdown under `store/docs/` remains the canonical full-body document store. SQLite intentionally stores excerpts, hashes, paths, normalized metadata, and a contentless FTS5 index rather than duplicating every full Markdown body.

## Permissions

Default agent behavior is read-only:

```bash
python3 scripts/kbs.py query "<terms>" --limit 10
python3 scripts/kbs.py get <doc-id> --follow-sources --include-code
python3 scripts/kbs.py schema
python3 scripts/kbs.py stats
python3 scripts/kbs.py doctor
```

Require explicit human permission before running write/maintenance commands:

```bash
python3 scripts/kbs.py put <markdown-path>
python3 scripts/kbs.py update <doc-id> ...
python3 scripts/kbs.py delete <doc-id>
python3 scripts/kbs.py restore <doc-id>
python3 scripts/kbs.py export <doc-id> --write-source
python3 scripts/kbs.py build-db --reset
python3 scripts/kbs.py reindex
python3 scripts/kbs.py optimize
python3 scripts/kbs.py vacuum
```

Hard delete needs separate explicit confirmation.

## SQLite Layer

Use CLI first. Raw SQL is only for debugging or compound analysis not exposed by the CLI.

| Table | Agent use |
|---|---|
| `docs` | one row per KBS page; use `id`, `kind`, `path`, `repo`, `confidence`, `body_excerpt` for result triage |
| `docs_fts` | contentless FTS5 index over title/body terms/search tags; join with `docs.doc_pk` |
| `tags` | precise filtering by `architecture`, `language`, `technique`, `hardware_feature`, `kernel_type`, `symptom`, alias |
| `doc_links` | graph edges from `sources`, `related`, `prerequisites`, `candidate_techniques` |
| `artifacts` | code/diff/provenance file metadata; open only specific files needed for evidence |
| `perf_claims` | structured performance claims; do not quote perf from prose when a row exists here |
| `schema_docs` | LLM-readable schema documentation embedded in SQLite |
| `kbs_meta` | store-level guidance such as canonical store, query/write/maintenance notes |
| `revisions` | lightweight audit trail for source-backed body/hash updates |

Query invariant:

```sql
SELECT d.id, d.kind, d.path, d.title
FROM docs_fts f
JOIN docs d ON d.doc_pk = f.rowid
WHERE docs_fts MATCH ?
  AND d.deleted_at IS NULL
ORDER BY bm25(docs_fts, 8.0, 1.0, 4.0);
```

The implementation follows SQLite FTS5's contentless/contentless-delete pattern, which avoids storing a second readable body copy in the FTS table while keeping full-text search efficient. SQLite also recommends `PRAGMA optimize` for lightweight query-planner maintenance; in this KBS use `python3 scripts/kbs.py optimize` after approved large imports/deletes or schema/index changes.

## Page Kinds

| Kind | ID pattern | Purpose |
|---|---|---|
| `source-pr` | `pr-<repo>-<N>` | merged/open/closed PR source note from a tracked repo |
| `source-doc` | `doc-*` | official doc, paper, release note, spec-like source |
| `source-blog` | `blog-*` | community or benchmark blog source note |
| `source-contest` | `contest-*` | contest problem, track, or submission note |
| `wiki-hardware` | `hw-*` | curated hardware feature page |
| `wiki-technique` | `technique-*` | reusable optimization technique |
| `wiki-kernel` | `kernel-*` | kernel case study, normally with `performance_claims` |
| `wiki-pattern` | `pattern-*` | symptom-to-technique diagnostic page |
| `wiki-language` | `lang-*` | CUDA/CuTe/Triton/PTX language guidance |
| `wiki-migration` | `migration-*` | architecture migration guidance |

## Metadata Fields That Matter

| Field | Why it matters |
|---|---|
| `id` | stable public key; prefer this in answers and commands |
| `title` | human-readable label |
| `type` / `kind` | distinguishes source notes from curated wiki pages |
| `repo`, `pr`, `status`, `merge_sha` | provenance for PR source pages |
| `architectures` | prevents mixing SM80/SM90/SM100 guidance |
| `tags`, `techniques`, `hardware_features`, `kernel_types`, `languages`, `symptoms` | normalized filters imported into SQLite `tags` |
| `confidence` | claim strength: `verified`, `source-reported`, `inferred`, `experimental` |
| `reproducibility` | expected implementation backing: `concept`, `pseudocode`, `snippet`, `runnable`, `benchmarked` |
| `sources` | evidence source IDs; follow before making strong claims |
| `related`, `prerequisites`, `candidate_techniques` | graph navigation; imported into `doc_links` |
| `performance_claims` | structured performance evidence; imported into `perf_claims` |
| `artifact_dir` | provenance/code bundle pointer; imported into `artifacts` when files exist |
| `version_sensitive` | links a claim to `store/state/versions/version-claims.yaml` |

## Writing Rules

When human-approved content changes are needed:

1. Check `store/schemas/tags.yaml` before adding new vocabulary.
2. Check `store/schemas/aliases.yaml` before adding user-facing aliases.
3. Keep full prose in Markdown under `store/docs/`; keep diff/code bundles under `store/corpus/artifacts/`.
4. Use `put` or `update --body-file` instead of hand-editing SQLite tables.
5. Use soft delete by default: `python3 scripts/kbs.py delete <doc-id>`.
6. Rebuild or reindex after bulk Markdown changes, then run `doctor`.

Recommended verification after approved writes:

```bash
python3 scripts/kbs.py doctor
python3 scripts/kbs.py query "<changed-topic>" --limit 3
python3 scripts/kbs.py get <changed-doc-id> --follow-sources --include-code
```

## Answer Guardrails

- Prefer curated `wiki-*` pages for synthesis, then follow `sources` to PR/blog/doc evidence.
- Use `source-pr` pages for implementation details and changed file provenance.
- Do not cite a page as `verified` unless its metadata has appropriate `evidence_basis`.
- Do not quote performance without the structured fields from `performance_claims`.
- Do not add or invent tags in answers; normalize to `store/schemas/tags.yaml`.
- If a page is SM90-only, require `blackwell_relevance` before applying it to Blackwell.
- If a claim is version-sensitive, mention the pinned version and verify upstream when the user asks for latest behavior.
