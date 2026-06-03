---
name: kernel-KBS
description: Corpus-backed GPU kernel knowledge base for CUDA, Triton, CuTe, CUTLASS, and Ampere/Hopper/Blackwell kernel research. Use when the user needs to search merged kernel PR pages, inspect PR diff/provenance artifacts, find KernelWiki synthesis pages, query blog/doc/contest notes, or retrieve evidence-backed implementation patterns by hardware feature, technique, repo, language, or kernel type. Do not use for environment checks, correctness checks, Nsight Compute profiling, benchmarking, or iterative optimization bookkeeping.
---

# kernel-KBS

`kernel-KBS` is a read/query skill. It retrieves evidence-backed GPU kernel knowledge from a SQLite-centered corpus. It does not run kernels, profile kernels, benchmark kernels, or manage optimization experiments.

## Default Authority

Default agent mode is **read-only**.

Allowed without asking first:
- Query and inspect KBS content.
- Run read-only commands such as `query`, `get`, `schema`, `stats`, `doctor`, and `check`.
- Read Markdown pages, references, schemas, ledgers, and artifact metadata needed to answer.

Requires explicit human permission in the current task:
- Add, update, delete, restore, or export back to source.
- Run `build-db`, `reindex`, `optimize`, or `vacuum`.
- Run PR refresh commands that apply changes or fetch artifacts.
- Mutate files, SQLite rows, indexes, state, ledgers, or artifacts.

`delete --hard` requires a separate explicit confirmation even when deletion has been approved.

## First Command

All commands run from the `kernel-KBS` skill root. Use `scripts/kbs.py` as the single front door.

Start with SQLite query:

```bash
python3 scripts/kbs.py query "<terms>" --limit 10
```

Then narrow with filters:

```bash
python3 scripts/kbs.py query tcgen05 --architecture sm100 --limit 10
python3 scripts/kbs.py query --repo cutlass --language cute-dsl --limit 20
python3 scripts/kbs.py query --tag nvfp4 --has-code --limit 20
python3 scripts/kbs.py query --symptom memory-bound --limit 10
```

Fetch evidence after selecting likely rows:

```bash
python3 scripts/kbs.py get <doc-id> --follow-sources --include-code
python3 scripts/kbs.py get <doc-id> --json
```

Use schema and health commands when needed:

```bash
python3 scripts/kbs.py schema
python3 scripts/kbs.py schema --table docs
python3 scripts/kbs.py stats
python3 scripts/kbs.py doctor
python3 scripts/kbs.py check all
```

Do not scan Markdown recursively for retrieval. Retrieval goes through SQLite: `docs`, `docs_fts`, `tags`, `doc_links`, `artifacts`, `perf_claims`, `kbs_meta`, and `schema_docs`.

## Query Workflow

1. Start broad with `query <terms> --limit 10`.
2. Add filters only when useful: `--kind`, `--repo`, `--architecture`, `--language`, `--tag`, `--symptom`, `--confidence`, or `--has-code`.
3. Fetch top rows with `get <id> --follow-sources --include-code`.
4. Follow `source` links before making evidence-backed claims.
5. Cite page ids and canonical paths from SQLite result rows.
6. Load references only if the query is too broad, the user asks for schema/query guidance, or you need a topic map.

## Navigation

| Need | Open / Run |
|---|---|
| Directory layout and storage boundaries | `store/store.md` |
| Script responsibilities and allowed entry points | `scripts/scripts.md` |
| SQLite schema and workflow notes | `python3 scripts/kbs.py schema` |
| Topic map and canonical page ids | `references/primer.md` |
| Frontmatter rules and controlled-vocabulary pointers | `references/schema.md` |
| Worked query examples | `references/examples.md` |
| Actual controlled tags and aliases | `store/schemas/tags.yaml`, `store/schemas/aliases.yaml` |
| Current corpus counts | `python3 scripts/kbs.py stats` |

Core storage map:

| Layer | Path | Role |
|---|---|---|
| Config | `store/config/` | Layout, corpus manifest, PR refresh configuration. |
| Schemas | `store/schemas/` | Page schemas, controlled tags, aliases. |
| Source docs | `store/docs/sources/` | PR/blog/doc/contest source Markdown. |
| Knowledge pages | `store/docs/wiki/` | Curated synthesis pages for answers. |
| Ledgers | `store/docs/ledgers/` | Candidate/core/source ledgers and policy inputs. |
| Artifacts | `store/corpus/artifacts/` | Diffs, code assets, provenance bundles. |
| Indexes | `store/indexes/kernel-KBS.sqlite` | SQLite metadata, relationships, and contentless FTS. |
| State | `store/state/` | Structured maintenance state: `refresh/`, `versions/`, and `audits/{content,refresh,validation}/`. |

Full Markdown bodies remain under `store/docs/`. SQLite stores metadata, paths, short excerpts, relationships, artifact pointers, performance claims, and FTS terms. It does not store a second readable copy of every Markdown body.

## Permissioned Maintenance

Only run these after explicit human approval:

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
python3 scripts/kbs.py update-prs --since <YYYY-MM-DD> --apply
```

Dry-run PR discovery is read-only from the corpus perspective, but it may require network and credentials:

```bash
python3 scripts/kbs.py update-prs --since <YYYY-MM-DD> --max-new 20
```

If approved maintenance changes Markdown or artifacts, run the approved rebuild/check sequence appropriate to the change:

```bash
python3 scripts/kbs.py build-db --reset
python3 scripts/kbs.py doctor
python3 scripts/kbs.py check all
```

## Answer Contract

When answering from this skill:

1. Cite concrete page ids and canonical paths, for example `kernel-flash-attention-4` at `store/docs/wiki/kernels/flash-attention-4.md`.
2. Use `sources`, `doc_links`, `artifact_dir`, and `artifacts` metadata before making evidence-backed claims.
3. Preserve confidence labels exactly: `verified`, `source-reported`, `inferred`, or `experimental`.
4. For performance claims, include available `gpu`, `dtype`, `shape`, `metric`, `value`, and `source_id`.
5. Keep profiling, benchmarking, environment checks, and correctness workflows outside this skill.
