#!/usr/bin/env python3
"""SQLite-backed metadata store and FTS index for kernel_KBS.

Markdown pages remain the canonical full-body content. SQLite stores normalized
metadata, relationships, artifact pointers, audit data, and a contentless FTS5
index for fast retrieval without duplicating Markdown bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import (  # noqa: E402
    ARTIFACTS_DIR,
    DOCS_DIR,
    INDEXES_DIR,
    RAW_SOURCES_DIR,
    SCHEMAS_DIR,
    WIKI_DIR,
    WIKI_ROOT,
    rel_to_root,
    resolve_rel_path,
)


DB_PATH = INDEXES_DIR / "kernel_kbs.sqlite"
MANIFEST_PATH = INDEXES_DIR / "kernel_kbs_manifest.json"
SCHEMA_VERSION = 3

TAG_FIELDS = {
    "tags": "tag",
    "techniques": "technique",
    "hardware_features": "hardware_feature",
    "kernel_types": "kernel_type",
    "languages": "language",
    "architectures": "architecture",
    "symptoms": "symptom",
    "aliases": "alias",
}

LINK_FIELDS = {
    "sources": "source",
    "related": "related",
    "candidate_techniques": "candidate_technique",
    "prerequisites": "prerequisite",
}

CODE_EXTS = {
    ".cu", ".cuh", ".ptx",
    ".cpp", ".cxx", ".cc", ".c",
    ".h", ".hpp", ".hxx", ".inl",
    ".py", ".pyx",
    ".patch",
    ".sh",
    ".md", ".yaml", ".yml", ".txt", ".json",
}

SCHEMA_DOCS = [
    {
        "name": "kbs_meta",
        "kind": "table",
        "description": "Small key/value metadata table for schema version, LLM usage notes, and build metadata.",
        "columns": {
            "key": "Metadata key. Stable keys include schema_version, llm_summary, llm_query_guide, llm_write_guide, canonical_store.",
            "value": "String value. Some values are prose; parse only if the key explicitly stores JSON.",
        },
        "llm_notes": "Read kbs_meta first when discovering the database. It tells you how to query and update the KBS safely.",
    },
    {
        "name": "schema_docs",
        "kind": "table",
        "description": "LLM-readable table and column documentation embedded inside the SQLite store.",
        "columns": {
            "name": "Table, virtual table, trigger, or index name.",
            "kind": "Object kind: table, virtual_table, trigger, index, relationship, or workflow.",
            "description": "Short purpose statement.",
            "columns_json": "JSON object mapping column names to column descriptions.",
            "llm_notes": "Extra guidance for agent/LLM usage.",
        },
        "llm_notes": "Use `kbs.py schema` or query this table to understand available tables before writing raw SQL.",
    },
    {
        "name": "docs",
        "kind": "table",
        "description": "Primary document metadata table. Each row is one KBS page; the full Markdown body stays in the source file pointed to by path.",
        "columns": {
            "doc_pk": "Integer primary key used by child tables and FTS rowid.",
            "id": "Stable public document id, e.g. hw-tcgen05-mma or pr-cutlass-2472. Prefer this in user-facing answers.",
            "kind": "Document class: source-pr, source-blog, source-doc, source-contest, wiki-hardware, wiki-technique, wiki-kernel, wiki-pattern, wiki-language, wiki-migration.",
            "title": "Human-readable title.",
            "body_excerpt": "Short normalized preview copied from the Markdown body for result snippets. Not the canonical body.",
            "meta_json": "Normalized JSON copy of frontmatter metadata.",
            "search_tags": "Flattened searchable metadata used by FTS: id, repo, pr, tags, languages, architectures, aliases, confidence, status.",
            "body_sha": "SHA-256 of canonical Markdown body text for drift/change detection.",
            "path": "Canonical source path relative to skill root when the row came from Markdown.",
            "repo": "Upstream repository string when applicable.",
            "pr": "Pull request number when applicable.",
            "date": "Best available date from frontmatter.",
            "confidence": "Evidence confidence when present: verified, source-reported, inferred, experimental.",
            "created_at": "UTC timestamp for DB row creation.",
            "updated_at": "UTC timestamp for latest DB row update.",
            "deleted_at": "UTC timestamp for soft delete. Active rows have NULL.",
        },
        "llm_notes": "Filter `deleted_at IS NULL` for normal reads. Use `id`, `title`, `kind`, `path`, `confidence`, and `body_excerpt` in search results; use `kbs.py get` to read full body from Markdown when citing.",
    },
    {
        "name": "docs_fts",
        "kind": "virtual_table",
        "description": "Contentless-delete FTS5 index over title, Markdown body text, and flattened metadata. It indexes terms without storing a readable duplicate of the body.",
        "columns": {
            "title": "Indexed document title.",
            "body": "Indexed Markdown body terms. Full body text is read from docs.path, not stored here.",
            "search_tags": "Indexed flattened metadata.",
        },
        "llm_notes": "Use `docs_fts MATCH ?` joined on `docs_fts.rowid = docs.doc_pk`; rank with `bm25(docs_fts, 8.0, 1.0, 4.0)` ascending. Prefer the CLI `kbs.py query` unless raw SQL is needed.",
    },
    {
        "name": "tags",
        "kind": "table",
        "description": "Normalized many-to-one metadata tags extracted from frontmatter.",
        "columns": {
            "doc_pk": "Foreign key to docs.doc_pk.",
            "category": "Tag category: tag, technique, hardware_feature, kernel_type, language, architecture, symptom, alias.",
            "tag": "Tag value.",
        },
        "llm_notes": "Use this table for precise filters such as architecture=sm100, language=cuda-cpp, hardware_feature=tma, technique=warp-specialization.",
    },
    {
        "name": "doc_links",
        "kind": "table",
        "description": "Graph edges extracted from frontmatter lists such as sources, related, prerequisites, and candidate_techniques.",
        "columns": {
            "doc_pk": "Foreign key to docs.doc_pk.",
            "rel_type": "Relation type: source, related, candidate_technique, prerequisite.",
            "target_id": "Target document id as written in frontmatter.",
        },
        "llm_notes": "Use `source` links to follow evidence and `related` links to expand context. Target rows can be joined by `docs.id = doc_links.target_id` when present.",
    },
    {
        "name": "artifacts",
        "kind": "table",
        "description": "Artifact files associated with documents, stored on disk and referenced by path/hash.",
        "columns": {
            "artifact_pk": "Integer primary key.",
            "doc_pk": "Foreign key to docs.doc_pk.",
            "kind": "File kind based on extension, e.g. cu, patch, yaml, py.",
            "path": "Artifact path relative to skill root when possible.",
            "sha256": "Content hash for provenance and drift checks.",
            "size_bytes": "File size in bytes.",
        },
        "llm_notes": "Do not paste large artifacts blindly. Use paths/hashes for provenance and open only specific files needed for evidence.",
    },
    {
        "name": "perf_claims",
        "kind": "table",
        "description": "Structured performance claims extracted from frontmatter performance_claims.",
        "columns": {
            "perf_pk": "Integer primary key.",
            "doc_pk": "Foreign key to docs.doc_pk.",
            "gpu": "GPU name or architecture for the claim.",
            "dtype": "Data type, e.g. fp8, bf16, nvfp4.",
            "shape": "Problem shape string when available.",
            "metric": "Performance metric name.",
            "value": "Metric value as text.",
            "source_id": "Evidence source id for the claim.",
            "meta_json": "Original normalized claim JSON.",
        },
        "llm_notes": "Only report performance numbers with gpu, dtype, shape, metric, value, and source_id when present.",
    },
    {
        "name": "revisions",
        "kind": "table",
        "description": "Lightweight audit trail for source-backed body hash changes and DB-side metadata updates.",
        "columns": {
            "rev_pk": "Integer primary key.",
            "doc_pk": "Foreign key to docs.doc_pk.",
            "old_body_sha": "Previous body SHA-256.",
            "new_body_sha": "New body SHA-256.",
            "patch_or_snapshot": "Small JSON audit context. Full historical body rollback should use the Markdown file history, not this table.",
            "created_at": "UTC revision timestamp.",
            "reason": "Caller-provided reason for the change.",
        },
        "llm_notes": "Use revision rows to explain edits. This table intentionally avoids storing old full bodies.",
    },
    {
        "name": "query_workflow",
        "kind": "workflow",
        "description": "Recommended retrieval workflow for LLMs.",
        "columns": {},
        "llm_notes": "1) Use `kbs.py query` with text and filters. 2) Use `kbs.py get <id>` for the top pages. 3) Follow `doc_links` with rel_type=source for evidence. 4) Cite ids and canonical paths. 5) Ignore soft-deleted rows unless explicitly asked.",
    },
    {
        "name": "write_workflow",
        "kind": "workflow",
        "description": "Recommended write workflow for LLMs.",
        "columns": {},
        "llm_notes": "Use `put` to import Markdown, `update` for DB-side metadata edits, `update --body-file` for source-backed body replacement, `delete` for soft delete, and `restore` to undo soft delete. Rebuild with `build --reset` after large Markdown moves.",
    },
    {
        "name": "maintenance_workflow",
        "kind": "workflow",
        "description": "Recommended SQLite maintenance workflow.",
        "columns": {},
        "llm_notes": "Use `doctor` to verify schema/path/FTS health, `reindex` after bulk Markdown edits, `optimize` after large import/delete batches, and `vacuum` only when a full database rewrite is acceptable.",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_json(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_json(v) for v in value]
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(normalize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(fm, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return fm, match.group(2)


def read_markdown_page(path: Path) -> tuple[dict[str, Any], str]:
    return split_frontmatter(path.read_text(encoding="utf-8"))


def stable_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_excerpt(body: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", body).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_or_abs(path: Path) -> str:
    try:
        return rel_to_root(path.resolve())
    except ValueError:
        return str(path.resolve())


def sqlite_sidecar_paths(db_path: Path) -> list[Path]:
    return [Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]


def manifest_path_for(db_path: Path) -> Path:
    if db_path.resolve() == DB_PATH.resolve():
        return MANIFEST_PATH
    return db_path.with_name(f"{db_path.stem}_manifest.json")


def iter_markdown_pages() -> Iterable[Path]:
    for base in (RAW_SOURCES_DIR, WIKI_DIR):
        if not base.exists():
            continue
        yield from sorted(p for p in base.rglob("*.md") if p.is_file())


def page_kind(path: Path, fm: dict[str, Any]) -> str:
    rel = Path(rel_or_abs(path))
    parts = rel.parts
    if len(parts) >= 4 and parts[:3] == ("store", "docs", "sources"):
        return f"source-{parts[3].rstrip('s')}"
    if len(parts) >= 4 and parts[:3] == ("store", "docs", "wiki"):
        return f"wiki-{fm.get('type') or parts[3]}"
    if fm.get("type"):
        return f"wiki-{fm['type']}"
    return "document"


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    if isinstance(value, dict):
        return [json_dumps(value)]
    return [str(value)]


def collect_tags(fm: dict[str, Any]) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, category in TAG_FIELDS.items():
        for value in flatten_values(fm.get(field)):
            value = value.strip()
            if not value:
                continue
            row = (category, value)
            if row not in seen:
                seen.add(row)
                tags.append(row)
    return tags


def collect_links(fm: dict[str, Any]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, rel_type in LINK_FIELDS.items():
        for value in flatten_values(fm.get(field)):
            value = value.strip()
            if not value:
                continue
            row = (rel_type, value)
            if row not in seen:
                seen.add(row)
                links.append(row)
    return links


def search_tags_text(fm: dict[str, Any]) -> str:
    values: list[str] = []
    for _, value in collect_tags(fm):
        values.append(value)
    for key in ("id", "repo", "pr", "source_category", "confidence", "reproducibility", "status"):
        values.extend(flatten_values(fm.get(key)))
    return " ".join(values)


def artifact_candidates(path: Path, fm: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    explicit = fm.get("artifact_dir")
    if explicit:
        out.append(resolve_rel_path(str(explicit)))

    rel = Path(rel_or_abs(path))
    parts = rel.parts
    if len(parts) >= 4 and parts[:3] == ("store", "docs", "sources"):
        source_kind = parts[3]
        if source_kind == "prs" and len(parts) >= 6:
            repo_part = parts[4]
            pr_stem = path.stem
            repo_short = str(fm.get("repo") or repo_part).split("/")[-1]
            out.extend(
                [
                    ARTIFACTS_DIR / "prs" / repo_part / pr_stem,
                    ARTIFACTS_DIR / "prs" / repo_part.lower() / pr_stem,
                    ARTIFACTS_DIR / "prs" / repo_short / pr_stem,
                    ARTIFACTS_DIR / "prs" / repo_short.lower() / pr_stem,
                ]
            )
        elif source_kind == "blogs":
            out.append(ARTIFACTS_DIR / "blogs" / path.stem)
        elif source_kind == "contests" and len(parts) >= 6:
            out.append(ARTIFACTS_DIR / "contests" / parts[4] / path.stem)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in out:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate)
    return deduped


def collect_artifacts(path: Path, fm: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for base in artifact_candidates(path, fm):
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in CODE_EXTS:
                continue
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(
                {
                    "kind": f.suffix.lower().lstrip(".") or "file",
                    "path": rel_or_abs(f),
                    "sha256": file_sha(f),
                    "size_bytes": f.stat().st_size,
                }
            )
    return rows


def collect_perf_claims(fm: dict[str, Any]) -> list[dict[str, Any]]:
    claims = fm.get("performance_claims") or []
    if not isinstance(claims, list):
        return []
    out: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        out.append(
            {
                "gpu": claim.get("gpu"),
                "dtype": claim.get("dtype"),
                "shape": claim.get("shape"),
                "metric": claim.get("metric"),
                "value": claim.get("value"),
                "source_id": claim.get("source_id"),
                "meta_json": json_dumps(claim),
            }
        )
    return out


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    con.execute("PRAGMA journal_mode = WAL")
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS kbs_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_docs (
          name TEXT NOT NULL,
          kind TEXT NOT NULL,
          description TEXT NOT NULL,
          columns_json TEXT NOT NULL DEFAULT '{}',
          llm_notes TEXT NOT NULL DEFAULT '',
          PRIMARY KEY (name, kind)
        );

        CREATE TABLE IF NOT EXISTS docs (
          doc_pk INTEGER PRIMARY KEY,
          id TEXT NOT NULL UNIQUE,
          kind TEXT NOT NULL,
          title TEXT NOT NULL,
          body_excerpt TEXT NOT NULL DEFAULT '',
          meta_json TEXT NOT NULL DEFAULT '{}',
          search_tags TEXT NOT NULL DEFAULT '',
          body_sha TEXT NOT NULL,
          path TEXT UNIQUE,
          repo TEXT,
          pr TEXT,
          date TEXT,
          confidence TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS tags (
          doc_pk INTEGER NOT NULL REFERENCES docs(doc_pk) ON DELETE CASCADE,
          category TEXT NOT NULL,
          tag TEXT NOT NULL,
          PRIMARY KEY (doc_pk, category, tag)
        );

        CREATE TABLE IF NOT EXISTS doc_links (
          doc_pk INTEGER NOT NULL REFERENCES docs(doc_pk) ON DELETE CASCADE,
          rel_type TEXT NOT NULL,
          target_id TEXT NOT NULL,
          PRIMARY KEY (doc_pk, rel_type, target_id)
        );

        CREATE TABLE IF NOT EXISTS artifacts (
          artifact_pk INTEGER PRIMARY KEY,
          doc_pk INTEGER NOT NULL REFERENCES docs(doc_pk) ON DELETE CASCADE,
          kind TEXT NOT NULL,
          path TEXT NOT NULL,
          sha256 TEXT,
          size_bytes INTEGER,
          UNIQUE (doc_pk, path)
        );

        CREATE TABLE IF NOT EXISTS perf_claims (
          perf_pk INTEGER PRIMARY KEY,
          doc_pk INTEGER NOT NULL REFERENCES docs(doc_pk) ON DELETE CASCADE,
          gpu TEXT,
          dtype TEXT,
          shape TEXT,
          metric TEXT,
          value TEXT,
          source_id TEXT,
          meta_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS revisions (
          rev_pk INTEGER PRIMARY KEY,
          doc_pk INTEGER NOT NULL REFERENCES docs(doc_pk) ON DELETE CASCADE,
          old_body_sha TEXT,
          new_body_sha TEXT,
          patch_or_snapshot TEXT,
          created_at TEXT NOT NULL,
          reason TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
          title,
          body,
          search_tags,
          content='',
          contentless_delete=1,
          tokenize='unicode61'
        );

        CREATE INDEX IF NOT EXISTS idx_docs_kind ON docs(kind);
        CREATE INDEX IF NOT EXISTS idx_docs_repo ON docs(repo);
        CREATE INDEX IF NOT EXISTS idx_docs_updated_at ON docs(updated_at);
        CREATE INDEX IF NOT EXISTS idx_docs_deleted_at ON docs(deleted_at);
        CREATE INDEX IF NOT EXISTS idx_docs_active_kind ON docs(kind, updated_at) WHERE deleted_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_docs_active_repo ON docs(repo, updated_at) WHERE deleted_at IS NULL AND repo IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
        CREATE INDEX IF NOT EXISTS idx_tags_category_tag ON tags(category, tag);
        CREATE INDEX IF NOT EXISTS idx_links_target ON doc_links(target_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_doc ON artifacts(doc_pk);
        """
    )
    install_schema_docs(con)
    con.commit()


def replace_fts_row(con: sqlite3.Connection, doc_pk: int, title: str, body: str, search_tags: str) -> None:
    con.execute("DELETE FROM docs_fts WHERE rowid = ?", (doc_pk,))
    con.execute(
        "INSERT INTO docs_fts(rowid, title, body, search_tags) VALUES (?, ?, ?, ?)",
        (doc_pk, title, body, search_tags),
    )


def install_schema_docs(con: sqlite3.Connection) -> None:
    meta_rows = {
        "schema_version": str(SCHEMA_VERSION),
        "canonical_store": "SQLite is the primary operational read/query/update/delete surface. Markdown files under store/docs are the canonical full-body document store; SQLite stores metadata, pointers, short excerpts, and a contentless FTS index.",
        "llm_summary": "kernel_KBS stores GPU kernel knowledge as Markdown-backed docs plus normalized tags, links, artifacts, performance claims, revisions, and contentless FTS5 search metadata.",
        "llm_query_guide": "Prefer CLI: `python3 scripts/kbs.py query <terms> --architecture sm100 --tag tcgen05`. For SQL, join docs_fts.rowid to docs.doc_pk and filter docs.deleted_at IS NULL.",
        "llm_write_guide": "Use `put`, `update`, `delete`, `restore`, and `export` commands. Delete is soft by default. `update --body-file` writes the canonical Markdown source before reindexing; metadata-only updates stay in SQLite until exported.",
        "llm_answer_contract": "Cite docs.id and docs.path. Follow doc_links rel_type=source for evidence. Report confidence and structured performance fields when present.",
        "llm_maintenance_guide": "Use `doctor` after schema/layout changes, `reindex` after bulk Markdown edits, and `optimize` after large import/delete batches.",
    }
    con.executemany(
        "INSERT OR REPLACE INTO kbs_meta(key, value) VALUES (?, ?)",
        sorted(meta_rows.items()),
    )
    con.execute("DELETE FROM schema_docs")
    con.executemany(
        """
        INSERT INTO schema_docs(name, kind, description, columns_json, llm_notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                row["name"],
                row["kind"],
                row["description"],
                json_dumps(row.get("columns", {})),
                row.get("llm_notes", ""),
            )
            for row in SCHEMA_DOCS
        ],
    )


def upsert_page(
    con: sqlite3.Connection,
    path: Path,
    *,
    reason: str = "import",
    deleted_at: str | None = None,
) -> tuple[str, bool]:
    fm, body = read_markdown_page(path)
    doc_id = str(fm.get("id") or "").strip()
    if not doc_id:
        raise ValueError(f"{rel_or_abs(path)}: missing frontmatter id")

    kind = page_kind(path, fm)
    title = str(fm.get("title") or doc_id)
    body_sha = stable_sha(body)
    rel_path = rel_or_abs(path)
    repo = str(fm.get("repo")) if fm.get("repo") is not None else None
    pr = str(fm.get("pr")) if fm.get("pr") is not None else None
    date = str(fm.get("date") or fm.get("retrieved_at") or fm.get("captured_at") or "") or None
    confidence = str(fm.get("confidence")) if fm.get("confidence") is not None else None
    meta_json = json_dumps(fm)
    tag_text = search_tags_text(fm)
    body_excerpt = make_excerpt(body)
    ts = now_utc()

    existing = con.execute(
        "SELECT doc_pk, body_sha, title, meta_json FROM docs WHERE id = ?",
        (doc_id,),
    ).fetchone()
    created = existing is None

    if existing is None:
        cur = con.execute(
            """
            INSERT INTO docs(
              id, kind, title, body_excerpt, meta_json, search_tags, body_sha, path,
              repo, pr, date, confidence, created_at, updated_at, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                kind,
                title,
                body_excerpt,
                meta_json,
                tag_text,
                body_sha,
                rel_path,
                repo,
                pr,
                date,
                confidence,
                ts,
                ts,
                deleted_at,
            ),
        )
        doc_pk = int(cur.lastrowid)
    else:
        doc_pk = int(existing["doc_pk"])
        if existing["body_sha"] != body_sha:
            con.execute(
                """
                INSERT INTO revisions(doc_pk, old_body_sha, new_body_sha, patch_or_snapshot, created_at, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_pk,
                    existing["body_sha"],
                    body_sha,
                    json_dumps({"old_title": existing["title"], "source_path": rel_path}),
                    ts,
                    reason,
                ),
            )
        con.execute(
            """
            UPDATE docs
            SET kind = ?, title = ?, body_excerpt = ?, meta_json = ?, search_tags = ?,
                body_sha = ?, path = ?, repo = ?, pr = ?, date = ?,
                confidence = ?, updated_at = ?, deleted_at = COALESCE(?, deleted_at)
            WHERE doc_pk = ?
            """,
            (
                kind,
                title,
                body_excerpt,
                meta_json,
                tag_text,
                body_sha,
                rel_path,
                repo,
                pr,
                date,
                confidence,
                ts,
                deleted_at,
                doc_pk,
            ),
        )

    replace_derived_rows(con, doc_pk, path, fm)
    replace_fts_row(con, doc_pk, title, body, tag_text)
    return doc_id, created


def replace_derived_rows(con: sqlite3.Connection, doc_pk: int, path: Path, fm: dict[str, Any]) -> None:
    con.execute("DELETE FROM tags WHERE doc_pk = ?", (doc_pk,))
    con.executemany(
        "INSERT OR IGNORE INTO tags(doc_pk, category, tag) VALUES (?, ?, ?)",
        [(doc_pk, category, tag) for category, tag in collect_tags(fm)],
    )

    con.execute("DELETE FROM doc_links WHERE doc_pk = ?", (doc_pk,))
    con.executemany(
        "INSERT OR IGNORE INTO doc_links(doc_pk, rel_type, target_id) VALUES (?, ?, ?)",
        [(doc_pk, rel_type, target) for rel_type, target in collect_links(fm)],
    )

    con.execute("DELETE FROM artifacts WHERE doc_pk = ?", (doc_pk,))
    con.executemany(
        """
        INSERT OR IGNORE INTO artifacts(doc_pk, kind, path, sha256, size_bytes)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (doc_pk, row["kind"], row["path"], row["sha256"], row["size_bytes"])
            for row in collect_artifacts(resolve_rel_path(path), fm)
        ],
    )

    con.execute("DELETE FROM perf_claims WHERE doc_pk = ?", (doc_pk,))
    con.executemany(
        """
        INSERT INTO perf_claims(doc_pk, gpu, dtype, shape, metric, value, source_id, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                doc_pk,
                row.get("gpu"),
                row.get("dtype"),
                row.get("shape"),
                row.get("metric"),
                str(row.get("value")) if row.get("value") is not None else None,
                row.get("source_id"),
                row.get("meta_json", "{}"),
            )
            for row in collect_perf_claims(fm)
        ],
    )


def rebuild_fts(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM docs_fts")
    rows = list(
        con.execute(
            "SELECT doc_pk, title, search_tags, path FROM docs WHERE deleted_at IS NULL ORDER BY doc_pk"
        )
    )
    for row in rows:
        if not row["path"]:
            continue
        path = resolve_rel_path(row["path"])
        if not path.is_file():
            continue
        _, body = read_markdown_page(path)
        replace_fts_row(con, int(row["doc_pk"]), row["title"], body, row["search_tags"])


def write_manifest(con: sqlite3.Connection, db_path: Path) -> None:
    counts = {
        "docs": con.execute("SELECT COUNT(*) FROM docs").fetchone()[0],
        "active_docs": con.execute("SELECT COUNT(*) FROM docs WHERE deleted_at IS NULL").fetchone()[0],
        "tags": con.execute("SELECT COUNT(*) FROM tags").fetchone()[0],
        "artifacts": con.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "db_path": rel_or_abs(db_path),
        "built_at": now_utc(),
        "counts": counts,
    }
    manifest_path = manifest_path_for(db_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fts_row_count(con: sqlite3.Connection) -> int:
    return int(con.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0])


def active_doc_count(con: sqlite3.Connection) -> int:
    return int(con.execute("SELECT COUNT(*) FROM docs WHERE deleted_at IS NULL").fetchone()[0])


def load_alias_expansions() -> dict[str, str]:
    aliases_path = SCHEMAS_DIR / "aliases.yaml"
    if not aliases_path.is_file():
        return {}
    try:
        raw = read_yaml(aliases_path) or {}
    except Exception:
        return {}
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out[canonical.lower()] = canonical
        for variant in variants or []:
            if isinstance(variant, str):
                out[variant.lower()] = canonical
    return out


def fts_query(query: str, *, raw: bool = False) -> str:
    if raw:
        return query
    aliases = load_alias_expansions()
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    groups: list[str] = []
    for token in tokens:
        variants = {token}
        canonical = aliases.get(token.lower())
        if canonical:
            variants.update(re.findall(r"[A-Za-z0-9_]+", canonical))
        quoted = [f'"{v.replace(chr(34), chr(34) + chr(34))}"' for v in sorted(variants, key=str.lower)]
        if len(quoted) == 1:
            groups.append(quoted[0])
        else:
            groups.append("(" + " OR ".join(quoted) + ")")
    return " AND ".join(groups)


def add_filter(sql: list[str], params: list[Any], condition: str, value: Any) -> None:
    if value is not None:
        sql.append(condition)
        params.append(value)


def tag_exists_clause(category: str | None = None) -> str:
    if category:
        return (
            "EXISTS (SELECT 1 FROM tags t WHERE t.doc_pk = d.doc_pk "
            "AND t.category = ? AND lower(t.tag) = lower(?))"
        )
    return "EXISTS (SELECT 1 FROM tags t WHERE t.doc_pk = d.doc_pk AND lower(t.tag) = lower(?))"


def metadata_query_groups(query: str) -> list[list[str]]:
    aliases = load_alias_expansions()
    groups: list[list[str]] = []
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        variants = {token}
        canonical = aliases.get(token.lower())
        if canonical:
            variants.update(re.findall(r"[A-Za-z0-9_]+", canonical))
        groups.append(sorted(variants, key=str.lower))
    return groups


def deleted_metadata_rows(
    con: sqlite3.Connection,
    *,
    base_where: list[str],
    base_params: list[Any],
    query_text: str,
    exact_lookup: str,
    limit: int,
) -> list[sqlite3.Row]:
    """Search soft-deleted docs that are intentionally absent from FTS.

    Normal retrieval uses docs_fts and filters active rows. Soft delete removes
    the FTS row so `doctor` can verify active_docs == docs_fts rows. When a
    caller explicitly asks for `--include-deleted`, fall back to metadata and
    body excerpts for deleted rows.
    """
    groups = metadata_query_groups(query_text)
    if not groups and not exact_lookup:
        return []

    where = ["d.deleted_at IS NOT NULL", *base_where]
    params: list[Any] = [exact_lookup, exact_lookup, *base_params]

    match_clauses: list[str] = []
    match_params: list[Any] = []
    if exact_lookup:
        match_clauses.append("d.id = ? OR d.path = ?")
        match_params.extend([exact_lookup, exact_lookup])
    for group in groups:
        per_token: list[str] = []
        for variant in group:
            pattern = f"%{variant}%"
            per_token.append(
                "("
                "lower(d.id) LIKE lower(?) OR "
                "lower(COALESCE(d.path, '')) LIKE lower(?) OR "
                "lower(d.title) LIKE lower(?) OR "
                "lower(d.search_tags) LIKE lower(?) OR "
                "lower(d.body_excerpt) LIKE lower(?)"
                ")"
            )
            match_params.extend([pattern, pattern, pattern, pattern, pattern])
        match_clauses.append("(" + " OR ".join(per_token) + ")")
    if exact_lookup:
        exact_clause = match_clauses[0]
        token_clauses = match_clauses[1:]
        if token_clauses:
            where.append(f"(({exact_clause}) OR ({' AND '.join(token_clauses)}))")
        else:
            where.append(f"({exact_clause})")
    else:
        where.append("(" + " AND ".join(match_clauses) + ")")
    params.extend(match_params)

    sql = f"""
        SELECT d.id, d.kind, d.title, d.path, d.repo, d.pr, d.date, d.confidence,
               d.updated_at, d.deleted_at,
               CASE WHEN d.id = ? OR d.path = ? THEN 0 ELSE 2 END AS exact_rank,
               NULL AS score,
               d.body_excerpt AS snippet
        FROM docs d
        WHERE {" AND ".join(where)}
        ORDER BY exact_rank ASC, d.updated_at DESC
        LIMIT ?
    """
    params.append(limit)
    return list(con.execute(sql, params))


def run_search(con: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    where: list[str] = []
    where_params: list[Any] = []
    query_text = " ".join(args.query or [])
    exact_lookup = query_text.strip() if args.query and len(args.query) == 1 else ""
    if not args.include_deleted:
        where.append("d.deleted_at IS NULL")
    add_filter(where, where_params, "d.kind = ?", args.kind)
    add_filter(where, where_params, "lower(d.repo) LIKE lower(?)", f"%{args.repo}%" if args.repo else None)
    add_filter(where, where_params, "d.confidence = ?", args.confidence)
    for tag in args.tag or []:
        where.append(tag_exists_clause())
        where_params.append(tag)
    for language in args.language or []:
        where.append(tag_exists_clause("language"))
        where_params.extend(["language", language])
    for arch in args.architecture or []:
        where.append(tag_exists_clause("architecture"))
        where_params.extend(["architecture", arch])
    for symptom in getattr(args, "symptom", None) or []:
        where.append(tag_exists_clause("symptom"))
        where_params.extend(["symptom", symptom])
    if getattr(args, "has_code", False):
        where.append(
            "EXISTS (SELECT 1 FROM artifacts a WHERE a.doc_pk = d.doc_pk "
            "AND a.kind IN ('cu','cuh','ptx','cpp','cxx','cc','c','h','hpp','hxx','inl','py','pyx','sh','txt','json','yaml','yml'))"
        )

    match = fts_query(query_text, raw=args.raw_match) if args.query else ""
    if match:
        base_where = list(where)
        base_params = list(where_params)
        sql = [
            """
            SELECT d.id, d.kind, d.title, d.path, d.repo, d.pr, d.date, d.confidence,
                   d.updated_at, d.deleted_at,
                   CASE WHEN d.id = ? OR d.path = ? THEN 0 ELSE 1 END AS exact_rank,
                   bm25(docs_fts, 8.0, 1.0, 4.0) AS score,
                   d.body_excerpt AS snippet
            FROM docs_fts
            JOIN docs d ON docs_fts.rowid = d.doc_pk
            """
        ]
        where.insert(0, "docs_fts MATCH ?")
        params = [exact_lookup, exact_lookup, match, *where_params]
        order = "ORDER BY exact_rank ASC, score ASC, d.updated_at DESC"
    else:
        sql = [
            """
            SELECT d.id, d.kind, d.title, d.path, d.repo, d.pr, d.date, d.confidence,
                   d.updated_at, d.deleted_at,
                   1 AS exact_rank,
                   NULL AS score,
                   d.body_excerpt AS snippet
            FROM docs d
            """
        ]
        params = [*where_params]
        order = "ORDER BY d.updated_at DESC, d.id ASC"

    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append(order)
    sql.append("LIMIT ?")
    params.append(args.limit)
    rows = list(con.execute("\n".join(sql), params))
    if match and args.include_deleted and len(rows) < args.limit:
        seen = {row["id"] for row in rows}
        deleted_rows = deleted_metadata_rows(
            con,
            base_where=base_where,
            base_params=base_params,
            query_text=query_text,
            exact_lookup=exact_lookup,
            limit=args.limit - len(rows),
        )
        rows.extend(row for row in deleted_rows if row["id"] not in seen)
    return rows[: args.limit]


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def print_search_rows(rows: list[sqlite3.Row], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps([row_to_dict(row) for row in rows], ensure_ascii=False, indent=2))
        return
    print(f"# {len(rows)} result(s)")
    print()
    for row in rows:
        score = "" if row["score"] is None else f" score={row['score']:.4f}"
        deleted = " deleted" if row["deleted_at"] else ""
        print(f"- [{row['kind']}] {row['id']}: {row['title']}{score}{deleted}")
        print(f"  path: {row['path']}")
        if row["repo"] or row["pr"]:
            print(f"  source: {row['repo'] or ''} {('#' + str(row['pr'])) if row['pr'] else ''}".rstrip())
        snippet = (row["snippet"] or "").strip().replace("\n", " ")
        if snippet:
            print(f"  snippet: {snippet[:260]}")


def get_doc(con: sqlite3.Connection, lookup: str, *, include_deleted: bool = False) -> sqlite3.Row | None:
    where = "(id = ? OR path = ?)"
    params: list[Any] = [lookup, lookup]
    if not include_deleted:
        where += " AND deleted_at IS NULL"
    return con.execute(f"SELECT * FROM docs WHERE {where} LIMIT 1", params).fetchone()


def read_doc_body(row: sqlite3.Row) -> str:
    if not row["path"]:
        return ""
    path = resolve_rel_path(row["path"])
    if not path.is_file():
        raise SystemExit(f"ERROR: Markdown source not found for {row['id']}: {row['path']}")
    _, body = read_markdown_page(path)
    return body


def write_doc_source(row: sqlite3.Row, fm: dict[str, Any], body: str) -> Path:
    if not row["path"]:
        raise SystemExit(f"ERROR: document {row['id']} has no source path; use put with a Markdown file first")
    path = resolve_rel_path(row["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.dump(fm, allow_unicode=True, sort_keys=False).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n{body.lstrip(chr(10))}", encoding="utf-8")
    return path


def doc_markdown(row: sqlite3.Row) -> str:
    fm = json.loads(row["meta_json"] or "{}")
    if "id" not in fm:
        fm["id"] = row["id"]
    if "title" not in fm:
        fm["title"] = row["title"]
    frontmatter = yaml.dump(fm, allow_unicode=True, sort_keys=False).strip()
    body = read_doc_body(row).lstrip("\n")
    return f"---\n{frontmatter}\n---\n\n{body}"


def parse_set(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected FIELD=VALUE")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("empty field name")
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        parsed = raw
    return key, parsed


def update_doc(con: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = get_doc(con, args.id, include_deleted=True)
    if row is None:
        raise SystemExit(f"ERROR: no document found for {args.id!r}")

    fm = json.loads(row["meta_json"] or "{}")
    body = read_doc_body(row)
    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")
    if args.title:
        fm["title"] = args.title
    for key, value in args.set or []:
        fm[key] = value
    for key in args.unset or []:
        fm.pop(key, None)
    for key, value in args.append or []:
        existing = fm.get(key)
        if existing is None:
            fm[key] = [value]
        elif isinstance(existing, list):
            if value not in existing:
                existing.append(value)
        else:
            fm[key] = [existing, value]
    for key, value in args.remove or []:
        existing = fm.get(key)
        if isinstance(existing, list):
            fm[key] = [item for item in existing if item != value]
        elif existing == value:
            fm.pop(key, None)

    fm["id"] = row["id"]

    if args.body_file:
        path = write_doc_source(row, fm, body)
        upsert_page(con, path, reason=args.reason or "update-body-source", deleted_at=row["deleted_at"])
        return

    title = str(fm.get("title") or row["title"])
    body_sha = stable_sha(body)
    meta_json = json_dumps(fm)
    tag_text = search_tags_text(fm)
    body_excerpt = make_excerpt(body)
    ts = now_utc()
    if row["body_sha"] != body_sha:
        con.execute(
            """
            INSERT INTO revisions(doc_pk, old_body_sha, new_body_sha, patch_or_snapshot, created_at, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["doc_pk"],
                row["body_sha"],
                body_sha,
                json_dumps({"old_title": row["title"], "source_path": row["path"]}),
                ts,
                args.reason or "update",
            ),
        )
    con.execute(
        """
        UPDATE docs
        SET title = ?, body_excerpt = ?, body_sha = ?, meta_json = ?, search_tags = ?,
            repo = ?, pr = ?, date = ?, confidence = ?, updated_at = ?
        WHERE doc_pk = ?
        """,
        (
            title,
            body_excerpt,
            body_sha,
            meta_json,
            tag_text,
            str(fm.get("repo")) if fm.get("repo") is not None else None,
            str(fm.get("pr")) if fm.get("pr") is not None else None,
            str(fm.get("date") or fm.get("retrieved_at") or fm.get("captured_at") or "") or None,
            str(fm.get("confidence")) if fm.get("confidence") is not None else None,
            ts,
            row["doc_pk"],
        ),
    )
    path = resolve_rel_path(row["path"]) if row["path"] else WIKI_ROOT
    replace_derived_rows(con, int(row["doc_pk"]), path, fm)
    replace_fts_row(con, int(row["doc_pk"]), title, body, tag_text)


def export_rows(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    if args.all:
        rows = list(con.execute("SELECT * FROM docs WHERE deleted_at IS NULL ORDER BY id"))
    else:
        row = get_doc(con, args.lookup, include_deleted=args.include_deleted)
        if row is None:
            raise SystemExit(f"ERROR: no document found for {args.lookup!r}")
        rows = [row]

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            out_path = out_dir / f"{row['id']}.md"
            out_path.write_text(doc_markdown(row), encoding="utf-8")
        return len(rows)

    if args.write_source:
        count = 0
        for row in rows:
            if not row["path"]:
                continue
            out_path = resolve_rel_path(row["path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(doc_markdown(row), encoding="utf-8")
            count += 1
        return count

    if len(rows) != 1:
        raise SystemExit("ERROR: use --out-dir or --write-source when exporting multiple docs")
    print(doc_markdown(rows[0]))
    return 1


def cmd_init(args: argparse.Namespace) -> int:
    missing_paths: list[dict[str, Any]] = []
    artifact_missing: list[dict[str, Any]] = []
    with connect(args.db) as con:
        init_schema(con)
    print(f"initialized {rel_or_abs(args.db)}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    if args.reset and args.db.exists():
        args.db.unlink()
    if args.reset:
        for sidecar in sqlite_sidecar_paths(args.db):
            if sidecar.exists():
                sidecar.unlink()
    with connect(args.db) as con:
        init_schema(con)
        imported = 0
        created = 0
        seen_ids: set[str] = set()
        errors: list[str] = []
        with con:
            for page in iter_markdown_pages():
                try:
                    doc_id, was_created = upsert_page(con, page, reason="build")
                except Exception as exc:
                    errors.append(f"{rel_or_abs(page)}: {exc}")
                    continue
                seen_ids.add(doc_id)
                imported += 1
                created += int(was_created)
            if args.prune_missing and seen_ids:
                placeholders = ",".join("?" for _ in seen_ids)
                con.execute(
                    f"UPDATE docs SET deleted_at = ? WHERE id NOT IN ({placeholders}) AND deleted_at IS NULL",
                    [now_utc(), *sorted(seen_ids)],
                )
            rebuild_fts(con)
        write_manifest(con, args.db)

    print(f"indexed {imported} markdown page(s) into {rel_or_abs(args.db)}")
    print(f"created {created} new doc(s)")
    if errors:
        print(f"WARNING: skipped {len(errors)} page(s)", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more", file=sys.stderr)
    return 0 if not errors else 1


def cmd_put(args: argparse.Namespace) -> int:
    path = resolve_rel_path(args.path)
    if not path.is_file():
        raise SystemExit(f"ERROR: markdown file not found: {args.path}")
    with connect(args.db) as con:
        init_schema(con)
        with con:
            doc_id, created = upsert_page(con, path, reason=args.reason or "put")
    print(f"{'created' if created else 'updated'} {doc_id}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        rows = run_search(con, args)
    print_search_rows(rows, json_output=args.json)
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        row = get_doc(con, args.lookup, include_deleted=args.include_deleted)
        if row is None:
            raise SystemExit(f"ERROR: no document found for {args.lookup!r}")
        if args.json:
            data = row_to_dict(row)
            data["meta"] = json.loads(data.pop("meta_json") or "{}")
            data["body"] = read_doc_body(row)
            if args.follow_sources:
                data["sources"] = [
                    row_to_dict(src)
                    for src in con.execute(
                        """
                        SELECT s.id, s.kind, s.title, s.path, s.confidence
                        FROM doc_links l
                        JOIN docs s ON s.id = l.target_id
                        WHERE l.doc_pk = ? AND l.rel_type = 'source'
                        ORDER BY s.kind, s.id
                        """,
                        (row["doc_pk"],),
                    )
                ]
            if args.include_code:
                data["artifacts"] = [
                    row_to_dict(artifact)
                    for artifact in con.execute(
                        "SELECT kind, path, sha256, size_bytes FROM artifacts WHERE doc_pk = ? ORDER BY path",
                        (row["doc_pk"],),
                    )
                ]
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.body_only:
            print(read_doc_body(row))
        elif args.frontmatter_only:
            print(yaml.dump(json.loads(row["meta_json"] or "{}"), allow_unicode=True, sort_keys=False))
        else:
            print(doc_markdown(row))
            if args.follow_sources:
                sources = list(
                    con.execute(
                        """
                        SELECT s.id, s.kind, s.title, s.path, s.confidence
                        FROM doc_links l
                        JOIN docs s ON s.id = l.target_id
                        WHERE l.doc_pk = ? AND l.rel_type = 'source'
                        ORDER BY s.kind, s.id
                        """,
                        (row["doc_pk"],),
                    )
                )
                if sources:
                    print()
                    print("---")
                    print("## Source Links")
                    print()
                    for src in sources:
                        confidence = f" confidence={src['confidence']}" if src["confidence"] else ""
                        print(f"- [{src['kind']}] {src['id']}: {src['title']} ({src['path']}){confidence}")
            if args.include_code:
                artifacts = list(
                    con.execute(
                        "SELECT kind, path, sha256, size_bytes FROM artifacts WHERE doc_pk = ? ORDER BY path",
                        (row["doc_pk"],),
                    )
                )
                if artifacts:
                    print()
                    print("---")
                    print("## Artifact Files")
                    print()
                    for artifact in artifacts:
                        print(
                            f"- `{artifact['path']}` "
                            f"kind={artifact['kind']} size={artifact['size_bytes']} sha256={artifact['sha256']}"
                        )
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        with con:
            update_doc(con, args)
    print(f"updated {args.id}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        row = get_doc(con, args.id, include_deleted=True)
        if row is None:
            raise SystemExit(f"ERROR: no document found for {args.id!r}")
        with con:
            if args.hard:
                con.execute(
                    """
                    INSERT INTO revisions(doc_pk, old_body_sha, new_body_sha, patch_or_snapshot, created_at, reason)
                    VALUES (?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        row["doc_pk"],
                        row["body_sha"],
                        json_dumps({"action": "hard-delete", "id": row["id"], "path": row["path"]}),
                        now_utc(),
                        args.reason or "hard-delete",
                    ),
                )
                con.execute("DELETE FROM docs WHERE doc_pk = ?", (row["doc_pk"],))
                con.execute("DELETE FROM docs_fts WHERE rowid = ?", (row["doc_pk"],))
                print(f"hard-deleted {args.id}")
            else:
                deleted_at = now_utc()
                con.execute(
                    "UPDATE docs SET deleted_at = ?, updated_at = ? WHERE doc_pk = ?",
                    (deleted_at, deleted_at, row["doc_pk"]),
                )
                con.execute(
                    """
                    INSERT INTO revisions(doc_pk, old_body_sha, new_body_sha, patch_or_snapshot, created_at, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["doc_pk"],
                        row["body_sha"],
                        row["body_sha"],
                        json_dumps({"action": "soft-delete", "deleted_at": deleted_at}),
                        deleted_at,
                        args.reason or "soft-delete",
                    ),
                )
                con.execute("DELETE FROM docs_fts WHERE rowid = ?", (row["doc_pk"],))
                print(f"soft-deleted {args.id}")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        row = get_doc(con, args.id, include_deleted=True)
        if row is None:
            raise SystemExit(f"ERROR: no document found for {args.id!r}")
        with con:
            restored_at = now_utc()
            con.execute(
                "UPDATE docs SET deleted_at = NULL, updated_at = ? WHERE doc_pk = ?",
                (restored_at, row["doc_pk"]),
            )
            con.execute(
                """
                INSERT INTO revisions(doc_pk, old_body_sha, new_body_sha, patch_or_snapshot, created_at, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["doc_pk"],
                    row["body_sha"],
                    row["body_sha"],
                    json_dumps({"action": "restore", "restored_at": restored_at}),
                    restored_at,
                    args.reason or "restore",
                ),
            )
            replace_fts_row(con, int(row["doc_pk"]), row["title"], read_doc_body(row), row["search_tags"])
    print(f"restored {args.id}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        count = export_rows(con, args)
    if args.out_dir or args.write_source:
        print(f"exported {count} document(s)")
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        with con:
            rebuild_fts(con)
        write_manifest(con, args.db)
    print(f"rebuilt FTS index for {rel_or_abs(args.db)}")
    return 0


def cmd_vacuum(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        con.execute("VACUUM")
    print(f"vacuumed {rel_or_abs(args.db)}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        with con:
            con.execute("INSERT INTO docs_fts(docs_fts) VALUES('optimize')")
            con.execute("PRAGMA optimize")
        checkpoint = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    print(f"optimized {rel_or_abs(args.db)}")
    if checkpoint:
        print(f"wal_checkpoint: busy={checkpoint[0]} log={checkpoint[1]} checkpointed={checkpoint[2]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    with connect(args.db) as con:
        init_schema(con)
        schema_version = con.execute("SELECT value FROM kbs_meta WHERE key = 'schema_version'").fetchone()
        docs_cols = [row["name"] for row in con.execute("PRAGMA table_info(docs)")]
        if schema_version is None or schema_version["value"] != str(SCHEMA_VERSION):
            problems.append(f"schema_version expected {SCHEMA_VERSION}, got {schema_version['value'] if schema_version else 'missing'}")
        if "body" in docs_cols:
            problems.append("docs.body still exists; rebuild with `kbs.py build-db --reset`")
        if "body_excerpt" not in docs_cols:
            problems.append("docs.body_excerpt missing; rebuild with `kbs.py build-db --reset`")
        active = active_doc_count(con)
        fts_rows = fts_row_count(con)
        if active != fts_rows:
            problems.append(f"FTS row count mismatch: active_docs={active}, docs_fts={fts_rows}; run `kbs.py reindex`")
        missing_paths = [
            row_to_dict(row)
            for row in con.execute(
                "SELECT id, path FROM docs WHERE deleted_at IS NULL AND path IS NOT NULL ORDER BY id"
            )
            if not resolve_rel_path(row["path"]).is_file()
        ]
        if missing_paths:
            problems.append(f"{len(missing_paths)} active doc path(s) are missing; run `kbs.py build-db --reset --prune-missing`")
        artifact_missing = [
            row_to_dict(row)
            for row in con.execute(
                """
                SELECT d.id, a.path
                FROM artifacts a
                JOIN docs d ON d.doc_pk = a.doc_pk
                WHERE d.deleted_at IS NULL
                ORDER BY d.id, a.path
                """
            )
            if not resolve_rel_path(row["path"]).is_file()
        ]
        if artifact_missing:
            problems.append(f"{len(artifact_missing)} artifact path(s) are missing; refresh artifacts or rebuild metadata")

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not problems,
                    "schema_version": SCHEMA_VERSION,
                    "db": rel_or_abs(args.db),
                    "problems": problems,
                    "missing_paths": missing_paths[: args.limit],
                    "missing_artifacts": artifact_missing[: args.limit],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("# kernel_KBS doctor")
        print(f"db: {rel_or_abs(args.db)}")
        print(f"schema_version: {SCHEMA_VERSION}")
        print(f"status: {'ok' if not problems else 'problems'}")
        for problem in problems:
            print(f"- {problem}")
        for row in missing_paths[: args.limit]:
            print(f"  missing doc: {row['id']} {row['path']}")
        for row in artifact_missing[: args.limit]:
            print(f"  missing artifact: {row['id']} {row['path']}")
    return 0 if not problems else 1


def cmd_stats(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        rows = list(
            con.execute(
                """
                SELECT kind,
                       COUNT(*) AS total,
                       SUM(CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN deleted_at IS NOT NULL THEN 1 ELSE 0 END) AS deleted
                FROM docs
                GROUP BY kind
                ORDER BY kind
                """
            )
        )
        totals = con.execute(
            """
            SELECT COUNT(*) AS docs,
                   SUM(CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END) AS active_docs,
                   SUM(CASE WHEN deleted_at IS NOT NULL THEN 1 ELSE 0 END) AS deleted_docs
            FROM docs
            """
        ).fetchone()
        tag_count = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        artifact_count = con.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        revision_count = con.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    size = args.db.stat().st_size if args.db.exists() else 0
    print("# kernel_KBS SQLite store")
    print(f"db: {rel_or_abs(args.db)}")
    print(f"size_bytes: {size}")
    print(f"docs: {totals['docs'] or 0} active={totals['active_docs'] or 0} deleted={totals['deleted_docs'] or 0}")
    print(f"tags: {tag_count}")
    print(f"artifacts: {artifact_count}")
    print(f"revisions: {revision_count}")
    print()
    print("kind  total  active  deleted")
    print("----  -----  ------  -------")
    for row in rows:
        print(f"{row['kind']}  {row['total']}  {row['active'] or 0}  {row['deleted'] or 0}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    with connect(args.db) as con:
        init_schema(con)
        meta = {
            row["key"]: row["value"]
            for row in con.execute("SELECT key, value FROM kbs_meta ORDER BY key")
        }
        docs = [
            {
                "name": row["name"],
                "kind": row["kind"],
                "description": row["description"],
                "columns": json.loads(row["columns_json"] or "{}"),
                "llm_notes": row["llm_notes"],
            }
            for row in con.execute(
                "SELECT name, kind, description, columns_json, llm_notes FROM schema_docs ORDER BY kind, name"
            )
        ]
    if args.json:
        print(json.dumps({"meta": meta, "schema": docs}, ensure_ascii=False, indent=2))
        return 0

    print("# kernel_KBS SQLite Schema")
    print()
    for key in (
        "canonical_store",
        "llm_summary",
        "llm_query_guide",
        "llm_write_guide",
        "llm_maintenance_guide",
        "llm_answer_contract",
    ):
        if key in meta:
            print(f"- **{key}**: {meta[key]}")
    print()
    for item in docs:
        if args.table and item["name"] != args.table:
            continue
        print(f"## {item['name']} ({item['kind']})")
        print(item["description"])
        columns = item.get("columns") or {}
        if columns:
            print()
            for col, desc in columns.items():
                print(f"- `{col}`: {desc}")
        if item["llm_notes"]:
            print()
            print(f"LLM notes: {item['llm_notes']}")
        print()
    return 0


def add_common_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=DB_PATH, help=f"SQLite DB path (default: {rel_to_root(DB_PATH)})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SQLite CRUD and FTS index for kernel_KBS")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Initialize an empty SQLite store")
    add_common_db_arg(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("build", help="Import all Markdown pages into SQLite and rebuild FTS")
    add_common_db_arg(p)
    p.add_argument("--reset", action="store_true", help="Remove the existing SQLite file before indexing")
    p.add_argument("--prune-missing", action="store_true", help="Soft-delete DB docs that no longer exist in Markdown")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("put", help="Create or update one DB document from a Markdown page")
    add_common_db_arg(p)
    p.add_argument("path", help="Markdown path under the canonical store/docs layout")
    p.add_argument("--reason", help="Revision reason")
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("search", help="Search the SQLite FTS index")
    add_common_db_arg(p)
    p.add_argument("query", nargs="*", help="Free-text query")
    p.add_argument("--raw-match", action="store_true", help="Treat query as raw FTS5 MATCH syntax")
    p.add_argument("--kind", help="Filter by kind, e.g. wiki-kernel or source-pr")
    p.add_argument("--repo", help="Filter by repo substring")
    p.add_argument("--confidence", help="Filter by confidence")
    p.add_argument("--tag", action="append", help="Require a tag, any category; repeatable")
    p.add_argument("--language", action="append", help="Require a language tag; repeatable")
    p.add_argument("--architecture", action="append", help="Require an architecture tag; repeatable")
    p.add_argument("--symptom", action="append", help="Require a symptom tag; repeatable")
    p.add_argument("--has-code", action="store_true", help="Only return docs with artifact source/code files")
    p.add_argument("--include-deleted", action="store_true", help="Include soft-deleted docs")
    p.add_argument("--limit", type=int, default=10, help="Max results")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="Read one DB document by id or path")
    add_common_db_arg(p)
    p.add_argument("lookup", help="Document id or canonical path")
    p.add_argument("--body-only", action="store_true")
    p.add_argument("--frontmatter-only", action="store_true")
    p.add_argument("--follow-sources", action="store_true", help="Print source-link metadata from SQLite")
    p.add_argument("--include-code", action="store_true", help="Print artifact file metadata from SQLite")
    p.add_argument("--include-deleted", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("update", help="Update one DB document in place")
    add_common_db_arg(p)
    p.add_argument("id", help="Document id")
    p.add_argument("--title", help="Set title")
    p.add_argument("--body-file", help="Replace body from a file")
    p.add_argument("--set", action="append", type=parse_set, metavar="FIELD=VALUE", help="Set frontmatter field")
    p.add_argument("--unset", action="append", metavar="FIELD", help="Remove frontmatter field")
    p.add_argument("--append", action="append", type=parse_set, metavar="FIELD=VALUE", help="Append to list field")
    p.add_argument("--remove", action="append", type=parse_set, metavar="FIELD=VALUE", help="Remove from list field")
    p.add_argument("--reason", help="Revision reason")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("delete", help="Soft-delete or hard-delete one DB document")
    add_common_db_arg(p)
    p.add_argument("id", help="Document id")
    p.add_argument("--hard", action="store_true", help="Physically delete the document and derived rows")
    p.add_argument("--reason", help="Revision reason")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("restore", help="Restore a soft-deleted DB document")
    add_common_db_arg(p)
    p.add_argument("id", help="Document id")
    p.add_argument("--reason", help="Revision reason")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("export", help="Export DB document(s) as Markdown")
    add_common_db_arg(p)
    p.add_argument("lookup", nargs="?", help="Document id or path")
    p.add_argument("--all", action="store_true", help="Export all active documents")
    p.add_argument("--out-dir", help="Write exported Markdown files to this directory")
    p.add_argument("--write-source", action="store_true", help="Write exported Markdown back to each document's source path")
    p.add_argument("--include-deleted", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("reindex", help="Rebuild the FTS table from docs")
    add_common_db_arg(p)
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("optimize", help="Optimize SQLite/FTS and truncate WAL")
    add_common_db_arg(p)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("doctor", help="Check SQLite schema, FTS, source paths, and artifact paths")
    add_common_db_arg(p)
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.add_argument("--limit", type=int, default=20, help="Max missing path details")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("vacuum", help="Run SQLite VACUUM")
    add_common_db_arg(p)
    p.set_defaults(func=cmd_vacuum)

    p = sub.add_parser("stats", help="Show SQLite store counts")
    add_common_db_arg(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("schema", help="Show LLM-readable SQLite schema and workflow notes")
    add_common_db_arg(p)
    p.add_argument("--table", help="Show one table/object")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.set_defaults(func=cmd_schema)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "all", False) and getattr(args, "lookup", None):
        parser.error("export accepts either LOOKUP or --all, not both")
    if getattr(args, "command", "") == "export" and not args.all and not args.lookup:
        parser.error("export requires LOOKUP or --all")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
