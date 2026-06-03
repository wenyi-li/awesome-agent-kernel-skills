#!/usr/bin/env python3
"""Unified SQLite entry point for the kernel_KBS corpus."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import (  # noqa: E402
    WIKI_ROOT,
    rel_to_root,
)


def rel(path: Path) -> str:
    return rel_to_root(path)


def forward(script_name: str, forwarded: list[str]) -> int:
    script = WIKI_ROOT / "scripts" / script_name
    if not script.is_file():
        print(f"ERROR: missing tool: {rel(script)}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(script), *forwarded], cwd=str(WIKI_ROOT))


HELP = dedent(
    """\
    usage: kbs.py <command> [args]

    Unified SQLite-centered entry point for kernel_KBS.

    SQLite CRUD / retrieval:
      build-db           Import Markdown pages into SQLite and rebuild FTS
      query, search      Search SQLite FTS
      get                Read one document by id or path
      put                Create/update one document from Markdown
      update             Update DB metadata or replace a source-backed body
      delete, restore    Soft-delete or restore documents
      export             Export DB documents as Markdown
      reindex            Rebuild FTS from docs
      doctor             Check schema, FTS, source paths, and artifact paths
      optimize, vacuum   SQLite maintenance
      stats, schema      Show counts or LLM-readable schema docs

    Maintenance groups:
      check              Offline governance checks: size, freshness, fixtures, all
      update-prs         Refresh PR source pages and optionally rebuild SQLite

    Examples:
      python3 scripts/kbs.py query tcgen05 --architecture sm100
      python3 scripts/kbs.py get hw-tcgen05-mma --follow-sources
      python3 scripts/kbs.py check all
      python3 scripts/kbs.py check --help
    """
)


def main() -> int:
    command_map = {
        "build-db": "build",
        "query": "search",
        "search": "search",
        "get": "get",
        "put": "put",
        "update": "update",
        "delete": "delete",
        "restore": "restore",
        "export": "export",
        "reindex": "reindex",
        "optimize": "optimize",
        "doctor": "doctor",
        "vacuum": "vacuum",
        "stats": "stats",
        "schema": "schema",
    }
    if len(sys.argv) <= 1:
        print(HELP)
        return 0
    command = sys.argv[1]
    if command in {"-h", "--help"}:
        print(HELP)
        return 0
    if command == "check":
        return forward("kbs_checks.py", sys.argv[2:])
    if command == "update-prs":
        return forward("update_pr_corpus.py", sys.argv[2:])
    if command not in command_map:
        print(f"ERROR: unknown command: {command}", file=sys.stderr)
        return forward("kbs_db.py", ["--help"])
    return forward("kbs_db.py", [command_map[command], *sys.argv[2:]])


if __name__ == "__main__":
    raise SystemExit(main())
