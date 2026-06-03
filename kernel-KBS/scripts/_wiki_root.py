"""Shared corpus-root and layout resolution for kernel_KBS scripts.

The skill uses a layered corpus layout: store/config, store/schemas,
store/docs, store/corpus, store/indexes, store/state, scripts, and references
live at the same root. Markdown documents use canonical store/docs paths.
By default the corpus root is this file's
grandparent:

    <kernel_KBS-root>/scripts/_wiki_root.py  ->  <kernel_KBS-root>

No environment variable is required for the common case. Optional
KERNEL_KBS_ROOT is honored for advanced setups. Any detected root is validated;
misconfiguration hard-errors rather than silently returning wrong results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _looks_like_wiki_root(p: Path) -> bool:
    return (p / "store" / "schemas" / "tags.yaml").is_file() and (p / "store" / "docs" / "wiki").is_dir()


def _error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_wiki_root() -> Path:
    # 1. explicit env override (advanced use)
    env = os.environ.get("KERNEL_KBS_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_wiki_root(p):
            return p
        _error(
            f"KERNEL_KBS_ROOT={env!r} does not point at a valid "
            f"kernel_KBS corpus (missing store/schemas/tags.yaml or store/docs/wiki/)."
        )

    # 2. default: script-file grandparent == skill/wiki root
    default_root = Path(__file__).resolve().parent.parent
    if _looks_like_wiki_root(default_root):
        return default_root

    # 3. walk up from script location and from cwd
    seen = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_wiki_root(candidate):
                return candidate

    _error(
        "Could not locate the kernel_KBS corpus root.\n"
       "       Expected a directory containing `store/schemas/tags.yaml` + `store/docs/wiki/`,\n"
       "       Fix: run scripts from inside the cloned skill directory, or\n"
       "       set KERNEL_KBS_ROOT to its absolute path."
    )
    return Path()  # unreachable


WIKI_ROOT = resolve_wiki_root()

STORE_DIR = WIKI_ROOT / "store"
CONFIG_DIR = STORE_DIR / "config"
SCHEMAS_DIR = STORE_DIR / "schemas"
DOCS_DIR = STORE_DIR / "docs"
CORPUS_DIR = STORE_DIR / "corpus"
RAW_SOURCES_DIR = DOCS_DIR / "sources"
ARTIFACTS_DIR = CORPUS_DIR / "artifacts"
LEDGERS_DIR = DOCS_DIR / "ledgers"
WIKI_DIR = DOCS_DIR / "wiki"
INDEXES_DIR = STORE_DIR / "indexes"
STATE_DIR = STORE_DIR / "state"
AUDITS_DIR = STATE_DIR / "audits"
STATE_REFRESH_DIR = STATE_DIR / "refresh"
STATE_VERSIONS_DIR = STATE_DIR / "versions"
AUDIT_CONTENT_DIR = AUDITS_DIR / "content"
AUDIT_REFRESH_DIR = AUDITS_DIR / "refresh"
AUDIT_VALIDATION_DIR = AUDITS_DIR / "validation"

def resolve_rel_path(path: str | Path) -> Path:
    """Resolve an absolute path, a canonical store path, or a skill-root path."""
    rel = Path(path)
    if rel.is_absolute():
        return rel
    parts = rel.parts
    if parts and parts[0] == "store":
        return WIKI_ROOT / rel
    return WIKI_ROOT / rel


def rel_to_root(path: Path) -> str:
    """Return a POSIX path relative to the skill root."""
    return path.relative_to(WIKI_ROOT).as_posix()
