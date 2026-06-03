#!/usr/bin/env python3
"""Unified offline governance checks for the kernel_KBS corpus.

This module replaces small one-off check scripts with one stable CLI:

  python3 scripts/kbs.py check size
  python3 scripts/kbs.py check freshness
  python3 scripts/kbs.py check fixtures
  python3 scripts/kbs.py check all

The checks are intentionally network-free. Networked source refresh and
upstream byte verification remain separate maintenance operations.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import (  # noqa: E402
    ARTIFACTS_DIR,
    DOCS_DIR,
    INDEXES_DIR,
    LEDGERS_DIR,
    RAW_SOURCES_DIR,
    AUDIT_VALIDATION_DIR,
    STATE_REFRESH_DIR,
    STATE_VERSIONS_DIR,
    WIKI_DIR,
    WIKI_ROOT,
    rel_to_root,
)

REPO_ROOT = WIKI_ROOT
CANDIDATES_DIR = LEDGERS_DIR / "candidates"
FIXTURES_PATH = AUDIT_VALIDATION_DIR / "phase3-dod-fixtures.yaml"
BUDGET_PATH = AUDIT_VALIDATION_DIR / "phase3-size-budget.yaml"

DEFAULT_VERSION_STALENESS_DAYS = 180
DEFAULT_LEDGER_STALENESS_DAYS = 30
FILE_COUNT_BUDGET = 6000

EXCLUDE_TOP = {
    ".git",
    ".humanize",
    ".codex",
    ".cache",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDE_ANY = {"__pycache__"}


def load_yaml(path: Path):
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_iso(value) -> date:
    """Return a date from a YAML scalar value or raise ValueError."""
    if hasattr(value, "isoformat") and not isinstance(value, str):
        if hasattr(value, "date"):
            return value.date()
        return value
    return date.fromisoformat(str(value))


def iter_tool_version_findings(tool_versions, today: date, version_staleness_days: int):
    threshold = today - timedelta(days=version_staleness_days)
    for tool in tool_versions.get("tools", []):
        for rel in tool.get("releases", []):
            name = rel.get("name", "?")
            released_at = rel.get("released_at")
            if released_at in (None, "needs-verification"):
                yield "warn", (
                    f"tool-versions: {tool['tool']} {name} has "
                    f"released_at={released_at!r} (needs-verification)"
                )
                continue
            try:
                release_date = parse_iso(released_at)
            except ValueError:
                yield "warn", f"tool-versions: {tool['tool']} {name} has unparseable released_at={released_at!r}"
                continue
            if release_date < threshold:
                age_days = (today - release_date).days
                yield "info", (
                    f"tool-versions: {tool['tool']} {name} released {age_days}d ago "
                    f"(>= {version_staleness_days}d staleness threshold)"
                )


def iter_version_claim_findings(claims_data, tool_versions, today: date, version_staleness_days: int):
    if claims_data is None:
        return
    threshold = today - timedelta(days=version_staleness_days)
    known_releases = {
        rel.get("name")
        for tool in tool_versions.get("tools", [])
        for rel in tool.get("releases", [])
    }
    for claim in claims_data.get("claims", []) or []:
        claim_id = claim.get("id", "?")
        verified_at = claim.get("last_verified_at")
        verified_release = claim.get("last_verified_release")
        if verified_at is None:
            yield "warn", f"version-claims: {claim_id} missing last_verified_at"
        else:
            try:
                verified_date = parse_iso(verified_at)
                if verified_date < threshold:
                    age_days = (today - verified_date).days
                    yield "warn", (
                        f"version-claims: {claim_id} last_verified_at={verified_date.isoformat()} "
                        f"({age_days}d ago, >= {version_staleness_days}d threshold)"
                    )
            except ValueError:
                yield "warn", f"version-claims: {claim_id} unparseable last_verified_at={verified_at!r}"
        if verified_release and verified_release not in known_releases:
            yield "warn", (
                f"version-claims: {claim_id} last_verified_release={verified_release!r} "
                f"not in {rel_to_root(STATE_VERSIONS_DIR / 'tool-versions.yaml')}"
            )


def iter_ledger_freshness_findings(today: date, ledger_staleness_days: int, refresh_cutoff: date | None):
    if not CANDIDATES_DIR.is_dir():
        return
    threshold = today - timedelta(days=ledger_staleness_days)
    for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
        data = load_yaml(ledger_file)
        if not isinstance(data, dict):
            continue
        searched_at = data.get("searched_at")
        if searched_at is None:
            yield "warn", f"ledger {ledger_file.name}: missing searched_at"
            continue
        try:
            searched_date = parse_iso(searched_at)
        except ValueError:
            yield "warn", f"ledger {ledger_file.name}: unparseable searched_at={searched_at!r}"
            continue
        if refresh_cutoff and searched_date != refresh_cutoff:
            yield "info", (
                f"ledger {ledger_file.name}: searched_at={searched_date.isoformat()} "
                f"disagrees with refresh-cutoff {refresh_cutoff.isoformat()}"
            )
        elif not refresh_cutoff and searched_date < threshold:
            age_days = (today - searched_date).days
            yield "warn", (
                f"ledger {ledger_file.name}: searched_at={searched_date.isoformat()} "
                f"({age_days}d ago, >= {ledger_staleness_days}d threshold)"
            )


def cmd_freshness(args: argparse.Namespace) -> int:
    today = date.fromisoformat(args.today) if args.today else date.today()

    tool_versions = load_yaml(STATE_VERSIONS_DIR / "tool-versions.yaml") or {}
    claims = load_yaml(STATE_VERSIONS_DIR / "version-claims.yaml") or {}
    refresh_cutoff_path = STATE_REFRESH_DIR / "refresh-cutoff.yaml"
    refresh_cutoff_data = load_yaml(refresh_cutoff_path)
    refresh_cutoff = None
    if refresh_cutoff_data and "cutoff_date" in refresh_cutoff_data:
        try:
            refresh_cutoff = parse_iso(refresh_cutoff_data["cutoff_date"])
        except ValueError:
            print(
                f"WARN: {rel_to_root(refresh_cutoff_path)}::cutoff_date is "
                f"unparseable: {refresh_cutoff_data['cutoff_date']!r}",
                file=sys.stderr,
            )

    findings = []
    findings.extend(iter_tool_version_findings(tool_versions, today, args.version_staleness_days))
    findings.extend(iter_version_claim_findings(claims, tool_versions, today, args.version_staleness_days))
    findings.extend(iter_ledger_freshness_findings(today, args.ledger_staleness_days, refresh_cutoff))

    warns = [m for severity, m in findings if severity == "warn"]
    infos = [m for severity, m in findings if severity == "info"]

    print(
        f"kbs_checks.py freshness - today={today.isoformat()}, "
        f"version-staleness={args.version_staleness_days}d, "
        f"ledger-staleness={args.ledger_staleness_days}d"
    )
    print(f"  {len(warns)} warnings, {len(infos)} info messages")
    for msg in warns:
        print(f"  WARN: {msg}")
    for msg in infos:
        print(f"  INFO: {msg}")

    if args.strict and warns:
        print(f"\n--strict: {len(warns)} warning(s) -> exit 1", file=sys.stderr)
        return 1
    return 0


def find_bundle_root_for(path: Path) -> Path | None:
    p = path.parent
    while p != p.parent:
        if (p / "PROVENANCE.yaml").is_file():
            return p
        p = p.parent
    return None


def load_modes_for_file(file_path: Path):
    root = find_bundle_root_for(file_path)
    if not root:
        return None
    prov_path = root / "PROVENANCE.yaml"
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        return None
    for entry in prov.get("files") or []:
        if isinstance(entry, dict) and entry.get("local_path") == rel:
            return entry.get("mode")
    return None


def load_bundle_mode_for_file(file_path: Path):
    root = find_bundle_root_for(file_path)
    if not root:
        return None
    prov_path = root / "PROVENANCE.yaml"
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    return prov.get("asset_mode")


def glob_canonical(pattern: str) -> list[Path]:
    """Glob files from canonical skill-root-relative paths."""
    rel = Path(pattern)
    if not rel.parts or rel.is_absolute() or rel.parts[0] != "store":
        return []
    return [p for p in REPO_ROOT.glob(pattern) if p.is_file()]


def check_fixture_entry(entry) -> list[str]:
    errors = []
    question = entry.get("question", "<unnamed>")
    required_assets = entry.get("required_assets") or []
    required_min_lines = entry.get("required_min_lines", 100)
    required_modes = set(entry.get("required_provenance_modes") or [])
    required_bundle_modes = set(entry.get("required_bundle_asset_mode") or [])
    required_patterns = entry.get("required_content_patterns") or []

    per_glob_matches = {}
    per_glob_min_lines = {}
    for asset in required_assets:
        if isinstance(asset, dict):
            glob = asset.get("glob")
            min_lines = asset.get("min_lines", required_min_lines)
        else:
            glob = asset
            min_lines = required_min_lines
        if not glob:
            continue
        hits = glob_canonical(glob)
        per_glob_matches[glob] = hits
        per_glob_min_lines[glob] = min_lines
        if not hits:
            errors.append(f"{question!r}: required_assets glob {glob!r} matched no files")

    if errors:
        return errors

    for glob, hits in per_glob_matches.items():
        floor = per_glob_min_lines[glob]
        long_enough = False
        for path in hits:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(lines) >= floor:
                    long_enough = True
                    break
            except OSError:
                pass
        if not long_enough:
            errors.append(f"{question!r}: no file matching {glob!r} reached required_min_lines={floor}")

    if required_modes:
        for glob, hits in per_glob_matches.items():
            if not any(load_modes_for_file(path) in required_modes for path in hits):
                errors.append(
                    f"{question!r}: no file matching {glob!r} has per-file provenance "
                    f"mode in {sorted(required_modes)}"
                )

    if required_bundle_modes:
        for glob, hits in per_glob_matches.items():
            if not any(load_bundle_mode_for_file(path) in required_bundle_modes for path in hits):
                errors.append(
                    f"{question!r}: no file matching {glob!r} lives in a bundle with "
                    f"asset_mode in {sorted(required_bundle_modes)}"
                )

    if required_patterns:
        aggregate = ""
        for hits in per_glob_matches.values():
            for path in hits:
                try:
                    aggregate += path.read_text(encoding="utf-8", errors="replace") + "\n"
                except OSError:
                    continue
        for pattern in required_patterns:
            try:
                if not re.search(pattern, aggregate, re.IGNORECASE):
                    errors.append(f"{question!r}: required_content_pattern {pattern!r} not found")
            except re.error as exc:
                errors.append(f"{question!r}: invalid regex {pattern!r}: {exc}")

    return errors


def cmd_fixtures(args: argparse.Namespace) -> int:
    if not FIXTURES_PATH.is_file():
        print(
            f"ERROR: {rel_to_root(FIXTURES_PATH)} not found; the DoD fixture gate "
            f"requires this file. If the gate is retired, remove this check from "
            f"kbs_checks.py instead of deleting only the fixture file.",
            file=sys.stderr,
        )
        return 2

    try:
        data = yaml.safe_load(FIXTURES_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"ERROR: could not parse {rel_to_root(FIXTURES_PATH)}: {exc}", file=sys.stderr)
        return 2

    entries = data.get("fixtures") or []
    if not entries:
        print(f"ERROR: {rel_to_root(FIXTURES_PATH)} has no `fixtures:` entries.", file=sys.stderr)
        return 2

    all_errors = []
    for entry in entries:
        if isinstance(entry, dict):
            all_errors.extend(check_fixture_entry(entry))

    print(f"Checked {len(entries)} DoD fixture entries.")
    if all_errors:
        for err in all_errors:
            print(f"  FAIL: {err}", file=sys.stderr)
        return 1
    print("All fixtures pass.")
    return 0


def working_tree_bytes(*, include_indexes: bool) -> tuple[int, int]:
    total = 0
    count = 0
    for path in REPO_ROOT.rglob("*"):
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] in EXCLUDE_TOP:
            continue
        if not include_indexes and len(parts) >= 2 and parts[:2] == ("store", "indexes"):
            continue
        if any(part in EXCLUDE_ANY for part in parts):
            continue
        if path.is_file():
            try:
                total += path.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def index_bytes() -> tuple[int, int]:
    if not INDEXES_DIR.is_dir():
        return 0, 0
    total = 0
    count = 0
    for path in INDEXES_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
            count += 1
        except OSError:
            continue
    return total, count


def artifacts_file_count() -> int:
    if not ARTIFACTS_DIR.is_dir():
        return 0
    return sum(1 for path in ARTIFACTS_DIR.rglob("*") if path.is_file())


def load_budget():
    if not BUDGET_PATH.is_file():
        return None
    try:
        return yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def cmd_size(args: argparse.Namespace) -> int:
    content_total, content_files = working_tree_bytes(include_indexes=False)
    indexes_total, indexes_files = index_bytes()
    total = content_total + indexes_total
    total_files = content_files + indexes_files
    artifacts_files = artifacts_file_count()
    content_mib = content_total / (1024 * 1024)
    indexes_mib = indexes_total / (1024 * 1024)
    total_mib = total / (1024 * 1024)

    budget = load_budget()
    if args.ceiling_mib is not None:
        ceiling_mib = args.ceiling_mib
        ceiling_src = "CLI override"
    elif budget and "active_ceiling_mib" in budget:
        ceiling_mib = budget["active_ceiling_mib"]
        ceiling_src = rel_to_root(BUDGET_PATH)
    else:
        ceiling_mib = None
        ceiling_src = None

    index_ceiling_mib = args.index_ceiling_mib
    if index_ceiling_mib is None and budget:
        index_ceiling_mib = budget.get("sqlite_index_ceiling_mib")
    if index_ceiling_mib is None:
        index_ceiling_mib = 32

    print(
        f"Working tree: {total_mib:.2f} MiB across {total_files} files "
        f"(content: {content_mib:.2f} MiB/{content_files} files, "
        f"SQLite indexes: {indexes_mib:.2f} MiB/{indexes_files} files, "
        f"store/corpus/artifacts: {artifacts_files} files)"
    )

    failed = False
    if ceiling_mib is not None:
        print(f"Active content ceiling: {ceiling_mib} MiB excluding store/indexes (source: {ceiling_src})")
        if content_mib > ceiling_mib:
            print(
                f"FAIL: content tree {content_mib:.2f} MiB exceeds ceiling {ceiling_mib} MiB",
                file=sys.stderr,
            )
            failed = True
    else:
        print("Active content ceiling: (no budget file present; skip content size gate)")

    print(f"Active SQLite index ceiling: {index_ceiling_mib} MiB")
    if indexes_mib > index_ceiling_mib:
        print(
            f"FAIL: store/indexes is {indexes_mib:.2f} MiB (> {index_ceiling_mib} MiB budget)",
            file=sys.stderr,
        )
        failed = True

    if artifacts_files > FILE_COUNT_BUDGET:
        print(
            f"FAIL: store/corpus/artifacts has {artifacts_files} files "
            f"(> {FILE_COUNT_BUDGET} budget)",
            file=sys.stderr,
        )
        failed = True

    return 1 if failed else 0


def cmd_all(args: argparse.Namespace) -> int:
    checks = [
        ("size", cmd_size),
        ("freshness", cmd_freshness),
        ("fixtures", cmd_fixtures),
    ]
    highest = 0
    for name, func in checks:
        print(f"\n## {name}")
        code = func(args)
        highest = max(highest, code)
    return highest


def add_freshness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict", action="store_true", help="Exit 1 if freshness warnings are present")
    parser.add_argument("--today", help="Override today's date for testing (YYYY-MM-DD)")
    parser.add_argument("--version-staleness-days", type=int, default=DEFAULT_VERSION_STALENESS_DAYS)
    parser.add_argument("--ledger-staleness-days", type=int, default=DEFAULT_LEDGER_STALENESS_DAYS)


def add_size_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ceiling-mib", type=int, help="Override the working-tree size ceiling (MiB)")
    parser.add_argument("--index-ceiling-mib", type=int, help="Override the SQLite index size ceiling (MiB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline governance checks for kernel_KBS")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freshness", help="Check local version and ledger freshness metadata")
    add_freshness_args(p)
    p.set_defaults(func=cmd_freshness)

    p = sub.add_parser("fixtures", help="Check Definition-of-Done fixture coverage")
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("size", help="Check working-tree and artifact size budgets")
    add_size_args(p)
    p.set_defaults(func=cmd_size)

    p = sub.add_parser("all", help="Run size, freshness, and fixture checks")
    add_size_args(p)
    add_freshness_args(p)
    p.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
