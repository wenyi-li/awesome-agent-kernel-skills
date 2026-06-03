#!/usr/bin/env python3
"""Refresh kernel_KBS PR pages from GitHub REST API.

Default mode is a dry-run. Pass --apply to write PR pages and update state.
Pass --fetch-artifacts with --apply to capture diff.patch + PROVENANCE.yaml for
newly written PR pages. Repository coverage and matching rules live in
store/config/pr-update.yaml.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import ARTIFACTS_DIR, CONFIG_DIR, RAW_SOURCES_DIR, SCHEMAS_DIR, STATE_REFRESH_DIR, WIKI_ROOT, rel_to_root  # noqa: E402


CONFIG_PATH = CONFIG_DIR / "pr-update.yaml"
STATE_PATH = STATE_REFRESH_DIR / "pr-update-state.yaml"
SOURCES_PRS = RAW_SOURCES_DIR / "prs"
ARTIFACTS_PRS = ARTIFACTS_DIR / "prs"
TAGS_PATH = SCHEMAS_DIR / "tags.yaml"


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None


def dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True, width=200),
        encoding="utf-8",
    )


def parse_iso_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def parse_github_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def repo_short(repo_full: str, slug: str | None = None) -> str:
    return slug or repo_full.split("/")[-1].lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class GitHubClient:
    api_base: str
    token: str | None = None
    sleep_seconds: float = 1.0

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "kernel-KBS-pr-updater",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request_json(self, path_or_url: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
        raw, headers = self.request_bytes(path_or_url, params=params, accept="application/vnd.github+json")
        return json.loads(raw.decode("utf-8")), headers

    def request_bytes(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
    ) -> tuple[bytes, dict[str, str]]:
        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
        else:
            url = self.api_base.rstrip("/") + "/" + path_or_url.lstrip("/")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers=self._headers(accept))
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                headers = {k.lower(): v for k, v in resp.headers.items()}
                self._maybe_sleep_for_rate(headers)
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return data, headers
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in e.headers.items()}
            raise RuntimeError(f"GitHub API HTTP {e.code} for {url}: {body[:500]}{self._rate_hint(headers)}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"GitHub API request failed for {url}: {e}") from e

    def _rate_hint(self, headers: dict[str, str]) -> str:
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining == "0" and reset:
            reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc).isoformat()
            return f" (rate limit exhausted; reset at {reset_dt})"
        retry_after = headers.get("retry-after")
        return f" (secondary rate limit; retry after {retry_after}s)" if retry_after else ""

    def _maybe_sleep_for_rate(self, headers: dict[str, str]) -> None:
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining == "0" and reset:
            sleep_for = max(0, int(reset) - int(time.time()) + 1)
            if sleep_for:
                print(f"rate limit exhausted; sleeping {sleep_for}s", file=sys.stderr)
                time.sleep(sleep_for)


def load_tag_sets() -> dict[str, set[str]]:
    tags = load_yaml(TAGS_PATH) or {}
    return {k: set(v or []) for k, v in tags.items()}


def existing_pr_numbers() -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for md in SOURCES_PRS.rglob("PR-*.md"):
        text = md.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text, re.DOTALL)
        if not match:
            continue
        fm = yaml.safe_load(match.group(1)) or {}
        repo, pr = fm.get("repo"), fm.get("pr")
        if repo and isinstance(pr, int):
            out.setdefault(repo, set()).add(pr)
    return out


def list_closed_prs(client: GitHubClient, repo: str, max_pages: int, per_page: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        chunk, _ = client.request_json(
            f"/repos/{owner}/{name}/pulls",
            {"state": "closed", "sort": "updated", "direction": "desc", "per_page": per_page, "page": page},
        )
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < per_page:
            break
    return rows


def search_merged_prs(
    client: GitHubClient,
    repo: str,
    keywords: list[str],
    since: date,
    until: date,
    max_pages: int,
    per_page: int,
) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for keyword in keywords or [""]:
        q_parts = [f"repo:{repo}", "is:pr", "is:merged", f"merged:{since.isoformat()}..{until.isoformat()}"]
        if keyword:
            q_parts.insert(0, keyword)
        for page in range(1, max_pages + 1):
            payload, _ = client.request_json(
                "/search/issues",
                {"q": " ".join(q_parts), "sort": "updated", "order": "desc", "per_page": per_page, "page": page},
            )
            items = payload.get("items", []) or []
            for item in items:
                number = item.get("number")
                if isinstance(number, int):
                    seen[number] = item
            if len(items) < per_page:
                break
    return [get_pr(client, repo, n) for n in sorted(seen)]


def get_pr(client: GitHubClient, repo: str, number: int) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    data, _ = client.request_json(f"/repos/{owner}/{name}/pulls/{number}")
    return data


def get_pr_files(client: GitHubClient, repo: str, number: int, per_page: int) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    files: list[dict[str, Any]] = []
    page = 1
    while True:
        chunk, _ = client.request_json(f"/repos/{owner}/{name}/pulls/{number}/files", {"per_page": per_page, "page": page})
        if not isinstance(chunk, list) or not chunk:
            break
        files.extend(chunk)
        if len(chunk) < per_page:
            break
        page += 1
    return files


def text_for_match(pr: dict[str, Any], files: list[dict[str, Any]]) -> str:
    parts = [str(pr.get("title") or ""), str(pr.get("body") or "")]
    parts.extend(str(f.get("filename") or "") for f in files)
    return " ".join(parts).lower()


def path_is_excluded(path: str, globs: list[str]) -> bool:
    posix = path.replace("\\", "/")
    return any(fnmatch.fnmatch(posix, g) for g in globs)


def path_is_included(path: str, globs: list[str]) -> bool:
    posix = path.replace("\\", "/")
    base = posix.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(posix, g) or fnmatch.fnmatch(base, g) for g in globs)


def is_kernel_related(pr: dict[str, Any], files: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[bool, str]:
    title = str(pr.get("title") or "").lower()
    if any(re.search(pat, title) for pat in cfg.get("exclude_title_regex") or []):
        return False, "excluded by title regex"

    kept = [f.get("filename", "") for f in files if f.get("filename")]
    kept = [p for p in kept if not path_is_excluded(p, cfg.get("exclude_path_globs") or [])]
    if any(path_is_included(p, cfg.get("include_path_globs") or []) for p in kept):
        return True, "matched kernel path"

    text = text_for_match(pr, files)
    for kw in cfg.get("include_keywords") or []:
        if str(kw).lower() in text:
            return True, f"matched keyword: {kw}"
    return False, "no kernel keyword/path match"


def detect_architectures(text: str, tag_sets: dict[str, set[str]]) -> list[str]:
    lowered = text.lower()
    checks = {
        "sm80": ["sm80", "a100", "ampere"],
        "sm86": ["sm86", "a10", "a10g", "a40", "rtx 3090", "rtx 3080"],
        "sm87": ["sm87"],
        "sm90": ["sm90", "hopper", "h100", "h200", "h800"],
        "sm100": ["sm100", "sm_100", "blackwell", "b200", "b100", "gb200", "tcgen05", "tmem"],
        "sm120": ["sm120", "sm_120"],
    }
    valid = tag_sets.get("architectures", set())
    hits = [arch for arch, needles in checks.items() if arch in valid and any(n in lowered for n in needles)]
    return hits or (["sm100"] if "sm100" in valid else sorted(valid)[:1])


def classify(pr: dict[str, Any], files: list[dict[str, Any]], tag_sets: dict[str, set[str]]) -> dict[str, list[str]]:
    text = text_for_match(pr, files)
    mapping = {
        "hardware_features": {
            "cp-async": ["cp.async", "async copy"],
            "tcgen05": ["tcgen05", "umma"],
            "tmem": ["tmem", "tensor memory"],
            "tma": ["tma", "cp.async.bulk"],
            "clc": ["clc", "cluster launch"],
            "nvfp4": ["nvfp4", "fp4"],
            "fp8": ["fp8"],
            "wgmma": ["wgmma"],
            "ldmatrix": ["ldmatrix"],
            "stmatrix": ["stmatrix"],
        },
        "kernel_types": {
            "gemm": ["gemm", "matmul"],
            "attention": ["attention", "attn"],
            "flash-attention": ["flash attention", "flash_attention", "flash-attention", "fmha"],
            "mla": ["mla", "latent attention"],
            "moe": ["moe", "mixture of experts"],
            "gemv": ["gemv"],
            "grouped-gemm": ["grouped gemm", "grouped_gemm"],
            "decode": ["decode"],
            "prefill": ["prefill"],
            "quantization": ["quant", "mxfp", "nvfp4"],
            "sparse-attention": ["sparse attention", "sparse-attention", "nsa"],
        },
        "techniques": {
            "ampere-optimization": ["ampere", "sm80", "sm86", "sm87", "cp.async", "ldmatrix"],
            "warp-specialization": ["warp special", "warp-special"],
            "persistent-kernel": ["persistent"],
            "pipeline-stages": ["pipeline", "num_stages"],
            "double-buffering": ["double buffer", "double-buffer"],
            "swizzling": ["swizzl"],
            "epilogue-fusion": ["epilogue"],
            "tile-scheduling": ["tile schedul"],
            "kernel-fusion": ["fusion", "fused"],
            "vectorized-loads": ["vectorized", "v4"],
            "cache-policy": ["cache policy", "evict", "no_allocate"],
            "register-budgeting": ["register pressure", "register"],
            "fine-grained-quantization": ["block scale", "block_scale", "mxfp", "nvfp4"],
        },
        "languages": {
            "cuda-cpp": [".cu", ".cuh", "cuda", "cpp"],
            "cute-dsl": ["cute", "cutlass"],
            "triton": ["triton"],
            "tilelang": ["tilelang"],
            "ptx": [".ptx", "ptx"],
            "python": [".py", "python"],
        },
    }
    result: dict[str, list[str]] = {}
    for field, rules in mapping.items():
        valid = tag_sets.get(field, set())
        result[field] = sorted({tag for tag, needles in rules.items() if tag in valid and any(n in text for n in needles)})
    result["tags"] = sorted(set(result["hardware_features"]) | set(result["kernel_types"]) | set(result["techniques"]) | set(result["languages"]))
    if not result["tags"] and "gemm" in tag_sets.get("kernel_types", set()):
        result["tags"] = ["gemm"]
    if not result["languages"] and "cuda-cpp" in tag_sets.get("languages", set()):
        result["languages"] = ["cuda-cpp"]
    return result


def frontmatter_text(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True, width=200).rstrip()


def summarize_body(body: str | None) -> str:
    if not body:
        return "No description provided."
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:1200].strip() or "No description provided."


def make_pr_page(repo: str, slug: str, pr: dict[str, Any], files: list[dict[str, Any]], reason: str, captured_at: str, tag_sets: dict[str, set[str]]) -> tuple[str, str]:
    number = int(pr["number"])
    repo_id = repo_short(repo, slug)
    changed_paths = [f.get("filename", "") for f in files if f.get("filename")][:50]
    labels = classify(pr, files, tag_sets)
    fm = {
        "id": f"pr-{repo_id}-{number}",
        "repo": repo,
        "pr": number,
        "title": pr.get("title") or "",
        "author": (pr.get("user") or {}).get("login", "unknown"),
        "date": (pr.get("created_at") or pr.get("merged_at") or captured_at)[:10],
        "url": pr.get("html_url") or f"https://github.com/{repo}/pull/{number}",
        "source_category": "upstream-code",
        "architectures": detect_architectures(text_for_match(pr, files), tag_sets),
        "tags": labels["tags"],
        "techniques": labels["techniques"],
        "hardware_features": labels["hardware_features"],
        "kernel_types": labels["kernel_types"],
        "languages": labels["languages"],
        "captured_at": captured_at,
        "status": "merged",
        "merge_sha": (pr.get("merge_commit_sha") or "unknown")[:40],
        "inclusion_reason": reason,
        "changed_paths": changed_paths,
    }
    lines = ["---", frontmatter_text(fm), "---", "", "## Summary", "", summarize_body(pr.get("body")), "", "## Problem", "", str(pr.get("title") or ""), "", "## Changed Files", ""]
    lines.extend(f"- `{p}`" for p in changed_paths[:30])
    lines.append("")
    return f"PR-{number}.md", "\n".join(lines)


def write_artifact_bundle(client: GitHubClient, repo: str, slug: str, pr: dict[str, Any], captured_at: str) -> str:
    number = int(pr["number"])
    pr_id = f"pr-{repo_short(repo, slug)}-{number}"
    bundle = ARTIFACTS_PRS / repo_short(repo, slug) / f"PR-{number}"
    bundle.mkdir(parents=True, exist_ok=True)
    diff_url = pr.get("diff_url") or f"https://github.com/{repo}/pull/{number}.diff"
    diff_bytes, _ = client.request_bytes(diff_url, accept="application/vnd.github.diff")
    (bundle / "diff.patch").write_bytes(diff_bytes)
    prov = {
        "origin_url": pr.get("html_url") or f"https://github.com/{repo}/pull/{number}",
        "upstream_repo": repo,
        "upstream_sha": pr.get("merge_commit_sha") or "unknown",
        "license": "inherits-from-upstream",
        "retrieved_at": captured_at,
        "asset_mode": "verbatim",
        "generated_by": "scripts/update_pr_corpus.py --fetch-artifacts",
        "source_pr_id": pr_id,
        "files": [{"local_path": "diff.patch", "role": "pr-diff", "mode": "upstream-patch", "sha256": sha256_bytes(diff_bytes)}],
    }
    dump_yaml(bundle / "PROVENANCE.yaml", prov)
    return rel_to_root(bundle)


def add_artifact_dir_to_page(path: Path, artifact_dir: str) -> None:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^(---\s*\r?\n)(.*?)(\r?\n---\s*\r?\n)(.*)", text, re.DOTALL)
    if not m:
        return
    fm = yaml.safe_load(m.group(2)) or {}
    fm["artifact_dir"] = artifact_dir
    for banned in ("diff_path", "code_path", "patch_path", "key_file_path"):
        fm.pop(banned, None)
    path.write_text(m.group(1) + frontmatter_text(fm) + m.group(3) + m.group(4), encoding="utf-8")


def normalize_repo_cfg(defaults: dict[str, Any], repo_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(repo_cfg)
    for key in ("include_keywords", "include_path_globs", "exclude_title_regex", "exclude_path_globs"):
        merged[key] = list(dict.fromkeys(list(defaults.get(key) or []) + list(repo_cfg.get(key) or [])))
    return merged


def select_repos(config: dict[str, Any], requested: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    for row in config.get("repositories") or []:
        if row.get("enabled") is False:
            continue
        if not row.get("slug") or not row.get("repo"):
            continue
        if requested and row["slug"] not in requested and row["repo"] not in requested:
            continue
        rows.append(row)
    return rows


def rebuild_sqlite_db() -> None:
    subprocess.run([sys.executable, str(WIKI_ROOT / "scripts" / "kbs_db.py"), "build"], cwd=str(WIKI_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(CONFIG_PATH), help="YAML update config")
    parser.add_argument("--repos", default=None, help="Comma-separated repo slugs or full names from config")
    parser.add_argument("--since", default=None, help="Only merged PRs on/after YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="Only merged PRs on/before YYYY-MM-DD")
    parser.add_argument("--mode", choices=["list", "search"], default="list", help="Discovery mode")
    parser.add_argument("--max-pages", type=int, default=None, help="Max GitHub API pages per repo")
    parser.add_argument("--max-new", type=int, default=None, help="Stop after N new matching PRs")
    parser.add_argument("--apply", action="store_true", help="Write PR pages and state")
    parser.add_argument("--fetch-artifacts", action="store_true", help="With --apply, fetch diff.patch bundles")
    parser.add_argument("--rebuild-db", action="store_true", help="With --apply, rebuild the SQLite CRUD/FTS store")
    parser.add_argument("--token-env", default=None, help="Environment variable holding GitHub token")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    if not isinstance(config, dict):
        raise SystemExit(f"invalid config: {args.config}")
    defaults = config.get("defaults") or {}
    until = parse_iso_date(args.until) or today_utc()
    since = parse_iso_date(args.since) or (until - timedelta(days=int(defaults.get("since_days", 30))))
    if since > until:
        raise SystemExit("--since must be <= --until")
    token_env = args.token_env or defaults.get("token_env") or "GITHUB_TOKEN"
    token = os.environ.get(token_env)
    if not token:
        print(f"note: {token_env} is not set; unauthenticated GitHub API limit is much lower", file=sys.stderr)
    client = GitHubClient(defaults.get("api_base", "https://api.github.com"), token, float(defaults.get("sleep_seconds", 1.0)))
    tag_sets = load_tag_sets()
    existing = existing_pr_numbers()
    requested = set(args.repos.split(",")) if args.repos else None
    repos = select_repos(config, requested)
    max_pages = args.max_pages if args.max_pages is not None else int(defaults.get("max_pages_per_repo", 3))
    per_page = int(defaults.get("per_page", 100))
    captured_at = today_utc().isoformat()

    if not args.apply:
        print("DRY-RUN: pass --apply to write PR pages/state.")
    print(f"window: {since.isoformat()}..{until.isoformat()} mode={args.mode}")

    state_rows = []
    total_new = 0
    for raw_repo_cfg in repos:
        repo_cfg = normalize_repo_cfg(defaults, raw_repo_cfg)
        repo, slug = repo_cfg["repo"], repo_cfg["slug"]
        print(f"\n[{slug}] {repo}")
        prs = search_merged_prs(client, repo, repo_cfg.get("include_keywords") or [], since, until, max_pages, min(per_page, 100)) if args.mode == "search" else list_closed_prs(client, repo, max_pages, per_page)

        seen = matched = written = skipped_existing = 0
        new_ids: list[str] = []
        outdir = SOURCES_PRS / repo_short(repo, slug)
        if args.apply:
            outdir.mkdir(parents=True, exist_ok=True)

        for pr in prs:
            number = pr.get("number")
            if not isinstance(number, int):
                continue
            merged_at = parse_github_dt(pr.get("merged_at"))
            updated_at = parse_github_dt(pr.get("updated_at"))
            if args.mode == "list" and updated_at and updated_at.date() < since:
                break
            if not merged_at or not (since <= merged_at.date() <= until):
                continue
            seen += 1
            if number in existing.get(repo, set()):
                skipped_existing += 1
                continue
            files = get_pr_files(client, repo, number, per_page)
            include, reason = is_kernel_related(pr, files, repo_cfg)
            if not include:
                continue
            matched += 1
            filename, content = make_pr_page(repo, slug, pr, files, reason, captured_at, tag_sets)
            pr_id = f"pr-{repo_short(repo, slug)}-{number}"
            print(f"  + {pr_id}: {str(pr.get('title') or '')[:100]} ({reason})")
            total_new += 1
            if args.apply:
                path = outdir / filename
                path.write_text(content, encoding="utf-8")
                written += 1
                existing.setdefault(repo, set()).add(number)
                new_ids.append(pr_id)
                if args.fetch_artifacts:
                    add_artifact_dir_to_page(path, write_artifact_bundle(client, repo, slug, pr, captured_at))
            if args.max_new and total_new >= args.max_new:
                break

        state_rows.append({"slug": slug, "repo": repo, "seen_merged_in_window": seen, "skipped_existing": skipped_existing, "matched_new": matched, "written": written, "new_ids": new_ids})
        if args.max_new and total_new >= args.max_new:
            break

    if args.apply:
        dump_yaml(STATE_PATH, {"generated_by": "scripts/update_pr_corpus.py", "run_at": datetime.now(timezone.utc).isoformat(), "window": {"since": since.isoformat(), "until": until.isoformat()}, "mode": args.mode, "repos": state_rows})
        if args.rebuild_db:
            rebuild_sqlite_db()
        print(f"\nwrote {rel_to_root(STATE_PATH)}")
    else:
        print("\ndry-run complete; no files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
