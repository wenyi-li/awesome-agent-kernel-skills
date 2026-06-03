#!/usr/bin/env python3
"""Validate kernel_KBS source docs, wiki pages, ledgers, and artifacts.

This is an offline corpus validator. Daily retrieval and CRUD go through
scripts/kbs.py and SQLite; this script verifies that canonical Markdown and
artifact inputs under store/ are structurally safe to import.
"""

import argparse
import hashlib
import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import (  # noqa: E402
    ARTIFACTS_DIR,
    AUDIT_REFRESH_DIR,
    LEDGERS_DIR,
    RAW_SOURCES_DIR,
    SCHEMAS_DIR,
    STATE_REFRESH_DIR,
    STATE_VERSIONS_DIR,
    WIKI_DIR,
    WIKI_ROOT,
    rel_to_root,
    resolve_rel_path,
)

REPO_ROOT = WIKI_ROOT
SOURCES_DIR = RAW_SOURCES_DIR
CANDIDATES_DIR = LEDGERS_DIR / "candidates"


REPRO_ORDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]

# Phase 3 per-file 1 MiB cap; bundle 5 MiB cap (see plan AC-10)
FILE_SIZE_CAP_BYTES = 1 * 1024 * 1024
BUNDLE_SIZE_CAP_BYTES = 5 * 1024 * 1024

# Source-file extensions that must live in an asset bundle.
# `.txt` was added in R23 to cover extract_blog_code.py's unlabeled-fence
# extraction fallback (R20). `.sh`, `.yaml`, `.json` were added in R33 to
# keep this set in sync with extract_blog_code.py's EXT_MAP (the extractor
# emits shell / yaml / json fences into bundles, and orphan + manifest-
# drift detection must cover them too — otherwise a stale `deploy.sh`
# under store/corpus/artifacts/blogs/<slug>/code/ would pass validate.py silently).
# Keep this set identical to the code-ext subset of kbs.py get
# --include-code and kbs.py query --has-code; the three together are the
# Phase-3 asset-source contract.
ASSET_SOURCE_EXTS = {
    ".cu", ".cuh", ".ptx",
    ".cpp", ".h", ".hpp",
    ".py", ".pyx",
    ".patch",
    ".inl",
    ".txt",
    ".sh", ".yaml", ".json",
}


def load_yaml_file(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}


def read_body(filepath):
    """Read the body (post-frontmatter) of a markdown file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\r?\n.*?\r?\n---\s*\r?\n', content, re.DOTALL)
    if match:
        return content[match.end():]
    return content


def detect_page_type(filepath, fm):
    """Detect page type from filepath and frontmatter."""
    rel = Path(rel_to_root(filepath))
    parts = rel.parts

    if parts and parts[0] == "store":
        parts = parts[1:]
        if parts and parts[0] == "docs":
            parts = parts[1:]

    if not parts:
        return "unknown"

    if parts[0] == "sources":
        if parts[1] == "prs":
            return "source-pr"
        elif parts[1] == "docs":
            return "source-doc"
        elif parts[1] == "blogs":
            return "source-blog"
        elif parts[1] == "contests":
            return "source-contest"
    elif parts[0] == "wiki":
        t = fm.get("type", "")
        if t:
            return f"wiki-{t}"
        subdir = parts[1] if len(parts) > 1 else ""
        type_map = {
            "hardware": "wiki-hardware",
            "techniques": "wiki-technique",
            "patterns": "wiki-pattern",
            "kernels": "wiki-kernel",
            "languages": "wiki-language",
            "migration": "wiki-migration",
        }
        return type_map.get(subdir, "unknown")
    return "unknown"


def repro_at_least(level, minimum):
    if level not in REPRO_ORDER or minimum not in REPRO_ORDER:
        return False
    return REPRO_ORDER.index(level) >= REPRO_ORDER.index(minimum)


# Base code languages + all DSLs from store/schemas/tags.yaml languages category
_BASE_CODE_LANGS = {
    "cuda", "c", "c++", "cpp", "python", "py", "ptx", "asm",
    "cuda-cpp", "cu", "rust", "shell", "bash", "yaml", "json",
}


def _load_code_langs():
    """Load recognized code fence languages from the canonical tag schema."""
    langs = set(_BASE_CODE_LANGS)
    tags_path = SCHEMAS_DIR / "tags.yaml"
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        langs.update(data.get("languages", []))
    return langs


# Patterns that indicate real code (not pseudocode or ASCII diagrams)
_CODE_INDICATORS = re.compile(
    # CUDA C++
    r'__global__|__device__|__shared__|__host__|'
    r'asm\s+volatile|#include|#define|#pragma|'
    r'\bvoid\b|\bint\b|\buint32_t\b|\buint64_t\b|\bfloat\b|\bhalf\b|'
    r'\bstruct\b|\btypedef\b|\btemplate\b|\bnamespace\b|'
    r'\bfor\s*\(|\bwhile\s*\(|\bif\s*\(|return\s|'
    # Python / Triton
    r'\bdef\s+\w+|import\s+\w+|@triton\.jit|tl\.\w+|'
    # PTX
    r'tcgen05|mbarrier|cp\.async|ld\.global|st\.global|'
    r'\.reg\s|\.pred\s|cvt\.\w+|mov\.b32|'
    # TileLang (TVM-based DSL)
    r'@T\.prim_func|T\.alloc_buffer|T\.grid|T\.block_attr|T\.reads|T\.writes|'
    # cuTile (NVIDIA Python DSL)
    r'cutile\.\w+|@cutile\.kernel|tile_load|tile_store|tile_mma|'
    # JAX Pallas
    r'pl\.\w+|@pl\.kernel|pallas\.|jax\.\w+|jnp\.\w+'
)


def has_compilable_code(body, code_langs):
    """Check if body contains a fenced code block with a known language, real code,
    and at least 3 non-blank non-comment lines (rejects stubs and placeholders)."""
    for m in re.finditer(r'^```(\S*)\s*\n(.*?)\n```', body, re.MULTILINE | re.DOTALL):
        info = m.group(1).lower()
        block = m.group(2)
        if info not in code_langs:
            continue
        if not _CODE_INDICATORS.search(block):
            continue
        # Count substantive code lines (not blank, not comment-only)
        code_lines = 0
        for line in block.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                code_lines += 1
        if code_lines >= 3:
            return True
    return False


def validate_file(filepath, schemas, valid_tags, all_source_ids, code_langs):
    """Validate a single file. Returns list of error strings."""
    errors = []
    rel = rel_to_root(filepath)

    fm = extract_frontmatter(filepath)
    if fm is None:
        errors.append(f"{rel}: missing YAML frontmatter")
        return errors
    if not isinstance(fm, dict):
        errors.append(f"{rel}: frontmatter must be a YAML mapping, got {type(fm).__name__}")
        return errors
    if "_parse_error" in fm:
        errors.append(f"{rel}: YAML parse error: {fm['_parse_error']}")
        return errors

    page_type = detect_page_type(filepath, fm)
    if page_type == "unknown":
        errors.append(f"{rel}: unknown page type")
        return errors

    schema = schemas.get(page_type)
    if not schema:
        errors.append(f"{rel}: no schema defined for type '{page_type}'")
        return errors

    constraints = schema.get("constraints", {})

    # Check required fields
    for field in schema.get("required", []):
        if field not in fm or fm[field] is None:
            errors.append(f"{rel}: missing required field '{field}'")

    # Validate id_prefix
    id_prefix = constraints.get("id_prefix")
    if id_prefix and "id" in fm:
        if not str(fm["id"]).startswith(id_prefix):
            errors.append(f"{rel}: id '{fm['id']}' must start with '{id_prefix}'")

    # Build per-field vocabulary sets
    # "tags" accepts only topical categories (not architectures/confidence/etc.)
    topical_categories = ["hardware_features", "techniques", "kernel_types", "languages"]
    tags_valid = set()
    for cat in topical_categories:
        tags_valid.update(valid_tags.get(cat, []))

    field_vocab = {
        "tags": tags_valid,
        "techniques": set(valid_tags.get("techniques", [])),
        "hardware_features": set(valid_tags.get("hardware_features", [])),
        "kernel_types": set(valid_tags.get("kernel_types", [])),
        "languages": set(valid_tags.get("languages", [])),
    }

    # Check list type and uniqueness for all list-valued fields
    list_fields = ["tags", "techniques", "hardware_features", "kernel_types", "languages",
                    "architectures", "related", "sources", "symptoms", "candidate_techniques",
                    "prerequisites", "aliases"]
    for tag_field in list_fields:
        if tag_field in fm:
            if not isinstance(fm[tag_field], list):
                errors.append(f"{rel}: field '{tag_field}' must be a YAML list, got {type(fm[tag_field]).__name__}")
                continue
            # Reject duplicates
            seen = set()
            for val in fm[tag_field]:
                if val in seen:
                    errors.append(f"{rel}: duplicate value '{val}' in field '{tag_field}'")
                seen.add(val)

    # Check hardware tags are reflected in hardware_features
    if "hardware_features" in fm and isinstance(fm["hardware_features"], list):
        hw_in_tags = set(fm.get("tags", [])) & set(field_vocab.get("hardware_features", set()))
        hw_explicit = set(fm["hardware_features"])
        missing_hw = hw_in_tags - hw_explicit
        if missing_hw:
            errors.append(
                f"{rel}: tags contain hardware features {sorted(missing_hw)} "
                f"not in hardware_features field"
            )

    # Validate each structured field against its own vocabulary
    for tag_field, vocab in field_vocab.items():
        if tag_field in fm and isinstance(fm[tag_field], list):
            for tag in fm[tag_field]:
                if tag not in vocab:
                    errors.append(f"{rel}: '{tag}' is not a valid {tag_field} value")

    # Validate candidate_techniques entries are known page ID prefixes
    valid_remedy_prefixes = ("technique-", "hw-", "migration-")
    if "candidate_techniques" in fm and isinstance(fm["candidate_techniques"], list):
        for ct in fm["candidate_techniques"]:
            if not str(ct).startswith(valid_remedy_prefixes):
                errors.append(
                    f"{rel}: candidate_techniques entry '{ct}' must use one of "
                    f"{valid_remedy_prefixes} prefixes"
                )

    # Validate architectures
    valid_archs = set(valid_tags.get("architectures", []))
    if "architectures" in fm and isinstance(fm["architectures"], list):
        for arch in fm["architectures"]:
            if arch not in valid_archs:
                errors.append(f"{rel}: unknown architecture '{arch}'")

    # Validate from_arch / to_arch on migration pages
    for arch_field in ["from_arch", "to_arch"]:
        if arch_field in fm:
            if fm[arch_field] not in valid_archs:
                errors.append(f"{rel}: {arch_field} '{fm[arch_field]}' is not a known architecture")

    # Validate confidence
    valid_conf = set(valid_tags.get("confidence", []))
    if "confidence" in fm and fm["confidence"] not in valid_conf:
        errors.append(f"{rel}: invalid confidence '{fm['confidence']}'")

    # Validate reproducibility
    valid_repro = set(valid_tags.get("reproducibility", []))
    if "reproducibility" in fm:
        if fm["reproducibility"] not in valid_repro:
            errors.append(f"{rel}: invalid reproducibility '{fm['reproducibility']}'")

    # Check reproducibility minimum
    repro_min = constraints.get("reproducibility_minimum")
    if repro_min and "reproducibility" in fm:
        if not repro_at_least(fm["reproducibility"], repro_min):
            errors.append(
                f"{rel}: reproducibility '{fm['reproducibility']}' below "
                f"minimum '{repro_min}' for {page_type}"
            )

    # Validate source_category against schema constraints
    valid_cats = set(valid_tags.get("source_categories", []))
    if "source_category" in fm:
        cat = fm["source_category"]
        if cat not in valid_cats:
            errors.append(f"{rel}: invalid source_category '{cat}'")
        # Check schema-specific category constraints
        cat_constraint = constraints.get("source_category")
        if cat_constraint:
            allowed = cat_constraint if isinstance(cat_constraint, list) else [cat_constraint]
            if cat not in allowed:
                errors.append(f"{rel}: source_category '{cat}' not in allowed {allowed}")

    # Validate status enum
    status_constraint = constraints.get("status")
    if status_constraint and "status" in fm:
        allowed = status_constraint if isinstance(status_constraint, list) else [status_constraint]
        if fm["status"] not in allowed:
            errors.append(f"{rel}: status '{fm['status']}' not in {allowed}")

    # Check merge_sha_required_when
    if constraints.get("merge_sha_required_when") == "status == merged":
        if fm.get("status") == "merged" and not fm.get("merge_sha"):
            errors.append(f"{rel}: merge_sha required when status is 'merged'")

    # Check type field matches constraint
    if "type" in constraints and "type" in fm:
        if fm["type"] != constraints["type"]:
            errors.append(
                f"{rel}: type '{fm['type']}' does not match "
                f"expected '{constraints['type']}' for {page_type}"
            )

    # Check blackwell_relevance required for Hopper-only wiki pages
    # Pages targeting both Hopper AND Blackwell are inherently Blackwell-relevant
    if page_type.startswith("wiki-"):
        archs = set(fm.get("architectures", []) if isinstance(fm.get("architectures"), list) else [])
        hopper_archs = archs & {"sm90", "sm90a"}
        blackwell_archs = archs & {"sm100", "sm100a", "sm120"}
        if hopper_archs and not blackwell_archs and "blackwell_relevance" not in fm:
            errors.append(
                f"{rel}: page targets only Hopper {hopper_archs} without Blackwell arch; "
                f"add 'blackwell_relevance' to justify inclusion in Blackwell-first scope"
            )

    # Check performance_claims structure (including shape and numeric value)
    if "performance_claims" in fm:
        pc = fm["performance_claims"]
        if not isinstance(pc, list):
            errors.append(f"{rel}: performance_claims must be a list, got {type(pc).__name__}")
        else:
            for i, claim in enumerate(pc):
                if not isinstance(claim, dict):
                    errors.append(f"{rel}: performance_claims[{i}] must be a mapping, got {type(claim).__name__}")
                    continue
                for req in ["gpu", "dtype", "shape", "metric", "value", "source_id"]:
                    if req not in claim:
                        errors.append(f"{rel}: performance_claims[{i}] missing '{req}'")
                if "value" in claim and not isinstance(claim["value"], (int, float)):
                    errors.append(
                        f"{rel}: performance_claims[{i}].value must be numeric, "
                        f"got {type(claim['value']).__name__}: {claim['value']}"
                    )
                # Cross-check source_id against known source IDs
                sid = claim.get("source_id", "")
                if sid and all_source_ids and sid not in all_source_ids:
                    errors.append(
                        f"{rel}: performance_claims[{i}].source_id '{sid}' "
                        f"not found in source corpus"
                    )

    # Check wiki sources reference existing source ids
    if page_type.startswith("wiki-") and "sources" in fm and isinstance(fm["sources"], list):
        for src_id in fm["sources"]:
            if all_source_ids and src_id not in all_source_ids:
                errors.append(f"{rel}: references unknown source id '{src_id}'")

    # AC-9: Enforce evidence_basis for verified wiki pages
    if page_type.startswith("wiki-") and fm.get("confidence") == "verified":
        eb = fm.get("evidence_basis")
        if not eb or not isinstance(eb, list) or len(eb) == 0:
            errors.append(
                f"{rel}: confidence 'verified' requires non-empty 'evidence_basis' field"
            )
        else:
            eb_types = {entry.get("evidence_type") for entry in eb if isinstance(entry, dict)}
            if "official-doc" not in eb_types:
                errors.append(
                    f"{rel}: evidence_basis for 'verified' must include at least one "
                    f"'official-doc' entry (found: {eb_types})"
                )
            if "upstream-code" not in eb_types:
                errors.append(
                    f"{rel}: evidence_basis for 'verified' must include at least one "
                    f"'upstream-code' entry (found: {eb_types})"
                )
            # Cross-check evidence_basis source_ids against page sources
            page_sources = set(fm.get("sources", []))
            for entry in eb:
                if isinstance(entry, dict):
                    sid = entry.get("source_id", "")
                    if sid and sid not in page_sources:
                        errors.append(
                            f"{rel}: evidence_basis references '{sid}' "
                            f"not listed in page sources"
                        )

    # Check technique/kernel/language pages have fenced code
    if page_type in ("wiki-technique", "wiki-kernel", "wiki-language"):
        body = read_body(filepath)
        if not has_compilable_code(body, code_langs):
            errors.append(f"{rel}: {page_type} page must contain fenced code block (reproducibility >= snippet)")

    # Enforce single artifact pointer at page frontmatter level.
    disallow = schemas.get(page_type, {}).get("disallow_peer_of_artifact_dir") or []
    for banned in disallow:
        if banned in fm:
            errors.append(
                f"{rel}: page frontmatter must not carry '{banned}:' at top level "
                f"(single-field contract: use 'artifact_dir:' only; per-file pointers "
                f"live inside PROVENANCE.yaml)"
            )

    # If artifact_dir is set, it must resolve to a real directory.
    if "artifact_dir" in fm:
        ad = fm["artifact_dir"]
        if not isinstance(ad, str):
            errors.append(f"{rel}: artifact_dir must be a string path")
        else:
            target = resolve_rel_path(ad)
            if not target.is_dir():
                errors.append(f"{rel}: artifact_dir '{ad}' does not resolve to an existing directory")
            else:
                # Resolve symlinks / `..` traversal before checking
                # containment under store/corpus/artifacts. kbs.py get
                # --include-code and kbs.py query --has-code both follow the
                # resolved path, so the validator compares resolved paths too.
                try:
                    resolved = target.resolve()
                    artifacts_root = ARTIFACTS_DIR.resolve()
                    if resolved != artifacts_root and artifacts_root not in resolved.parents:
                        errors.append(
                            f"{rel}: artifact_dir '{ad}' must resolve under "
                            f"{rel_to_root(ARTIFACTS_DIR)} "
                            f"(code assets are quarantined from docs/wiki); "
                            f"resolves to '{resolved}' which is outside "
                            f"'{artifacts_root}'"
                        )
                except (OSError, RuntimeError) as e:
                    errors.append(f"{rel}: artifact_dir '{ad}' could not be resolved: {e}")

    # Validate nested submissions[*] on source-contest pages.
    if page_type == "source-contest" and "submissions" in fm:
        errors.extend(validate_contest_submissions(fm, filepath, schemas))

    return errors


def validate_contest_submissions(fm, filepath, schemas):
    """Enforce truth-model enum, conditional code_path / reason, and
    contest-bundle containment (code_path must resolve inside the page's own
    implicit submission bundle, not just anywhere under store/corpus/artifacts/contests/)."""
    rel = rel_to_root(filepath)
    errors = []
    subs = fm.get("submissions") or []
    sub_schema = schemas.get("source-contest", {}).get("submissions_schema", {})
    required = sub_schema.get("required", [])
    optional = sub_schema.get("optional", [])
    allowed_truths = (sub_schema.get("constraints") or {}).get("submission_truth") or []

    # Implicit submission bundle root:
    # store/corpus/artifacts/contests/<contest>/<problem>/submissions/
    # where <contest> is the page's parent directory name and <problem> is the
    # page's filename stem.
    contest_slug = filepath.parent.name
    problem_slug = filepath.stem
    expected_prefix = f"store/corpus/artifacts/contests/{contest_slug}/{problem_slug}/submissions/"
    # R34: resolve-vs-resolve containment check. A raw startswith
    # would accept a `code_path` like `store/corpus/artifacts/contests/<c>/<p>/
    # submissions/../rank-2-other/file.cpp` which escapes the
    # submission bundle for this row.
    try:
        expected_root = (ARTIFACTS_DIR / "contests" / contest_slug / problem_slug / "submissions").resolve()
    except (OSError, RuntimeError):
        expected_root = ARTIFACTS_DIR / "contests" / contest_slug / problem_slug / "submissions"

    for i, entry in enumerate(subs):
        if not isinstance(entry, dict):
            errors.append(f"{rel}: submissions[{i}] must be a mapping, got {type(entry).__name__}")
            continue
        for req in required:
            if req not in entry:
                errors.append(f"{rel}: submissions[{i}] missing required '{req}'")
        # Enum
        truth = entry.get("submission_truth")
        if truth is not None and allowed_truths and truth not in allowed_truths:
            errors.append(f"{rel}: submissions[{i}].submission_truth '{truth}' not in {allowed_truths}")
        # Conditional requirements
        if truth == "unavailable":
            if not entry.get("code_unavailable_reason"):
                errors.append(
                    f"{rel}: submissions[{i}] has submission_truth='unavailable' "
                    f"but no 'code_unavailable_reason'"
                )
        elif truth is not None:
            cp = entry.get("code_path")
            if not cp:
                errors.append(
                    f"{rel}: submissions[{i}] has submission_truth='{truth}' but no 'code_path'"
                )
            else:
                target = resolve_rel_path(cp)
                if not target.exists():
                    errors.append(
                        f"{rel}: submissions[{i}].code_path '{cp}' does not exist"
                    )
                else:
                    # R34: compare RESOLVED paths so `..` traversal is
                    # caught. A raw startswith string check accepts
                    # `.../submissions/../rank-2-other/...` which
                    # points outside this row's submission bundle.
                    try:
                        resolved_cp = target.resolve()
                        if expected_root not in resolved_cp.parents and resolved_cp != expected_root:
                            errors.append(
                                f"{rel}: submissions[{i}].code_path '{cp}' must live under "
                                f"'{expected_prefix}' (the page's own implicit submission bundle), "
                                f"not an arbitrary location inside store/corpus/artifacts/contests/; "
                                f"resolves to '{resolved_cp}' which is outside "
                                f"'{expected_root}'"
                            )
                    except (OSError, RuntimeError) as e:
                        errors.append(
                            f"{rel}: submissions[{i}].code_path '{cp}' could not be resolved: {e}"
                        )
        # Reject unknown fields strictly
        allowed_fields = set(required) | set(optional)
        for k in entry.keys():
            if k not in allowed_fields:
                errors.append(f"{rel}: submissions[{i}] has unknown field '{k}'")
    return errors


# ---------------------------------------------------------------------------
# Artifact bundle validation
# ---------------------------------------------------------------------------

## Ledger-shape check. Every candidates/*.yaml must carry the canonical
## top-level fields and its summary counts must match its row decisions.
LEDGER_REQUIRED_TOP_FIELDS = [
    "repo",
    "searched_at",
    "keywords_used",
    "total_candidates",
    "included",
    "excluded",
    "deferred",
    "prs",
]
LEDGER_REQUIRED_PR_FIELDS = ["number", "title", "date", "decision", "reason"]


def validate_ledger(ledger_path):
    """Validate a candidate ledger file's top-level shape and summary
    consistency. Returns a list of error strings (empty if valid)."""
    errors = []
    rel = rel_to_root(ledger_path)
    try:
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"{rel}: invalid YAML ({e})"]
    if not isinstance(data, dict):
        return [f"{rel}: top-level value must be a mapping"]
    for field in LEDGER_REQUIRED_TOP_FIELDS:
        if field not in data:
            errors.append(f"{rel}: missing required top-level field '{field}'")
    if errors:
        # Don't continue with summary-count check if shape is broken.
        return errors
    prs = data["prs"]
    if not isinstance(prs, list):
        return [f"{rel}: 'prs' must be a list, got {type(prs).__name__}"]
    inc = exc = dfr = 0
    for i, row in enumerate(prs):
        if not isinstance(row, dict):
            errors.append(f"{rel}: prs[{i}] must be a mapping")
            continue
        for f in LEDGER_REQUIRED_PR_FIELDS:
            if f not in row:
                errors.append(f"{rel}: prs[{i}] missing required field '{f}'")
        d = str(row.get("decision", "")).lower()
        if d == "include":
            inc += 1
        elif d == "exclude":
            exc += 1
        elif d == "defer":
            dfr += 1
        else:
            errors.append(f"{rel}: prs[{i}] has unknown decision '{row.get('decision')}'")
    if data["total_candidates"] != len(prs):
        errors.append(
            f"{rel}: total_candidates={data['total_candidates']} disagrees with len(prs)={len(prs)}"
        )
    if data["included"] != inc:
        errors.append(f"{rel}: included={data['included']} disagrees with row count {inc}")
    if data["excluded"] != exc:
        errors.append(f"{rel}: excluded={data['excluded']} disagrees with row count {exc}")
    if data["deferred"] != dfr:
        errors.append(f"{rel}: deferred={data['deferred']} disagrees with row count {dfr}")
    if (
        data["total_candidates"] != data["included"] + data["excluded"] + data["deferred"]
    ):
        errors.append(
            f"{rel}: total_candidates ({data['total_candidates']}) != "
            f"included + excluded + deferred ("
            f"{data['included']} + {data['excluded']} + {data['deferred']})"
        )
    return errors


## Version-claim registry consistency. Pages in scope that carry a per-page
## `version_sensitive: <id>` pointer must resolve to a claim in
## store/state/versions/version-claims.yaml; reverse direction is also enforced.
def validate_version_claims_registry(all_source_ids):
    """Return list of error strings for version-registry consistency."""
    errors = []
    claims_path = STATE_VERSIONS_DIR / "version-claims.yaml"
    if not claims_path.is_file():
        return [f"{rel_to_root(claims_path)}: missing version-claims registry"]
    try:
        data = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"{rel_to_root(claims_path)}: invalid YAML ({e})"]
    claims = (data or {}).get("claims") or []

    # Build registry-id -> applies_to mapping for reverse-direction check.
    claim_by_id = {}
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"{rel_to_root(claims_path)}::claims[{i}]: must be a mapping")
            continue
        cid = claim.get("id")
        if not cid:
            errors.append(f"{rel_to_root(claims_path)}::claims[{i}]: missing id")
            continue
        claim_by_id[cid] = claim
        # source_ids resolution
        for sid in claim.get("source_ids", []) or []:
            if sid not in all_source_ids:
                errors.append(f"{rel_to_root(claims_path)}::{cid}: source_id '{sid}' does not resolve")
        # applies_to: every target must EXIST and CARRY the matching pointer.
        # Reverse direction enforcement (the AC-2 strict check Codex Round 3
        # required): existence alone is insufficient.
        for applies in claim.get("applies_to", []) or []:
            file_part, _, scalar_pointer = applies.partition("::")
            target_path = resolve_rel_path(file_part)
            if not target_path.exists():
                errors.append(f"{rel_to_root(claims_path)}::{cid}: applies_to '{applies}' (file '{file_part}') does not exist")
                continue
            if scalar_pointer:
                # YAML JSON-pointer form, e.g.
                # store/docs/ledgers/inclusion-policy.yaml::triton.description.
                # The reverse-direction proof for these is the existence of
                # an authoring rule recorded elsewhere; we don't try to rewrite
                # YAML scalars to embed a pointer. The `applies_to` path itself
                # is the authoring-time anchor.
                continue
            # Markdown file: must carry version_sensitive: <id> in frontmatter.
            try:
                fm = extract_frontmatter(target_path)
            except Exception as e:
                errors.append(f"{rel_to_root(claims_path)}::{cid}: applies_to '{applies}' frontmatter parse failed: {e}")
                continue
            if not fm or not isinstance(fm, dict):
                errors.append(f"{rel_to_root(claims_path)}::{cid}: applies_to '{applies}' has no frontmatter (target must carry version_sensitive: {cid})")
                continue
            vs = fm.get("version_sensitive")
            ptr_id = vs.get("id") if isinstance(vs, dict) else vs
            if ptr_id != cid:
                errors.append(f"{rel_to_root(claims_path)}::{cid}: applies_to '{applies}' carries version_sensitive id={ptr_id!r}, expected {cid!r}")

    # Forward direction: every page in scope with a per-page pointer must
    # resolve to a registry entry. Pages without a pointer are not flagged
    # here — flag-on-missing-pointer is the job of the AC-2 surface check
    # below (parsed YAML scalar detection).
    in_scope = []
    if WIKI_DIR.exists():
        in_scope.extend(sorted(WIKI_DIR.rglob("*.md")))
    for ref in (REPO_ROOT / "references" / "primer.md", REPO_ROOT / "references" / "examples.md"):
        if ref.is_file():
            in_scope.append(ref)
    for md_file in in_scope:
        fm = extract_frontmatter(md_file)
        if not fm or not isinstance(fm, dict):
            continue
        vs = fm.get("version_sensitive")
        if vs is None:
            continue
        # vs may be a dict {id: ...} or a string id
        ptr = vs.get("id") if isinstance(vs, dict) else vs
        if not ptr:
            errors.append(f"{rel_to_root(md_file)}: version_sensitive block has no id")
            continue
        if ptr not in claim_by_id:
            errors.append(f"{rel_to_root(md_file)}: version_sensitive id '{ptr}' does not resolve to {rel_to_root(claims_path)}")

    return errors


## Inclusion-policy YAML scalar guard. The Triton lane's `description`
## scalar must NOT contain the obsolete "no direct tcgen05/TMEM access"
## phrase. Validated by parsing the YAML data, never by reading comments.
def validate_inclusion_policy_scalars():
    errors = []
    ip_path = LEDGERS_DIR / "inclusion-policy.yaml"
    if not ip_path.is_file():
        return errors
    try:
        data = yaml.safe_load(ip_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return [f"{rel_to_root(ip_path)}: invalid YAML ({e})"]
    triton_desc = (data or {}).get("triton", {}).get("description", "") or ""
    if "no direct tcgen05/tmem access" in triton_desc.lower():
        errors.append(
            f"{rel_to_root(ip_path)}::triton.description: still contains "
            "obsolete substring 'no direct tcgen05/TMEM access'"
        )
    return errors


## Skip-audit coverage check. Every ledger row with `decision: include`
## must appear EITHER as a generated
## `store/docs/sources/prs/<repo-slug>/PR-<N>.md` page OR as a
## `store/state/audits/refresh/pr-page-skipped.yaml` row.
def validate_skip_audit_coverage():
    errors = []
    audit_path = AUDIT_REFRESH_DIR / "pr-page-skipped.yaml"
    audit_rows = []
    if audit_path.is_file():
        try:
            data = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
            audit_rows = (data or {}).get("rows") or []
        except yaml.YAMLError as e:
            return [f"{rel_to_root(audit_path)}: invalid YAML ({e})"]
    audit_keys = {(row["repo"], row["pr_number"]) for row in audit_rows
                  if isinstance(row, dict) and "repo" in row and "pr_number" in row}

    if not CANDIDATES_DIR.exists():
        return errors
    for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
        ledger = yaml.safe_load(ledger_file.read_text(encoding="utf-8")) or {}
        repo_full = ledger.get("repo")
        if not repo_full:
            continue
        repo_slug = repo_full.split("/")[1] if "/" in repo_full else ledger_file.stem
        outdir = SOURCES_DIR / "prs" / repo_slug
        existing_pages = set()
        if outdir.is_dir():
            for p in outdir.glob("PR-*.md"):
                try:
                    existing_pages.add(int(p.stem.split("-")[1]))
                except (ValueError, IndexError):
                    pass
        for row in ledger.get("prs", []) or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("decision", "")).lower() != "include":
                continue
            num = row.get("number")
            if num in existing_pages:
                continue
            if (repo_full, num) in audit_keys:
                continue
            errors.append(
                f"refresh coverage: {repo_full} PR #{num} is `decision: include` "
                f"but has neither store/docs/sources/prs/{repo_slug}/PR-{num}.md nor a "
                f"{rel_to_root(audit_path)} row"
            )
    return errors


## Optional refresh-governance checks. These enforce invariants for a manual
## refresh round. They are not part of the default corpus import validation
## because the SQLite-first KBS can be queried and maintained independently of
## a candidate-refresh window.
def validate_refresh_cutoff_alignment():
    errors = []
    cutoff_path = STATE_REFRESH_DIR / "refresh-cutoff.yaml"
    if not cutoff_path.is_file():
        return errors  # Advisory: no cutoff means no enforcement
    try:
        cutoff_data = yaml.safe_load(cutoff_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [f"{rel_to_root(cutoff_path)}: invalid YAML ({e})"]
    cutoff = cutoff_data.get("cutoff_date")
    if cutoff is None:
        return [f"{rel_to_root(cutoff_path)}: missing cutoff_date"]
    cutoff_str = cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff)
    if not CANDIDATES_DIR.exists():
        return errors
    for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
        ledger = yaml.safe_load(ledger_file.read_text(encoding="utf-8")) or {}
        sa = ledger.get("searched_at")
        sa_str = sa.isoformat() if hasattr(sa, "isoformat") else str(sa)
        if sa_str != cutoff_str:
            errors.append(
                f"refresh cutoff: {rel_to_root(ledger_file)}::searched_at "
                f"({sa_str!r}) != {rel_to_root(cutoff_path)}::cutoff_date ({cutoff_str!r})"
            )
    return errors


## Optional subset check. Every PR number in
## store/state/refresh/refresh-search-results.yaml::repos[].pr_numbers_seen must appear
## in the corresponding candidates/<repo>.yaml::prs[*].number set.
def validate_refresh_subset():
    errors = []
    results_path = STATE_REFRESH_DIR / "refresh-search-results.yaml"
    if not results_path.is_file():
        return errors  # Advisory when artifact absent
    try:
        results = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [f"{rel_to_root(results_path)}: invalid YAML ({e})"]
    for repo_block in results.get("repos", []) or []:
        slug = repo_block.get("repo_slug")
        if not slug:
            continue
        ledger_path = CANDIDATES_DIR / f"{slug}.yaml"
        if not ledger_path.is_file():
            continue
        ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
        ledger_nums = {row.get("number") for row in (ledger.get("prs") or [])
                       if isinstance(row, dict)}
        seen = set(repo_block.get("pr_numbers_seen", []) or [])
        missing = seen - ledger_nums
        if missing:
            errors.append(
                f"refresh subset: {len(missing)} pr_numbers_seen for "
                f"{slug} are NOT in {rel_to_root(ledger_path)}::prs[*].number "
                f"(first 5: {sorted(missing)[:5]})"
            )
    return errors


## AC-2 missing-pointer for claim-bearing pages. A page is "claim-bearing"
## when its body contains any of the obsolete claim signatures listed
## here (case-insensitive). All listed in-scope pages with such a hit
## must carry a `version_sensitive` frontmatter pointer.
##
## This is the missing-pointer failure mode Codex Round 4 flagged.
CLAIM_SIGNATURE_PATTERNS = [
    # The exact obsolete Triton-3.5 framings — narrow enough to avoid
    # matching legitimate hardware claims like "Hopper (SM90) has no TMEM".
    r"\bno direct tcgen05 access\b",
    r"\bno TMEM:\s*accumulators stay in registers\b",
    r"\btriton compiler generates wgmma\b",
]


def validate_claim_bearing_pages_have_pointer():
    """If an in-scope page contains any obsolete claim signature, it MUST
    carry a version_sensitive frontmatter pointer. Pages with the
    signatures inside an explicitly-marked historical-context block
    are exempt (the wiki/languages/triton-blackwell.md historical
    subsection)."""
    errors = []
    in_scope = []
    if WIKI_DIR.exists():
        in_scope.extend(sorted(WIKI_DIR.rglob("*.md")))
    for ref in (REPO_ROOT / "references" / "primer.md", REPO_ROOT / "references" / "examples.md"):
        if ref.is_file():
            in_scope.append(ref)
    sig_re = re.compile("|".join(CLAIM_SIGNATURE_PATTERNS), re.IGNORECASE)
    for md_file in in_scope:
        text = md_file.read_text(encoding="utf-8")
        if not sig_re.search(text):
            continue
        # Strip "Pre-3.6 historical context" (and similar) sections from text
        # before re-checking. The historical subsection is allowed to contain
        # the signatures.
        stripped = re.sub(
            r"##\s*Pre-3\.6 historical context.*?(?=\n##\s|\Z)",
            "",
            text,
            flags=re.S | re.I,
        )
        if not sig_re.search(stripped):
            continue
        fm = extract_frontmatter(md_file)
        if not fm or not isinstance(fm, dict):
            errors.append(
                f"{rel_to_root(md_file)}: contains claim signature "
                f"outside historical-context block but has no frontmatter "
                f"(AC-2 missing-pointer)"
            )
            continue
        if fm.get("version_sensitive") is None:
            errors.append(
                f"{rel_to_root(md_file)}: contains claim signature "
                f"outside historical-context block but lacks version_sensitive "
                f"frontmatter pointer (AC-2 missing-pointer)"
            )
    return errors


## DEC-4 CUTLASS dev-pinning rule. Any wiki page whose body (excluding
## fenced code blocks) mentions the literal string "4.5-dev" must:
##   (1) carry confidence: source-reported or confidence: experimental
##       (NOT confidence: verified — verified pages cite stable releases
##       only per DEC-4 mixed policy);
##   (2) carry a version_sensitive frontmatter pointer whose registry
##       entry pins a specific dev_branch_sha.
def validate_cutlass_dev_pinning():
    errors = []
    if not WIKI_DIR.exists():
        return errors
    # Load registry once for dev_branch_sha lookups.
    claims_path = STATE_VERSIONS_DIR / "version-claims.yaml"
    claims_by_id = {}
    if claims_path.is_file():
        try:
            cdata = yaml.safe_load(claims_path.read_text(encoding="utf-8")) or {}
            for c in cdata.get("claims", []) or []:
                if isinstance(c, dict) and "id" in c:
                    claims_by_id[c["id"]] = c
        except yaml.YAMLError:
            pass
    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Strip fenced code blocks; the rule applies to prose only.
        body = re.sub(r"```.*?```", "", text, flags=re.S)
        if "4.5-dev" not in body:
            continue
        fm = extract_frontmatter(md_file)
        rel = rel_to_root(md_file)
        if not fm or not isinstance(fm, dict):
            errors.append(
                f"{rel}: mentions '4.5-dev' outside code fences but has no frontmatter "
                f"(DEC-4: dev-branch references require version_sensitive pointer)"
            )
            continue
        confidence = fm.get("confidence")
        if confidence == "verified":
            errors.append(
                f"{rel}: mentions '4.5-dev' outside code fences with confidence: verified "
                f"(DEC-4: verified pages cite stable releases only)"
            )
            continue
        if confidence not in ("source-reported", "experimental"):
            errors.append(
                f"{rel}: mentions '4.5-dev' but confidence={confidence!r} "
                f"(DEC-4: must be source-reported or experimental)"
            )
            continue
        vs = fm.get("version_sensitive")
        ptr_id = vs.get("id") if isinstance(vs, dict) else vs
        if not ptr_id or ptr_id not in claims_by_id:
            errors.append(
                f"{rel}: mentions '4.5-dev' but version_sensitive pointer is "
                f"absent or unresolved (DEC-4: requires registry entry with dev_branch_sha)"
            )
            continue
        dev_sha = claims_by_id[ptr_id].get("dev_branch_sha")
        if not dev_sha or str(dev_sha).lower() in ("none", "null", "needs-verification", ""):
            errors.append(
                f"{rel}: mentions '4.5-dev' and version_sensitive resolves to "
                f"{ptr_id!r}, but the registry entry has no concrete dev_branch_sha "
                f"(DEC-4)"
            )
    return errors


## Layout discoverability checks.
def validate_discoverability():
    errors = []
    upstreams_dir = SOURCES_DIR / "upstreams"
    if upstreams_dir.exists():
        for md_file in upstreams_dir.rglob("*.md"):
            errors.append(
                f"{rel_to_root(md_file)}: store/docs/sources/upstreams/ paths "
                f"are forbidden; use source-blog or source-doc pages"
            )
    return errors


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_bundle_roots():
    """Yield bundle-root Paths under store/corpus/artifacts.

    Standard layout:
      store/corpus/artifacts/prs/<repo>/PR-<N>/
      store/corpus/artifacts/contests/<contest>/<problem>/submissions/<rank-N-author>/
      store/corpus/artifacts/blogs/<slug>/code/
      store/corpus/artifacts/kernels/<slug>/full/
      store/corpus/artifacts/kernels/<slug>/variants/
    """
    if not ARTIFACTS_DIR.is_dir():
        return
    # PR bundles
    prs = ARTIFACTS_DIR / "prs"
    if prs.is_dir():
        for repo in sorted(prs.iterdir()):
            if repo.is_dir():
                for pr_dir in sorted(repo.iterdir()):
                    if pr_dir.is_dir() and pr_dir.name.startswith("PR-"):
                        yield pr_dir
    # Contest submissions
    contests = ARTIFACTS_DIR / "contests"
    if contests.is_dir():
        for contest in sorted(contests.iterdir()):
            if contest.is_dir():
                for problem in sorted(contest.iterdir()):
                    if problem.is_dir():
                        subs = problem / "submissions"
                        if subs.is_dir():
                            for sub in sorted(subs.iterdir()):
                                if sub.is_dir():
                                    yield sub
    # Blog code
    blogs = ARTIFACTS_DIR / "blogs"
    if blogs.is_dir():
        for blog in sorted(blogs.iterdir()):
            if blog.is_dir():
                code = blog / "code"
                if code.is_dir():
                    yield code
    # Kernel deep pages
    kernels = ARTIFACTS_DIR / "kernels"
    if kernels.is_dir():
        for slug in sorted(kernels.iterdir()):
            if slug.is_dir():
                for sub in ("full", "variants"):
                    d = slug / sub
                    if d.is_dir():
                        yield d


def find_orphan_source_files():
    """Return source files under store/corpus/artifacts outside known bundles."""
    bundle_roots = set(discover_bundle_roots())
    orphans = []
    if not ARTIFACTS_DIR.is_dir():
        return orphans
    # R33: with `.yaml` now in ASSET_SOURCE_EXTS, per-blog MANIFEST.yaml
    # files at `store/corpus/artifacts/blogs/<slug>/MANIFEST.yaml` would otherwise be
    # flagged as orphans (the recognized bundle root is the `code/`
    # subdir). MANIFEST.yaml is metadata the extractor writes at the
    # parent by design; validate_bundle's drift check already excludes
    # it by name, so mirror that exclusion here. `approach.md` and
    # `bench.txt` are similar bundle-adjacent metadata from earlier
    # rounds that may live above the recognized root.
    _ORPHAN_EXCLUDE_NAMES = {"MANIFEST.yaml", "approach.md", "bench.txt"}
    for path in ARTIFACTS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ASSET_SOURCE_EXTS:
            continue
        if path.name in _ORPHAN_EXCLUDE_NAMES:
            continue
        # find nearest bundle root
        in_bundle = any(str(path).startswith(str(root) + "/") or str(path) == str(root) for root in bundle_roots)
        if not in_bundle:
            orphans.append(path)
    return orphans


def validate_bundle(bundle_root, known_source_ids):
    """Validate a single asset bundle root per plan AC-2/AC-9/AC-10."""
    rel = rel_to_root(bundle_root)
    errors = []
    prov_path = bundle_root / "PROVENANCE.yaml"
    if not prov_path.is_file():
        errors.append(f"{rel}: asset bundle missing PROVENANCE.yaml")
        return errors
    # Disallow nested PROVENANCE.yaml anywhere beneath
    for extra in bundle_root.rglob("PROVENANCE.yaml"):
        if extra != prov_path:
            errors.append(f"{rel_to_root(extra)}: nested PROVENANCE.yaml disallowed (flat ownership)")
    # Load + schema check
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        errors.append(f"{rel}/PROVENANCE.yaml: YAML parse error: {e}")
        return errors
    if not isinstance(prov, dict):
        errors.append(f"{rel}/PROVENANCE.yaml: top-level must be a mapping")
        return errors
    for req in ("origin_url", "upstream_repo", "license", "retrieved_at", "asset_mode", "files"):
        if req not in prov:
            errors.append(f"{rel}/PROVENANCE.yaml: missing required '{req}'")
    mode = prov.get("asset_mode")
    if mode not in ("verbatim", "extracted", "derived"):
        errors.append(f"{rel}/PROVENANCE.yaml: asset_mode '{mode}' must be one of verbatim/extracted/derived")
    if mode == "derived":
        dfrom = prov.get("derived_from")
        if not dfrom or not isinstance(dfrom, list):
            errors.append(f"{rel}/PROVENANCE.yaml: asset_mode=derived requires derived_from list")
        else:
            for sid in dfrom:
                if sid not in known_source_ids:
                    errors.append(f"{rel}/PROVENANCE.yaml: derived_from '{sid}' not a known source id")
    elif mode == "verbatim":
        if not prov.get("upstream_sha"):
            errors.append(f"{rel}/PROVENANCE.yaml: asset_mode=verbatim requires upstream_sha")

    # AC-9 directory-to-mode rule
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[:2] == ("corpus", "artifacts"):
        # Variants dir allows derived; prs/contests disallow derived
        artifact_kind = parts[2]
        in_variants = len(parts) >= 4 and artifact_kind == "kernels" and parts[-1] == "variants"
        if mode == "derived" and not in_variants:
            errors.append(f"{rel}: asset_mode=derived only allowed under corpus/artifacts/kernels/*/variants/")
        if not in_variants and artifact_kind in ("prs", "contests") and mode not in ("verbatim", "extracted"):
            errors.append(f"{rel}: bundles under corpus/artifacts/{artifact_kind}/** must use asset_mode verbatim or extracted")

    # Files list validation
    files = prov.get("files") or []
    if not isinstance(files, list):
        errors.append(f"{rel}/PROVENANCE.yaml: files must be a list")
        files = []

    declared_paths = set()
    bundle_total = 0
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] must be a mapping")
            continue
        lp = entry.get("local_path")
        role = entry.get("role")
        e_mode = entry.get("mode")
        sha = entry.get("sha256")
        if not lp:
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] missing local_path")
            continue
        if role not in ("pr-diff", "upstream-file", "extracted-block", "derived-source", "approach-notes", "bench-record"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].role '{role}' not in allowed set")
        if e_mode not in ("verbatim", "extracted", "derived", "upstream-patch"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].mode '{e_mode}' not in allowed set")
        if not sha:
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] missing sha256")
        abs_path = bundle_root / lp
        if not abs_path.is_file():
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' does not exist in bundle")
            continue
        # R34: reject manifest entries whose resolved path escapes the
        # bundle root (e.g. `../outside.py`). Without this check,
        # files[*].local_path can satisfy is_file() while pointing at
        # content that isn't actually part of the bundle, undermining
        # the flat-ownership / manifest-drift invariants.
        try:
            resolved_path = abs_path.resolve()
            resolved_root = bundle_root.resolve()
            if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
                errors.append(
                    f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' escapes "
                    f"the bundle root (resolves to '{resolved_path}', outside '{resolved_root}')"
                )
                continue
        except (OSError, RuntimeError) as e:
            errors.append(
                f"{rel}/PROVENANCE.yaml: files[{i}].local_path '{lp}' could not be resolved: {e}"
            )
            continue
        declared_paths.add(resolved_path)
        # SHA verification (unless size_cap_truncated: true on this entry)
        truncated = bool(entry.get("size_cap_truncated"))
        if sha and not truncated:
            actual = sha256_of_file(abs_path)
            if actual != sha:
                errors.append(f"{rel}/PROVENANCE.yaml: files[{i}] sha256 mismatch (got {actual[:12]}..., declared {str(sha)[:12]}...)")
        # Size cap
        size = abs_path.stat().st_size
        bundle_total += size
        if size > FILE_SIZE_CAP_BYTES and not truncated:
            errors.append(
                f"{rel}/PROVENANCE.yaml: files[{i}] local_path '{lp}' is {size} bytes "
                f"(> {FILE_SIZE_CAP_BYTES} cap); set size_cap_truncated: true or split"
            )
        # Extracted requires heading_path
        if e_mode == "extracted" and not entry.get("heading_path"):
            errors.append(f"{rel}/PROVENANCE.yaml: files[{i}].mode=extracted requires heading_path")

    # Bundle-level size cap
    bundle_truncated = bool(prov.get("size_cap_truncated"))
    if bundle_total > BUNDLE_SIZE_CAP_BYTES and not bundle_truncated:
        errors.append(
            f"{rel}: bundle aggregate is {bundle_total} bytes (> {BUNDLE_SIZE_CAP_BYTES} cap); "
            f"set PROVENANCE.yaml size_cap_truncated: true or downgrade the bundle"
        )

    # Filesystem-vs-manifest drift detection: every source file in the bundle
    # (recursive) must appear in declared_paths
    for f in bundle_root.rglob("*"):
        if not f.is_file():
            continue
        if f.name == "PROVENANCE.yaml":
            continue
        if f.suffix.lower() not in ASSET_SOURCE_EXTS and f.name not in ("MANIFEST.yaml", "approach.md", "bench.txt"):
            continue
        # MANIFEST.yaml (blog extraction) lives in parent, not bundle root
        if f.resolve() not in declared_paths and f.suffix.lower() in ASSET_SOURCE_EXTS:
            errors.append(
                f"{rel_to_root(f)}: source file present in bundle but not listed in "
                f"{rel}/PROVENANCE.yaml files[*] (manifest drift)"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--include-refresh-governance",
        action="store_true",
        help=(
            "Also enforce manual refresh-round invariants for candidate ledgers, "
            "refresh cutoff/search state, and pr-page-skipped audits."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    tags = load_yaml_file(SCHEMAS_DIR / "tags.yaml")
    schemas = load_yaml_file(SCHEMAS_DIR / "page-schemas.yaml")

    all_errors = []
    file_count = 0
    ids_seen = {}

    code_langs = _load_code_langs()

    # First pass: collect all source IDs (for cross-referencing wiki->source).
    # Also collect all known page IDs (source + wiki) for provenance
    # derived_from checks (wiki-hardware/wiki-technique IDs like hw-tcgen05-mma
    # and technique-warp-specialization are legitimate provenance citations).
    all_source_ids = set()
    for md_file in sorted(SOURCES_DIR.rglob("*.md")) if SOURCES_DIR.exists() else []:
        fm = extract_frontmatter(md_file)
        if fm and isinstance(fm, dict) and "id" in fm:
            all_source_ids.add(fm["id"])
    all_known_ids = set(all_source_ids)
    for md_file in sorted(WIKI_DIR.rglob("*.md")) if WIKI_DIR.exists() else []:
        fm = extract_frontmatter(md_file)
        if fm and isinstance(fm, dict) and "id" in fm:
            all_known_ids.add(fm["id"])

    # Second pass: validate everything
    for search_dir in [SOURCES_DIR, WIKI_DIR]:
        if not search_dir.exists():
            continue
        for md_file in sorted(search_dir.rglob("*.md")):
            file_count += 1
            fm = extract_frontmatter(md_file)

            # Check for duplicate ids
            if fm and isinstance(fm, dict) and "id" in fm:
                fid = fm["id"]
                if fid in ids_seen:
                    all_errors.append(
                        f"{rel_to_root(md_file)}: duplicate id '{fid}' "
                        f"(also in {ids_seen[fid]})"
                    )
                else:
                    ids_seen[fid] = rel_to_root(md_file)

            errors = validate_file(md_file, schemas, tags, all_source_ids, code_langs)
            all_errors.extend(errors)

    # Phase 3: artifact bundle validation
    bundle_count = 0
    bundle_errors = 0
    verbatim_count = 0
    extracted_count = 0
    derived_count = 0
    for bundle_root in discover_bundle_roots():
        bundle_count += 1
        berrs = validate_bundle(bundle_root, all_known_ids)
        if berrs:
            bundle_errors += 1
        all_errors.extend(berrs)
        # Collect asset-mode breakdown for summary
        prov_path = bundle_root / "PROVENANCE.yaml"
        if prov_path.is_file():
            try:
                prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
                m = prov.get("asset_mode")
                if m == "verbatim":
                    verbatim_count += 1
                elif m == "extracted":
                    extracted_count += 1
                elif m == "derived":
                    derived_count += 1
            except yaml.YAMLError:
                pass

    # Orphan source-file scan
    orphans = find_orphan_source_files()
    for op in orphans:
        all_errors.append(f"{rel_to_root(op)}: source file outside any recognized asset bundle")

    # Candidate-ledger shape check.
    ledger_count = 0
    if CANDIDATES_DIR.exists():
        for ledger_file in sorted(CANDIDATES_DIR.glob("*.yaml")):
            ledger_count += 1
            all_errors.extend(validate_ledger(ledger_file))

    # AC-2 hybrid version-claim registry consistency.
    all_errors.extend(validate_version_claims_registry(all_source_ids))

    # AC-11 inclusion-policy YAML scalar guard.
    all_errors.extend(validate_inclusion_policy_scalars())

    if args.include_refresh_governance:
        # Manual refresh-round governance checks.
        all_errors.extend(validate_skip_audit_coverage())
        all_errors.extend(validate_refresh_cutoff_alignment())
        all_errors.extend(validate_refresh_subset())

    # Missing-pointer detection on claim-bearing pages.
    all_errors.extend(validate_claim_bearing_pages_have_pointer())

    # DEC-4 CUTLASS dev-pinning rule.
    all_errors.extend(validate_cutlass_dev_pinning())

    # Layout discoverability.
    all_errors.extend(validate_discoverability())

    print(f"Validated {file_count} files ({len(all_source_ids)} source IDs collected)")
    if bundle_count or orphans:
        print(f"Validated {bundle_count} asset bundles "
              f"(verbatim={verbatim_count}, extracted={extracted_count}, derived={derived_count}, "
              f"orphan-source-files={len(orphans)})")
    if ledger_count:
        print(f"Validated {ledger_count} candidate ledgers")
    if args.include_refresh_governance:
        print("Included manual refresh-governance checks")
    if all_errors:
        print(f"\n{len(all_errors)} errors found:\n")
        for err in all_errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print("All files valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
