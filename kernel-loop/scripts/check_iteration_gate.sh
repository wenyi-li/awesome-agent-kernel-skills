#!/usr/bin/env bash
# Kernel-loop iteration gate.
# Usage:
#   check_iteration_gate.sh <output_dir>/vK [--verbose]
#   check_iteration_gate.sh <output_dir>/vN --final [--verbose]
#
# Default mode checks that the current version's required files exist before
# vK+1 may be created and advances current_iteration.txt on success. Final mode
# audits v0..vN without advancing state and does not require vN/hypothesis.txt.

set -uo pipefail

VERSION_DIR="${1:-}"
shift 2>/dev/null || true

VERBOSE=0
FINAL_MODE=0
FAILURES=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

usage() {
    echo "Usage:"
    echo "  $0 <output_dir>/vK [--verbose]"
    echo "  $0 <output_dir>/vN --final [--verbose]"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    FAILURES=$((FAILURES + 1))
}

pass() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo -e "${GREEN}[OK]${NC}   $*"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --verbose)
            VERBOSE=1
            ;;
        --final)
            FINAL_MODE=1
            ;;
        *)
            usage
            exit 2
            ;;
    esac
    shift
done

if [ -z "$VERSION_DIR" ] || [ ! -d "$VERSION_DIR" ]; then
    usage
    exit 2
fi

VNAME="$(basename "$VERSION_DIR")"
OUTPUT_DIR="$(dirname "$VERSION_DIR")"
CURRENT_FILE="$OUTPUT_DIR/current_iteration.txt"

if ! [[ "$VNAME" =~ ^v[0-9]+$ ]]; then
    echo "Invalid version directory name: $VNAME (expected v0, v1, ...)"
    exit 2
fi

VK="${VNAME#v}"

required_common_files=(
    "correctness.md"
    "ncu_summary.md"
    "ncu_details.md"
    "kbs_evidence.md"
)

HYPOTHESIS_TEMPLATE="references/hypothesis.md"
KBS_EVIDENCE_TEMPLATE="references/kbs_evidence.md"

hypothesis_headings=(
    "Version:"
    "Current evidence:"
    "Decision link:"
    "Single change:"
    "Rationale:"
    "Expected metric movement:"
    "Risk:"
    "Evidence used:"
    "Decision rule:"
)

hypothesis_heading_patterns=(
    "^[[:space:]]*Version:[[:space:]]*$"
    "^[[:space:]]*Current evidence:[[:space:]]*$"
    "^[[:space:]]*Decision link:[[:space:]]*$"
    "^[[:space:]]*Single change:[[:space:]]*$"
    "^[[:space:]]*Rationale:[[:space:]]*$"
    "^[[:space:]]*Expected metric movement:[[:space:]]*$"
    "^[[:space:]]*Risk:[[:space:]]*$"
    "^[[:space:]]*Evidence used:[[:space:]]*$"
    "^[[:space:]]*Decision rule:[[:space:]]*$"
)

kbs_headings=(
    "# KBS Evidence"
    "## NCU Facts"
    "## Queries"
    "## Selected Evidence"
    "## Rejected / Limits"
    "## Decision Link"
)

kbs_heading_patterns=(
    "^#[[:space:]]+KBS Evidence([[:space:]]|[[:space:]]*-[[:space:]]|$)"
    "^##[[:space:]]+NCU Facts[[:space:]]*$"
    "^##[[:space:]]+Queries[[:space:]]*$"
    "^##[[:space:]]+Selected Evidence[[:space:]]*$"
    "^##[[:space:]]+Rejected / Limits[[:space:]]*$"
    "^##[[:space:]]+Decision Link[[:space:]]*$"
)

find_kernel() {
    local dir="$1"
    local matches=()
    local candidate

    for candidate in "$dir/kernel.py" "$dir/kernel.cu"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    while IFS= read -r candidate; do
        matches+=("$candidate")
    done < <(find "$dir" -maxdepth 1 -type f \( -name '*.py' -o -name '*.cu' \) ! -name 'ref.py' | sort)

    if [ "${#matches[@]}" -eq 1 ]; then
        echo "${matches[0]}"
        return 0
    fi

    if [ "${#matches[@]}" -gt 1 ]; then
        printf '%s\n' "${matches[@]}"
        return 2
    fi

    return 1
}

version_has_artifacts() {
    local dir="$1"
    local artifact
    local found

    [ -d "$dir" ] || return 1

    for artifact in "${required_common_files[@]}" "hypothesis.txt" "kernel.py" "kernel.cu"; do
        if [ -e "$dir/$artifact" ]; then
            return 0
        fi
    done

    found="$(find "$dir" -maxdepth 1 -type f \( -name '*.py' -o -name '*.cu' \) ! -name 'ref.py' | head -n 1)"
    [ -n "$found" ]
}

highest_artifact_version() {
    local dir
    local name
    local num
    local highest=-1

    shopt -s nullglob
    for dir in "$OUTPUT_DIR"/v[0-9]*; do
        [ -d "$dir" ] || continue
        name="$(basename "$dir")"
        [[ "$name" =~ ^v[0-9]+$ ]] || continue
        version_has_artifacts "$dir" || continue
        num="${name#v}"
        if [ "$num" -gt "$highest" ]; then
            highest="$num"
        fi
    done
    shopt -u nullglob

    echo "$highest"
}

check_required_file() {
    local file="$1"
    local label="$2"

    if [ -s "$file" ]; then
        pass "$label exists and is non-empty"
    elif [ -f "$file" ]; then
        fail "$label is empty"
    else
        fail "$label MISSING"
    fi
}

find_heading_after_line() {
    local file="$1"
    local pattern="$2"
    local after_line="$3"

    awk -v pat="$pattern" -v after="$after_line" '
        NR > after && $0 ~ pat {
            print NR
            found = 1
            exit
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' "$file"
}

check_hypothesis_heading_hierarchy() {
    local file="$1"
    local previous_line=0
    local line
    local i

    [ -s "$file" ] || return

    for i in "${!hypothesis_headings[@]}"; do
        line="$(find_heading_after_line "$file" "${hypothesis_heading_patterns[$i]}" "$previous_line")"
        if [ -n "$line" ]; then
            pass "hypothesis.txt heading order: ${hypothesis_headings[$i]}"
            previous_line="$line"
        else
            fail "hypothesis.txt format mismatch: missing heading in order: ${hypothesis_headings[$i]}. Please consult template: $HYPOTHESIS_TEMPLATE"
            return
        fi
    done
}

check_kbs_heading_hierarchy() {
    local file="$1"
    local previous_line=0
    local line
    local i

    [ -s "$file" ] || return

    for i in "${!kbs_headings[@]}"; do
        line="$(find_heading_after_line "$file" "${kbs_heading_patterns[$i]}" "$previous_line")"
        if [ -n "$line" ]; then
            pass "kbs_evidence.md heading order: ${kbs_headings[$i]}"
            previous_line="$line"
        else
            fail "kbs_evidence.md format mismatch: missing heading in order: ${kbs_headings[$i]}. Please consult template: $KBS_EVIDENCE_TEMPLATE"
            return
        fi
    done
}

check_kernel() {
    local dir="$1"
    local vnum="$2"
    local kernel_file
    local kernel_status
    local prev_dir
    local prev_kernel
    local prev_status

    kernel_file="$(find_kernel "$dir")"
    kernel_status="$?"

    if [ "$kernel_status" -eq 0 ]; then
        if [ -s "$kernel_file" ]; then
            pass "kernel file exists and is non-empty: $(basename "$kernel_file")"
        else
            fail "kernel file is empty: $(basename "$kernel_file")"
            return
        fi
    elif [ "$kernel_status" -eq 2 ]; then
        fail "multiple candidate kernel files found; use one kernel.py/kernel.cu or one non-ref .py/.cu"
        return
    else
        fail "kernel file MISSING"
        return
    fi

    if [ "$vnum" -eq 0 ]; then
        pass "v0 has no previous kernel to compare"
        return
    fi

    prev_dir="$OUTPUT_DIR/v$((vnum - 1))"
    if [ ! -d "$prev_dir" ]; then
        fail "previous version directory missing: $prev_dir"
        return
    fi

    prev_kernel="$(find_kernel "$prev_dir")"
    prev_status="$?"
    if [ "$prev_status" -ne 0 ]; then
        fail "previous version has no unambiguous kernel file: $prev_dir"
        return
    fi

    if cmp -s "$kernel_file" "$prev_kernel"; then
        fail "kernel file is identical to previous version"
    else
        pass "kernel file differs from previous version"
    fi
}

check_version_files() {
    local dir="$1"
    local require_hypothesis="$2"
    local vname
    local vnum
    local file

    if [ ! -d "$dir" ]; then
        fail "version directory missing: $dir"
        return
    fi

    vname="$(basename "$dir")"
    if ! [[ "$vname" =~ ^v[0-9]+$ ]]; then
        fail "invalid version directory name: $vname"
        return
    fi
    vnum="${vname#v}"

    echo "=== Checking $vname ==="

    for file in "${required_common_files[@]}"; do
        check_required_file "$dir/$file" "$file"
    done

    if [ "$require_hypothesis" -eq 1 ]; then
        check_required_file "$dir/hypothesis.txt" "hypothesis.txt"
        check_hypothesis_heading_hierarchy "$dir/hypothesis.txt"
    elif [ -f "$dir/hypothesis.txt" ]; then
        check_required_file "$dir/hypothesis.txt" "hypothesis.txt"
        check_hypothesis_heading_hierarchy "$dir/hypothesis.txt"
    else
        pass "final version does not require a next-change hypothesis"
    fi

    check_kbs_heading_hierarchy "$dir/kbs_evidence.md"
    check_kernel "$dir" "$vnum"
}

check_state() {
    local expected="$1"
    local recorded
    local recorded_num
    local highest_num

    if [ "$expected" = "v0" ] && [ ! -f "$CURRENT_FILE" ] && [ "$FINAL_MODE" -eq 0 ]; then
        echo "v0" > "$CURRENT_FILE"
        pass "current_iteration.txt initialized to v0"
        return
    fi

    if [ ! -f "$CURRENT_FILE" ]; then
        fail "current_iteration.txt missing; run the v0 transition gate first"
        return
    fi

    recorded="$(tr -d '[:space:]' < "$CURRENT_FILE")"
    if [ "$recorded" = "$expected" ]; then
        pass "current_iteration.txt matches $expected"
    else
        fail "current_iteration.txt says $recorded, but checking $expected. This usually means the gate was not run sequentially."
    fi

    if [[ "$recorded" =~ ^v[0-9]+$ ]]; then
        recorded_num="${recorded#v}"
        highest_num="$(highest_artifact_version)"
        if [ "$highest_num" -gt "$recorded_num" ]; then
            fail "state drift detected: current_iteration.txt says $recorded, but artifacts already exist through v$highest_num. Future iterations appear to have been generated before their gates passed; do not continue from this state without review."
        fi
    fi
}

check_future_artifacts() {
    local current_num="$1"
    local dir
    local name
    local num
    local artifact
    local found

    shopt -s nullglob
    for dir in "$OUTPUT_DIR"/v[0-9]*; do
        [ -d "$dir" ] || continue
        name="$(basename "$dir")"
        [[ "$name" =~ ^v[0-9]+$ ]] || continue
        num="${name#v}"
        [ "$num" -gt "$current_num" ] || continue

        for artifact in "${required_common_files[@]}" "hypothesis.txt" "kernel.py" "kernel.cu"; do
            if [ -e "$dir/$artifact" ]; then
                fail "future artifact exists before gate pass: $dir/$artifact"
            fi
        done

        found="$(find "$dir" -maxdepth 1 -type f \( -name '*.py' -o -name '*.cu' \) ! -name 'ref.py' | head -n 1)"
        if [ -n "$found" ]; then
            fail "future kernel artifact exists before gate pass: $found"
        fi
    done
    shopt -u nullglob
}

if [ "$FINAL_MODE" -eq 1 ]; then
    echo "=== Kernel-loop Final Gate: $VNAME ==="
    check_state "$VNAME"
    check_future_artifacts "$VK"

    i=0
    while [ "$i" -lt "$VK" ]; do
        check_version_files "$OUTPUT_DIR/v$i" 1
        i=$((i + 1))
    done
    check_version_files "$VERSION_DIR" 0
else
    echo "=== Kernel-loop Transition Gate: $VNAME ==="
    check_state "$VNAME"
    check_future_artifacts "$VK"
    check_version_files "$VERSION_DIR" 1
fi

echo ""
if [ "$FAILURES" -eq 0 ]; then
    if [ "$FINAL_MODE" -eq 1 ]; then
        echo -e "${GREEN}=== Gate PASSED: final audit complete through $VNAME ===${NC}"
    else
        NEXT="v$((VK + 1))"
        echo "$NEXT" > "$CURRENT_FILE"
        echo -e "${GREEN}=== Gate PASSED: $VNAME complete; next allowed version is $NEXT ===${NC}"
    fi
    exit 0
fi

echo -e "${RED}=== Gate FAILED: $FAILURES check(s) failed ===${NC}"
echo "Fix the current iteration artifacts and rerun the gate. Do not create the next version yet."
exit 1
