#!/bin/bash
# NCU automated deep-performance profiling script (output → ncu_reports/)
# Supports Native CUDA / CUTLASS / Triton / CuTe DSL kernels
#
# Usage:
#   ./ncu_profile.sh <kernel_or_command> [output_prefix] [extra_ncu_args...]
#
# Examples:
#   ./ncu_profile.sh ./cuda_kernel cuda_report
#   ./ncu_profile.sh ./cuda_kernel cuda_report --kernel-name "cuda_"
#   ./ncu_profile.sh ./cutlass_kernel cutlass_report --kernel-name "cutlass_"
#   ./ncu_profile.sh ./cutlass_kernel cutlass_report --launch-skip 2 --launch-count 1
#   ./ncu_profile.sh "python triton_kernel.py" triton_report
#   ./ncu_profile.sh "python triton_kernel.py" triton_report --kernel-name "triton_"
#   ./ncu_profile.sh "python cutedsl_kernel.py" cutedsl_report
#   ./ncu_profile.sh "python cutedsl_kernel.py" cutedsl_report --kernel-name "cutedsl_"

set -e

KERNEL=$1
PREFIX=${2:-"report_$(date +%Y%m%d_%H%M%S)"}
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"
REPORT_DIR="ncu_reports"

if [ -z "$KERNEL" ]; then
    echo "Usage: ./ncu_profile.sh <kernel_or_command> [output_prefix] [extra_ncu_args...]"
    echo ""
    echo "Examples (Native CUDA / CUTLASS):"
    echo "  ./ncu_profile.sh ./cuda_kernel cuda_report"
    echo "  ./ncu_profile.sh ./cuda_kernel cuda_report --kernel-name 'cuda_'"
    echo "  ./ncu_profile.sh ./cutlass_kernel cutlass_report --kernel-name 'cutlass_'"
    echo ""
    echo "Examples (Triton / Python):"
    echo "  ./ncu_profile.sh 'python triton_kernel.py' triton_report"
    echo "  ./ncu_profile.sh 'python triton_kernel.py' triton_report --kernel-name 'triton_'"
    echo "  ./ncu_profile.sh 'python triton_kernel.py' triton_report --launch-skip 5 --launch-count 1"
    echo ""
    echo "Examples (CuTe DSL / Python):"
    echo "  ./ncu_profile.sh 'python cutedsl_kernel.py' cutedsl_report"
    echo "  ./ncu_profile.sh 'python cutedsl_kernel.py' cutedsl_report --kernel-name 'cutedsl_'"
    echo "  ./ncu_profile.sh 'python cutedsl_kernel.py' cutedsl_report --launch-skip 1 --launch-count 1"
    echo ""
    echo "Generates:"
    echo "  ncu_reports/<prefix>.ncu-rep        Full NCU report"
    echo "  ncu_reports/<prefix>.csv            CSV raw metrics (for deep analysis)"
    echo "  ncu_reports/<prefix>_analysis.md    AI analysis report"
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12.* /usr/local/cuda-11.*; do
        if [ -x "${cuda_dir}/bin/ncu" ]; then
            export PATH="${cuda_dir}/bin:$PATH"
            break
        fi
    done
    if ! command -v ncu &> /dev/null; then
        echo "Error: ncu not found. Please ensure CUDA toolkit is installed."
        exit 1
    fi
fi

mkdir -p "$REPORT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IS_PYTHON=false
IS_TRITON=false
IS_CUTEDSL=false
if echo "$KERNEL" | grep -qE '\.py(\s|$)|^python'; then
    IS_PYTHON=true
    # Detect CuTe DSL vs Triton by checking source file for imports
    PY_FILE=$(echo "$KERNEL" | grep -oE '[^ ]+\.py')
    if [ -n "$PY_FILE" ] && [ -f "$PY_FILE" ]; then
        if grep -qE 'cutlass\.cute|cute\.compile|cute\.kernel|cute\.jit' "$PY_FILE" 2>/dev/null; then
            IS_CUTEDSL=true
        fi
        if grep -qE 'triton\.jit|@triton\.jit|triton\.language|triton\.autotune' "$PY_FILE" 2>/dev/null; then
            IS_TRITON=true
        fi
    else
        IS_TRITON=true
    fi
fi

# Determine display mode
if [ "$IS_CUTEDSL" = true ] && [ "$IS_TRITON" = true ]; then
    MODE_DISPLAY="CuTe DSL + Triton / Python"
elif [ "$IS_CUTEDSL" = true ]; then
    MODE_DISPLAY="CuTe DSL / Python"
elif [ "$IS_TRITON" = true ]; then
    MODE_DISPLAY="Triton / Python"
elif [ "$IS_PYTHON" = true ]; then
    MODE_DISPLAY="Python (auto-detect kernel type)"
else
    MODE_DISPLAY="Native CUDA / CUTLASS"
fi

echo "=================================================="
echo "  NCU deep performance profiling"
echo "=================================================="
echo "  Target:  $KERNEL"
echo "  Mode:    $MODE_DISPLAY"
echo "  Prefix:  $PREFIX"
echo "  Output:  $REPORT_DIR/"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  Extra:   $EXTRA_ARGS"
fi
echo ""

if [ "$IS_TRITON" = true ]; then
    TRITON_CACHE="${REPORT_DIR}/${PREFIX}_triton_cache"
    mkdir -p "$TRITON_CACHE"
    export TRITON_CACHE_DIR="$TRITON_CACHE"
    echo "  [Triton] Cache dir: $TRITON_CACHE"
    echo "  [Triton] Clearing stale cache for clean profiling..."
    rm -rf "${TRITON_CACHE:?}"/*
    echo ""
fi

if [ "$IS_CUTEDSL" = true ]; then
    echo "  [CuTe DSL] Detected cutlass.cute imports in source"
    echo "  [CuTe DSL] Note: clear Python _compiled_cache when re-profiling after code changes"
    echo ""
fi

# Phase 1: Full metrics collection
echo "[1/4] Collecting full metrics (--set full)..."
if [ "$IS_PYTHON" = true ]; then
    echo "      Python mode ($MODE_DISPLAY): profiling with --target-processes all"
fi
echo "      This may take a while for complex kernels."
echo ""

ncu --set full \
    -o "${REPORT_DIR}/${PREFIX}" \
    --target-processes all \
    --force-overwrite \
    $EXTRA_ARGS \
    $KERNEL 2>&1 | tee "${REPORT_DIR}/${PREFIX}_log.txt"

echo ""
echo "  -> ${REPORT_DIR}/${PREFIX}.ncu-rep"

# Phase 2: Export CSV raw metrics
echo ""
echo "[2/4] Exporting CSV raw metrics..."

ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" \
    --page raw \
    --csv \
    > "${REPORT_DIR}/${PREFIX}.csv" 2>/dev/null

CSV_LINES=$(wc -l < "${REPORT_DIR}/${PREFIX}.csv" 2>/dev/null || echo "0")
echo "  -> ${REPORT_DIR}/${PREFIX}.csv ($CSV_LINES lines)"

if [ "$IS_TRITON" = true ]; then
    TRITON_KERNEL_COUNT=$(grep -c "triton_\|_kernel_.*d.*d.*e\b" "${REPORT_DIR}/${PREFIX}.csv" 2>/dev/null || echo "0")
    echo "  [Triton] Detected ~${TRITON_KERNEL_COUNT} Triton kernel entries in CSV"
fi

if [ "$IS_CUTEDSL" = true ]; then
    TOTAL_KERNEL_COUNT=$(tail -n +2 "${REPORT_DIR}/${PREFIX}.csv" 2>/dev/null | grep -c '.' || echo "0")
    echo "  [CuTe DSL] Total kernel entries in CSV: ~${TOTAL_KERNEL_COUNT}"
    echo "  [CuTe DSL] Tip: Use --kernel-name to filter CuTe DSL kernel by @cute.kernel method name"
fi

# Phase 3: Summary
echo ""
echo "[3/4] Generating summary..."

ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" \
    --print-summary per-kernel \
    > "${REPORT_DIR}/${PREFIX}_summary.txt" 2>/dev/null

echo "  -> ${REPORT_DIR}/${PREFIX}_summary.txt"

# Phase 4: Deep analysis with Python
echo ""
echo "[4/4] Running deep analysis..."

ANALYSE_EXTRA_ARGS=""
if [ "$IS_CUTEDSL" = true ] && [ "$IS_TRITON" != true ]; then
    ANALYSE_EXTRA_ARGS="--type cutedsl"
fi

if [ -f "${SCRIPT_DIR}/ncu_analyse.py" ] && command -v python3 &> /dev/null; then
    python3 "${SCRIPT_DIR}/ncu_analyse.py" \
        "${REPORT_DIR}/${PREFIX}.csv" \
        -o "${REPORT_DIR}/${PREFIX}_analysis.md" \
        $ANALYSE_EXTRA_ARGS 2>&1 || {
        echo "  Warning: Python analysis failed, skipping."
    }
    if [ -f "${REPORT_DIR}/${PREFIX}_analysis.md" ]; then
        echo "  -> ${REPORT_DIR}/${PREFIX}_analysis.md"
    fi
else
    echo "  Warning: python3 or ncu_analyse.py not available, skipping deep analysis."
fi

if [ "$IS_TRITON" = true ] && [ -d "$TRITON_CACHE" ]; then
    echo ""
    echo "[Triton] Collecting compilation artifacts..."
    PTX_COUNT=$(find "$TRITON_CACHE" -name "*.ptx" 2>/dev/null | wc -l)
    CUBIN_COUNT=$(find "$TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
    echo "  [Triton] Found ${PTX_COUNT} PTX file(s), ${CUBIN_COUNT} CUBIN file(s)"

    if [ "$PTX_COUNT" -gt 0 ]; then
        TENSOR_CORE_USAGE=$(grep -cl "mma\.\|hmma\." "$TRITON_CACHE"/*/*.ptx 2>/dev/null | wc -l || echo "0")
        ASYNC_COPY_USAGE=$(grep -cl "cp\.async" "$TRITON_CACHE"/*/*.ptx 2>/dev/null | wc -l || echo "0")
        echo "  [Triton] Tensor Core (mma/hmma) used in ${TENSOR_CORE_USAGE}/${PTX_COUNT} PTX file(s)"
        echo "  [Triton] Async copy (cp.async) used in ${ASYNC_COPY_USAGE}/${PTX_COUNT} PTX file(s)"
    fi
fi

if [ "$IS_CUTEDSL" = true ]; then
    echo ""
    echo "[CuTe DSL] Analysis notes:"
    echo "  [CuTe DSL] Kernel names from @cute.kernel method names (check source code for mapping)"
    echo "  [CuTe DSL] To get compilation artifacts, add '--keep --verbose' to cute.compile() options"
fi

echo ""
echo "=================================================="
echo "  Step 1 COMPLETE: NCU data collected"
echo "=================================================="
echo ""
echo "  Generated files:"
echo "    ${REPORT_DIR}/${PREFIX}.ncu-rep          Full NCU report"
echo "    ${REPORT_DIR}/${PREFIX}.csv              CSV raw metrics"
echo "    ${REPORT_DIR}/${PREFIX}_summary.txt      Performance summary"
echo "    ${REPORT_DIR}/${PREFIX}_log.txt          NCU log"
if [ -f "${REPORT_DIR}/${PREFIX}_analysis.md" ]; then
    echo "    ${REPORT_DIR}/${PREFIX}_analysis.md      Deep analysis report"
fi
if [ "$IS_TRITON" = true ] && [ -d "$TRITON_CACHE" ]; then
    echo "    ${TRITON_CACHE}/                        Triton compilation cache (PTX/CUBIN)"
fi
echo ""
echo "  *** MANDATORY WORKFLOW (do NOT skip steps) ***"
echo ""
echo "  Step 2: Analyze the profiling data"
echo "    cat ${REPORT_DIR}/${PREFIX}_analysis.md"
echo "    python3 ${SCRIPT_DIR}/ncu_analyse.py ${REPORT_DIR}/${PREFIX}.csv"
echo ""

# Type-specific guidance in next steps
if [ "$IS_CUTEDSL" = true ]; then
    echo "  Step 3: Apply CuTe DSL playbook (based on analysis findings)"
    echo "    Focus: threads_per_cta, elems_per_thread, CopyAtom (num_bits_per_copy),"
    echo "           tiled_copy layout, cta_reduce pattern, smem staging"
    echo "    IMPORTANT: clear compilation cache before re-profiling:"
    echo "      rm -rf __pycache__/ .cache/ /tmp/cutlass_cute_cache/"
elif [ "$IS_TRITON" = true ]; then
    echo "  Step 3: Apply Triton playbook (based on analysis findings)"
    echo "    Focus: num_warps, num_stages, BLOCK_* tile sizes,"
    echo "           tl.multiple_of / tl.max_contiguous hints, tl.dot config"
    echo "    IMPORTANT: clear Triton JIT cache before re-profiling:"
    echo "      rm -rf ~/.triton/cache"
else
    # Native CUDA or CUTLASS
    echo "  Step 3: Apply optimization playbook (based on analysis findings)"
    echo "    Native CUDA: launch config, memory access, cp.async, Tensor Core"
    echo "    CUTLASS:     ThreadblockShape, stages, alignment, schedule, epilogue fusion"
fi
echo ""
echo "  Step 4: Re-profile and compare (verify improvement)"
echo "    bash ${SCRIPT_DIR}/ncu_profile.sh ${KERNEL} ${PREFIX}_v2"
echo "    python3 ${SCRIPT_DIR}/ncu_analyse.py ${REPORT_DIR}/${PREFIX}_v2.csv --diff ${REPORT_DIR}/${PREFIX}.csv"
echo ""
echo "  WARNING: Do NOT modify kernel code before completing Step 2 analysis."
echo "           Every code change MUST be justified by NCU metric evidence."
echo ""
echo "    # View in NCU GUI"
echo "    ncu --import ${REPORT_DIR}/${PREFIX}.ncu-rep"
if [ "$IS_TRITON" = true ]; then
    echo ""
    echo "  Triton-specific:"
    echo "    # Inspect generated PTX"
    echo "    find ${TRITON_CACHE} -name '*.ptx' -exec head -50 {} \\;"
    echo ""
    echo "    # Check Tensor Core usage in PTX"
    echo "    grep -c 'mma\\.\|hmma\\.' ${TRITON_CACHE}/*/*.ptx"
    echo ""
    echo "    # Clear Triton cache before re-profiling"
    echo "    rm -rf ${TRITON_CACHE}/*"
fi
if [ "$IS_CUTEDSL" = true ]; then
    echo ""
    echo "  CuTe DSL-specific:"
    echo "    # Re-analyze specifying kernel type"
    echo "    python3 ${SCRIPT_DIR}/ncu_analyse.py ${REPORT_DIR}/${PREFIX}.csv --type cutedsl"
    echo ""
    echo "    # Clear CuTe DSL compilation cache (in Python code)"
    echo "    # Add '_compiled_cache.clear()' before re-profiling, or change cache key"
    echo ""
    echo "    # Get CuTe DSL compilation artifacts"
    echo "    # Add '--keep --verbose' to cute.compile() options"
fi
