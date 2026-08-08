#!/bin/bash
# Phase 1: clean runs  -> results_hilbert_${TS}.txt
# Phase 2: ncu         -> results_hilbert_ncu_${TS}.txt

TIMES=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/results_hilbert_${TIMES}.txt"
OUTPUT_NCU="results/hilbert_ncu_${TIMES}.txt"

MATRICES=(
    "data/sciComputingMatrix/cant/cant.mtx"
    "data/sciComputingMatrix/ct20stif/ct20stif.mtx"
    "data/randomSparseMatrix/barrier2-1/barrier2-1.mtx"
    "data/randomSparseMatrix/nlpkkt80/nlpkkt80.mtx"
    "data/graphMatrix/roadNet-CA/roadNet-CA.mtx"
    "data/graphMatrix/roadNet-TX/roadNet-TX.mtx"
    "data/largeScaleGraphMatrix/com-Youtube/com-Youtube.mtx"
)

# ══════════════════════════════════════════════
# Phase 1: Benchmark runs
# ══════════════════════════════════════════════
{
nvidia-smi
echo ""
echo "Start time: $(date)"
echo ""

echo "=========================================="
echo " Phase 1: Benchmark Runs"
echo "=========================================="

echo "------------------------------------------"
echo " rowReorder"
echo "------------------------------------------"
for MTX in "${MATRICES[@]}"; do
    echo "--- $MTX ---"
    ./build/rowReorder "$MTX"
    echo ""
done

for VEC_SIZE in 256 512 1024 2048 4096; do
    echo "------------------------------------------"
    echo " blockedHilbert (B=$VEC_SIZE)"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ./build/blockedHilbert "$MTX" $VEC_SIZE
        echo ""
    done
done


for VEC_SIZE in 256 512 1024 2048 4096; do
    echo "------------------------------------------"
    echo " globalHilbert (B=$VEC_SIZE)"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ./build/globalHilbert "$MTX" $VEC_SIZE
        echo ""
    done
done
} 2>&1 | tee "$OUTPUT"

# ══════════════════════════════════════════════
# Phase 2: ncu Memory Workload Analysis
# ══════════════════════════════════════════════
{
echo "Start time: $(date)"
echo ""

echo "=========================================="
echo " Phase 2: ncu Memory Workload Analysis"
echo "=========================================="

echo "------------------------------------------"
echo " ncu: rowReorder"
echo "------------------------------------------"
for MTX in "${MATRICES[@]}"; do
    echo "--- $MTX ---"
    ncu --section MemoryWorkloadAnalysis \
        --launch-count 1 --launch-skip 1 \
        ./build/rowReorder "$MTX"
    echo ""
done

for VEC_SIZE in 256 512 1024 2048 4096; do
echo "------------------------------------------"
    echo " ncu: blockedHilbert (B=$VEC_SIZE)"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ncu --section MemoryWorkloadAnalysis \
        --launch-count 1 --launch-skip 1 \
        ./build/blockedHilbert "$MTX" $VEC_SIZE
        echo ""
    done
done
for VEC_SIZE in 256 512 1024 2048 4096; do
    echo "------------------------------------------"
    echo " ncu: globalHilbert (B=$VEC_SIZE)"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ncu --section MemoryWorkloadAnalysis \
        --launch-count 1 --launch-skip 1 \
        ./build/globalHilbert "$MTX" $VEC_SIZE
        echo ""
    done
done
} 2>&1 | tee "$OUTPUT_NCU"