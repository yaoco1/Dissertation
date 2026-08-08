#!/bin/bash
# Phase 1: clean runs for naiveCsr / warpCsr / sharedCsr
# Phase 2: ncu Memory Workload Analysis for all three kernels

TIMES=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/results_${TIMES}.txt"
NCU_OUTPUT="results/ncu_${TIMES}.txt"

MATRICES=(
    "data/sciComputingMatrix/cant/cant.mtx"
    "data/sciComputingMatrix/ct20stif/ct20stif.mtx"
    "data/randomSparseMatrix/barrier2-1/barrier2-1.mtx"
    "data/randomSparseMatrix/nlpkkt80/nlpkkt80.mtx"
    "data/graphMatrix/roadNet-CA/roadNet-CA.mtx"
    "data/graphMatrix/roadNet-TX/roadNet-TX.mtx"
    "data/largeScaleGraphMatrix/com-Youtube/com-Youtube.mtx"
)

NAMES=("cant" "ct20stif" "barrier2-1" "nlpkkt80" "roadNet-CA" "roadNet-TX" "com-Youtube")

{
nvidia-smi
echo ""
echo "Start time: $(date)"
echo ""


# Phase 1: Benchmark runs
echo "=========================================="
echo " Phase 1: Benchmark Runs"
echo "=========================================="

echo "------------------------------------------"
echo " Kernel: baseline (cuSPARSE)"
echo "------------------------------------------"
for MTX in "${MATRICES[@]}"; do
    echo "--- $MTX ---"
    ./build/baseline "$MTX"
    echo ""
done

for KERNEL in naiveCsr warpCsr; do
    echo "------------------------------------------"
    echo " Kernel: $KERNEL"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ./build/$KERNEL "$MTX"
        echo ""
    done
done

for VEC_SIZE in 256 512 1024 2048 4096; do
    echo "------------------------------------------"
    echo " Kernel: sharedCsr (sharedVecSize=$VEC_SIZE)"
    echo "------------------------------------------"
    for MTX in "${MATRICES[@]}"; do
        echo "--- $MTX ---"
        ./build/sharedCsr "$MTX" $VEC_SIZE
        echo ""
	done
done
} 2>&1 | tee "$OUTPUT"

{
echo "Start time: $(date)"
echo ""

echo "=========================================="
echo " Phase 2: ncu Memory Workload Analysis"
echo "=========================================="

echo "------------------------------------------"
echo " ncu: baseline (cuSPARSE)"
echo "------------------------------------------"
for i in "${!MATRICES[@]}"; do
    MTX="${MATRICES[$i]}"
    NAME="${NAMES[$i]}"
    echo "--- $MTX ---"
    ncu --section MemoryWorkloadAnalysis \
        --launch-count 1 --launch-skip 1 \
        ./build/baseline "$MTX"
    echo ""
done

for KERNEL in naiveCsr warpCsr; do
    echo "------------------------------------------"
    echo " ncu: $KERNEL"
    echo "------------------------------------------"
    for i in "${!MATRICES[@]}"; do
        MTX="${MATRICES[$i]}"
        NAME="${NAMES[$i]}"
        echo "--- $MTX ---"
        ncu --section MemoryWorkloadAnalysis \
            --launch-count 1 --launch-skip 1 \
            ./build/$KERNEL "$MTX"
        echo ""
    done
done
 
for VEC_SIZE in 256 512 1024 2048 4096; do
    echo "------------------------------------------"
    echo " ncu: sharedCsr (sharedVecSize=$VEC_SIZE)"
    echo "------------------------------------------"
    for i in "${!MATRICES[@]}"; do
        MTX="${MATRICES[$i]}"
        NAME="${NAMES[$i]}"
        echo "--- $MTX ---"
        ncu --section MemoryWorkloadAnalysis \
			--launch-count 1 --launch-skip 1 \
			./build/sharedCsr "$MTX" $VEC_SIZE
        echo ""
		done
done

} 2>&1 | tee "$NCU_OUTPUT"