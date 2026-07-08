/**
 * @file rowReorder.cu
 * @brief This file implements row reordering for CSR SpMV. It implements the improvement of L2 cache
 * reuse across consecutive thread blocks by sorting rows in ascending order based on the average 
 * column index.
 *
 * @author  Congcong Yao
 * @version 1.0
 * @date    07/08/2026
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "reorder.h"

/**
 * @brief This function reorders matrix rows by ascending average column index.
 * This reordering is a 1D strategy that is only rows are permuted and column indices
 * within each row remain unchanged.
 *
 * @param[in]  rows    Number of rows in the sparse matrix.
 * @param[in]  nnz     Total number of non-zeros.
 * @param[in]  rowPtr  CSR row pointer array.
 * @param[in]  colIdx  CSR column index array.
 * @param[in]  val     CSR non-zero value array.
 *
 * @return ReorderResult containing the reordered CSR arrays (rowPtr, colIdx, val) and
 *         the permutation vectors (perm, invPerm). All arrays are heap-allocated and
 *         must be freed by the caller.
 *
 */
ReorderResult rowReorder(int rows, int nnz, const int* rowPtr, const int* colIdx, const float* val)
{
    std::vector<double> avgCol(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        int len = rowPtr[i+1] - rowPtr[i];
        
        if (len == 0)
            continue;
        
        double s = 0.0;
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++)
            s += colIdx[j];
        
        avgCol[i] = s / len;
    }

    std::vector<int> perm(rows);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b){return avgCol[a] < avgCol[b]; });

    std::vector<int> invPerm(rows);
    for (int i = 0; i < rows; i++)
        invPerm[perm[i]] = i;

    int* newRp = (int*)malloc((rows+1) * sizeof(int));
    int* newCi = (int*)malloc(nnz * sizeof(int));
    float* newVal = (float*)malloc(nnz * sizeof(float));

    newRp[0] = 0;
    for (int i = 0; i < rows; i++) {
        int oldRow = perm[i];
        int len = rowPtr[oldRow+1] - rowPtr[oldRow];
        newRp[i+1] = newRp[i] + len;
        memcpy(newCi + newRp[i], colIdx + rowPtr[oldRow], len * sizeof(int));
        memcpy(newVal + newRp[i], val + rowPtr[oldRow], len * sizeof(float));
    }

    ReorderResult r;
    r.rowPtr = newRp;
    r.colIdx = newCi;
    r.val = newVal;
    r.perm = (int*)malloc(rows * sizeof(int));
    r.invPerm = (int*)malloc(rows * sizeof(int));
    memcpy(r.perm, perm.data(), rows * sizeof(int));
    memcpy(r.invPerm, invPerm.data(), rows * sizeof(int));
    
    return r;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./row_reorder <matrix.mtx>\n");
        return 1;
    }

    // Read matrix from file
    int rows, cols, nnz;
    int *rowPtr, *colIdx;
    float *val;
    loadMtx(argv[1], &rows, &cols, &nnz, &rowPtr, &colIdx, &val);
    printf("Matrix: %d x %d, nnz = %d\n\n", rows, cols, nnz);

    // Build input and output vectors
    float* x = (float*) malloc(cols * sizeof(float));
    float* y = (float*) malloc(rows * sizeof(float));
    double* yRef = (double*)calloc(rows, sizeof(double));
    
    for (int i = 0; i < cols; i++)
        x[i] = 1.0f;
    
    for (int i = 0; i < rows; i++)
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++)
            yRef[i] += (double)val[j] * (double)x[colIdx[j]];

    // Row reorder（CPU preprocessing）
    auto t0 = std::chrono::high_resolution_clock::now();
    ReorderResult rr = rowReorder(rows, nnz, rowPtr, colIdx, val);
    auto t1 = std::chrono::high_resolution_clock::now();
    double preprocessMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Preprocessing (row reorder): %.2f ms\n\n", preprocessMs);

    // Allocate GPU memory
    int *rpGpu, *ciGpu, *rp2Gpu, *ci2Gpu;
    float *valGpu, *val2Gpu, *xGpu, *yGpu;
    cudaMalloc(&rpGpu, (rows+1)*sizeof(int));
    cudaMalloc(&ciGpu, nnz*sizeof(int));
    cudaMalloc(&valGpu, nnz*sizeof(float));
    cudaMalloc(&rp2Gpu, (rows+1)*sizeof(int));
    cudaMalloc(&ci2Gpu, nnz*sizeof(int));
    cudaMalloc(&val2Gpu, nnz*sizeof(float));
    cudaMalloc(&xGpu, cols*sizeof(float));
    cudaMalloc(&yGpu, rows*sizeof(float));

    cudaMemcpy(rpGpu, rowPtr, (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ciGpu, colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valGpu, val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rp2Gpu, rr.rowPtr, (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ci2Gpu, rr.colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(val2Gpu, rr.val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xGpu, x, cols*sizeof(float), cudaMemcpyHostToDevice);

    const int RUNS = 100;
    float bytes = (float)(nnz*4 + nnz*4 + (rows+1)*4 + cols*4 + rows*4);
    float* yRecovered = (float*)malloc(rows * sizeof(float));
    int shMem = SHARED_VEC_SIZE * sizeof(float);
    const int blockSize = 256;
    int gridNaive = (rows + blockSize - 1) / blockSize;
    int gridWarp = (rows + blockSize/32 - 1) / (blockSize/32);

    auto checkReordered = [&]() {
        cudaMemcpy(y, yGpu, rows * sizeof(float), cudaMemcpyDeviceToHost);
        free(yRecovered);
        yRecovered = (float*)malloc(rows * sizeof(float));
        for (int i = 0; i < rows; i++)
            yRecovered[rr.perm[i]] = y[i];
        
        double err = l2Error(yRef, yRecovered, rows);
        printf("  L2 error: %.2e  [%s]\n", err, err < 1e-3 ? "PASSED" : "FAILED");
    };

    auto checkOriginal = [&]() {
        cudaMemcpy(y, yGpu, rows * sizeof(float), cudaMemcpyDeviceToHost);
        double err = l2Error(yRef, y, rows);
        printf("  L2 error: %.2e  [%s]\n", err, err < 1e-3 ? "PASSED" : "FAILED");
    };

    auto bw = [&](float ms){
        return (bytes/1e9f)/(ms/1e3f);
    };

    float t[6];
    printf("[1] Naive CSR      |  original\n");
    naiveCsrSpmv<<<gridNaive, blockSize>>>(rpGpu, ciGpu, valGpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkOriginal();
    t[0] = timeKernel(NAIVE, rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[0], bw(t[0]));

    printf("\n[2] Naive CSR      |  row-reordered\n");
    naiveCsrSpmv<<<gridNaive, blockSize>>>(rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkReordered();
    t[1] = timeKernel(NAIVE, rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [1]: %.3fx\n", t[1], bw(t[1]), t[0]/t[1]);

    printf("\n[3] Warp CSR       |  original\n");
    warpCsrSpmv<<<gridWarp, blockSize>>>(rpGpu, ciGpu, valGpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkOriginal();
    t[2] = timeKernel(WARP, rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[2], bw(t[2]));
    
    printf("\n[4] Warp CSR       |  row-reordered\n");
    warpCsrSpmv<<<gridWarp, blockSize>>>(rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkReordered();
    t[3] = timeKernel(WARP, rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [3]: %.3fx\n", t[3], bw(t[3]), t[2]/t[3]);

    printf("\n[5] Shared CSR     |  original  (window=%d)\n", SHARED_VEC_SIZE);
    sharedCsrSpmv<<<gridWarp, blockSize, shMem>>>(rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, SHARED_VEC_SIZE);
    cudaDeviceSynchronize();
    checkOriginal();
    t[4] = timeKernel(SHARED, rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[4], bw(t[4]));

    printf("\n[6] Shared CSR     |  row-reordered  (window=%d)\n", SHARED_VEC_SIZE);
    sharedCsrSpmv<<<gridWarp, blockSize, shMem>>>(rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows, cols, SHARED_VEC_SIZE);
    cudaDeviceSynchronize();
    checkReordered();
    t[5] = timeKernel(SHARED, rp2Gpu, ci2Gpu, val2Gpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [5]: %.3fx\n", t[5], bw(t[5]), t[4]/t[5]);

    // t[0]=Naive-orig  t[1]=Naive-reo
    // t[2]=Warp-orig   t[3]=Warp-reo
    // t[4]=Shared-orig t[5]=Shared-reo
    printf("\n════════════════════════════════════════════════════════\n");
    printf("Preprocessing (row reorder): %.2f ms\n\n", preprocessMs);

    // Timetable
    printf("%-20s  %10s  %10s  %10s\n", "Kernel", "Original", "Reordered", "Reo/Orig");
    printf("────────────────────────────────────────────────────────\n");
    printf("%-20s  %8.4f ms  %8.4f ms  %8.3fx\n", "Naive",  t[0], t[1], t[0]/t[1]);
    printf("%-20s  %8.4f ms  %8.4f ms  %8.3fx\n", "Warp",   t[2], t[3], t[2]/t[3]);
    printf("%-20s  %8.4f ms  %8.4f ms  %8.3fx\n", "Shared", t[4], t[5], t[4]/t[5]);

    printf("\nKernel-level comparison（Baseline = Naive original %.4f ms）:\n", t[0]);
    printf("────────────────────────────────────────────────────────\n");
    printf("  Warp   original:   %.3fx\n", t[0]/t[2]);
    printf("  Warp   reordered:  %.3fx\n", t[0]/t[3]);
    printf("  Shared original:   %.3fx\n", t[0]/t[4]);
    printf("  Shared reordered:  %.3fx\n", t[0]/t[5]);

    // Break-even
    printf("\n");
    const char* names[] = {"Naive", "Warp", "Shared"};
    for (int ki = 0; ki < 3; ki++) {
        float orig = t[ki*2], reo = t[ki*2+1];
        if (reo < orig)
            printf("%s break-even: %.0f iterations\n", names[ki], preprocessMs / (orig - reo));
    }
    printf("════════════════════════════════════════════════════════\n");

    // Release resources
    cudaFree(rpGpu);
    cudaFree(ciGpu);
    cudaFree(valGpu);
    cudaFree(rp2Gpu);
    cudaFree(ci2Gpu);
    cudaFree(val2Gpu);
    cudaFree(xGpu);
    cudaFree(yGpu);
    free(rowPtr);
    free(colIdx);
    free(val);
    free(rr.rowPtr);
    free(rr.colIdx);
    free(rr.val);
    free(rr.perm);
    free(rr.invPerm);
    free(x);
    free(y);
    free(yRef);
    free(yRecovered);
    return 0;
}