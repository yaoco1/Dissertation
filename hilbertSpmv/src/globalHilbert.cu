/**
 * @file globalHilbert.cu
 * @brief This file contains the implementation for the Global Hilbert curve symmetric permutation CSR SpMV benchmark.
 * It implements a full symmetric permutation A' = P·A·P^T, where P is derived from the Hilbert curve ordering of row indices.
 * Unlike the row-only strategies, the symmetric permutation also reorders column indices within each row, ensuring that both 
 * the row access pattern and the x-vector access pattern reflect the same spatial ordering.
 *
 * @author Congcong Yao
 * @version 1.0
 * @date   07/08/2026
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "reorder.h"

/**
 * @brief This function computes the full symmetric permutation and x-vector permutation via Hilbert curve ordering.
 * It computes a symmetric permutation A' = P·A·P^T where P is derived from the Hilbert curve ordering of (row, column-block) 
 * coordinates. The symmetric permutation remaps both rows and column indices, ensuring that the x-vector access pattern
 * reflects the same spatial ordering as the row ordering.
 *
 * @param[in]  rows        Number of rows in the sparse matrix.
 * @param[in]  cols        Number of columns in the sparse matrix.
 * @param[in]  nnz         Total number of non-zeros.
 * @param[in]  rowPtr      CSR row pointer array.
 * @param[in]  colIdx      CSR column index array.
 * @param[in]  val         CSR non-zero value array.
 * @param[in]  x           Input vector. Permuted to match the new column ordering and stored in the
 *                         returned ReorderResult.
 * @param[in]  blockSize   Column block size for the Hilbert coordinate mapping.
 *
 * @return ReorderResult containing:
 *         - Reordered CSR arrays (rowPtr, colIdx, val) for A' = PAP^T.
 *         - perm[]    — Maps new row index.
 *         - xPerm[]   — Permuted input vector x[] to match the new column ordering.
 *         - invPerm[] — Maps old row index.
 *         All arrays are heap-allocated and must be freed by the caller.
 *
 */
ReorderResult globalHilbertReorder(int rows, int cols, int nnz, const int* rowPtr, const int* colIdx,
                                     const float* val, const float* x, int blockSize)
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

    uint64_t gridRows = ((uint64_t)rows + blockSize - 1) / blockSize;
    uint64_t gridCols = ((uint64_t)cols + blockSize - 1) / blockSize;
    uint64_t gridN = nextPow264(std::max(gridRows, gridCols));

    std::vector<uint64_t> hkey(rows);
    for (int i = 0; i < rows; i++) {
        uint64_t brow = (uint64_t)i / blockSize;
        uint64_t bcol = (uint64_t)avgCol[i] / blockSize;
        if (brow >= gridN)
            brow = gridN - 1;
        if (bcol >= gridN)
            bcol = gridN - 1;
        
        hkey[i] = xy2Hilbert(gridN, brow, bcol);
    }

    std::vector<int> perm(rows);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (hkey[a] != hkey[b])
            return hkey[a] < hkey[b];
        
        return avgCol[a] < avgCol[b];
    });

    std::vector<int> invPerm(rows);
    for (int i = 0; i < rows; i++)
        invPerm[perm[i]] = i;
    
    // Build new CSR arrays after row reordering
    int* newRp = (int*)malloc((rows+1) * sizeof(int));
    int* newCi = (int*)malloc(nnz * sizeof(int));
    float* newVal = (float*)malloc(nnz * sizeof(float));

    newRp[0] = 0;
    for (int i = 0; i < rows; i++) {
        int oldRow = perm[i];
        int start = rowPtr[oldRow];
        int len = rowPtr[oldRow+1] - start;
        newRp[i+1] = newRp[i] + len;
        
        for (int k = 0; k < len; k++) {
            newCi[newRp[i] + k] = invPerm[colIdx[start + k]];
            newVal[newRp[i] + k] = val[start + k];
        }
    }

    float* xPerm = (float*)malloc(cols * sizeof(float));
    for (int i = 0; i < cols; i++)
        xPerm[i] = x[perm[i]];

    ReorderResult r;
    r.rowPtr = newRp;
    r.colIdx = newCi;
    r.val = newVal;
    r.xPerm = xPerm;
    r.perm = (int*)malloc(rows * sizeof(int));
    r.invPerm = (int*)malloc(rows * sizeof(int));
    memcpy(r.perm, perm.data(), rows * sizeof(int));
    memcpy(r.invPerm, invPerm.data(), rows * sizeof(int));

    return r;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./global_hilbert <matrix.mtx> [blockSize]\n");
        printf("  blockSize: Hilbert tile size (default: 512)\n");
        return 1;
    }

    int blockSize = (argc >= 3) ? atoi(argv[2]) : 512;

    // Read matrix from file
    int rows, cols, nnz;
    int *rowPtr, *colIdx; float *val;
    loadMtx(argv[1], &rows, &cols, &nnz, &rowPtr, &colIdx, &val);
    printf("Matrix: %d x %d, nnz = %d\n", rows, cols, nnz);
    printf("Block size: %d\n\n", blockSize);

    // Build input and output vectors
    float* x = (float*) malloc(cols * sizeof(float));
    float* y = (float*) malloc(rows * sizeof(float));
    double* yRef = (double*)calloc(rows, sizeof(double));
    for (int i = 0; i < cols; i++)
        x[i] = 1.0f;

    for (int i = 0; i < rows; i++)
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++)
            yRef[i] += (double)val[j] * (double)x[colIdx[j]];

    // Global Hilbert Reordering
    auto t0 = std::chrono::high_resolution_clock::now();
    ReorderResult rr = globalHilbertReorder(rows, cols, nnz, rowPtr, colIdx, val, x, blockSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    double preprocessMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Preprocessing (global Hilbert, B=%d): %.2f ms\n\n", blockSize, preprocessMs);

    // Allocate GPU memory
    int *rpGpu, *ciGpu, *rp2Gpu, *ci2Gpu;
    float *valGpu, *val2Gpu, *xGpu, *x2Gpu, *yGpu;
    cudaMalloc(&rpGpu, (rows+1)*sizeof(int));
    cudaMalloc(&ciGpu, nnz*sizeof(int));
    cudaMalloc(&valGpu, nnz*sizeof(float));
    cudaMalloc(&rp2Gpu, (cols+1)*sizeof(int));
    cudaMalloc(&ci2Gpu, nnz*sizeof(int));
    cudaMalloc(&val2Gpu, nnz*sizeof(float));
    cudaMalloc(&xGpu, cols*sizeof(float));
    cudaMalloc(&x2Gpu, cols*sizeof(float));
    cudaMalloc(&yGpu, rows*sizeof(float));

    cudaMemcpy(rpGpu, rowPtr, (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ciGpu, colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valGpu, val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xGpu, x, cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rp2Gpu, rr.rowPtr, (cols+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ci2Gpu, rr.colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(val2Gpu, rr.val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2Gpu, rr.xPerm, cols*sizeof(float), cudaMemcpyHostToDevice);

    const int RUNS = 100;
    float bytes = (float)(nnz*4 + nnz*4 + (rows+1)*4 + cols*4 + rows*4);
    float* yRecovered = (float*)malloc(rows * sizeof(float));
    int shMem = SHARED_VEC_SIZE * sizeof(float);
    int gridNaive = (rows + blockSize - 1) / blockSize;
    int gridWarp = (rows + blockSize/32 - 1) / (blockSize/32);

    // lambda
    auto checkReordered = [&]() {
        cudaMemcpy(y, yGpu, rows * sizeof(float), cudaMemcpyDeviceToHost);

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
    t[0] = timeKernel(NAIVE, rpGpu, ciGpu,  valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[0], bw(t[0]));

    printf("\n[2] Naive CSR      |  global-hilbert-reordered  (B=%d)\n", blockSize);
    naiveCsrSpmv<<<gridNaive, blockSize>>>(rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkReordered();
    t[1] = timeKernel(NAIVE, rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [1]: %.3fx\n", t[1], bw(t[1]), t[0]/t[1]);

    printf("\n[3] Warp CSR       |  original\n");
    warpCsrSpmv<<<gridWarp, blockSize>>>(rpGpu, ciGpu, valGpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkOriginal();
    t[2] = timeKernel(WARP, rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[2], bw(t[2]));

    printf("\n[4] Warp CSR       |  global-hilbert-reordered  (B=%d)\n", blockSize);
    warpCsrSpmv<<<gridWarp, blockSize>>>(rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows);
    cudaDeviceSynchronize();
    checkReordered();
    t[3] = timeKernel(WARP, rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [3]: %.3fx\n", t[3], bw(t[3]), t[2]/t[3]);

    printf("\n[5] Shared CSR     |  original  (window=%d)\n", SHARED_VEC_SIZE);
    sharedCsrSpmv<<<gridWarp, blockSize, shMem>>>(rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, SHARED_VEC_SIZE);
    cudaDeviceSynchronize();
    checkOriginal();
    t[4] = timeKernel(SHARED, rpGpu, ciGpu, valGpu, xGpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s\n", t[4], bw(t[4]));

    printf("\n[6] Shared CSR     |  global-hilbert-reordered  (B=%d, window=%d)\n", blockSize, SHARED_VEC_SIZE);
    sharedCsrSpmv<<<gridWarp, blockSize, shMem>>>(rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows, cols, SHARED_VEC_SIZE);
    cudaDeviceSynchronize();
    checkReordered();
    t[5] = timeKernel(SHARED, rp2Gpu, ci2Gpu, val2Gpu, x2Gpu, yGpu, rows, cols, RUNS);
    printf("  Time: %.4f ms  |  BW: %.2f GB/s  |  Speedup vs [5]: %.3fx\n", t[5], bw(t[5]), t[4]/t[5]);

    printf("\n════════════════════════════════════════════════════════\n");
    printf("Preprocessing (global Hilbert, B=%d): %.2f ms\n\n", blockSize, preprocessMs);
    printf("%-20s  %10s  %12s  %10s\n", "Kernel", "Original", "GH-Reordered", "Reo/Orig");
    printf("────────────────────────────────────────────────────────\n");
    printf("%-20s  %8.4f ms  %10.4f ms  %8.3fx\n", "Naive",  t[0], t[1], t[0]/t[1]);
    printf("%-20s  %8.4f ms  %10.4f ms  %8.3fx\n", "Warp",   t[2], t[3], t[2]/t[3]);
    printf("%-20s  %8.4f ms  %10.4f ms  %8.3fx\n", "Shared", t[4], t[5], t[4]/t[5]);

    printf("\nKernel-level comparison（Baseline = Naive original %.4f ms）:\n", t[0]);
    printf("────────────────────────────────────────────────────────\n");
    printf("  Warp   original:    %.3fx\n", t[0]/t[2]);
    printf("  Warp   GH-reorder:  %.3fx\n", t[0]/t[3]);
    printf("  Shared original:    %.3fx\n", t[0]/t[4]);
    printf("  Shared GH-reorder:  %.3fx\n", t[0]/t[5]);

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
    cudaFree(x2Gpu);
    cudaFree(yGpu);
    free(rowPtr);
    free(colIdx);
    free(val);
    free(rr.rowPtr);
    free(rr.colIdx);
    free(rr.val);
    free(rr.xPerm);
    free(rr.perm);
    free(rr.invPerm);
    free(x);
    free(y);
    free(yRef);
    free(yRecovered);

    return 0;
}