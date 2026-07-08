/**
 * @file warpCsr.cu
 * @brief This file contains the Warp-cooperative CSR SpMV kernel and driver. Each warp of 32 threads
 * cooperatively computes one output element y[i] via strided accumulation and butterfly reduction
 * with __shfl_down_sync.
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
#include <vector>


/**
 * @brief This function parses the Matrix header to detect matrix properties (symmetric, pattern),
 * reads all non-zero entries, expands symmetric matrices to their full representation,
 * skips explicit zero values, and converts the data into compressed Sparse Row (CSR) arrays.
 *
 * @param[in]  filename   Path to the Matrix Market file.
 * @param[out] m          Number of rows in the matrix.
 * @param[out] n          Number of columns in the matrix.
 * @param[out] nnz        Total number of non-zeros.
 * @param[out] rowPtr     Host array of row pointers. h_rowPtr[i+1] - h_rowPtr[i] gives the number
 *                        of non-zeros in row i.
 * @param[out] colIdx     Host array of column indices for each non-zero.
 * @param[out] val        Host array of non-zero float values.
 *
 */
void loadMtx(const char* fileName, int* m, int* n, int* nnz, int** rowPtr, int** colIdx, float** val)
{
    FILE* fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", fileName);
        exit(1);
    }

    char line[512];
    bool isSymmetric = false;
    bool isPattern = false;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric"))
                isSymmetric = true;
            if (strstr(line, "pattern"))
               isPattern = true;
            continue;
        }
        break;
    }

    int rows, cols, nz;
    sscanf(line, "%d %d %d", &rows, &cols, &nz);
    *m = rows;
    *n = cols;
    struct Entry {
        int row, col;
        float val;
    };
    std::vector<Entry> entries;
    entries.reserve(isSymmetric ? nz * 2 : nz);

    for (int i = 0; i < nz; i++) {
        int row, col;
        double data = 1.0;
        if (isPattern)
            fscanf(fp, "%d %d", &row, &col);
        else
            fscanf(fp, "%d %d %lf", &row, &col, &data);
        row--;
        col--;

        if (data == 0.0)
            continue;
        
        float fData = (float)data;
        entries.push_back({row, col, fData});
        if (isSymmetric && row != col)
            entries.push_back({col, row, fData});
    }
    fclose(fp);

    int totalNNZ = (int)entries.size();
    *nnz = totalNNZ;

    *rowPtr = (int*)calloc(rows + 1, sizeof(int));
    *colIdx = (int*)malloc(totalNNZ * sizeof(int));
    *val = (float*)malloc(totalNNZ * sizeof(float));

    for (auto& e : entries)
        (*rowPtr)[e.row + 1]++;
    for (int i = 1; i <= rows; i++)
        (*rowPtr)[i] += (*rowPtr)[i-1];

    std::vector<int> cursor((*rowPtr), (*rowPtr) + rows + 1);
    for (auto& e : entries) {
        int pos = cursor[e.row]++;
        (*colIdx)[pos] = e.col;
        (*val)[pos] = e.val;
    }
}

/**
 * @brief This function implements a Warp-cooperative CSR SpMV kernel (one warp per row). Each warp 
 * of 32 threads cooperatively computes one output element y[i].
 *
 * @param[in]  rowPtr  CSR row pointer array. rowPtr[i+1] - rowPtr[i] gives the number
 *                     of non-zeros in row i.
 * @param[in]  colIdx  CSR column index array.
 * @param[in]  val     CSR non-zero value array.
 * @param[in]  x       Input vector.
 * @param[out] y       Output vector.
 * @param[in]  rows    Number of rows in the sparse matrix.
 *
 */
__global__ void warpCsrSpmv(const int* rowPtr, const int* colIdx, const float* val, const float* x, float* y, int rows)
{
    int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneIdx = threadIdx.x % 32;

    if (warpIdx >= rows)
        return;

    int rowStart = rowPtr[warpIdx];
    int rowEnd = rowPtr[warpIdx + 1];

    float sum = 0.0f;
    for (int j = rowStart + laneIdx; j < rowEnd; j += 32)
        sum += val[j] * x[colIdx[j]];

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (laneIdx == 0)
        y[warpIdx] = sum;
}

/**
 * @brief This function verifies GPU SpMV output against a double-precision CPU reference.
 *
 * @param[in] yRef   Double-precision CPU reference vector.
 * @param[in] yTest  Single-precision GPU output vector.
 * @param[in] rows   Number of matrix rows.
 * @param[in] tol    Relative L2 error tolerance (default 1e-3).
 *
 * @return @c true  if relative L2 error < @p tol.
 * @return @c false if relative L2 error >= @p tol.
 *
 */
bool verify(const double* yRef, const float* yTest, int rows, float tol = 1e-3f)
{
    double normDiff = 0.0, normRef = 0.0;

    for (int i = 0; i < rows; i++) {
        double diff = (double)yTest[i] - yRef[i];
        normDiff += diff * diff;
        normRef += yRef[i] * yRef[i];
    }

    double relErr = sqrt(normDiff / (normRef + 1e-30));
    printf("Relative L2 error: %.2e\n", relErr);
    
    return relErr < tol;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./warp_csr <matrix.mtx>\n");
        return 1;
    }

    // Read matrix from file
    int rows, cols, nnz;
    int *rowPtr, *colIdx;
    float *val;
    loadMtx(argv[1], &rows, &cols, &nnz, &rowPtr, &colIdx, &val);
    printf("Matrix: %d x %d, nnz = %d\n", rows, cols, nnz);

    // Build input and output vectors
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));
    double* yRef = (double*)calloc(rows, sizeof(double));
    for (int i = 0; i < cols; i++)
        x[i] = 1.0f;

    // CPU reference result (double precision)
    for (int i = 0; i < rows; i++)
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++)
            yRef[i] += (double)val[j] * (double)x[colIdx[j]];

    // Allocate GPU memory
    int *rowPtrGpu, *colIdxGpu;
    float *valGpu, *xGpu, *yGpu;
    cudaMalloc(&rowPtrGpu, (rows+1)*sizeof(int));
    cudaMalloc(&colIdxGpu, nnz*sizeof(int));
    cudaMalloc(&valGpu, nnz*sizeof(float));
    cudaMalloc(&xGpu, cols*sizeof(float));
    cudaMalloc(&yGpu, rows*sizeof(float));

    cudaMemcpy(rowPtrGpu, rowPtr, (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIdxGpu, colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valGpu, val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xGpu, x, cols*sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel parameters
    int blockSize = 256;   // 256 = 8 warps
    int warpPerBlock = blockSize / 32;  // = 8
    int gridSize = (rows + warpPerBlock - 1) / warpPerBlock;

    // Warmup
    cudaMemset(yGpu, 0, rows*sizeof(float));
    warpCsrSpmv<<<gridSize, blockSize>>>(rowPtrGpu, colIdxGpu, valGpu, xGpu, yGpu, rows);
    cudaDeviceSynchronize();

    // Verify correctness
    cudaMemcpy(y, yGpu, rows*sizeof(float), cudaMemcpyDeviceToHost);
    if (verify(yRef, y, rows))
        printf("Correctness: PASSED\n");
    else
        printf("Correctness: FAILED\n");

    // Timing
    int RUNS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(yGpu, 0, rows*sizeof(float));
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++) {
        warpCsrSpmv<<<gridSize, blockSize>>>(rowPtrGpu, colIdxGpu, valGpu, xGpu, yGpu, rows);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avgMs = ms / RUNS;

    // Computational performance indicators
    float bytes = (float)(nnz*4 + nnz*4 + (rows+1)*4 + cols*4 + rows*4);
    float bandwidth = (bytes / 1e9f) / (avgMs / 1e3f);
    float gflops = (2.0f * nnz / 1e9f) / (avgMs / 1e3f);

    printf("\n[Warp-Cooperative CSR Kernel] block_size=%d:\n", blockSize);
    printf("  Time:      %.4f ms\n", avgMs);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    printf("  GFLOPS:    %.2f\n", gflops);

    // Release resources
    cudaFree(rowPtrGpu);
    cudaFree(colIdxGpu);
    cudaFree(valGpu);
    cudaFree(xGpu);
    cudaFree(yGpu);
    free(rowPtr);
    free(colIdx);
    free(val);
    free(x);
    free(y);
    free(yRef);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}