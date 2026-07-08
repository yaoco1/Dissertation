/**
 * @file baseline.cu
 * @brief This file contains the cuSPARSE CSR SpMV baseline benchmark. It provides the industry-standard
 * reference implementation using NVIDIA's cuSPARSE library.
 * 
 * @author Congcong Yao
 * @version 1.0
 * @date   07/08/2026
 */

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

/**
 * @brief This function parses the Matrix header to detect matrix properties (symmetric, pattern),
 * reads all non-zero entries, expands symmetric matrices to their full representation,
 * skips explicit zero values, and converts the data into compressed Sparse Row (CSR) arrays.
 *
 * @param[in]  filename   Path to the Matrix Market file.
 * @param[out] m          Number of rows in the matrix.
 * @param[out] n          Number of columns in the matrix.
 * @param[out] nnz        Total number of stored non-zeros.
 * @param[out] rowPtr     Host array of row pointers. h_rowPtr[i+1] - h_rowPtr[i] gives the number
 *                        of non-zeros in row i.
 * @param[out] colIdx     Host array of column indices for each non-zero.
 * @param[out] val        Host array of non-zero float values.
 *
 */
void loadMtx(const char* filename, int* m, int* n, int* nnz, int** rowPtr, int** colIdx, float** val)
{
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", filename);
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

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./baseline <matrix.mtx>\n");
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
    for (int i = 0; i < cols; i++)
        x[i] = 1.0f;
    for (int i = 0; i < rows; i++)
        y[i] = 0.0f;

    // Allocate GPU memory
    int *rowPtrGpu, *colIdxGpu;
    float *valGpu, *xGpu, *yGpu;
    cudaMalloc(&rowPtrGpu, (rows+1) * sizeof(int));
    cudaMalloc(&colIdxGpu, nnz * sizeof(int));
    cudaMalloc(&valGpu, nnz * sizeof(float));
    cudaMalloc(&xGpu, cols * sizeof(float));
    cudaMalloc(&yGpu, rows * sizeof(float));

    cudaMemcpy(rowPtrGpu, rowPtr, (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIdxGpu, colIdx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valGpu, val, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xGpu, x, cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(yGpu, y, rows*sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&matA, rows, cols, nnz, rowPtrGpu, colIdxGpu, valGpu, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, cols, xGpu, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, rows, yGpu, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* buffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&buffer, bufferSize);

    // warmup
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    cudaDeviceSynchronize();

    // CUDA Timing
    int RUNS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avgMs = ms / RUNS;

    // Computational performance indicators
    float bytes = (float)(nnz * 4 + nnz * 4 + (rows+1) * 4 + cols * 4 + rows * 4);
    float bandwidth = (bytes / 1e9f) / (avgMs / 1e3f);

    printf("cuSPARSE baseline:\n");
    printf("  Average time: %.4f ms\n", avgMs);
    printf("  Bandwidth:    %.2f GB/s\n", bandwidth);
    printf("  GFLOPS:       %.2f\n", (2.0f * nnz / 1e9f) / (avgMs / 1e3f));

    // Release resources
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(rowPtrGpu);
    cudaFree(colIdxGpu);
    cudaFree(valGpu);
    cudaFree(xGpu);
    cudaFree(yGpu);
    cudaFree(buffer);
    free(rowPtr);
    free(colIdx);
    free(val);
    free(x);
    free(y);

    return 0;
}