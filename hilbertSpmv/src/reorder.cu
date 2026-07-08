/**
 * @file reorder.cu
 * @brief This file contains the implementation of shared sparse matrix reordering utilities.
 * It provides all CPU-side reordering functions shared across the three reordering benchmarks.
 *
 * @author  Congcong Yao
 * @version 1.0
 * @date    07/08/2026
 *
 */


#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "reorder.h"


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
    bool isSymmetric = false, isPattern = false;
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
        (*val)[pos]    = e.val;
    }
}

/**
 * @brief This function rounds up a 64-bit integer to the next power of two.
 *
 * @param[in] v  Input value.
 *
 * @return Smallest power of two >= @p v.
 *
 */
uint64_t nextPow264(uint64_t v)
{
    if (v == 0)
        return 1;
    
    v--;
    v |= v >> 1; 
    v |= v >> 2; 
    v |= v >> 4;
    v |= v >> 8; 
    v |= v >> 16;
    v |= v >> 32;
    
    return v + 1;
}

/**
 * @brief This function converts 2D grid coordinates to a Hilbert curve index.
 *
 * @param[in] n  The grid has side length 2^n. Must satisfy 2^n >= max(num_rows, num_cols) of the matrix.             
 * @param[in] x  Row coordinate in [0, 2^n).
 * @param[in] y  Column coordinate in [0, 2^n).
 *
 * @return Hilbert curve index in [0, 2^(2n)), representing the position of point (x, y)
 *         along the Hilbert curve.
 *
 */
uint64_t xy2Hilbert(uint64_t n, uint64_t x, uint64_t y)
{
    uint64_t d = 0;

    for (uint64_t s = n >> 1; s > 0; s >>= 1) {
        uint64_t rx = (x & s) ? 1ULL : 0ULL;
        uint64_t ry = (y & s) ? 1ULL : 0ULL;
        d += s * s * ((3ULL * rx) ^ ry);

        if (ry == 0) {
            if (rx == 1) { 
                x = s - 1 - x;
                y = s - 1 - y;
            }
            
            uint64_t t = x;
            x = y;
            y = t;
        }
    }

    return d;
}

/**
 * @brief This function implements a naive CSR SpMV kernel. Each CUDA thread is assigned
 * one matrix row and accumulates the dot product by iterating over all non-zeros in that row.
 *
 * @param[in]  rowPtr  CSR Row pointer array. rowPtr[i+1] - rowPtr[i] gives the number of non-zeros in row i.
 * @param[in]  colIdx  CSR Column index array.
 * @param[in]  val     CSR non-zero value array.
 * @param[in]  x       Input vector.
 * @param[out] y       Output vector.
 * @param[in]  rows    Number of rows in the sparse matrix.
 *
 */
__global__ void naiveCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                    const float* __restrict__ x, float* y, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows)
        return;
    
    float sum = 0.0f;
    for (int j = rowPtr[row]; j < rowPtr[row+1]; j++)
        sum += val[j] * x[colIdx[j]];
    
    y[row] = sum;
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
__global__ void warpCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                   const float* __restrict__ x, float* y, int rows)
{
    int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneIdx = threadIdx.x % 32;
    
    if (warpIdx >= rows)
        return;

    float sum = 0.0f;
    for (int j = rowPtr[warpIdx] + laneIdx; j < rowPtr[warpIdx+1]; j += 32)
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
 * @brief This function implements a Shared Memory CSR SpMV kernel with diagonal x-vector prefetch.
 * Each thread block prefetches a contiguous window of the input vector x[] into shared memory 
 * before processing its assigned rows. Threads then read x values from shared memory (low latency)
 * whenever the required index falls within the cached window, falling back to global memory only 
 * for indices outside the window.
 *
 * @param[in]  rowPtr          CSR row pointer array. rowPtr[i+1] - rowPtr[i] gives the number
 *                             of non-zeros in row i.
 * @param[in]  colIdx          CSR column index array.
 * @param[in]  val             CSR non-zero value array.
 * @param[in]  x               Input vector.
 * @param[out] y               Output vector.
 * @param[in]  rows            Number of rows in the sparse matrix.
 * @param[in]  cols            Number of columns in the sparse matrix.
 * @param[in]  sharedVecSize   Number of x[] elements to prefetch into shared memory per block.
 *
 */
__global__ void sharedCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                    const float* __restrict__ x, float* y, int rows, int cols, int sharedVecSize)
{
    extern __shared__ float xShared[];
    int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneIdx = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;
    int blockFirstRow = blockIdx.x * warpsPerBlock;
    int colStart = blockFirstRow - sharedVecSize / 2;

    if (colStart < 0)
        colStart = 0;

    if (colStart + sharedVecSize > cols)
        colStart = (cols - sharedVecSize > 0) ? cols - sharedVecSize : 0;

    int colEnd = colStart + sharedVecSize;

    for (int i = threadIdx.x; i < sharedVecSize; i += blockDim.x)
        xShared[i] = (colStart + i < cols) ? x[colStart + i] : 0.0f;
    __syncthreads();

    if (warpIdx >= rows)
        return;

    float sum = 0.0f;
    for (int j = rowPtr[warpIdx] + laneIdx; j < rowPtr[warpIdx+1]; j += 32) {
        int col = colIdx[j];
        float xVal = (col >= colStart && col < colEnd) ? xShared[col - colStart] : x[col];
        sum += val[j] * xVal;
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (laneIdx == 0)
        y[warpIdx] = sum;
}

/**
 * @brief This function computes the relative L2 error between a double-precision reference
 *        and a single-precision GPU result.
 *
 * @param[in] yRef   Double-precision CPU reference vector,
 * @param[in] yTest  Single-precision GPU output vector.
 * @param[in] rows   Number of rows in the sparse matrix.
 *
 * @return Relative L2 error as a double.
 *
 */
double l2Error(const double* yRef, const float* yTest, int rows)
{
    double nd = 0.0, nr = 0.0;

    for (int i = 0; i < rows; i++) {
        double d = (double)yTest[i] - yRef[i];
        nd += d * d; nr += yRef[i] * yRef[i];
    }

    return sqrt(nd / (nr + 1e-30));
}

/**
 * @brief This function launches a CSR SpMV kernel repeatedly and return the average execution time.
 * It selects and launches the specified kernel @p RUNS times using cudaEvent timing. The first launch
 * is a warm-up and is excluded from the average to avoid cold-start effects.
 *
 * @param[in]  type  Kernel variant to launch (NAIVE, WARP, or SHARED).
 * @param[in]  rp    CSR row pointer array on device.
 * @param[in]  ci    CSR column index array on device.
 * @param[in]  val   CSR non-zero value array on device.
 * @param[in]  x     Input vector on device.
 * @param[out] y     Output vector on device. Overwritten on each launch.
 * @param[in]  rows  Number of rows in the sparse matrix.
 * @param[in]  cols  Number of columns in the sparse matrix.
 * @param[in]  RUNS  Total number of kernel launches including the warm-up. Must be >= 2. 
 *                   Typical value: 101 (1 warm-up + 100 timed).
 *
 * @return Average kernel execution time in milliseconds over (RUNS - 1) timed launches.
 *
 */
float timeKernel(kernelType type, int* rp, int* ci, float* val, float* x, float* y, int rows, int cols, int RUNS)
{
    const int blockSize = 256;
    int gridNaive = (rows + blockSize - 1) / blockSize;
    int gridWarpshared = (rows + blockSize/32 - 1) / (blockSize/32);
    int shMem = SHARED_VEC_SIZE * sizeof(float);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaMemset(y, 0, rows * sizeof(float));
    cudaEventRecord(s);
    for (int i = 0; i < RUNS; i++) {
        if (type == NAIVE)
            naiveCsrSpmv<<<gridNaive, blockSize>>>(rp, ci, val, x, y, rows);
        else if (type == WARP)
            warpCsrSpmv<<<gridWarpshared, blockSize>>>(rp, ci, val, x, y, rows);
        else
            sharedCsrSpmv<<<gridWarpshared, blockSize, shMem>>>(rp, ci, val, x, y, rows, cols, SHARED_VEC_SIZE);
    }
    cudaEventRecord(e);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    
    return ms / RUNS;
}