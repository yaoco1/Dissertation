#pragma once

struct ReorderResult {
    int*   rowPtr;
    int*   colIdx;
    float* val;
    int*   perm;     // perm[newRow]    = oldRow
    float* xPerm;    // xPerm[newRow]   = x[oldRow]
    int*   invPerm;  // invPerm[oldRow] = newRow
};

enum kernelType { 
    NAIVE,
    WARP,
    SHARED
};

const int SHARED_VEC_SIZE = 512;

void loadMtx(const char* filename, int* m, int* n, int* nnz, int** rowPtr, int** colIdx, float** val) ;

uint64_t nextPow264(uint64_t v);

uint64_t xy2Hilbert(uint64_t n, uint64_t x, uint64_t y);

__global__ void naiveCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                    const float* __restrict__ x, float* y, int rows);

__global__ void warpCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                   const float* __restrict__ x, float* y, int rows);

__global__ void sharedCsrSpmv(const int* __restrict__ rowPtr, const int* __restrict__ colIdx, const float* __restrict__ val,
                    const float* __restrict__ x, float* y, int rows, int cols, int sharedVecSize);

double l2Error(const double* y_ref, const float* y_test, int rows);

float timeKernel(kernelType type, int* rp, int* ci, float* val, float* x, float* y, int rows, int cols, int RUNS);