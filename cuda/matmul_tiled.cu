#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A(m, n) x B(n, k)
#define TILE_WIDTH 16
__global__ void tiledMatMul(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output elem we're working on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // loop over required tiles
    float Cvalue = 0.0f;
    for (int ph = 0; ph < ceil(n / (float)TILE_WIDTH); ++ph) {
        // collaboartive effort to load values into shared memory from each thread
        if ((Row < m) && (ph * TILE_WIDTH + tx < n)) {
            Ads[ty][tx] = A[Row * n + ph * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }

        if ((ph * TILE_WIDTH + ty < n) && (Col < k)) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * k + Col];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();
    }

    if ((Row < m) && (Col < k)) {
        C[Row * k + Col] = Cvalue;
    }
}

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(1);
    }
}

void checkCublas(cublasStatus_t result, const char* message) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

int main() {
    const int m = 37;
    const int n = 29;
    const int k = 41;

    const int aSize = m * n;
    const int bSize = n * k;
    const int cSize = m * k;

    float* hA = new float[aSize];
    float* hB = new float[bSize];
    float* hKernelC = new float[cSize];
    float* hCublasC = new float[cSize];

    std::srand(1234);
    for (int i = 0; i < aSize; ++i) {
        hA[i] = static_cast<float>((std::rand() % 21) - 10);
    }
    for (int i = 0; i < bSize; ++i) {
        hB[i] = static_cast<float>((std::rand() % 21) - 10);
    }

    float* dA;
    float* dB;
    float* dKernelC;
    float* dCublasC;

    checkCuda(cudaMalloc((void**)&dA, aSize * sizeof(float)), "cudaMalloc dA failed");
    checkCuda(cudaMalloc((void**)&dB, bSize * sizeof(float)), "cudaMalloc dB failed");
    checkCuda(cudaMalloc((void**)&dKernelC, cSize * sizeof(float)), "cudaMalloc dKernelC failed");
    checkCuda(cudaMalloc((void**)&dCublasC, cSize * sizeof(float)), "cudaMalloc dCublasC failed");

    checkCuda(cudaMemcpy(dA, hA, aSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy hA -> dA failed");
    checkCuda(cudaMemcpy(dB, hB, bSize * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy hB -> dB failed");

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((k + TILE_WIDTH - 1) / TILE_WIDTH,
                 (m + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatMul<<<dimGrid, dimBlock>>>(dA, dB, dKernelC, m, n, k);
    checkCuda(cudaGetLastError(), "kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "kernel execution failed");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate failed");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS assumes column-major matrices, so compute C^T = B^T * A^T.
    checkCublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    k,
                    m,
                    n,
                    &alpha,
                    dB,
                    k,
                    dA,
                    n,
                    &beta,
                    dCublasC,
                    k),
        "cublasSgemm failed");

    checkCuda(cudaMemcpy(hKernelC, dKernelC, cSize * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy dKernelC -> hKernelC failed");
    checkCuda(cudaMemcpy(hCublasC, dCublasC, cSize * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy dCublasC -> hCublasC failed");

    bool matches = true;
    float maxAbsDiff = 0.0f;
    int mismatchIndex = -1;

    for (int i = 0; i < cSize; ++i) {
        float diff = std::fabs(hKernelC[i] - hCublasC[i]);
        if (diff > maxAbsDiff) {
            maxAbsDiff = diff;
        }
        if (diff > 1e-3f && mismatchIndex == -1) {
            matches = false;
            mismatchIndex = i;
        }
    }

    std::cout << "A shape: " << m << " x " << n << std::endl;
    std::cout << "B shape: " << n << " x " << k << std::endl;
    std::cout << "C shape: " << m << " x " << k << std::endl;
    std::cout << "Max absolute difference: " << maxAbsDiff << std::endl;

    if (matches) {
        std::cout << "Check passed: kernel output matches cuBLAS." << std::endl;
    } else {
        std::cout << "Check failed: kernel output does not match cuBLAS." << std::endl;
        std::cout << "First mismatch index: " << mismatchIndex << std::endl;
        std::cout << "Kernel value: " << hKernelC[mismatchIndex] << std::endl;
        std::cout << "cuBLAS value: " << hCublasC[mismatchIndex] << std::endl;
    }

    checkCublas(cublasDestroy(handle), "cublasDestroy failed");
    checkCuda(cudaFree(dA), "cudaFree dA failed");
    checkCuda(cudaFree(dB), "cudaFree dB failed");
    checkCuda(cudaFree(dKernelC), "cudaFree dKernelC failed");
    checkCuda(cudaFree(dCublasC), "cudaFree dCublasC failed");

    delete[] hA;
    delete[] hB;
    delete[] hKernelC;
    delete[] hCublasC;

    return 0;
}
