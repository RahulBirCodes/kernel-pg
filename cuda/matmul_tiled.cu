#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void tiledMatMul(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

int main() {
    const int Width = 16;
    const int size = Width * Width;

    float* hM = new float[size];
    float* hN = new float[size];
    float* hP = new float[size];

    std::srand(1234);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            hM[row * Width + col] = static_cast<float>(std::rand() % 10);
            hN[row * Width + col] = (row == col) ? 1.0f : 0.0f;
        }
    }

    float* dM;
    float* dN;
    float* dP;

    cudaMalloc((void**)&dM, size * sizeof(float));
    cudaMalloc((void**)&dN, size * sizeof(float));
    cudaMalloc((void**)&dP, size * sizeof(float));

    cudaMemcpy(dM, hM, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dN, hN, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

    tiledMatMul<<<dimGrid, dimBlock>>>(dM, dN, dP, Width);
    cudaDeviceSynchronize();

    cudaMemcpy(hP, dP, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Random matrix M:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hM[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Identity matrix N:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hN[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix P = M x N:" << std::endl;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            std::cout << hP[row * Width + col] << "\t";
        }
        std::cout << std::endl;
    }

    bool matches = true;
    for (int i = 0; i < size; ++i) {
        if (static_cast<int>(hM[i]) != static_cast<int>(hP[i])) {
            matches = false;
            break;
        }
    }

    std::cout << std::endl;
    if (matches) {
        std::cout << "Check passed: hM and hP match." << std::endl;
    } else {
        std::cout << "Check failed: hM and hP do not match." << std::endl;
    }

    cudaFree(dM);
    cudaFree(dN);
    cudaFree(dP);
    delete[] hM;
    delete[] hN;
    delete[] hP;

    return 0;
}
