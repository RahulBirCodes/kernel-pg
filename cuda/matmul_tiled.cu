#include <iostream>
#include <cuda_runtime.h>

__global__ void tiledMatMul(float* a, float* b, float* c, int m, int k, int n) {
    extern __shared__ float shared[];

    int tileWidth = blockDim.x;
    float* aTile = shared;
    float* bTile = shared + tileWidth * tileWidth;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * tileWidth + ty;
    int col = bx * tileWidth + tx;

    float cValue = 0.0f;

    int numPhases = (k + tileWidth - 1) / tileWidth;

    for (int ph = 0; ph < numPhases; ++ph) {
        int aCol = ph * tileWidth + tx;
        int bRow = ph * tileWidth + ty;

        if (row < m && aCol < k) {
            aTile[ty * tileWidth + tx] = a[row * k + aCol];
        } else {
            aTile[ty * tileWidth + tx] = 0.0f;
        }

        if (bRow < k && col < n) {
            bTile[ty * tileWidth + tx] = b[bRow * n + col];
        } else {
            bTile[ty * tileWidth + tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < tileWidth; ++i) {
            cValue += aTile[ty * tileWidth + i] * bTile[i * tileWidth + tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = cValue;
    }
}

int main() {
    const int m = 5;
    const int k = 7;
    const int n = 6;
    const int tileWidth = 4;

    const int aSize = m * k;
    const int bSize = k * n;
    const int cSize = m * n;

    float hA[aSize];
    float hB[bSize];
    float hC[cSize];

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            hA[row * k + col] = static_cast<float>(row * k + col + 1);
        }
    }

    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            hB[row * n + col] = static_cast<float>(row * n + col + 1);
        }
    }

    float* dA;
    float* dB;
    float* dC;

    cudaMalloc((void**)&dA, aSize * sizeof(float));
    cudaMalloc((void**)&dB, bSize * sizeof(float));
    cudaMalloc((void**)&dC, cSize * sizeof(float));

    cudaMemcpy(dA, hA, aSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(tileWidth, tileWidth);
    dim3 gridSize((n + tileWidth - 1) / tileWidth,
                  (m + tileWidth - 1) / tileWidth);

    size_t sharedBytes = 2 * tileWidth * tileWidth * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Shared memory available per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared memory used by one block: " << sharedBytes << " bytes" << std::endl;
    std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
    std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;

    tiledMatMul<<<gridSize, blockSize, sharedBytes>>>(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, cSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    std::cout << "Matrix A (" << m << "x" << k << "):" << std::endl;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            std::cout << hA[row * k + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix B (" << k << "x" << n << "):" << std::endl;
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << hB[row * n + col] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Matrix C = A x B (" << m << "x" << n << "):" << std::endl;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << hC[row * n + col] << "\t";
        }
        std::cout << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
