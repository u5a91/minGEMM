#include <cuda_runtime.h>

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "mingemm/cuda.cuh"

#define TILE 16

/*
 * One thread computes one element of C:
 *
 *   C[i][j] = sum_k A[i][k] * B[k][j]
 *
 * Threads are laid out so that threadIdx.x maps to the column j, which makes
 * the loads of B (and the stores of C) coalesced. The A loads are the same for
 * every thread in a row, so the hardware broadcasts them.
 */
__global__ void matmul_naive_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= N) {
        return;
    }

    const float *a = A + (size_t)i * K;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += a[k] * B[(size_t)k * N + j];
    }

    C[(size_t)i * N + j] = sum;
}

static int cuda_ok(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "mingemm cuda: %s failed: %s\n",
                what, cudaGetErrorString(err));
        return 0;
    }

    return 1;
}

cudaError_t matmul_cuda_naive_async(
    const float *dA,
    const float *dB,
    float *dC,
    int M, int N, int K,
    cudaStream_t stream
)
{
    if (M < 0 || N < 0 || K < 0) {
        return cudaErrorInvalidValue;
    }

    if (M == 0 || N == 0) {
        return cudaSuccess;
    }

    if (K == 0) {
        return cudaMemsetAsync(dC, 0, (size_t)M * N * sizeof(float), stream);
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);

    return cudaGetLastError();
}

/*
 * Synchronous host-copy adapter used by the shared Python tests and benchmark.
 * The device-pointer async entry point is matmul_cuda_naive_async().
 */
extern "C" void matmul(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    if (M < 0 || N < 0 || K < 0) {
        return;
    }

    if (M == 0 || N == 0) {
        return;
    }

    size_t size_a = (size_t)M * K;
    size_t size_b = (size_t)K * N;
    size_t size_c = (size_t)M * N;

    if (K == 0) {
        memset(C, 0, size_c * sizeof(float));
        return;
    }

    float *dA = NULL;
    float *dB = NULL;
    float *dC = NULL;

    int ok = (cuda_ok(cudaMalloc((void **)&dA, size_a * sizeof(float)), "cudaMalloc(A)")
            && cuda_ok(cudaMalloc((void **)&dB, size_b * sizeof(float)), "cudaMalloc(B)")
            && cuda_ok(cudaMalloc((void **)&dC, size_c * sizeof(float)), "cudaMalloc(C)"));

    ok = ok
         && cuda_ok(cudaMemcpy(dA, A, size_a * sizeof(float),
                            cudaMemcpyHostToDevice), "cudaMemcpy(A)")
         && cuda_ok(cudaMemcpy(dB, B, size_b * sizeof(float),
                            cudaMemcpyHostToDevice), "cudaMemcpy(B)");

    if (ok) {
        ok = cuda_ok(matmul_cuda_naive_async(dA, dB, dC, M, N, K, 0),
                     "matmul_cuda_naive_async");
    }

    ok = ok
         && cuda_ok(cudaMemcpy(C, dC, size_c * sizeof(float),
                            cudaMemcpyDeviceToHost), "cudaMemcpy(C)");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    if (!ok) {
        cudaGetLastError(); /* clear the sticky error state */
    }
}
