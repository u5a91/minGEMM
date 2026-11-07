#ifndef MINGEMM_CUDA_CUH
#define MINGEMM_CUDA_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Asynchronous CUDA GEMM: C = A * B
 *
 * A: M x K device pointer, row-major, contiguous, FP32
 * B: K x N device pointer, row-major, contiguous, FP32
 * C: M x N device pointer, row-major, contiguous, FP32
 *
 * C is overwritten, not accumulated. A, B, and C must not overlap.
 * M, N, and K must be non-negative; zero-sized dimensions are allowed.
 *
 * The caller owns all device buffers. The operation is enqueued on `stream`
 * and is not synchronized by this function. The caller must check
 * asynchronous execution errors (for example with cudaStreamSynchronize).
 *
 * Returns cudaSuccess if the operation was enqueued, or an error from
 * validation, cudaMemsetAsync, or kernel launch.
 */
cudaError_t matmul_cuda_naive_async(
    const float *dA,
    const float *dB,
    float *dC,
    int M, int N, int K,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* MINGEMM_CUDA_CUH */
