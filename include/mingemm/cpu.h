#ifndef MINGEMM_CPU_H
#define MINGEMM_CPU_H

/*
 * restrict qualifier that works from both C and C++ (including CUDA host code).
 */
#if defined(__cplusplus)
#  if defined(_MSC_VER)
#    define MINGEMM_RESTRICT __restrict
#  else
#    define MINGEMM_RESTRICT __restrict__
#  endif
#elif defined(_MSC_VER)
#  define MINGEMM_RESTRICT __restrict
#else
#  define MINGEMM_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * C = A * B
 *
 * A: M x K, row-major, contiguous, FP32
 * B: K x N, row-major, contiguous, FP32
 * C: M x N, row-major, contiguous, FP32
 *
 * C is overwritten, not accumulated. A, B, and C must not overlap.
 * M, N, and K must be non-negative; zero-sized dimensions are allowed.
 * This function performs no memory allocation.
 */
void matmul(
    const float *MINGEMM_RESTRICT A,
    const float *MINGEMM_RESTRICT B,
    float *MINGEMM_RESTRICT C,
    int M, int N, int K
);

#ifdef __cplusplus
}
#endif

#endif /* MINGEMM_CPU_H */
