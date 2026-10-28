#include <immintrin.h>
#include <stddef.h>

#ifndef BS
#define BS 128
#endif

void matmul(
    const float *restrict A,
    const float *restrict B,
    float *restrict C,
    int M, int N, int K
)
{
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (int ii = 0; ii < M; ii += BS) {
        for (int kk = 0; kk < K; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {

                int i_end = (ii + BS < M) ? ii + BS : M;
                int k_end = (kk + BS < K) ? kk + BS : K;
                int j_end = (jj + BS < N) ? jj + BS : N;

                int i = ii;

                // Process 4 rows at a time
                for (; i + 4 <= i_end; i += 4) {
                    int j = jj;

                    // Process 8 columns at a time
                    for (; j + 8 <= j_end; j += 8) {
                        __m256 c0 = _mm256_loadu_ps(&C[(i + 0) * N + j]);
                        __m256 c1 = _mm256_loadu_ps(&C[(i + 1) * N + j]);
                        __m256 c2 = _mm256_loadu_ps(&C[(i + 2) * N + j]);
                        __m256 c3 = _mm256_loadu_ps(&C[(i + 3) * N + j]);

                        for (int k = kk; k < k_end; k++) {
                            __m256 b = _mm256_loadu_ps(&B[k * N + j]);

                            __m256 a0 = _mm256_set1_ps(A[(i + 0) * K + k]);
                            __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);
                            __m256 a2 = _mm256_set1_ps(A[(i + 2) * K + k]);
                            __m256 a3 = _mm256_set1_ps(A[(i + 3) * K + k]);

                            c0 = _mm256_fmadd_ps(a0, b, c0);
                            c1 = _mm256_fmadd_ps(a1, b, c1);
                            c2 = _mm256_fmadd_ps(a2, b, c2);
                            c3 = _mm256_fmadd_ps(a3, b, c3);
                        }

                        _mm256_storeu_ps(&C[(i + 0) * N + j], c0);
                        _mm256_storeu_ps(&C[(i + 1) * N + j], c1);
                        _mm256_storeu_ps(&C[(i + 2) * N + j], c2);
                        _mm256_storeu_ps(&C[(i + 3) * N + j], c3);
                    }

                    // Remaining columns for these 4 rows
                    for (; j < j_end; j++) {
                        for (int r = 0; r < 4; r++) {
                            float acc = C[(i + r) * N + j];
                            for (int k = kk; k < k_end; k++) {
                                acc += A[(i + r) * K + k] * B[k * N + j];
                            }
                            C[(i + r) * N + j] = acc;
                        }
                    }
                }

                // Remaining rows
                for (; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        float acc = C[i * N + j];
                        for (int k = kk; k < k_end; k++) {
                            acc += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = acc;
                    }
                }
            }
        }
    }
}
