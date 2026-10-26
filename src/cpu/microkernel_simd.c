#include <immintrin.h>
#include <stddef.h>

void matmul(
    const float *restrict A,
    const float *restrict B,
    float *restrict C,
    int M, int N, int K
)
{
    for (int i = 0; i < M; i++) {
        int jj = 0;

        // SIMD: 8 columns at a time
        for (; jj + 8 <= N; jj += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 a = _mm256_set1_ps(A[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + jj]);

                acc = _mm256_fmadd_ps(a, b, acc);
            }

            _mm256_storeu_ps(&C[i * N + jj], acc);
        }

        // Remaining columns
        for (; jj < N; jj++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + jj];
            }
            C[i * N + jj] = acc;
        }
    }
}
