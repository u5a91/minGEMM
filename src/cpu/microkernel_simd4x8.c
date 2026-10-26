
#include <immintrin.h>
#include <stddef.h>

void matmul(
    const float *restrict A,
    const float *restrict B,
    float *restrict C,
    int M, int N, int K
)
{
    int i = 0;

    // 4 rows at a time
    for (; i + 4 <= M; i += 4) {
        int j = 0;

        for (; j + 8 <= N; j += 8) {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                __m256 b = _mm256_loadu_ps(
                    &B[k * N + j]
                );

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

        for (; j < N; j++) {
            for (int r = 0; r < 4; r++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += A[(i + r) * K + k] * B[k * N + j];
                }
                C[(i + r) * N + j] = acc;
            }
        }
    }

    // Remaining rows
    for (; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}
