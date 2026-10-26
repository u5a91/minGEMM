#include <stddef.h>

void matmul(
    const float *restrict A,
    const float *restrict B,
    float *restrict C,
    int M, int N, int K
)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
