#include <stddef.h>

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

    for (int i = 0; i < M; i++) {
        int jj = 0;

        for (; jj + 7 < N; jj += 8) {
            float c0 = C[i * N + jj + 0];
            float c1 = C[i * N + jj + 1];
            float c2 = C[i * N + jj + 2];
            float c3 = C[i * N + jj + 3];
            float c4 = C[i * N + jj + 4];
            float c5 = C[i * N + jj + 5];
            float c6 = C[i * N + jj + 6];
            float c7 = C[i * N + jj + 7];

            for (int k = 0; k < K; k++) {
                float a = A[i * K + k];

                c0 += a * B[k * N + jj + 0];
                c1 += a * B[k * N + jj + 1];
                c2 += a * B[k * N + jj + 2];
                c3 += a * B[k * N + jj + 3];
                c4 += a * B[k * N + jj + 4];
                c5 += a * B[k * N + jj + 5];
                c6 += a * B[k * N + jj + 6];
                c7 += a * B[k * N + jj + 7];
            }

            C[i * N + jj + 0] = c0;
            C[i * N + jj + 1] = c1;
            C[i * N + jj + 2] = c2;
            C[i * N + jj + 3] = c3;
            C[i * N + jj + 4] = c4;
            C[i * N + jj + 5] = c5;
            C[i * N + jj + 6] = c6;
            C[i * N + jj + 7] = c7;
        }

        for (; jj < N; jj++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + jj];
            }
            C[i * N + jj] = acc;
        }
    }
}
