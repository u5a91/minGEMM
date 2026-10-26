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

                for (int i = ii; i < i_end; i++) {
                    int j = jj;
                    for (; j + 7 < j_end; j += 8) {

                        float c0 = C[i*N + j + 0];
                        float c1 = C[i*N + j + 1];
                        float c2 = C[i*N + j + 2];
                        float c3 = C[i*N + j + 3];
                        float c4 = C[i*N + j + 4];
                        float c5 = C[i*N + j + 5];
                        float c6 = C[i*N + j + 6];
                        float c7 = C[i*N + j + 7];

                        for (int k = kk; k < k_end; k++) {
                            float a = A[i*K + k];

                            c0 += a * B[k*N + j + 0];
                            c1 += a * B[k*N + j + 1];
                            c2 += a * B[k*N + j + 2];
                            c3 += a * B[k*N + j + 3];
                            c4 += a * B[k*N + j + 4];
                            c5 += a * B[k*N + j + 5];
                            c6 += a * B[k*N + j + 6];
                            c7 += a * B[k*N + j + 7];
                        }

                        C[i*N + j + 0] = c0;
                        C[i*N + j + 1] = c1;
                        C[i*N + j + 2] = c2;
                        C[i*N + j + 3] = c3;
                        C[i*N + j + 4] = c4;
                        C[i*N + j + 5] = c5;
                        C[i*N + j + 6] = c6;
                        C[i*N + j + 7] = c7;
                    }

                    for (; j < j_end; ++j) {
                        float acc = C[i * N + j];
                        for (int k = kk; k < k_end; ++k) {
                            acc += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = acc;
                    }
                }
            }
        }
    }
}
