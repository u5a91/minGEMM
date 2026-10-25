#include <stddef.h>

#ifndef BS
#define BS 128
#endif

void matmul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K
)
{
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (int ii = 0; ii < M; ii += BS) {
        for (int kk = 0; kk < K; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int i_end = (ii + BS < M) ? ii + BS : M;
                int j_end = (jj + BS < N) ? jj + BS : N;
                int k_end = (kk + BS < K) ? kk + BS : K;

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a = A[i * K + k];
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
