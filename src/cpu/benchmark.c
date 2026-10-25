#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K
);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random(float *x, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        x[i] = (float)rand() / (float) RAND_MAX;
    }
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s n\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        fprintf(stderr, "n must be positive\n");
        return 1;
    }

    size_t size = (size_t)n * n;

    float *A = malloc(size * sizeof(float));
    float *B = malloc(size * sizeof(float));
    float *C = malloc(size * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    srand(0);

    fill_random(A, size);
    fill_random(B, size);

    double t0 = now_sec();

    matmul(A, B, C, n, n, n);

    double t1 = now_sec();

    double sec = t1 - t0;
    double flop_count = 2.0 * n * n * n;
    double gflops = flop_count / sec / 1e9;

    printf("n = %d\n", n);
    printf("time = %.6f s\n", sec);
    printf("performance = %.3f GFLOPS/s\n", gflops);

    /* compiler の最適化が計算全体を消すのを防ぐダミー処理 */
    printf("C[0] = %f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
