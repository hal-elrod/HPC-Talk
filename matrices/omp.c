#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static void multiply(const double *A, const double *B, double *C, int n) {
    /* Transpose B so the inner k-loop accesses BT row-wise (cache-friendly) */
    double *BT = malloc((size_t)n * n * sizeof(double));
    if (!BT) { fprintf(stderr, "Allocation failed\n"); exit(1); }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j * n + i] = B[i * n + j];

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * BT[j * n + k];
            C[i * n + j] = sum;
        }

    free(BT);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "n must be a positive integer\n");
        return 1;
    }

    double *A = malloc((size_t)n * n * sizeof(double));
    double *B = malloc((size_t)n * n * sizeof(double));
    double *C = malloc((size_t)n * n * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(A); free(B); free(C);
        return 1;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    multiply(A, B, C, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("n=%d  threads=%d  time=%.6f s\n", n, omp_get_max_threads(), elapsed);

    free(A); free(B); free(C);
    return 0;
}
