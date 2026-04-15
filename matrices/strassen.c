#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Switch to naive below this size to avoid recursion overhead */
#define THRESHOLD 64

static void naive_mul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

static void mat_add(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

static void mat_sub(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

/* Copy h×h block at (r,c) of an m-wide matrix into a contiguous h×h buffer */
static void extract(const double *src, double *dst, int m, int r, int c) {
    int h = m / 2;
    for (int i = 0; i < h; i++)
        memcpy(dst + i * h, src + (i + r) * m + c, h * sizeof(double));
}

/* Write contiguous h×h buffer into block (r,c) of an m-wide matrix */
static void insert(const double *src, double *dst, int m, int r, int c) {
    int h = m / 2;
    for (int i = 0; i < h; i++)
        memcpy(dst + (i + r) * m + c, src + i * h, h * sizeof(double));
}

/*
 * Strassen on m×m matrices where m is a power of two.
 * All matrices are contiguous row-major buffers of size m*m.
 */
static void strassen(const double *A, const double *B, double *C, int m) {
    if (m <= THRESHOLD) {
        naive_mul(A, B, C, m);
        return;
    }

    int h = m / 2;
    size_t sz = (size_t)h * h;

    /*
     * Single allocation: 21 sub-matrices of size h×h.
     * Layout: A11 A12 A21 A22  B11 B12 B21 B22
     *         M1..M7  T1 T2  C11 C12 C21 C22
     */
    double *buf = malloc(21 * sz * sizeof(double));
    if (!buf) { fprintf(stderr, "OOM at m=%d\n", m); exit(1); }

    double *A11=buf+ 0*sz, *A12=buf+ 1*sz, *A21=buf+ 2*sz, *A22=buf+ 3*sz;
    double *B11=buf+ 4*sz, *B12=buf+ 5*sz, *B21=buf+ 6*sz, *B22=buf+ 7*sz;
    double *M1 =buf+ 8*sz, *M2 =buf+ 9*sz, *M3 =buf+10*sz, *M4 =buf+11*sz;
    double *M5 =buf+12*sz, *M6 =buf+13*sz, *M7 =buf+14*sz;
    double *T1 =buf+15*sz, *T2 =buf+16*sz;
    double *C11=buf+17*sz, *C12=buf+18*sz, *C21=buf+19*sz, *C22=buf+20*sz;

    extract(A, A11, m, 0, 0); extract(A, A12, m, 0, h);
    extract(A, A21, m, h, 0); extract(A, A22, m, h, h);
    extract(B, B11, m, 0, 0); extract(B, B12, m, 0, h);
    extract(B, B21, m, h, 0); extract(B, B22, m, h, h);

    /* M1 = (A11+A22)(B11+B22) */
    mat_add(A11, A22, T1, h); mat_add(B11, B22, T2, h);
    strassen(T1, T2, M1, h);

    /* M2 = (A21+A22)B11 */
    mat_add(A21, A22, T1, h);
    strassen(T1, B11, M2, h);

    /* M3 = A11(B12-B22) */
    mat_sub(B12, B22, T1, h);
    strassen(A11, T1, M3, h);

    /* M4 = A22(B21-B11) */
    mat_sub(B21, B11, T1, h);
    strassen(A22, T1, M4, h);

    /* M5 = (A11+A12)B22 */
    mat_add(A11, A12, T1, h);
    strassen(T1, B22, M5, h);

    /* M6 = (A21-A11)(B11+B12) */
    mat_sub(A21, A11, T1, h); mat_add(B11, B12, T2, h);
    strassen(T1, T2, M6, h);

    /* M7 = (A12-A22)(B21+B22) */
    mat_sub(A12, A22, T1, h); mat_add(B21, B22, T2, h);
    strassen(T1, T2, M7, h);

    /* C11 = M1+M4-M5+M7 */
    mat_add(M1, M4, T1, h); mat_sub(T1, M5, T2, h); mat_add(T2, M7, C11, h);
    /* C12 = M3+M5 */
    mat_add(M3, M5, C12, h);
    /* C21 = M2+M4 */
    mat_add(M2, M4, C21, h);
    /* C22 = M1-M2+M3+M6 */
    mat_sub(M1, M2, T1, h); mat_add(T1, M3, T2, h); mat_add(T2, M6, C22, h);

    insert(C11, C, m, 0, 0); insert(C12, C, m, 0, h);
    insert(C21, C, m, h, 0); insert(C22, C, m, h, h);

    free(buf);
}

/* Round up to the next power of two */
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void multiply(const double *A, const double *B, double *C, int n) {
    int m = next_pow2(n);

    if (m == n) {
        strassen(A, B, C, n);
        return;
    }

    /* Pad A and B to m×m with zeros, run Strassen, extract n×n result */
    double *Ap = calloc((size_t)m * m, sizeof(double));
    double *Bp = calloc((size_t)m * m, sizeof(double));
    double *Cp = calloc((size_t)m * m, sizeof(double));
    if (!Ap || !Bp || !Cp) { fprintf(stderr, "OOM\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        memcpy(Ap + i * m, A + i * n, n * sizeof(double));
        memcpy(Bp + i * m, B + i * n, n * sizeof(double));
    }

    strassen(Ap, Bp, Cp, m);

    for (int i = 0; i < n; i++)
        memcpy(C + i * n, Cp + i * m, n * sizeof(double));

    free(Ap); free(Bp); free(Cp);
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
    printf("n=%d  time=%.6f s\n", n, elapsed);

    free(A); free(B); free(C);
    return 0;
}
