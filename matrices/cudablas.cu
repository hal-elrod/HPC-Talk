#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void multiply_gpu(cublasHandle_t handle, const double *dA, const double *dB,
                         double *dC, int n) {
    double alpha = 1.0, beta = 0.0;
    /* For row-major data, swap operands: compute what cuBLAS sees as B*A,
       which is the correct A*B for row-major. */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha, dA, n, dB, n,
                &beta, dC, n);
    cudaDeviceSynchronize();
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

    size_t bytes = (size_t)n * n * sizeof(double);

    /* Host memory for initialization only */
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        free(A); free(B); free(C);
        return 1;
    }

    /* Fill host matrices with random data (not timed) */
    srand((unsigned)time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    /* Allocate GPU memory (not timed) */
    double *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    /* Setup cuBLAS */
    cublasHandle_t handle;
    cublasCreate(&handle);

    /* Warm up the GPU before timing */
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
    multiply_gpu(handle, dA, dB, dC, n);
    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    /* Time entire operation: H2D transfer + compute + D2H transfer */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
    multiply_gpu(handle, dA, dB, dC, n);
    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("n=%d  gpu=\"%s\"  time=%.6f s\n", n, prop.name, elapsed);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}

