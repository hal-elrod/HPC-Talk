#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*
 * Each thread computes one element C[i][j].
 * Mirrors the OMP approach: the i and j loops become the thread grid,
 * and the k loop runs serially inside each thread.
 */
__global__ static void multiply_kernel(const double *A, const double *B,
                                       double *C, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    double sum = 0.0;
    for (int k = 0; k < n; k++)
        sum += A[i * n + k] * B[k * n + j];
    C[i * n + j] = sum;
}

static void multiply(const double *A, const double *B, double *C, int n) {
    size_t bytes = (size_t)n * n * sizeof(double);

    double *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    /* 16×16 thread block → each block covers a 16×16 tile of the output */
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    multiply_kernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
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

    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    double *C = (double *)malloc((size_t)n * n * sizeof(double));
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

    /* Warm up the GPU before timing */
    cudaFree(0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    multiply(A, B, C, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("n=%d  gpu=\"%s\"  time=%.6f s\n", n, prop.name, elapsed);

    free(A); free(B); free(C);
    return 0;
}
