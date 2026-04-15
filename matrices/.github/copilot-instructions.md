# Copilot Instructions

## Build Commands

```sh
make          # build all five binaries: naive, strassen, omp, cuda, cudablas
make naive    # build a single target
make clean    # remove all binaries
make py-naive N=512  # run NumPy naive Python implementation
make py-torch N=512  # run PyTorch CUDA Python implementation
```

Requires `gcc`, `nvcc` (CUDA toolkit), and an OpenMP-capable compiler. The `cuda` target can be skipped if no GPU is available.

## Python Dependencies

```sh
python3 -m pip install numpy
python3 -m pip install torch
```

`torch_cuda.py` requires a CUDA-enabled PyTorch install and an available CUDA GPU.

## Running / Benchmarking

Each binary takes a single argument — the matrix dimension `n` — and prints timing to stdout:

```sh
./naive 512
./strassen 512
./omp 512
OMP_NUM_THREADS=4 ./omp 512
./cuda 512
./cudablas 512
python3 naive_numpy.py 512
python3 torch_cuda.py 512
```

There is no automated test suite. Correctness is checked manually by comparing outputs across implementations for the same `n`.

## Architecture

Seven independent implementations of square matrix multiplication:

| File | Algorithm | Key technique |
|------|-----------|---------------|
| `naive.c` | O(n³) triple loop | Baseline reference |
| `strassen.c` | Strassen's algorithm | Recursive, falls back to naive below `THRESHOLD 64`; pads non-power-of-2 inputs to the next power of two |
| `omp.c` | Naïve loop + OpenMP | Transposes B before the parallel region for cache-friendly access; uses `collapse(2) schedule(static)` |
| `cuda.cu` | One thread per output element | 16×16 thread blocks; GPU warm-up (`cudaFree(0)`) done before timing |
| `cudablas.cu` | cuBLAS DGEMM | Calls optimized CUDA Basic Linear Algebra Subprograms routine for GPU-accelerated matrix multiply |
| `naive_numpy.py` | O(n³) triple loop using NumPy arrays | Keeps naive loop structure while storing matrices as `numpy.float64` arrays |
| `torch_cuda.py` | GPU-accelerated matmul using PyTorch | Uses CUDA tensors with warm-up and synchronization around timed multiply |

## Key Conventions

- All matrices are flat, row-major `double *` buffers of size `n*n`. Element `[i][j]` is at index `i*n + j`.
- Timing uses `clock_gettime(CLOCK_MONOTONIC)` and covers only the `multiply()` call, not allocation or initialization.
- `main()` is identical across all four files (allocate → fill with `rand()/RAND_MAX` → time → print → free). Core logic lives in a static `multiply()` function.
- Strassen's single-allocation strategy: 21 sub-matrices of size `h×h` are carved out of one `malloc(21 * h*h * sizeof(double))` block to avoid 21 separate allocations per recursion level.
- CUDA kernel threads map `blockIdx.x/threadIdx.x → j` (column) and `blockIdx.y/threadIdx.y → i` (row), mirroring the OMP `i`/`j` loop nesting.
- Compiler flags: `-O2 -Wall` for GCC targets; `-O2` for nvcc; `-fopenmp` only on the `omp` target.
