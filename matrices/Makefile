CC      = gcc
NVCC    = $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CFLAGS  = -O2 -Wall
LDFLAGS = -lm
PYTHON  = .venv/bin/python
N      ?= 512

all: naive strassen omp cuda cudablas

naive: naive.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

strassen: strassen.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

omp: omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

cuda: cuda.cu
	$(NVCC) -O2 -o $@ $<

cudablas: cudablas.cu
	$(NVCC) -O2 -o $@ $< -lcublas

py-naive:
	$(PYTHON) naive_numpy.py $(N)

py-torch:
	$(PYTHON) torch_cuda.py $(N)

clean:
	rm -f naive strassen omp cuda cudablas

.PHONY: all clean py-naive py-torch cudablas
