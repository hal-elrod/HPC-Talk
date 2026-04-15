#!/usr/bin/env python3
import sys
import time

import numpy as np


def multiply(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> None:
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += float(a[i, k]) * float(b[k, j])
            c[i, j] = s


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <n>", file=sys.stderr)
        return 1

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("n must be a positive integer", file=sys.stderr)
        return 1

    if n <= 0:
        print("n must be a positive integer", file=sys.stderr)
        return 1

    rng = np.random.default_rng(seed=int(time.time()))
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)
    c = np.empty((n, n), dtype=np.float64)

    t0 = time.perf_counter()
    multiply(a, b, c, n)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"n={n}  time={elapsed:.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
