#!/usr/bin/env python3
import sys
import time

import torch


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

    if not torch.cuda.is_available():
        print("CUDA is required for this program, but no CUDA device is available", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create data on CPU first
    a_cpu = torch.rand((n, n), dtype=torch.float64)
    b_cpu = torch.rand((n, n), dtype=torch.float64)

    # Warm up GPU before timing: transfer to GPU, compute, transfer back
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = torch.matmul(a, b)
    c_cpu = c.to("cpu")
    torch.cuda.synchronize(device)

    # Time entire operation: H2D transfer + compute + D2H transfer
    t0 = time.perf_counter()
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = torch.matmul(a, b)
    c_cpu = c.to("cpu")
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    gpu_name = torch.cuda.get_device_name(device)
    print(f'n={n}  gpu="{gpu_name}"  time={elapsed:.6f} s')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
