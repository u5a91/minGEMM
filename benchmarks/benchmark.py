import ctypes
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sizes = [256, 512, 1024]
warmups = 3
runs = 10

for n in sizes:
    print(f"n = {n}")

    rng = np.random.default_rng(42)
    A = rng.random((n, n), dtype=np.float32)
    B = rng.random((n, n), dtype=np.float32)
    C = np.empty((n, n), dtype=np.float32)

    for lib_path in sys.argv[1:]:
        lib = ctypes.CDLL(lib_path)
        matmul = lib.matmul

        float_ptr = ctypes.POINTER(ctypes.c_float)
        matmul.argtypes = [
            float_ptr, float_ptr, float_ptr,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        matmul.restype = None

        def run():
            matmul(
                A.ctypes.data_as(float_ptr),
                B.ctypes.data_as(float_ptr),
                C.ctypes.data_as(float_ptr),
                n, n, n,
            )

        for _ in range(warmups):
            run()

        times = []

        for _ in range(runs):
            start = time.perf_counter_ns()
            run()
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed * 1e-9)

        flops = 2 * n**3
        performances = [flops / t / 1e9 for t in times]

        median_time = statistics.median(times)
        median_gflops = flops / median_time / 1e9

        print(f"  {Path(lib_path).stem}")
        print(f"    runs   = {[round(p, 3) for p in performances]}")
        print(f"    time   = {median_time:.6f} s")
        print(f"    median = {median_gflops:.3f} GFLOP/s")
