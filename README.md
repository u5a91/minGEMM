# minGEMM

A small, educational implementation of **FP32 general matrix multiplication (GEMM)** in C and CUDA.

The goal is to understand how implementation choices—loop ordering, cache blocking, microkernels, SIMD, and GPU thread mapping—affect matrix multiplication performance. This repository is a learning and benchmarking project, **not a replacement for optimized BLAS libraries**.

The implementations compute

$$
C = AB, \qquad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N},\; C \in \mathbb{R}^{M \times N}.
$$

Matrices are single-precision (`float32`) and stored contiguously in row-major order. The CPU variants and the CUDA host adapter expose a `matmul` entry point for the Python test and benchmark harness; the CUDA device API is exposed separately as `matmul_cuda_naive_async`.

## Implementations

### CPU (C)

| Variant | Idea |
| --- | --- |
| `naive` | Straightforward triple-loop baseline |
| `ikj` | Loop reordering to change memory access patterns |
| `blocked` | Cache blocking |
| `microkernel` | Small output-tile microkernel |
| `microkernel_simd` | SIMD microkernel |
| `microkernel_simd4x8` | 4x8 SIMD microkernel |
| `blocked_microkernel` | Cache blocking with a microkernel |
| `blocked_microkernel_simd4x8` | Cache blocking with a 4x8 SIMD microkernel |
| `bs16`, `bs32`, `bs64`, `bs128`, `bs256`, `bs512` | Variants of `blocked` with different block sizes |

### GPU (CUDA)

The current `naive_cuda` implementation assigns **one output element to each CUDA thread**. The x dimension of the thread block maps to output columns, so adjacent threads access adjacent elements of `B` and `C`. The kernel uses a 16x16 thread block and does not yet use shared-memory tiling.

`naive_cuda.so` exports two entry points:

- `matmul_cuda_naive_async` in [`include/mingemm/cuda.cuh`](include/mingemm/cuda.cuh) takes device pointers and a CUDA stream. It enqueues the operation without synchronizing the stream.
- A synchronous host-pointer `matmul` adapter supports the same Python tests and benchmark harness as the CPU variants. This adapter is a test/benchmark shim, not part of the public CUDA API. It allocates device buffers, transfers inputs and outputs, and synchronizes execution.

CUDA performance is measured in two ways. The C++ driver in [`src/cuda/benchmark.cu`](src/cuda/benchmark.cu) (`make bench-cuda`) allocates device buffers once and times only kernel execution with CUDA events, excluding host-to-device and device-to-host copies. The Python harness (`make bench-cuda-e2e`) goes through the synchronous host-copy adapter, so its numbers include allocation, transfers, and synchronization.

## Build and run

### Requirements

- x86-64 Linux environment with GCC and Make for CPU builds; the Makefile uses `-march=native`, and the SIMD variants require AVX2/FMA
- Python 3 and the dependencies listed in [`requirements.txt`](requirements.txt) (`numpy`, `pytest`)
- For CUDA: an NVIDIA GPU, CUDA Toolkit (`nvcc`), and a compatible driver

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# CPU
make
make test
make bench

# CUDA (optional)
make cuda
make test-cuda
make bench-cuda        # kernel-only throughput
make bench-cuda-e2e    # end-to-end throughput (host-copy adapter)
```

The Makefile currently defaults to `CUDA_ARCH=-arch=sm_86` (used for the RTX A5000). Override it for another GPU, for example:

```bash
make cuda CUDA_ARCH=-arch=sm_80
```

`make test` runs the CPU variants, and `make test-cuda` runs the CUDA variant. Both compare against NumPy matrix multiplication over the same 15 shapes, including non-square matrices and zero-sized dimensions, using floating-point tolerances.

## Benchmarks

The benchmark uses square FP32 matrices of sizes **256, 512, and 1024**, with **3 warm-up runs** and **10 timed runs** per implementation. Throughput is calculated from the median elapsed time:

$$
\text{GFLOP/s} = \frac{2n^3}{10^9 \cdot \text{median}(\text{elapsed seconds})}.
$$

Values below are from individual experimental sessions, not repeated independent benchmark sessions. Performance is hardware- and environment-dependent.

### CPU results

**Environment:** Intel Core i5-1135G7 (11th Gen, 2.40 GHz); Linux under WSL2. Values are in **GFLOP/s** (higher is better).

| Implementation | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: |
| `naive` | 0.689 | 0.629 | 0.400 |
| `ikj` | 17.279 | 7.215 | 8.555 |
| `blocked` | 14.560 | 7.168 | 14.438 |
| `microkernel_simd4x8` | 24.644 | 8.175 | 9.774 |
| `blocked_microkernel_simd4x8` | 19.330 | 8.125 | **23.353** |

For `n = 1024`, `blocked_microkernel_simd4x8` achieved roughly **50–60x** the measured throughput of `naive` in this run. Several variants show substantial run-to-run variation; these numbers illustrate the optimization experiments rather than stable peak performance.

The table shows selected implementations. Run `make bench` to benchmark all CPU variants, including the block-size sweep.

### CUDA results

Kernel-only timing is the primary CUDA metric because it isolates kernel
execution from host-side allocation and transfer costs. End-to-end timing is
reported separately because it depends heavily on the environment.

#### Kernel-only

**Environment:** NVIDIA RTX A5000; CUDA Toolkit 13.2; containerized Linux
environment. **Command:** `make bench-cuda` (3 warm-up launches; median of 10
timed launches).

| Implementation | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: |
| `naive_cuda` | 1369.792 | 1622.241 | 1701.090 |

Values are in **GFLOP/s**. The driver in
[`src/cuda/benchmark.cu`](src/cuda/benchmark.cu) allocates device buffers once
and times only the asynchronous kernel entry point with CUDA events, so device
allocation and H2D/D2H copies are excluded. At n = 1024 the naive kernel
reaches about 6% of the A5000's approximately 27.8 TFLOP/s FP32 peak, leaving
substantial room for shared-memory tiling and register blocking.

#### End-to-end

`make bench-cuda-e2e` goes through the synchronous host-copy adapter and
includes device allocation, H2D/D2H transfers, and synchronization. In a
containerized run on the same machine, the medians were 6.382 / 41.421 /
351.938 GFLOP/s at n = 256 / 512 / 1024, with individual runs at n = 256
varying by more than an order of magnitude (about 1.4–38 GFLOP/s).

Median wall-clock time stayed at roughly 5–6 ms across all sizes even though
the kernel alone took 0.024–1.262 ms. Because the adapter allocates and frees
device buffers on every call, these values are dominated by fixed host-side API
overhead in that environment rather than by kernel execution or transfer
volume. No end-to-end table is reported; use the kernel-only results to compare
implementations.

The CPU and GPU benchmarks were also run in different environments, so the
results should not be used to claim a controlled CPU-versus-GPU speedup.

## Project layout

```text
benchmarks/       Python benchmark harness
include/mingemm/  CPU and CUDA interface headers
src/cpu/          CPU implementation variants
src/cuda/         CUDA kernels and the kernel-only benchmark driver
tests/            NumPy-based correctness tests
Makefile          Build, test, and benchmark targets
```

## Next steps

- Implement a shared-memory tiled CUDA GEMM kernel and compare kernel-only throughput with the naive kernel.
- Compare with cuBLAS under matched data types, matrix sizes, and timing boundaries.
- Reuse device buffers in the host-copy adapter so end-to-end timing is not dominated by per-call allocation.
- Improve benchmark repeatability (for example, by controlling background load and documenting compiler/CPU settings).

## AI use

ChatGPT and DeepSeek were used as development assistants for code generation, refactoring, documentation, and review. The author reviewed and tested the resulting changes. AI-assisted output can contain mistakes; the test suite and the benchmark caveats above define what has actually been verified.

## Scope

This repository is intended to make the optimization process visible. The focus is on understanding and measuring implementation trade-offs, not providing a production-ready GEMM library.
