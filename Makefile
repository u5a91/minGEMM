CC := gcc
CFLAGS := -O3 -march=native -std=c11 -ffp-contract=fast -Wall -Wextra
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17
PYTHON := .venv/bin/python

SRC := src/cpu
SRC_CUDA := src/cuda
INCLUDE := include
BUILD := build
LIB := $(BUILD)/lib
HEADER := $(INCLUDE)/mingemm/cpu.h
HEADER_CUDA := $(INCLUDE)/mingemm/cuda.cuh

CUDA_ARCH := -arch=sm_86

CPPFLAGS := -I$(INCLUDE)

VARIANTS := naive ikj blocked microkernel microkernel_simd microkernel_simd4x8 blocked_microkernel blocked_microkernel_simd4x8
BLOCK_SIZES := 16 32 64 128 256 512

LIB_NORMAL := $(addprefix $(LIB)/,$(addsuffix .so,$(VARIANTS)))
LIB_BS := $(addprefix $(LIB)/bs,$(addsuffix .so,$(BLOCK_SIZES)))
LIB_TARGETS := $(LIB_NORMAL) $(LIB_BS)

CUDA_VARIANTS := naive
LIB_CUDA := $(addprefix $(LIB)/,$(addsuffix _cuda.so,$(CUDA_VARIANTS)))
BENCH_CUDA := $(BUILD)/bench_cuda

.PHONY: all test bench cuda test-cuda bench-cuda bench-cuda-e2e clean

all: $(LIB_TARGETS)

$(BUILD):
		mkdir -p $@

$(LIB): | $(BUILD)
		mkdir -p $@

# Shared libraries
$(LIB_NORMAL): $(LIB)/%.so: $(SRC)/%.c $(HEADER) | $(LIB)
		$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $< -o $@

$(LIB_BS): $(LIB)/bs%.so: $(SRC)/blocked.c $(HEADER) | $(LIB)
		$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -DBS=$* $< -o $@

# CUDA shared libraries
$(LIB)/%_cuda.so: $(SRC_CUDA)/%.cu $(HEADER_CUDA) | $(LIB)
		$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) --cudart shared $(CPPFLAGS) -Xcompiler -fPIC -shared $< -o $@

# Kernel-only benchmark (dlopen's the variant libraries)
$(BENCH_CUDA): $(SRC_CUDA)/benchmark.cu | $(BUILD)
		$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) --cudart shared $(CPPFLAGS) $< -o $@ -ldl

cuda: $(LIB_CUDA)

# Correctness tests
test: $(LIB_TARGETS)
		MINGEMM_TEST_LIBS="$(LIB_TARGETS)" $(PYTHON) -m pytest tests/ -v

# Benchmarks
bench: $(LIB_TARGETS)
		$(PYTHON) benchmarks/benchmark.py $(LIB_TARGETS)

# CUDA tests and benchmarks
test-cuda: $(LIB_CUDA)
		MINGEMM_TEST_LIBS="$(LIB_CUDA)" $(PYTHON) -m pytest tests/ -v

# Kernel-only: times the asynchronous device entry point with preallocated
# device buffers, excluding H2D/D2H transfers.
bench-cuda: $(LIB_CUDA) $(BENCH_CUDA)
		@echo "note: kernel-only timing (device buffers preallocated; H2D/D2H excluded)."
		$(BENCH_CUDA) $(LIB_CUDA)

# End-to-end: times the synchronous host-copy adapter, including allocations
# and H2D/D2H transfers.
bench-cuda-e2e: $(LIB_CUDA)
		@echo "note: end-to-end timing through the synchronous host-copy adapter; includes allocations and H2D/D2H transfers."
		$(PYTHON) benchmarks/benchmark.py $(LIB_CUDA)

clean:
		rm -rf $(BUILD)
