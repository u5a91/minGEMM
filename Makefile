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
LIB_CUDA := $(LIB)/naive_cuda.so
LIB_TARGETS := $(LIB_NORMAL) $(LIB_BS)

.PHONY: all test bench cuda test-cuda bench-cuda clean

all: $(LIB_TARGETS)

$(LIB):
	mkdir -p $@

# Shared libraries
$(LIB_NORMAL): $(LIB)/%.so: $(SRC)/%.c $(HEADER) | $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared $< -o $@

$(LIB_BS): $(LIB)/bs%.so: $(SRC)/blocked.c $(HEADER) | $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -DBS=$* $< -o $@

# CUDA shared libraries
$(LIB)/%_cuda.so: $(SRC_CUDA)/%.cu $(HEADER_CUDA) | $(LIB)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $(CPPFLAGS) -Xcompiler -fPIC -shared $< -o $@

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

bench-cuda: $(LIB_CUDA)
	@echo "note: naive_cuda uses the synchronous host-copy adapter; timings include H2D/D2H transfers and allocations."
	$(PYTHON) benchmarks/benchmark.py $(LIB_CUDA)

clean:
	rm -rf $(BUILD)
