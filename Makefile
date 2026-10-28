
CC := gcc
CFLAGS := -O3 -march=native -std=c11 -ffp-contract=fast -Wall -Wextra
PYTHON := .venv/bin/python

SRC := src/cpu
BUILD := build
LIB := $(BUILD)/lib

VARIANTS := naive ikj blocked microkernel microkernel_simd microkernel_simd4x8 blocked_microkernel blocked_microkernel_simd4x8
BLOCK_SIZES := 16 32 64 128 256 512

LIB_NORMAL := $(addprefix $(LIB)/,$(addsuffix .so,$(VARIANTS)))
LIB_BS := $(addprefix $(LIB)/bs,$(addsuffix .so,$(BLOCK_SIZES)))
LIB_TARGETS := $(LIB_NORMAL) $(LIB_BS)

.PHONY: all test bench clean

all: $(LIB_TARGETS)

$(LIB):
	mkdir -p $@

# Shared libraries
$(LIB_NORMAL): $(LIB)/%.so: $(SRC)/%.c | $(LIB)
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

$(LIB_BS): $(LIB)/bs%.so: $(SRC)/blocked.c | $(LIB)
	$(CC) $(CFLAGS) -fPIC -shared -DBS=$* $< -o $@

# Correctness tests
test: $(LIB_TARGETS)
	MINGEMM_TEST_LIBS="$(LIB_TARGETS)" $(PYTHON) -m pytest tests/ -v

# Benchmarks
bench: $(LIB_TARGETS)
	$(PYTHON) benchmarks/benchmark.py $(LIB_TARGETS)

clean:
	rm -rf $(BUILD)
