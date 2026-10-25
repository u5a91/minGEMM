CC := gcc
CFLAGS := -O3 -march=native -std=c11 -Wall -Wextra
PYTHON := .venv/bin/python

SRC := src/cpu
BUILD := build
BIN := $(BUILD)/bin
LIB := $(BUILD)/lib

VARIANTS := naive ikj blocked microkernel blocked_microkernel
BLOCK_SIZES := 16 32 64 128 256 512

BENCH_NORMAL := $(addprefix $(BIN)/benchmark_,$(VARIANTS))
BENCH_BS := $(addprefix $(BIN)/benchmark_bs,$(BLOCK_SIZES))
BENCH_TARGETS := $(BENCH_NORMAL) $(BENCH_BS)

LIB_NORMAL := $(addprefix $(LIB)/,$(addsuffix .so,$(VARIANTS)))
LIB_BS := $(addprefix $(LIB)/bs,$(addsuffix .so,$(BLOCK_SIZES)))
LIB_TARGETS := $(LIB_NORMAL) $(LIB_BS)

.PHONY: all test bench clean

all: $(BENCH_TARGETS)

$(BIN) $(LIB):
	mkdir -p $@

# Benchmark executables
$(BENCH_NORMAL): $(BIN)/benchmark_%: $(SRC)/%.c $(SRC)/benchmark.c | $(BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(BENCH_BS): $(BIN)/benchmark_bs%: $(SRC)/blocked.c $(SRC)/benchmark.c | $(BIN)
	$(CC) $(CFLAGS) -DBS=$* $^ -o $@

# Shared libraries for correctness tests
$(LIB_NORMAL): $(LIB)/%.so: $(SRC)/%.c | $(LIB)
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

$(LIB_BS): $(LIB)/bs%.so: $(SRC)/blocked.c | $(LIB)
	$(CC) $(CFLAGS) -fPIC -shared -DBS=$* $< -o $@

test: $(LIB_TARGETS)
	MINGEMM_TEST_LIBS="$(LIB_TARGETS)" $(PYTHON) -m pytest tests/ -v

bench: $(BENCH_TARGETS)
	$(PYTHON) benchmarks/benchmark.py $(BENCH_TARGETS)

clean:
	rm -rf $(BUILD)
