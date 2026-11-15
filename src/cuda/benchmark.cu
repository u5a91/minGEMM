#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using matmul_async_fn = cudaError_t (*)(
    const float *, const float *, float *,
    int, int, int, cudaStream_t);

const char *const kLaunchSymbols[] = {
    "matmul_cuda_async",
    "matmul_cuda_naive_async",
};

struct library {
    std::string name;
    void *handle;
    matmul_async_fn launch;
};

[[noreturn]] void die(const std::string &message)
{
    fprintf(stderr, "bench_cuda: %s\n", message.c_str());
    exit(EXIT_FAILURE);
}

void check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        die(std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

void usage(const char *prog)
{
    printf(
        "usage: %s [options] <library.so> [<library.so> ...]\n"
        "\n"
        "Times the asynchronous device entry point of each CUDA GEMM library.\n"
        "Device buffers are allocated once and H2D/D2H copies are excluded, so\n"
        "the reported time is kernel execution only.\n"
        "\n"
        "options:\n"
        "  --sizes 256,512,1024  square matrix sizes to benchmark (default)\n"
        "  --warmups N           warm-up launches per size (default 3)\n"
        "  --runs N              timed launches per size (default 10)\n"
        "  -h, --help            show this help\n"
        "\n"
        "Each library should export matmul_cuda_async (matmul_cuda_naive_async\n"
        "is also accepted).\n",
        prog);
}

int parse_positive(const char *text, const char *name)
{
    char *end = nullptr;
    errno = 0;
    long value = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 1000000) {
        die(std::string("invalid ") + name + ": " + text);
    }

    return (int)value;
}

void parse_sizes(const char *text, std::vector<int> &sizes)
{
    std::string input(text);
    size_t pos = 0;

    sizes.clear();

    while (true) {
        size_t comma = input.find(',', pos);
        std::string item = input.substr(
            pos, comma == std::string::npos ? std::string::npos : comma - pos);

        if (item.empty()) {
            die("empty value in --sizes");
        }

        sizes.push_back(parse_positive(item.c_str(), "size"));

        if (comma == std::string::npos) {
            break;
        }

        pos = comma + 1;
    }
}

std::string library_stem(const std::string &path)
{
    size_t slash = path.find_last_of('/');
    std::string base =
        (slash == std::string::npos) ? path : path.substr(slash + 1);

    if (base.size() > 3 && base.compare(base.size() - 3, 3, ".so") == 0) {
        base.erase(base.size() - 3);
    }

    return base;
}

library load_library(const std::string &path)
{
    library lib;
    lib.name = library_stem(path);
    lib.handle = nullptr;
    lib.launch = nullptr;

    lib.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (lib.handle == nullptr) {
        const char *error = dlerror();
        die(std::string("cannot load ") + path + ": " +
            (error != nullptr ? error : "unknown error"));
    }

    for (const char *symbol : kLaunchSymbols) {
        dlerror(); /* clear any previous error */
        void *entry = dlsym(lib.handle, symbol);

        if (entry != nullptr) {
            lib.launch = reinterpret_cast<matmul_async_fn>(entry);
            break;
        }
    }

    if (lib.launch == nullptr) {
        die(path + ": no CUDA launch entry point found "
            "(expected matmul_cuda_async or matmul_cuda_naive_async)");
    }

    return lib;
}

void fill_random(float *x, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        x[i] = (float)rand() / (float)RAND_MAX;
    }
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    return values[middle];
}

} /* namespace */

int main(int argc, char **argv)
{
    std::vector<int> sizes = {256, 512, 1024};
    std::vector<std::string> paths;
    int warmups = 3;
    int runs = 10;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return 0;
        }

        if (strcmp(arg, "--sizes") == 0) {
            if (++i >= argc) {
                die("--sizes requires a value");
            }

            parse_sizes(argv[i], sizes);
        } else if (strcmp(arg, "--warmups") == 0) {
            if (++i >= argc) {
                die("--warmups requires a value");
            }

            warmups = parse_positive(argv[i], "warmups");
        } else if (strcmp(arg, "--runs") == 0) {
            if (++i >= argc) {
                die("--runs requires a value");
            }

            runs = parse_positive(argv[i], "runs");
        } else if (arg[0] == '-') {
            die(std::string("unknown option: ") + arg);
        } else {
            paths.push_back(arg);
        }
    }

    if (paths.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<library> libraries;

    for (const std::string &path : paths) {
        libraries.push_back(load_library(path));
    }

    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;

    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    srand(42);

    printf("kernel-only timing: device buffers are preallocated, "
           "H2D/D2H copies excluded\n");

    for (int n : sizes) {
        size_t count = (size_t)n * (size_t)n;
        size_t bytes = count * sizeof(float);

        float *hA = (float *)malloc(bytes);
        float *hB = (float *)malloc(bytes);

        if (hA == nullptr || hB == nullptr) {
            die("host allocation failed");
        }

        fill_random(hA, count);
        fill_random(hB, count);

        float *dA = nullptr;
        float *dB = nullptr;
        float *dC = nullptr;

        check(cudaMalloc((void **)&dA, bytes), "cudaMalloc(A)");
        check(cudaMalloc((void **)&dB, bytes), "cudaMalloc(B)");
        check(cudaMalloc((void **)&dC, bytes), "cudaMalloc(C)");
        check(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
        check(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");

        double flops = 2.0 * (double)n * (double)n * (double)n;

        printf("n = %d\n", n);

        for (const library &lib : libraries) {
            for (int i = 0; i < warmups; i++) {
                check(lib.launch(dA, dB, dC, n, n, n, stream), "kernel launch");
            }

            check(cudaStreamSynchronize(stream), "warm-up synchronize");

            std::vector<double> times;
            times.reserve(runs);

            for (int r = 0; r < runs; r++) {
                check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
                check(lib.launch(dA, dB, dC, n, n, n, stream), "kernel launch");
                check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
                check(cudaEventSynchronize(stop), "cudaEventSynchronize");

                float milliseconds = 0.0f;
                check(cudaEventElapsedTime(&milliseconds, start, stop),
                      "cudaEventElapsedTime");
                times.push_back((double)milliseconds * 1e-3);
            }

            double seconds = median(times);

            printf("  %s\n", lib.name.c_str());
            printf("    kernel time (median) = %.6f s\n", seconds);
            printf("    throughput (median)  = %.3f GFLOP/s\n",
                   flops / seconds / 1e9);
        }

        free(hA);
        free(hB);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    for (library &lib : libraries) {
        dlclose(lib.handle);
    }

    return 0;
}
