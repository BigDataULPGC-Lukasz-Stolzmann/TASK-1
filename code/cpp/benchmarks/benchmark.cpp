#include <benchmark/benchmark.h>
#include "matrix_mul.h"

// Global test matrices for consistent benchmarking
static Matrix test_matrix_a;
static Matrix test_matrix_b;
static int current_size = 0;

static void SetupMatrices(int size) {
    if (size != current_size) {
        test_matrix_a = random_matrix(size, 42);
        test_matrix_b = random_matrix(size, 43);
        current_size = size;
    }
}

static void BM_MatMul_IJK(benchmark::State& state) {
    int size = state.range(0);
    SetupMatrices(size);

    for (auto _ : state) {
        auto result = matmul_ijk(test_matrix_a, test_matrix_b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(double) * 3);
}

static void BM_MatMul_IKJ(benchmark::State& state) {
    int size = state.range(0);
    SetupMatrices(size);

    for (auto _ : state) {
        auto result = matmul_ikj(test_matrix_a, test_matrix_b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(double) * 3);
}

static void BM_MatMul_KIJ(benchmark::State& state) {
    int size = state.range(0);
    SetupMatrices(size);

    for (auto _ : state) {
        auto result = matmul_kij(test_matrix_a, test_matrix_b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(double) * 3);
}

static void BM_MatMul_Blocked(benchmark::State& state) {
    int size = state.range(0);
    SetupMatrices(size);

    for (auto _ : state) {
        auto result = matmul_blocked(test_matrix_a, test_matrix_b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(double) * 3);
}

static void BM_MatMul_Blocked32(benchmark::State& state) {
    int size = state.range(0);
    SetupMatrices(size);

    for (auto _ : state) {
        auto result = matmul_blocked(test_matrix_a, test_matrix_b, 32);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(size);
    state.SetItemsProcessed(state.iterations() * size * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(double) * 3);
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_MatMul_IJK)->RangeMultiplier(2)->Range(64, 512)->Complexity(benchmark::oN3);
BENCHMARK(BM_MatMul_IKJ)->RangeMultiplier(2)->Range(64, 512)->Complexity(benchmark::oN3);
BENCHMARK(BM_MatMul_KIJ)->RangeMultiplier(2)->Range(64, 512)->Complexity(benchmark::oN3);
BENCHMARK(BM_MatMul_Blocked)->RangeMultiplier(2)->Range(64, 512)->Complexity(benchmark::oN3);
BENCHMARK(BM_MatMul_Blocked32)->RangeMultiplier(2)->Range(64, 512)->Complexity(benchmark::oN3);

BENCHMARK_MAIN();