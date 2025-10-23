#include "include/matrix_mul.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <iomanip>

double calculate_mean(const std::vector<double>& times) {
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

double calculate_stddev(const std::vector<double>& times) {
    double mean = calculate_mean(times);
    double sum = 0.0;
    for (double time : times) {
        sum += (time - mean) * (time - mean);
    }
    return std::sqrt(sum / times.size());
}

int main() {
    std::vector<int> sizes = {64, 128, 256};
    int runs = 5;

    std::cout << "C++ Matrix Multiplication Benchmark" << std::endl;
    std::cout << "====================================" << std::endl;

    for (int size : sizes) {
        std::cout << "\nMatrix size: " << size << "x" << size << std::endl;

        auto A = random_matrix(size, 42);
        auto B = random_matrix(size, 43);

        // Benchmark i-j-k
        std::vector<double> times_ijk(runs);
        for (int r = 0; r < runs; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto C = matmul_ijk(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            times_ijk[r] = std::chrono::duration<double>(end - start).count();
        }

        // Benchmark i-k-j
        std::vector<double> times_ikj(runs);
        for (int r = 0; r < runs; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto C = matmul_ikj(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            times_ikj[r] = std::chrono::duration<double>(end - start).count();
        }

        double mean_ijk = calculate_mean(times_ijk);
        double mean_ikj = calculate_mean(times_ikj);
        double std_ijk = calculate_stddev(times_ijk);
        double std_ikj = calculate_stddev(times_ikj);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  i-j-k: " << mean_ijk << "s ± " << std_ijk << "s" << std::endl;
        std::cout << "  i-k-j: " << mean_ikj << "s ± " << std_ikj << "s" << std::endl;
        std::cout << "  Speedup: " << std::setprecision(2) << mean_ijk/mean_ikj << "x" << std::endl;
    }

    return 0;
}