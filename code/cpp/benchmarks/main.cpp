#include "matrix_mul.h"
#include <chrono>
#include <iostream>
#include <numeric>

int main() {
    int N = 256;         
    int RUNS = 10;    
    
    auto A = random_matrix(N, 42);
    auto B = random_matrix(N, 43);

    std::vector<double> times(RUNS);

    for(int r=0;r<RUNS;r++){
        auto start = std::chrono::high_resolution_clock::now();
        auto C = matmul_ijk(A,B);
        auto end = std::chrono::high_resolution_clock::now();
        times[r] = std::chrono::duration<double>(end-start).count();
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / RUNS;
    std::cout << "C++ ijk " << N << "x" << N << std::endl;
    std::cout << "Mean time: " << mean << " s" << std::endl;
}
