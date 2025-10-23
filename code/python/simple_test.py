#!/usr/bin/env python3
"""Simple benchmark without external dependencies"""

import sys
import os
import time
import statistics

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mat_mul import random_matrix, matmul_ijk, matmul_ikj
from data_handler import save_matrices, save_result_matrix, save_benchmark_results

def benchmark_algorithm(func, A, B, runs=5):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(A, B)
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0

def main():
    sizes = [64, 128, 256]
    results = []

    print("Python Matrix Multiplication Benchmark")
    print("="*40)

    # Create data directories
    os.makedirs("../../data/input", exist_ok=True)
    os.makedirs("../../data/output", exist_ok=True)

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")

        A = random_matrix(size, seed=42)
        B = random_matrix(size, seed=43)

        # Save input matrices
        save_matrices(A, B, size, f"../../data/input/python_matrices_{size}x{size}.json")

        mean_ijk, std_ijk = benchmark_algorithm(matmul_ijk, A, B)
        mean_ikj, std_ikj = benchmark_algorithm(matmul_ikj, A, B)

        # Save result matrices
        result_ijk = matmul_ijk(A, B)
        result_ikj = matmul_ikj(A, B)
        save_result_matrix(result_ijk, size, "i-j-k", f"../../data/output/python_result_ijk_{size}x{size}.json")
        save_result_matrix(result_ikj, size, "i-k-j", f"../../data/output/python_result_ikj_{size}x{size}.json")

        print(f"  i-j-k: {mean_ijk:.4f}s ± {std_ijk:.4f}s")
        print(f"  i-k-j: {mean_ikj:.4f}s ± {std_ikj:.4f}s")
        print(f"  Speedup: {mean_ijk/mean_ikj:.2f}x")

        # Store results
        results.extend([
            {
                'language': 'Python',
                'algorithm': 'i-j-k',
                'size': size,
                'mean_time': mean_ijk,
                'std_time': std_ijk,
                'speedup': ''
            },
            {
                'language': 'Python',
                'algorithm': 'i-k-j',
                'size': size,
                'mean_time': mean_ikj,
                'std_time': std_ikj,
                'speedup': f"{mean_ijk/mean_ikj:.2f}x"
            }
        ])

    # Save benchmark results
    save_benchmark_results(results, "../../data/output/python_benchmark_results.csv")
    print(f"\nData saved to ../../data/input/ and ../../data/output/")

if __name__ == "__main__":
    main()