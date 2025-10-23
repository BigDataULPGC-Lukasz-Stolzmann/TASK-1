#!/usr/bin/env python3
"""
Professional benchmarks using pytest-benchmark.
Run with: pytest test_benchmark.py --benchmark-json=results.json
"""

import sys
import os
import pytest
from memory_profiler import profile

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mat_mul import (
    random_matrix, random_matrix_numpy,
    matmul_ijk, matmul_ikj, matmul_kij, matmul_numpy, matmul_blocked
)

# Test parameters
SIZES = [64, 128, 256, 512]
SEED_A, SEED_B = 42, 43

class TestMatrixMultiplication:
    """Professional benchmark suite for matrix multiplication algorithms."""

    @pytest.fixture(params=SIZES)
    def matrix_size(self, request):
        return request.param

    @pytest.fixture
    def test_matrices(self, matrix_size):
        """Generate test matrices for benchmarking."""
        A = random_matrix(matrix_size, seed=SEED_A)
        B = random_matrix(matrix_size, seed=SEED_B)
        return A, B

    @pytest.fixture
    def test_matrices_numpy(self, matrix_size):
        """Generate numpy test matrices for benchmarking."""
        A = random_matrix_numpy(matrix_size, seed=SEED_A)
        B = random_matrix_numpy(matrix_size, seed=SEED_B)
        return A, B

    def test_benchmark_ijk(self, benchmark, test_matrices):
        """Benchmark i-j-k algorithm."""
        A, B = test_matrices
        benchmark.pedantic(matmul_ijk, args=(A, B), rounds=5, iterations=3)

    def test_benchmark_ikj(self, benchmark, test_matrices):
        """Benchmark i-k-j algorithm."""
        A, B = test_matrices
        benchmark.pedantic(matmul_ikj, args=(A, B), rounds=5, iterations=3)

    def test_benchmark_kij(self, benchmark, test_matrices):
        """Benchmark k-i-j algorithm."""
        A, B = test_matrices
        benchmark.pedantic(matmul_kij, args=(A, B), rounds=5, iterations=3)

    def test_benchmark_blocked(self, benchmark, test_matrices):
        """Benchmark blocked algorithm."""
        A, B = test_matrices
        benchmark.pedantic(matmul_blocked, args=(A, B), rounds=5, iterations=3)

    def test_benchmark_numpy(self, benchmark, test_matrices_numpy):
        """Benchmark NumPy algorithm."""
        A, B = test_matrices_numpy
        benchmark.pedantic(matmul_numpy, args=(A, B), rounds=10, iterations=5)

    @profile
    def test_memory_profile_ijk(self, test_matrices):
        """Memory profiling for i-j-k algorithm."""
        A, B = test_matrices
        result = matmul_ijk(A, B)
        return result