#!/usr/bin/env python3
"""
Comprehensive correctness tests for matrix multiplication implementations.
"""

import sys
import os
import numpy as np
import pytest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mat_mul import (
    random_matrix, random_matrix_numpy,
    matmul_ijk, matmul_ikj, matmul_kij, matmul_numpy, matmul_blocked
)

class TestMatrixMultiplicationCorrectness:
    """Test correctness of all matrix multiplication implementations."""

    @pytest.fixture(params=[4, 8, 16])
    def small_size(self, request):
        """Small matrix sizes for thorough testing."""
        return request.param

    @pytest.fixture
    def test_matrices_small(self, small_size):
        """Generate small test matrices."""
        A = random_matrix(small_size, seed=42)
        B = random_matrix(small_size, seed=43)
        return A, B

    @pytest.fixture
    def test_matrices_numpy_small(self, small_size):
        """Generate small numpy test matrices."""
        A = random_matrix_numpy(small_size, seed=42)
        B = random_matrix_numpy(small_size, seed=43)
        return A, B

    def test_identity_matrix(self):
        """Test multiplication with identity matrix."""
        # Create a simple test matrix
        A = [[1, 2], [3, 4]]
        I = [[1, 0], [0, 1]]  # Identity matrix

        result = matmul_ijk(A, I)
        assert result == A, "A * I should equal A"

        result = matmul_ijk(I, A)
        assert result == A, "I * A should equal A"

    def test_zero_matrix(self):
        """Test multiplication with zero matrix."""
        A = [[1, 2], [3, 4]]
        Z = [[0, 0], [0, 0]]  # Zero matrix

        result = matmul_ijk(A, Z)
        expected = [[0, 0], [0, 0]]
        assert result == expected, "A * 0 should equal 0"

    def test_algorithm_consistency(self, test_matrices_small):
        """Test that all algorithms produce the same result."""
        A, B = test_matrices_small

        # Compute results with different algorithms
        result_ijk = matmul_ijk(A, B)
        result_ikj = matmul_ikj(A, B)
        result_kij = matmul_kij(A, B)
        result_blocked = matmul_blocked(A, B)

        # Check that all results are approximately equal
        def matrices_equal(M1, M2, tolerance=1e-10):
            if len(M1) != len(M2) or len(M1[0]) != len(M2[0]):
                return False
            for i in range(len(M1)):
                for j in range(len(M1[0])):
                    if abs(M1[i][j] - M2[i][j]) > tolerance:
                        return False
            return True

        assert matrices_equal(result_ijk, result_ikj), "i-j-k and i-k-j should produce same result"
        assert matrices_equal(result_ijk, result_kij), "i-j-k and k-i-j should produce same result"
        assert matrices_equal(result_ijk, result_blocked), "i-j-k and blocked should produce same result"

    def test_numpy_consistency(self, test_matrices_numpy_small):
        """Test that implementations match NumPy."""
        A_np, B_np = test_matrices_numpy_small

        # Convert to list format for algorithms
        A = A_np.tolist()
        B = B_np.tolist()

        # Compute with algorithm and NumPy
        result = matmul_ijk(A, B)
        numpy_result = matmul_numpy(A_np, B_np)

        # Compare results (allowing for small floating point differences)
        result_np = np.array(result)
        np.testing.assert_allclose(result_np, numpy_result, rtol=1e-10, atol=1e-12)

    def test_associativity(self):
        """Test matrix multiplication associativity: (AB)C = A(BC)."""
        # Small matrices for this test
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = [[9, 10], [11, 12]]

        # Compute (AB)C
        AB = matmul_ijk(A, B)
        ABC_left = matmul_ijk(AB, C)

        # Compute A(BC)
        BC = matmul_ijk(B, C)
        ABC_right = matmul_ijk(A, BC)

        # Check equality
        def matrices_equal(M1, M2, tolerance=1e-10):
            for i in range(len(M1)):
                for j in range(len(M1[0])):
                    if abs(M1[i][j] - M2[i][j]) > tolerance:
                        return False
            return True

        assert matrices_equal(ABC_left, ABC_right), "Matrix multiplication should be associative"

    def test_distributivity(self):
        """Test matrix multiplication distributivity: A(B+C) = AB + AC."""
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = [[1, 1], [1, 1]]

        # Compute A(B+C)
        B_plus_C = [[B[i][j] + C[i][j] for j in range(len(B[0]))] for i in range(len(B))]
        A_BC = matmul_ijk(A, B_plus_C)

        # Compute AB + AC
        AB = matmul_ijk(A, B)
        AC = matmul_ijk(A, C)
        AB_plus_AC = [[AB[i][j] + AC[i][j] for j in range(len(AB[0]))] for i in range(len(AB))]

        # Check equality
        def matrices_equal(M1, M2, tolerance=1e-10):
            for i in range(len(M1)):
                for j in range(len(M1[0])):
                    if abs(M1[i][j] - M2[i][j]) > tolerance:
                        return False
            return True

        assert matrices_equal(A_BC, AB_plus_AC), "Matrix multiplication should be distributive"

    def test_block_size_variations(self):
        """Test that different block sizes produce the same result."""
        size = 16
        A = random_matrix(size, seed=42)
        B = random_matrix(size, seed=43)

        result_default = matmul_blocked(A, B)
        result_block_4 = matmul_blocked(A, B, block_size=4)
        result_block_8 = matmul_blocked(A, B, block_size=8)

        def matrices_equal(M1, M2, tolerance=1e-10):
            for i in range(len(M1)):
                for j in range(len(M1[0])):
                    if abs(M1[i][j] - M2[i][j]) > tolerance:
                        return False
            return True

        assert matrices_equal(result_default, result_block_4), "Different block sizes should produce same result"
        assert matrices_equal(result_default, result_block_8), "Different block sizes should produce same result"