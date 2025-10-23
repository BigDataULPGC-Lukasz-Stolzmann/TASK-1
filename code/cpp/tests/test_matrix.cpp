#include "matrix_mul.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

const double TOLERANCE = 1e-10;

bool matrices_equal(const Matrix& M1, const Matrix& M2, double tolerance = TOLERANCE) {
    if (M1.size() != M2.size() || M1[0].size() != M2[0].size()) {
        return false;
    }
    for (size_t i = 0; i < M1.size(); i++) {
        for (size_t j = 0; j < M1[0].size(); j++) {
            if (std::abs(M1[i][j] - M2[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

Matrix matrix_add(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

void test_identity_matrix() {
    std::cout << "Testing identity matrix multiplication..." << std::endl;

    Matrix A = {{1, 2}, {3, 4}};
    Matrix I = {{1, 0}, {0, 1}}; // Identity matrix

    Matrix result = matmul_ijk(A, I);
    assert(matrices_equal(A, result));

    result = matmul_ijk(I, A);
    assert(matrices_equal(A, result));

    std::cout << "✓ Identity matrix test passed" << std::endl;
}

void test_zero_matrix() {
    std::cout << "Testing zero matrix multiplication..." << std::endl;

    Matrix A = {{1, 2}, {3, 4}};
    Matrix Z = {{0, 0}, {0, 0}}; // Zero matrix
    Matrix expected = {{0, 0}, {0, 0}};

    Matrix result = matmul_ijk(A, Z);
    assert(matrices_equal(expected, result));

    std::cout << "✓ Zero matrix test passed" << std::endl;
}

void test_algorithm_consistency() {
    std::cout << "Testing algorithm consistency..." << std::endl;

    int size = 8;
    auto A = random_matrix(size, 42);
    auto B = random_matrix(size, 43);

    auto result_ijk = matmul_ijk(A, B);
    auto result_ikj = matmul_ikj(A, B);
    auto result_kij = matmul_kij(A, B);
    auto result_blocked = matmul_blocked(A, B);

    assert(matrices_equal(result_ijk, result_ikj));
    assert(matrices_equal(result_ijk, result_kij));
    assert(matrices_equal(result_ijk, result_blocked));

    std::cout << "✓ Algorithm consistency test passed" << std::endl;
}

void test_associativity() {
    std::cout << "Testing associativity..." << std::endl;

    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};
    Matrix C = {{9, 10}, {11, 12}};

    // Compute (AB)C
    auto AB = matmul_ijk(A, B);
    auto ABC_left = matmul_ijk(AB, C);

    // Compute A(BC)
    auto BC = matmul_ijk(B, C);
    auto ABC_right = matmul_ijk(A, BC);

    assert(matrices_equal(ABC_left, ABC_right));

    std::cout << "✓ Associativity test passed" << std::endl;
}

void test_distributivity() {
    std::cout << "Testing distributivity..." << std::endl;

    Matrix A = {{1, 2}, {3, 4}};
    Matrix B = {{5, 6}, {7, 8}};
    Matrix C = {{1, 1}, {1, 1}};

    // Compute A(B+C)
    auto B_plus_C = matrix_add(B, C);
    auto A_BC = matmul_ijk(A, B_plus_C);

    // Compute AB + AC
    auto AB = matmul_ijk(A, B);
    auto AC = matmul_ijk(A, C);
    auto AB_plus_AC = matrix_add(AB, AC);

    assert(matrices_equal(A_BC, AB_plus_AC));

    std::cout << "✓ Distributivity test passed" << std::endl;
}

void test_block_size_variations() {
    std::cout << "Testing block size variations..." << std::endl;

    int size = 16;
    auto A = random_matrix(size, 42);
    auto B = random_matrix(size, 43);

    auto result_default = matmul_blocked(A, B);
    auto result_block_4 = matmul_blocked(A, B, 4);
    auto result_block_8 = matmul_blocked(A, B, 8);

    assert(matrices_equal(result_default, result_block_4));
    assert(matrices_equal(result_default, result_block_8));

    std::cout << "✓ Block size variations test passed" << std::endl;
}

void test_random_matrix_generation() {
    std::cout << "Testing random matrix generation..." << std::endl;

    int size = 5;
    auto M1 = random_matrix(size, 42);
    auto M2 = random_matrix(size, 42); // Same seed

    assert(matrices_equal(M1, M2));

    auto M3 = random_matrix(size, 43); // Different seed
    assert(!matrices_equal(M1, M3));

    std::cout << "✓ Random matrix generation test passed" << std::endl;
}

int main() {
    std::cout << "Running C++ matrix multiplication tests..." << std::endl << std::endl;

    test_identity_matrix();
    test_zero_matrix();
    test_algorithm_consistency();
    test_associativity();
    test_distributivity();
    test_block_size_variations();
    test_random_matrix_generation();

    std::cout << std::endl << "All tests passed! ✅" << std::endl;
    return 0;
}