#include "matrix_mul.h"
#include <random>
#include <algorithm>

Matrix random_matrix(int n, unsigned seed) {
    Matrix M(n, std::vector<double>(n));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = dist(rng);
        }
    }
    return M;
}

Matrix matmul_ijk(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix matmul_ikj(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}

Matrix matmul_kij(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix matmul_blocked(const Matrix &A, const Matrix &B, int block_size) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));

    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Block boundaries
                int i_end = std::min(ii + block_size, n);
                int j_end = std::min(jj + block_size, n);
                int k_end = std::min(kk + block_size, n);

                // Block multiplication
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        for (int k = kk; k < k_end; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
    return C;
}
