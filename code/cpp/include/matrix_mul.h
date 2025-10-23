#pragma once
#include <vector>

using Matrix = std::vector<std::vector<double>>;

// Matrix generation
Matrix random_matrix(int n, unsigned seed = 0);

// Matrix multiplication algorithms
Matrix matmul_ijk(const Matrix &A, const Matrix &B);
Matrix matmul_ikj(const Matrix &A, const Matrix &B);
Matrix matmul_kij(const Matrix &A, const Matrix &B);
Matrix matmul_blocked(const Matrix &A, const Matrix &B, int block_size = 64);
