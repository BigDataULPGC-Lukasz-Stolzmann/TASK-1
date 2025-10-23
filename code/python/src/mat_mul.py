import random

def random_matrix(n, seed=None):
    """Generate a random n×n matrix with values between 0 and 1."""
    if seed is not None:
        random.seed(seed)
    return [[random.random() for _ in range(n)] for _ in range(n)]

def matmul_ijk(A, B):
    """Standard i-j-k matrix multiplication."""
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matmul_ikj(A, B):
    """i-k-j matrix multiplication (better cache locality)."""
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matmul_kij(A, B):
    """k-i-j matrix multiplication."""
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matmul_blocked(A, B, block_size=64):
    """Block matrix multiplication for better cache performance."""
    n = len(A)
    C = [[0.0]*n for _ in range(n)]

    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                # Block boundaries
                i_end = min(ii + block_size, n)
                j_end = min(jj + block_size, n)
                k_end = min(kk + block_size, n)

                # Block multiplication
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        for k in range(kk, k_end):
                            C[i][j] += A[i][k] * B[k][j]
    return C
