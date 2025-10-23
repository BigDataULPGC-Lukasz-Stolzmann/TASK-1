import time
import statistics
from mat_mul import random_matrix, matmul_ijk

N = 256
RUNS = 10

A = random_matrix(N, seed=42)
B = random_matrix(N, seed=43)

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    matmul_ijk(A, B)
    t1 = time.perf_counter()
    times.append(t1 - t0)

print(f'Python ijk {N}*{N}')
print(f'Mean: {statistics.mean(times):.6f}s, Median: {statistics.median(times):.6f}s')
