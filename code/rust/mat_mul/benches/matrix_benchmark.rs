use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mat_mul::{matmul_blocked, matmul_ijk, matmul_ikj, matmul_kij, random_matrix};

fn benchmark_algorithms(c: &mut Criterion) {
    let sizes = vec![64, 128, 256, 512];

    for size in sizes {
        let mut group = c.benchmark_group(format!("matrix_multiplication_{}", size));

        // Set throughput for better reporting
        group.throughput(Throughput::Elements((size * size * size) as u64));

        // Generate test matrices once per size
        let a = match size {
            64 => random_matrix::<64>(42),
            128 => random_matrix::<128>(42),
            256 => random_matrix::<256>(42),
            512 => random_matrix::<512>(42),
            _ => panic!("Unsupported size"),
        };

        let b = match size {
            64 => random_matrix::<64>(43),
            128 => random_matrix::<128>(43),
            256 => random_matrix::<256>(43),
            512 => random_matrix::<512>(43),
            _ => panic!("Unsupported size"),
        };

        // Benchmark each algorithm
        match size {
            64 => {
                group.bench_with_input(BenchmarkId::new("ijk", size), &size, |b, _| {
                    b.iter(|| matmul_ijk::<64>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("ikj", size), &size, |b, _| {
                    b.iter(|| matmul_ikj::<64>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("kij", size), &size, |b, _| {
                    b.iter(|| matmul_kij::<64>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<64>(black_box(a), black_box(b), 64))
                });
                group.bench_with_input(BenchmarkId::new("blocked_32", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<64>(black_box(a), black_box(b), 32))
                });
            }
            128 => {
                group.bench_with_input(BenchmarkId::new("ijk", size), &size, |b, _| {
                    b.iter(|| matmul_ijk::<128>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("ikj", size), &size, |b, _| {
                    b.iter(|| matmul_ikj::<128>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("kij", size), &size, |b, _| {
                    b.iter(|| matmul_kij::<128>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<128>(black_box(a), black_box(b), 64))
                });
                group.bench_with_input(BenchmarkId::new("blocked_32", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<128>(black_box(a), black_box(b), 32))
                });
            }
            256 => {
                group.bench_with_input(BenchmarkId::new("ijk", size), &size, |b, _| {
                    b.iter(|| matmul_ijk::<256>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("ikj", size), &size, |b, _| {
                    b.iter(|| matmul_ikj::<256>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("kij", size), &size, |b, _| {
                    b.iter(|| matmul_kij::<256>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<256>(black_box(a), black_box(b), 64))
                });
                group.bench_with_input(BenchmarkId::new("blocked_32", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<256>(black_box(a), black_box(b), 32))
                });
            }
            512 => {
                group.bench_with_input(BenchmarkId::new("ijk", size), &size, |b, _| {
                    b.iter(|| matmul_ijk::<512>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("ikj", size), &size, |b, _| {
                    b.iter(|| matmul_ikj::<512>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("kij", size), &size, |b, _| {
                    b.iter(|| matmul_kij::<512>(black_box(a), black_box(b)))
                });
                group.bench_with_input(BenchmarkId::new("blocked", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<512>(black_box(a), black_box(b), 64))
                });
                group.bench_with_input(BenchmarkId::new("blocked_32", size), &size, |b, _| {
                    b.iter(|| matmul_blocked::<512>(black_box(a), black_box(b), 32))
                });
            }
            _ => panic!("Unsupported size"),
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_algorithms);
criterion_main!(benches);
