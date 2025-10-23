use mat_mul::{matmul_ijk, matmul_ikj, random_matrix};
use std::time::Instant;

fn main() {
    let sizes = [64, 128, 256];
    let runs = 5;

    println!("Rust Matrix Multiplication Benchmark");
    println!("=====================================");

    for size in sizes {
        println!("\nMatrix size: {}x{}", size, size);

        match size {
            64 => {
                let a = random_matrix::<64>();
                let b = random_matrix::<64>();

                // Benchmark i-j-k
                let mut times_ijk = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ijk(a, b);
                    times_ijk.push(start.elapsed().as_secs_f64());
                }

                // Benchmark i-k-j
                let mut times_ikj = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ikj(a, b);
                    times_ikj.push(start.elapsed().as_secs_f64());
                }

                let mean_ijk: f64 = times_ijk.iter().sum::<f64>() / runs as f64;
                let mean_ikj: f64 = times_ikj.iter().sum::<f64>() / runs as f64;

                println!("  i-j-k: {:.4}s ± {:.4}s", mean_ijk, stddev(&times_ijk));
                println!("  i-k-j: {:.4}s ± {:.4}s", mean_ikj, stddev(&times_ikj));
                println!("  Speedup: {:.2}x", mean_ijk / mean_ikj);
            }
            128 => {
                let a = random_matrix::<128>();
                let b = random_matrix::<128>();

                let mut times_ijk = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ijk(a, b);
                    times_ijk.push(start.elapsed().as_secs_f64());
                }

                let mut times_ikj = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ikj(a, b);
                    times_ikj.push(start.elapsed().as_secs_f64());
                }

                let mean_ijk: f64 = times_ijk.iter().sum::<f64>() / runs as f64;
                let mean_ikj: f64 = times_ikj.iter().sum::<f64>() / runs as f64;

                println!("  i-j-k: {:.4}s ± {:.4}s", mean_ijk, stddev(&times_ijk));
                println!("  i-k-j: {:.4}s ± {:.4}s", mean_ikj, stddev(&times_ikj));
                println!("  Speedup: {:.2}x", mean_ijk / mean_ikj);
            }
            256 => {
                let a = random_matrix::<256>();
                let b = random_matrix::<256>();

                let mut times_ijk = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ijk(a, b);
                    times_ijk.push(start.elapsed().as_secs_f64());
                }

                let mut times_ikj = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let start = Instant::now();
                    let _c = matmul_ikj(a, b);
                    times_ikj.push(start.elapsed().as_secs_f64());
                }

                let mean_ijk: f64 = times_ijk.iter().sum::<f64>() / runs as f64;
                let mean_ikj: f64 = times_ikj.iter().sum::<f64>() / runs as f64;

                println!("  i-j-k: {:.4}s ± {:.4}s", mean_ijk, stddev(&times_ijk));
                println!("  i-k-j: {:.4}s ± {:.4}s", mean_ikj, stddev(&times_ikj));
                println!("  Speedup: {:.2}x", mean_ijk / mean_ikj);
            }
            _ => {}
        }
    }
}

fn stddev(times: &[f64]) -> f64 {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64;
    variance.sqrt()
}
