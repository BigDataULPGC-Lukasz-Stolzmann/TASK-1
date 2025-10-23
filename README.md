# Matrix Multiplication Performance Benchmark

Performance analysis of matrix multiplication algorithms across four programming languages: **Python**, **Java**, **C++**, and **Rust**.

## Project Overview

Matrix multiplication algorithms implemented and benchmarked to analyze:
- **Algorithmic performance differences** (loop ordering: i-j-k vs i-k-j)
- **Programming language performance characteristics**
- **Cache optimization effectiveness**
- **Scalability across different matrix sizes**

## Project Structure

```
LanguageBenchmarkMatMul/
├── code/                     # Source code implementations
│   ├── python/              # Python implementation
│   │   ├── src/             # Production code
│   │   ├── tests/           # Unit tests
│   │   └── simple_test.py   # Simple benchmark
│   ├── java/                # Java implementation
│   │   └── SimpleBenchmark.java  # Simple benchmark
│   ├── cpp/                 # C++ implementation
│   │   ├── src/             # Source files
│   │   ├── include/         # Header files
│   │   ├── simple_benchmark.cpp
│   │   └── Makefile         # Build configuration
│   └── rust/                # Rust implementation
│       └── mat_mul/         # Cargo project
│           ├── src/         # Source code
│           └── Cargo.toml   # Cargo configuration
├── data/                    # Benchmark data
│   ├── input/              # Test matrices (JSON)
│   ├── output/             # Result matrices and CSV files
│   └── results/            # Raw benchmark outputs
├── paper/                   # LaTeX research paper
│   ├── matrix_multiplication_benchmark.tex
│   └── references.bib
└── run_benchmarks.sh       # Automated benchmark execution
```

## Algorithms Implemented

### 1. Standard Loop Variants
- **i-j-k**: Standard nested loop order
- **i-k-j**: Cache-optimized loop order

### 2. Cache Optimization
- **Blocked Matrix Multiplication**: Tiled approach for better cache locality

## Benchmarking Approach

Each language uses direct time measurements:

| Language | Tool | Features |
|----------|------|----------|
| **Python** | `time.perf_counter()` | High-resolution timing |
| **Java** | `System.nanoTime()` | Nanosecond precision |
| **C++** | `std::chrono::high_resolution_clock` | Standard library timing |
| **Rust** | `std::time::Instant` | High-precision timing |

## Quick Start

### Prerequisites
- **Python 3.8+**
- **Java 11+**
- **C++17** compiler (g++/clang++)
- **Rust 1.70+** with Cargo

### Running All Benchmarks

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

### Running Individual Benchmarks

#### Python
```bash
cd code/python
python3 simple_test.py
```

#### Java
```bash
cd code/java
javac SimpleBenchmark.java
java SimpleBenchmark
```

#### C++
```bash
cd code/cpp
make && ./simple_benchmark
```

#### Rust
```bash
cd code/rust/mat_mul
cargo run --release
```

## Running Tests

Correctness tests available for each language:

```bash
# Python
cd code/python && python3 tests/test_correctness.py

# Java
cd code/java && javac SimpleBenchmark.java && java SimpleBenchmark

# C++
cd code/cpp && make && ./matrix_test

# Rust
cd code/rust/mat_mul && cargo test
```

## Performance Analysis

Benchmark results include:
- **Execution time measurements** with statistical analysis
- **Algorithm comparison** (i-j-k vs i-k-j)
- **Scalability trends** across matrix sizes (64x64, 128x128, 256x256)

### Hardware Specifications
- **Apple M3 Pro** with 18GB unified memory
- **ARM64 architecture**
- **Optimized compilation** (-O3 for C++, --release for Rust)

## Data Output

Benchmarks generate:
- **Input matrices**: JSON files in `data/input/`
- **Result matrices**: JSON files in `data/output/`
- **Benchmark results**: CSV files and text outputs in `data/results/`

## Research Paper

Analysis documented in LaTeX:
- **Location**: `paper/matrix_multiplication_benchmark.tex`
- **Compilation**: Requires LaTeX distribution (TeXLive, MiKTeX, or `brew install --cask mactex`)
- **Results**: Real benchmark data included
- **PDF Generation**: `cd paper && pdflatex matrix_multiplication_benchmark.tex`

## Configuration

### Matrix Sizes
Default benchmark sizes: 64×64, 128×128, 256×256

Modify in each language's benchmark file to test different sizes.

### Algorithms
Each implementation includes:
- `matmul_ijk`: Standard algorithm
- `matmul_ikj`: Cache-optimized algorithm
- `matmul_blocked`: Block-based algorithm (selected languages)

## Code Quality

### Implementation Features
- **Separation of concerns**: Production code, tests, and benchmarks separated
- **Reproducible results**: Fixed random seeds for consistent matrix generation
- **Build systems**: Makefile, javac, cargo
- **Statistical analysis**: Multiple runs with mean and standard deviation

### Results Pattern
- **C++/Rust**: Fastest due to native compilation and optimization
- **Java**: Good performance after JIT warmup
- **Python**: Slower due to interpretation overhead
- **Algorithm impact**: i-k-j typically 10-15% faster than i-j-k

## Key Findings

1. **Loop order matters**: i-k-j consistently outperforms i-j-k
2. **Language hierarchy**: C++ ≈ Rust > Java > Python
3. **Cache effects**: More pronounced in larger matrices
4. **M3 Pro performance**: Excellent for all languages due to unified memory

## License

This project is licensed under the MIT License.

## Authors

Matrix multiplication performance analysis by Lukasz Stolzmann.

---

**Note**: This benchmark uses timing methods for educational purposes. For production applications, consider specialized libraries like Intel MKL, OpenBLAS, or CUDA.