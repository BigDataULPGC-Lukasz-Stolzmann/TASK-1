#!/bin/bash

# Professional Matrix Multiplication Benchmark Suite
# This script runs all benchmarks and collects results

set -e

echo "=== Matrix Multiplication Benchmark Suite ==="
echo "Running professional benchmarks for Python, Java, C++, and Rust"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create results directory
mkdir -p data/results

# Python benchmarks (simple version)
echo "🐍 Running Python benchmarks..."
cd "$SCRIPT_DIR/code/python"
python3 simple_test.py > ../../data/results/python_results.txt
echo "Python benchmarks completed ✅"
echo ""

# Java benchmarks (simple version)
echo "☕ Running Java benchmarks..."
cd "$SCRIPT_DIR/code/java"
javac SimpleBenchmark.java
if [ -f SimpleBenchmark.class ]; then
    java SimpleBenchmark > ../../data/results/java_results.txt
    echo "Java benchmarks completed ✅"
else
    echo "Java compilation failed ⚠️"
fi
echo ""

# C++ benchmarks (simple version)
echo "⚡ Running C++ benchmarks..."
cd "$SCRIPT_DIR/code/cpp"
make clean && make
if [ -f simple_benchmark ]; then
    ./simple_benchmark > ../../data/results/cpp_results.txt
    echo "C++ benchmarks completed ✅"
else
    echo "C++ compilation failed ⚠️"
fi
echo ""

# Rust benchmarks (simple version)
echo "🦀 Running Rust benchmarks..."
cd "$SCRIPT_DIR/code/rust/mat_mul"
if command -v cargo &> /dev/null; then
    cargo run --release > ../../../data/results/rust_results.txt
    echo "Rust benchmarks completed ✅"
else
    echo "Rust not installed - skipping Rust benchmarks"
fi
echo ""

echo "=== All benchmarks completed! ==="
echo "Results saved in data/results/"
echo ""
echo "Files generated:"
echo "  - python_results.txt (simple benchmark)"
echo "  - java_results.txt (simple benchmark)"
echo "  - cpp_results.txt (simple benchmark)"
echo "  - rust_results.txt (simple benchmark)"
echo ""
echo "Input/Output data saved in:"
echo "  - data/input/ (test matrices)"
echo "  - data/output/ (result matrices and CSV files)"