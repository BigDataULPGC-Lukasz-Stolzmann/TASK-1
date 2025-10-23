import json
import csv
from datetime import datetime

def save_matrices(A, B, size, filepath):
    """Save input matrices to JSON file."""
    data = {
        "size": size,
        "timestamp": datetime.now().isoformat(),
        "matrix_A": A,
        "matrix_B": B
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_result_matrix(C, size, algorithm, filepath):
    """Save result matrix to JSON file."""
    data = {
        "size": size,
        "algorithm": algorithm,
        "timestamp": datetime.now().isoformat(),
        "result_matrix": C
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_benchmark_results(results, filepath):
    """Save benchmark results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Language', 'Algorithm', 'Size', 'Mean_Time_s', 'Std_Time_s', 'Speedup'])
        for result in results:
            writer.writerow([
                result['language'],
                result['algorithm'],
                result['size'],
                result['mean_time'],
                result['std_time'],
                result.get('speedup', '')
            ])

def load_matrices(filepath):
    """Load matrices from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['matrix_A'], data['matrix_B'], data['size']