use mat_mul::{random_matrix, matmul_ijk, matmul_ikj, matmul_kij, matmul_blocked};

const TOLERANCE: f64 = 1e-10;

fn matrices_equal<const N: usize>(m1: [[f64; N]; N], m2: [[f64; N]; N]) -> bool {
    for i in 0..N {
        for j in 0..N {
            if (m1[i][j] - m2[i][j]).abs() > TOLERANCE {
                return false;
            }
        }
    }
    true
}

fn matrix_add<const N: usize>(a: [[f64; N]; N], b: [[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

#[test]
fn test_identity_matrix() {
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let i: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]]; // Identity matrix

    let result = matmul_ijk(a, i);
    assert!(matrices_equal(a, result), "A * I should equal A");

    let result = matmul_ijk(i, a);
    assert!(matrices_equal(a, result), "I * A should equal A");
}

#[test]
fn test_zero_matrix() {
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let z: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 0.0]]; // Zero matrix
    let expected: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 0.0]];

    let result = matmul_ijk(a, z);
    assert!(matrices_equal(expected, result), "A * 0 should equal 0");
}

#[test]
fn test_algorithm_consistency() {
    const SIZE: usize = 8;
    let a = random_matrix::<SIZE>(42);
    let b = random_matrix::<SIZE>(43);

    let result_ijk = matmul_ijk(a, b);
    let result_ikj = matmul_ikj(a, b);
    let result_kij = matmul_kij(a, b);
    let result_blocked = matmul_blocked(a, b, 4);

    assert!(matrices_equal(result_ijk, result_ikj), "i-j-k and i-k-j should produce same result");
    assert!(matrices_equal(result_ijk, result_kij), "i-j-k and k-i-j should produce same result");
    assert!(matrices_equal(result_ijk, result_blocked), "i-j-k and blocked should produce same result");
}

#[test]
fn test_associativity() {
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let b: [[f64; 2]; 2] = [[5.0, 6.0], [7.0, 8.0]];
    let c: [[f64; 2]; 2] = [[9.0, 10.0], [11.0, 12.0]];

    // Compute (AB)C
    let ab = matmul_ijk(a, b);
    let abc_left = matmul_ijk(ab, c);

    // Compute A(BC)
    let bc = matmul_ijk(b, c);
    let abc_right = matmul_ijk(a, bc);

    assert!(matrices_equal(abc_left, abc_right), "Matrix multiplication should be associative");
}

#[test]
fn test_distributivity() {
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let b: [[f64; 2]; 2] = [[5.0, 6.0], [7.0, 8.0]];
    let c: [[f64; 2]; 2] = [[1.0, 1.0], [1.0, 1.0]];

    // Compute A(B+C)
    let b_plus_c = matrix_add(b, c);
    let a_bc = matmul_ijk(a, b_plus_c);

    // Compute AB + AC
    let ab = matmul_ijk(a, b);
    let ac = matmul_ijk(a, c);
    let ab_plus_ac = matrix_add(ab, ac);

    assert!(matrices_equal(a_bc, ab_plus_ac), "Matrix multiplication should be distributive");
}

#[test]
fn test_block_size_variations() {
    const SIZE: usize = 16;
    let a = random_matrix::<SIZE>(42);
    let b = random_matrix::<SIZE>(43);

    let result_default = matmul_blocked(a, b, 64);
    let result_block_4 = matmul_blocked(a, b, 4);
    let result_block_8 = matmul_blocked(a, b, 8);

    assert!(matrices_equal(result_default, result_block_4), "Different block sizes should produce same result");
    assert!(matrices_equal(result_default, result_block_8), "Different block sizes should produce same result");
}

#[test]
fn test_random_matrix_generation() {
    const SIZE: usize = 5;
    let m1 = random_matrix::<SIZE>(42);
    let m2 = random_matrix::<SIZE>(42); // Same seed

    assert!(matrices_equal(m1, m2), "Same seed should produce same matrix");

    let m3 = random_matrix::<SIZE>(43); // Different seed
    assert!(!matrices_equal(m1, m3), "Different seeds should produce different matrices");
}