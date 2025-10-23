use rand::prelude::*;

pub fn random_matrix<const N: usize>() -> [[f32; N]; N] {
    let mut rng = rand::rng();
    let mut matrix = [[0.0f32; N]; N];

    matrix
        .iter_mut()
        .take(N)
        .for_each(|row| row.iter_mut().take(N).for_each(|x| *x = rng.random()));
    matrix
}

pub fn matmul_ijk<const N: usize>(a: [[f32; N]; N], b: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut c = [[0.0f32; N]; N];
    for i in 0..N {
        for (k, b_row) in b.iter().enumerate().take(N) {
            for (j, b_el) in b_row.iter().enumerate().take(N) {
                c[i][j] += a[i][k] * b_el;
            }
        }
    }
    c
}

pub fn matmul_ikj<const N: usize>(a: [[f32; N]; N], b: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut c = [[0.0f32; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            for j in 0..N {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

pub fn matmul_kij<const N: usize>(a: [[f32; N]; N], b: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut c = [[0.0f32; N]; N];
    for k in 0..N {
        for i in 0..N {
            for j in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

pub fn matmul_blocked<const N: usize>(
    a: [[f32; N]; N],
    b: [[f32; N]; N],
    block_size: usize,
) -> [[f32; N]; N] {
    let mut c = [[0.0f32; N]; N];

    let mut ii = 0;
    while ii < N {
        let mut jj = 0;
        while jj < N {
            let mut kk = 0;
            while kk < N {
                let i_end = (ii + block_size).min(N);
                let j_end = (jj + block_size).min(N);
                let k_end = (kk + block_size).min(N);

                for i in ii..i_end {
                    for j in jj..j_end {
                        for k in kk..k_end {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                kk += block_size;
            }
            jj += block_size;
        }
        ii += block_size;
    }
    c
}
