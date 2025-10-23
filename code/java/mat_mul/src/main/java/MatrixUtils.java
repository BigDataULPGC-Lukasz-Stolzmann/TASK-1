import java.util.Random;

public class MatrixUtils {

    /**
     * Generate a random n×n matrix with values between 0 and 1.
     */
    public static double[][] randomMatrix(int n, long seed) {
        double[][] M = new double[n][n];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = rng.nextDouble();
            }
        }
        return M;
    }

    /**
     * Standard i-j-k matrix multiplication.
     */
    public static double[][] matmulIJK(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    /**
     * i-k-j matrix multiplication (better cache locality).
     */
    public static double[][] matmulIKJ(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = A[i][k];
                for (int j = 0; j < n; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
        return C;
    }

    /**
     * k-i-j matrix multiplication.
     */
    public static double[][] matmulKIJ(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    /**
     * Block matrix multiplication for better cache performance.
     */
    public static double[][] matmulBlocked(double[][] A, double[][] B, int blockSize) {
        int n = A.length;
        double[][] C = new double[n][n];

        for (int ii = 0; ii < n; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {
                    // Block boundaries
                    int iEnd = Math.min(ii + blockSize, n);
                    int jEnd = Math.min(jj + blockSize, n);
                    int kEnd = Math.min(kk + blockSize, n);

                    // Block multiplication
                    for (int i = ii; i < iEnd; i++) {
                        for (int j = jj; j < jEnd; j++) {
                            for (int k = kk; k < kEnd; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
        return C;
    }

    /**
     * Block matrix multiplication with default block size.
     */
    public static double[][] matmulBlocked(double[][] A, double[][] B) {
        return matmulBlocked(A, B, 64);
    }
}
