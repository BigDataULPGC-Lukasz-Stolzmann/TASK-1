import org.junit.Test;
import static org.junit.Assert.*;

public class MatrixUtilsTest {

    private static final double TOLERANCE = 1e-10;

    @Test
    public void testIdentityMatrix() {
        double[][] A = {{1, 2}, {3, 4}};
        double[][] I = {{1, 0}, {0, 1}}; // Identity matrix

        double[][] result = MatrixUtils.matmulIJK(A, I);
        assertMatricesEqual(A, result, "A * I should equal A");

        result = MatrixUtils.matmulIJK(I, A);
        assertMatricesEqual(A, result, "I * A should equal A");
    }

    @Test
    public void testZeroMatrix() {
        double[][] A = {{1, 2}, {3, 4}};
        double[][] Z = {{0, 0}, {0, 0}}; // Zero matrix
        double[][] expected = {{0, 0}, {0, 0}};

        double[][] result = MatrixUtils.matmulIJK(A, Z);
        assertMatricesEqual(expected, result, "A * 0 should equal 0");
    }

    @Test
    public void testAlgorithmConsistency() {
        int size = 8;
        double[][] A = MatrixUtils.randomMatrix(size, 42);
        double[][] B = MatrixUtils.randomMatrix(size, 43);

        double[][] resultIJK = MatrixUtils.matmulIJK(A, B);
        double[][] resultIKJ = MatrixUtils.matmulIKJ(A, B);
        double[][] resultKIJ = MatrixUtils.matmulKIJ(A, B);
        double[][] resultBlocked = MatrixUtils.matmulBlocked(A, B);

        assertMatricesEqual(resultIJK, resultIKJ, "i-j-k and i-k-j should produce same result");
        assertMatricesEqual(resultIJK, resultKIJ, "i-j-k and k-i-j should produce same result");
        assertMatricesEqual(resultIJK, resultBlocked, "i-j-k and blocked should produce same result");
    }

    @Test
    public void testAssociativity() {
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] C = {{9, 10}, {11, 12}};

        // Compute (AB)C
        double[][] AB = MatrixUtils.matmulIJK(A, B);
        double[][] ABC_left = MatrixUtils.matmulIJK(AB, C);

        // Compute A(BC)
        double[][] BC = MatrixUtils.matmulIJK(B, C);
        double[][] ABC_right = MatrixUtils.matmulIJK(A, BC);

        assertMatricesEqual(ABC_left, ABC_right, "Matrix multiplication should be associative");
    }

    @Test
    public void testDistributivity() {
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] C = {{1, 1}, {1, 1}};

        // Compute A(B+C)
        double[][] B_plus_C = matrixAdd(B, C);
        double[][] A_BC = MatrixUtils.matmulIJK(A, B_plus_C);

        // Compute AB + AC
        double[][] AB = MatrixUtils.matmulIJK(A, B);
        double[][] AC = MatrixUtils.matmulIJK(A, C);
        double[][] AB_plus_AC = matrixAdd(AB, AC);

        assertMatricesEqual(A_BC, AB_plus_AC, "Matrix multiplication should be distributive");
    }

    @Test
    public void testBlockSizeVariations() {
        int size = 16;
        double[][] A = MatrixUtils.randomMatrix(size, 42);
        double[][] B = MatrixUtils.randomMatrix(size, 43);

        double[][] resultDefault = MatrixUtils.matmulBlocked(A, B);
        double[][] resultBlock4 = MatrixUtils.matmulBlocked(A, B, 4);
        double[][] resultBlock8 = MatrixUtils.matmulBlocked(A, B, 8);

        assertMatricesEqual(resultDefault, resultBlock4, "Different block sizes should produce same result");
        assertMatricesEqual(resultDefault, resultBlock8, "Different block sizes should produce same result");
    }

    @Test
    public void testRandomMatrixGeneration() {
        int size = 5;
        double[][] M1 = MatrixUtils.randomMatrix(size, 42);
        double[][] M2 = MatrixUtils.randomMatrix(size, 42); // Same seed

        assertMatricesEqual(M1, M2, "Same seed should produce same matrix");

        double[][] M3 = MatrixUtils.randomMatrix(size, 43); // Different seed
        assertFalse("Different seeds should produce different matrices",
                   matricesEqual(M1, M3, TOLERANCE));
    }

    private void assertMatricesEqual(double[][] expected, double[][] actual, String message) {
        assertTrue(message, matricesEqual(expected, actual, TOLERANCE));
    }

    private boolean matricesEqual(double[][] M1, double[][] M2, double tolerance) {
        if (M1.length != M2.length || M1[0].length != M2[0].length) {
            return false;
        }
        for (int i = 0; i < M1.length; i++) {
            for (int j = 0; j < M1[0].length; j++) {
                if (Math.abs(M1[i][j] - M2[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    private double[][] matrixAdd(double[][] A, double[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    }
}