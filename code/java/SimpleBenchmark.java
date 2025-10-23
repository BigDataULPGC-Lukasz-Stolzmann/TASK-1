import java.util.Random;

public class SimpleBenchmark {

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

    public static void main(String[] args) {
        int[] sizes = {64, 128, 256};
        int runs = 5;

        System.out.println("Java Matrix Multiplication Benchmark");
        System.out.println("=====================================");

        for (int size : sizes) {
            System.out.println("\nMatrix size: " + size + "x" + size);

            double[][] A = randomMatrix(size, 42);
            double[][] B = randomMatrix(size, 43);

            // Benchmark i-j-k
            double[] timesIJK = new double[runs];
            for (int r = 0; r < runs; r++) {
                long start = System.nanoTime();
                matmulIJK(A, B);
                long end = System.nanoTime();
                timesIJK[r] = (end - start) / 1e9;
            }

            // Benchmark i-k-j
            double[] timesIKJ = new double[runs];
            for (int r = 0; r < runs; r++) {
                long start = System.nanoTime();
                matmulIKJ(A, B);
                long end = System.nanoTime();
                timesIKJ[r] = (end - start) / 1e9;
            }

            double meanIJK = mean(timesIJK);
            double meanIKJ = mean(timesIKJ);

            System.out.printf("  i-j-k: %.4fs ± %.4fs%n", meanIJK, stddev(timesIJK));
            System.out.printf("  i-k-j: %.4fs ± %.4fs%n", meanIKJ, stddev(timesIKJ));
            System.out.printf("  Speedup: %.2fx%n", meanIJK / meanIKJ);
        }
    }

    private static double mean(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }

    private static double stddev(double[] values) {
        double m = mean(values);
        double sum = 0;
        for (double v : values) sum += (v - m) * (v - m);
        return Math.sqrt(sum / values.length);
    }
}