import java.util.Arrays;

public class Benchmark {
    public static void main(String[] args) {
        int N = 256;
        int RUNS = 10;

        double[][] A = MatrixUtils.randomMatrix(N, 42);
        double[][] B = MatrixUtils.randomMatrix(N, 43);
        double[] times = new double[RUNS];

        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            double[][] C = MatrixUtils.matmulIJK(A, B);
            long end = System.nanoTime();
            times[r] = (end - start) / 1_000_000_000.0; 
        }

        double mean = Arrays.stream(times).average().orElse(0.0);

        System.out.printf("Java ijk %dx%d%n", N, N);
        System.out.printf("Mean time: %.6f s%n", mean);
    }
}
