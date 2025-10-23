import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 2, jvmArgs = {"-Xms2G", "-Xmx2G"})
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
public class MatrixBenchmark {

    @Param({"64", "128", "256", "512"})
    private int size;

    private double[][] matrixA;
    private double[][] matrixB;

    @Setup(Level.Trial)
    public void setup() {
        matrixA = MatrixUtils.randomMatrix(size, 42);
        matrixB = MatrixUtils.randomMatrix(size, 43);
    }

    @Benchmark
    public double[][] benchmarkIJK() {
        return MatrixUtils.matmulIJK(matrixA, matrixB);
    }

    @Benchmark
    public double[][] benchmarkIKJ() {
        return MatrixUtils.matmulIKJ(matrixA, matrixB);
    }

    @Benchmark
    public double[][] benchmarkKIJ() {
        return MatrixUtils.matmulKIJ(matrixA, matrixB);
    }

    @Benchmark
    public double[][] benchmarkBlocked() {
        return MatrixUtils.matmulBlocked(matrixA, matrixB);
    }

    @Benchmark
    public double[][] benchmarkBlockedCustom() {
        return MatrixUtils.matmulBlocked(matrixA, matrixB, 32);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(MatrixBenchmark.class.getSimpleName())
                .resultFormat(org.openjdk.jmh.results.format.ResultFormatType.JSON)
                .result("benchmark-results.json")
                .build();

        new Runner(opt).run();
    }
}