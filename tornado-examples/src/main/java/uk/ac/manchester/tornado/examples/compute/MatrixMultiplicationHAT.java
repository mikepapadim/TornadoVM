/*
 * Copyright (c) 2025, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package uk.ac.manchester.tornado.examples.compute;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Matrix Multiplication implementations inspired by the HAT (Heterogeneous Accelerated Transformations) framework
 * from the OpenJDK Babylon project. This demonstrates various optimization strategies from naive to highly optimized kernels.
 *
 * <p>
 * Implementations include:
 * <ul>
 * <li>Naive 2D kernel - Basic parallel matrix multiplication</li>
 * <li>2D Tiled kernel - Uses local memory for improved cache efficiency</li>
 * <li>Register Tiling - Exploits GPU memory hierarchy with work-item private memory</li>
 * </ul>
 * </p>
 *
 * <p>
 * How to run:
 * </p>
 * <code>
 * $ tornado -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplicationHAT [matrixSize] [tileSize] [workPerThread]
 * </code>
 */
public class MatrixMultiplicationHAT {

    private static final int WARM_UP_ITERATIONS = 15;
    private static final int BENCHMARK_ITERATIONS = 10;
    private static final float DELTA = 0.01f;

    /**
     * Naive 2D matrix multiplication kernel.
     * Each thread computes one element of the result matrix.
     */
    public static void matrixMultiplyNaive2D(KernelContext context, FloatArray A, FloatArray B, FloatArray C, int N) {
        int row = context.globalIdx;
        int col = context.globalIdy;

        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A.get(row * N + k) * B.get(k * N + col);
            }
            C.set(row * N + col, sum);
        }
    }

    /**
     * 2D matrix multiplication with tiling and local memory.
     * Uses shared local memory to reduce global memory accesses.
     * Based on the myGEMM pattern (https://github.com/cnugteren/myGEMM).
     */
    public static void matrixMultiplyTiled2D(KernelContext context, FloatArray A, FloatArray B, FloatArray C, int N, int tileSize) {
        // Thread identifiers
        int localRow = context.localIdx;     // Local row ID (max: tileSize)
        int localCol = context.localIdy;     // Local col ID (max: tileSize)
        int globalRow = tileSize * context.groupIdx + localRow;   // Row ID of C
        int globalCol = tileSize * context.groupIdy + localCol;   // Col ID of C

        // Allocate local memory tiles
        float[] localA = context.allocateFloatLocalArray(tileSize * tileSize);
        float[] localB = context.allocateFloatLocalArray(tileSize * tileSize);

        float sum = 0.0f;

        // Loop over all tiles in the K dimension
        int numTiles = N / tileSize;
        for (int t = 0; t < numTiles; t++) {
            // Load one tile of A and B into local memory
            int tiledRow = tileSize * t + localRow;
            int tiledCol = tileSize * t + localCol;

            // Load tile of A (transposed in local memory for coalescing)
            localA[localCol * tileSize + localRow] = A.get(tiledCol * N + globalRow);
            // Load tile of B
            localB[localCol * tileSize + localRow] = B.get(globalCol * N + tiledRow);

            // Synchronize to ensure tile is loaded
            context.localBarrier();

            // Compute partial result for this tile
            for (int k = 0; k < tileSize; k++) {
                sum += localA[k * tileSize + localRow] * localB[localCol * tileSize + k];
            }

            // Synchronize before loading next tile
            context.localBarrier();
        }

        // Store final result
        C.set(globalCol * N + globalRow, sum);
    }

    /**
     * 2D matrix multiplication with work-item level optimization.
     * Each thread computes multiple elements (workPerThread) to increase computational intensity.
     * Uses register blocking to reduce memory traffic.
     */
    public static void matrixMultiplyRegisterTiled2D(KernelContext context, FloatArray A, FloatArray B, FloatArray C, int N, int tileSize, int workPerThread) {
        // Thread identifiers
        int localRow = context.localIdx;
        int localCol = context.localIdy;
        int globalRow = tileSize * context.groupIdx + localRow;
        int globalCol = workPerThread * (tileSize * context.groupIdy + localCol);

        // Allocate local memory
        float[] localA = context.allocateFloatLocalArray(tileSize * tileSize);
        float[] localB = context.allocateFloatLocalArray(tileSize * tileSize);

        // Register accumulators for workPerThread results
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        // Loop over tiles
        int numTiles = N / tileSize;
        for (int t = 0; t < numTiles; t++) {
            int tiledRow = tileSize * t + localRow;
            int tiledCol = tileSize * t + localCol;

            // Load tile of A (transposed)
            localA[localCol * tileSize + localRow] = A.get(tiledCol * N + globalRow);

            context.localBarrier();

            // Compute workPerThread results using the loaded tile of A
            for (int w = 0; w < workPerThread; w++) {
                int bCol = globalCol + w;
                // Load one column of B for this work item
                localB[localCol * tileSize + localRow] = B.get(bCol * N + tiledRow);

                context.localBarrier();

                // Accumulate partial results
                for (int k = 0; k < tileSize; k++) {
                    float aVal = localA[k * tileSize + localRow];
                    float bVal = localB[localCol * tileSize + k];
                    if (w == 0) sum0 += aVal * bVal;
                    else if (w == 1) sum1 += aVal * bVal;
                    else if (w == 2) sum2 += aVal * bVal;
                    else if (w == 3) sum3 += aVal * bVal;
                }

                context.localBarrier();
            }
        }

        // Write results
        C.set(globalCol * N + globalRow, sum0);
        if (globalCol + 1 < N) C.set((globalCol + 1) * N + globalRow, sum1);
        if (globalCol + 2 < N) C.set((globalCol + 2) * N + globalRow, sum2);
        if (globalCol + 3 < N) C.set((globalCol + 3) * N + globalRow, sum3);
    }

    /**
     * Simple parallel matrix multiplication using @Parallel annotation.
     * This is the baseline TornadoVM implementation.
     */
    public static void matrixMultiplyParallel(FloatArray A, FloatArray B, FloatArray C, int N) {
        for (@Parallel int i = 0; i < N; i++) {
            for (@Parallel int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A.get(i * N + k) * B.get(k * N + j);
                }
                C.set(i * N + j, sum);
            }
        }
    }

    /**
     * Sequential implementation for verification.
     */
    private static void matrixMultiplySequential(FloatArray A, FloatArray B, FloatArray C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A.get(i * N + k) * B.get(k * N + j);
                }
                C.set(i * N + j, sum);
            }
        }
    }

    private static void initializeMatrix(FloatArray matrix, int N) {
        var random = new Random(42);
        for (int i = 0; i < N * N; i++) {
            matrix.set(i, random.nextFloat());
        }
    }

    private static boolean verify(FloatArray result, FloatArray expected, int N) {
        for (int i = 0; i < N * N; i++) {
            if (Math.abs(result.get(i) - expected.get(i)) > DELTA) {
                System.out.printf("Mismatch at index %d: expected %.6f, got %.6f (diff: %.6f)\n",
                    i, expected.get(i), result.get(i), Math.abs(result.get(i) - expected.get(i)));
                return false;
            }
        }
        return true;
    }

    private static void printStats(String name, LongSummaryStatistics stats, int N) {
        double avgTimeMs = stats.getAverage() / 1_000_000.0;
        double minTimeMs = stats.getMin() / 1_000_000.0;
        double maxTimeMs = stats.getMax() / 1_000_000.0;
        double flops = 2.0 * N * N * N;
        double gflops = (flops * 1e-9) / (avgTimeMs / 1000.0);

        System.out.printf("%s:\n", name);
        System.out.printf("  Average time: %.3f ms\n", avgTimeMs);
        System.out.printf("  Min time: %.3f ms\n", minTimeMs);
        System.out.printf("  Max time: %.3f ms\n", maxTimeMs);
        System.out.printf("  Performance: %.2f GFLOP/s\n", gflops);
    }

    public static void main(String[] args) {
        // Parse arguments
        int N = 512;
        int tileSize = 16;
        int workPerThread = 4;

        if (args.length >= 1) {
            N = Integer.parseInt(args[0]);
        }
        if (args.length >= 2) {
            tileSize = Integer.parseInt(args[1]);
        }
        if (args.length >= 3) {
            workPerThread = Integer.parseInt(args[2]);
        }

        System.out.println("Matrix Multiplication HAT-Inspired Implementation");
        System.out.println("==================================================");
        System.out.printf("Matrix size: %d x %d\n", N, N);
        System.out.printf("Tile size: %d\n", tileSize);
        System.out.printf("Work per thread: %d\n", workPerThread);
        System.out.println();

        // Allocate matrices
        var A = new FloatArray(N * N);
        var B = new FloatArray(N * N);
        var C_sequential = new FloatArray(N * N);
        var C_parallel = new FloatArray(N * N);
        var C_naive = new FloatArray(N * N);
        var C_tiled = new FloatArray(N * N);
        var C_register = new FloatArray(N * N);

        // Initialize input matrices
        initializeMatrix(A, N);
        initializeMatrix(B, N);

        // Timing lists
        var sequentialTimes = new ArrayList<Long>();
        var parallelTimes = new ArrayList<Long>();
        var naiveTimes = new ArrayList<Long>();
        var tiledTimes = new ArrayList<Long>();
        var registerTimes = new ArrayList<Long>();

        // Setup TornadoVM task graphs

        // 1. Parallel (@Parallel annotation)
        var taskGraphParallel = new TaskGraph("parallel")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
            .task("t0", MatrixMultiplicationHAT::matrixMultiplyParallel, A, B, C_parallel, N)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, C_parallel);
        var immutableParallel = taskGraphParallel.snapshot();
        var executorParallel = new TornadoExecutionPlan(immutableParallel);

        // 2. Naive 2D kernel
        var workerNaive = new WorkerGrid2D(N, N);
        workerNaive.setLocalWork(16, 16, 1);
        var schedulerNaive = new GridScheduler("naive.t0", workerNaive);

        var taskGraphNaive = new TaskGraph("naive")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
            .task("t0", MatrixMultiplicationHAT::matrixMultiplyNaive2D, new KernelContext(), A, B, C_naive, N)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, C_naive);
        var immutableNaive = taskGraphNaive.snapshot();
        var executorNaive = new TornadoExecutionPlan(immutableNaive);
        executorNaive.withGridScheduler(schedulerNaive);

        // 3. Tiled 2D kernel
        var workerTiled = new WorkerGrid2D(N, N);
        workerTiled.setLocalWork(tileSize, tileSize, 1);
        var schedulerTiled = new GridScheduler("tiled.t0", workerTiled);

        var taskGraphTiled = new TaskGraph("tiled")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
            .task("t0", MatrixMultiplicationHAT::matrixMultiplyTiled2D, new KernelContext(), A, B, C_tiled, N, tileSize)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, C_tiled);
        var immutableTiled = taskGraphTiled.snapshot();
        var executorTiled = new TornadoExecutionPlan(immutableTiled);
        executorTiled.withGridScheduler(schedulerTiled);

        // 4. Register tiled kernel - each thread computes workPerThread columns
        var workerRegister = new WorkerGrid2D(N, N / workPerThread);
        workerRegister.setLocalWork(tileSize, tileSize, 1);
        var schedulerRegister = new GridScheduler("register.t0", workerRegister);

        var taskGraphRegister = new TaskGraph("register")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
            .task("t0", MatrixMultiplicationHAT::matrixMultiplyRegisterTiled2D, new KernelContext(), A, B, C_register, N, tileSize, workPerThread)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, C_register);
        var immutableRegister = taskGraphRegister.snapshot();
        var executorRegister = new TornadoExecutionPlan(immutableRegister);
        executorRegister.withGridScheduler(schedulerRegister);

        // Warm-up
        System.out.println("Warming up implementations...");
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            matrixMultiplySequential(A, B, C_sequential, N);
            executorParallel.execute();
            executorNaive.execute();
            executorTiled.execute();
            executorRegister.execute();
        }

        // Benchmark sequential
        System.out.println("Benchmarking sequential implementation...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            matrixMultiplySequential(A, B, C_sequential, N);
            long end = System.nanoTime();
            sequentialTimes.add(end - start);
        }

        // Benchmark parallel
        System.out.println("Benchmarking @Parallel implementation...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            executorParallel.execute();
            long end = System.nanoTime();
            parallelTimes.add(end - start);
        }

        // Benchmark naive
        System.out.println("Benchmarking naive 2D kernel...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            executorNaive.execute();
            long end = System.nanoTime();
            naiveTimes.add(end - start);
        }

        // Benchmark tiled
        System.out.println("Benchmarking tiled 2D kernel...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            executorTiled.execute();
            long end = System.nanoTime();
            tiledTimes.add(end - start);
        }

        // Benchmark register tiled
        System.out.println("Benchmarking register tiled kernel...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            executorRegister.execute();
            long end = System.nanoTime();
            registerTimes.add(end - start);
        }

        // Compute statistics
        var statsSeq = sequentialTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        var statsParallel = parallelTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        var statsNaive = naiveTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        var statsTiled = tiledTimes.stream().mapToLong(Long::longValue).summaryStatistics();
        var statsRegister = registerTimes.stream().mapToLong(Long::longValue).summaryStatistics();

        // Print results
        System.out.println("\nPerformance Results:");
        System.out.println("====================");
        printStats("Sequential", statsSeq, N);
        printStats("@Parallel", statsParallel, N);
        printStats("Naive 2D Kernel", statsNaive, N);
        printStats("Tiled 2D Kernel", statsTiled, N);
        printStats("Register Tiled Kernel", statsRegister, N);

        // Verification
        System.out.println("\nVerification:");
        System.out.println("=============");
        System.out.println("@Parallel vs Sequential: " + (verify(C_parallel, C_sequential, N) ? "PASSED" : "FAILED"));
        System.out.println("Naive 2D vs Sequential: " + (verify(C_naive, C_sequential, N) ? "PASSED" : "FAILED"));
        System.out.println("Tiled 2D vs Sequential: " + (verify(C_tiled, C_sequential, N) ? "PASSED" : "FAILED"));
        System.out.println("Register Tiled vs Sequential: " + (verify(C_register, C_sequential, N) ? "PASSED" : "FAILED"));

        // Speedups
        System.out.println("\nSpeedups (vs Sequential):");
        System.out.println("=========================");
        System.out.printf("@Parallel: %.2fx\n", statsSeq.getAverage() / statsParallel.getAverage());
        System.out.printf("Naive 2D: %.2fx\n", statsSeq.getAverage() / statsNaive.getAverage());
        System.out.printf("Tiled 2D: %.2fx\n", statsSeq.getAverage() / statsTiled.getAverage());
        System.out.printf("Register Tiled: %.2fx\n", statsSeq.getAverage() / statsRegister.getAverage());

        System.out.println("\nSpeedups (Optimized vs Naive):");
        System.out.println("================================");
        System.out.printf("Tiled vs Naive: %.2fx\n", statsNaive.getAverage() / statsTiled.getAverage());
        System.out.printf("Register Tiled vs Naive: %.2fx\n", statsNaive.getAverage() / statsRegister.getAverage());
        System.out.printf("Register Tiled vs Tiled: %.2fx\n", statsTiled.getAverage() / statsRegister.getAverage());
    }
}
