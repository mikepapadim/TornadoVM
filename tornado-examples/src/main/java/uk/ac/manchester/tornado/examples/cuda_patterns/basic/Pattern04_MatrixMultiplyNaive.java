/*
 * Copyright (c) 2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.examples.cuda_patterns.basic;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * CUDA Pattern 04: Matrix Multiplication (Naive)
 *
 * Demonstrates:
 * - Basic matrix multiplication algorithm
 * - Global memory access patterns
 * - Computational intensity
 * - Performance baseline for optimization
 *
 * Computes C = A × B where:
 * - A is M×K matrix
 * - B is K×N matrix
 * - C is M×N matrix
 *
 * Each thread computes one element of the output matrix.
 *
 * CUDA equivalent:
 * __global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     if (row < M && col < N) {
 *         float sum = 0.0f;
 *         for (int k = 0; k < K; k++) {
 *             sum += A[row * K + k] * B[k * N + col];
 *         }
 *         C[row * N + col] = sum;
 *     }
 * }
 */
public class Pattern04_MatrixMultiplyNaive {

    /**
     * Naive matrix multiplication kernel
     *
     * @param context Kernel execution context
     * @param A Input matrix A (M×K)
     * @param B Input matrix B (K×N)
     * @param C Output matrix C (M×N)
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     */
    public static void matmulNaive(KernelContext context, FloatArray A, FloatArray B, FloatArray C, int M, int N, int K) {
        int row = context.globalIdy;  // Output row
        int col = context.globalIdx;  // Output column

        if (row < M && col < N) {
            float sum = 0.0f;

            // Compute dot product of row from A and column from B
            for (int k = 0; k < K; k++) {
                sum += A.get(row * K + k) * B.get(k * N + col);
            }

            C.set(row * N + col, sum);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void matmulCPU(FloatArray A, FloatArray B, FloatArray C, int M, int N, int K) {
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A.get(row * K + k) * B.get(k * N + col);
                }
                C.set(row * N + col, sum);
            }
        }
    }

    public static void main(String[] args) {
        final int M = 256;  // Rows in A and C
        final int N = 256;  // Columns in B and C
        final int K = 256;  // Columns in A, rows in B
        final int blockSize = 16;

        // Initialize matrices
        FloatArray A = new FloatArray(M * K);
        FloatArray B = new FloatArray(K * N);
        FloatArray C = new FloatArray(M * N);
        FloatArray expected = new FloatArray(M * N);

        // Fill with test data
        for (int i = 0; i < M * K; i++) {
            A.set(i, (i % 10) / 10.0f);
        }
        for (int i = 0; i < K * N; i++) {
            B.set(i, (i % 10) / 10.0f);
        }

        // Compute expected result on CPU
        matmulCPU(A, B, expected, M, N, K);

        // Create KernelContext and configure 2D grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid2D(N, M);  // Note: N=cols, M=rows
        worker.setGlobalWork(N, M, 1);
        worker.setLocalWork(blockSize, blockSize, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B) //
                .task("t0", Pattern04_MatrixMultiplyNaive::matmulNaive, context, A, B, C, M, N, K) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, C);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < M * N; i++) {
            if (Math.abs(C.get(i) - expected.get(i)) > 0.01f) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + C.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 04 - Matrix Multiplication (Naive): PASSED");
            System.out.println("Successfully computed " + M + "×" + K + " × " + K + "×" + N + " = " + M + "×" + N);
        } else {
            System.out.println("Pattern 04 - Matrix Multiplication (Naive): FAILED");
        }
    }
}
