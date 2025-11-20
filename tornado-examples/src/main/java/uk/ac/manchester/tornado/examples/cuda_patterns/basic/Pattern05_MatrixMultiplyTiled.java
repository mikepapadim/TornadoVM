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
 * CUDA Pattern 05: Matrix Multiplication (Tiled with Shared Memory)
 *
 * Demonstrates:
 * - Tiling optimization for matrix multiplication
 * - Local/shared memory usage to reduce global memory access
 * - Thread synchronization with barriers
 * - Memory coalescing and reuse
 *
 * This optimized version loads tiles of A and B into shared memory,
 * significantly reducing global memory traffic and improving performance.
 *
 * Performance improvement: ~10-20x over naive implementation for large matrices.
 *
 * CUDA equivalent:
 * __global__ void matmulTiled(float *A, float *B, float *C, int M, int N, int K) {
 *     __shared__ float As[TILE_SIZE][TILE_SIZE];
 *     __shared__ float Bs[TILE_SIZE][TILE_SIZE];
 *
 *     int row = blockIdx.y * TILE_SIZE + threadIdx.y;
 *     int col = blockIdx.x * TILE_SIZE + threadIdx.x;
 *
 *     float sum = 0.0f;
 *     for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
 *         // Load tile into shared memory
 *         As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
 *         Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
 *         __syncthreads();
 *
 *         // Compute partial dot product
 *         for (int k = 0; k < TILE_SIZE; k++)
 *             sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
 *         __syncthreads();
 *     }
 *     C[row * N + col] = sum;
 * }
 */
public class Pattern05_MatrixMultiplyTiled {

    private static final int TILE_SIZE = 16;

    /**
     * Tiled matrix multiplication kernel with shared memory
     *
     * @param context Kernel execution context
     * @param A Input matrix A (M×K)
     * @param B Input matrix B (K×N)
     * @param C Output matrix C (M×N)
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     */
    public static void matmulTiled(KernelContext context, FloatArray A, FloatArray B, FloatArray C, int M, int N, int K) {
        int row = context.groupIdy * TILE_SIZE + context.localIdy;
        int col = context.groupIdx * TILE_SIZE + context.localIdx;

        int localRow = context.localIdy;
        int localCol = context.localIdx;

        // Allocate shared memory for tiles
        float[] As = context.allocateFloatLocalArray(TILE_SIZE * TILE_SIZE);
        float[] Bs = context.allocateFloatLocalArray(TILE_SIZE * TILE_SIZE);

        float sum = 0.0f;

        // Loop over tiles
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            // Load tile from A into shared memory
            int aCol = t * TILE_SIZE + localCol;
            if (row < M && aCol < K) {
                As[localRow * TILE_SIZE + localCol] = A.get(row * K + aCol);
            } else {
                As[localRow * TILE_SIZE + localCol] = 0.0f;
            }

            // Load tile from B into shared memory
            int bRow = t * TILE_SIZE + localRow;
            if (bRow < K && col < N) {
                Bs[localRow * TILE_SIZE + localCol] = B.get(bRow * N + col);
            } else {
                Bs[localRow * TILE_SIZE + localCol] = 0.0f;
            }

            // Synchronize to ensure tiles are loaded
            context.localBarrier();

            // Compute partial dot product using shared memory
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[localRow * TILE_SIZE + k] * Bs[k * TILE_SIZE + localCol];
            }

            // Synchronize before loading next tile
            context.localBarrier();
        }

        // Write result to global memory
        if (row < M && col < N) {
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
        final int M = 512;  // Rows in A and C
        final int N = 512;  // Columns in B and C
        final int K = 512;  // Columns in A, rows in B

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
        WorkerGrid worker = new WorkerGrid2D(N, M);
        worker.setGlobalWork(N, M, 1);
        worker.setLocalWork(TILE_SIZE, TILE_SIZE, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B) //
                .task("t0", Pattern05_MatrixMultiplyTiled::matmulTiled, context, A, B, C, M, N, K) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, C);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < M * N; i++) {
            if (Math.abs(C.get(i) - expected.get(i)) > 0.1f) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + C.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 05 - Matrix Multiplication (Tiled): PASSED");
            System.out.println("Successfully computed " + M + "×" + K + " × " + K + "×" + N + " = " + M + "×" + N);
            System.out.println("Using tile size: " + TILE_SIZE + "×" + TILE_SIZE);
        } else {
            System.out.println("Pattern 05 - Matrix Multiplication (Tiled): FAILED");
        }
    }
}
