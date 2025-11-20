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
package uk.ac.manchester.tornado.examples.cuda_patterns.sparse;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * CUDA Pattern 22: Sparse Matrix-Vector Multiplication (SpMV) - COO Format
 *
 * Demonstrates:
 * - Sparse matrix operations
 * - COO (Coordinate) format representation
 * - Irregular memory access patterns
 * - Atomic operations for accumulation
 *
 * COO format stores non-zero elements as (row, col, value) triplets.
 * This is simple but requires atomic operations when multiple threads
 * update the same output element.
 *
 * Matrix A (4×4) with 6 non-zeros:
 * [1  0  2  0]
 * [0  3  0  0]
 * [4  0  5  6]
 * [0  0  0  7]
 *
 * COO: rows=[0,0,1,2,2,2,3], cols=[0,2,1,0,2,3,3], vals=[1,2,3,4,5,6,7]
 *
 * CUDA equivalent:
 * __global__ void spMV_COO(int *rows, int *cols, float *vals, float *x,
 *                          float *y, int nnz) {
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     if (idx < nnz) {
 *         int row = rows[idx];
 *         int col = cols[idx];
 *         float val = vals[idx];
 *
 *         atomicAdd(&y[row], val * x[col]);
 *     }
 * }
 */
public class Pattern22_SpMV_COO {

    /**
     * SpMV kernel using COO format
     *
     * @param context Kernel execution context
     * @param rows Row indices of non-zero elements
     * @param cols Column indices of non-zero elements
     * @param vals Values of non-zero elements
     * @param x Input vector
     * @param y Output vector (result of A × x)
     * @param nnz Number of non-zero elements
     */
    public static void spMV_COO(KernelContext context, IntArray rows, IntArray cols, FloatArray vals, FloatArray x, FloatArray y, int nnz) {
        int idx = context.globalIdx;

        if (idx < nnz) {
            int row = rows.get(idx);
            int col = cols.get(idx);
            float val = vals.get(idx);

            // Compute contribution: val * x[col]
            float contribution = val * x.get(col);

            // Atomically add to output (multiple threads may write to same row)
            context.atomicAdd(y, row, contribution);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void spMV_COO_CPU(IntArray rows, IntArray cols, FloatArray vals, FloatArray x, FloatArray y, int nnz, int numRows) {
        // Initialize output
        for (int i = 0; i < numRows; i++) {
            y.set(i, 0.0f);
        }

        // Compute SpMV
        for (int i = 0; i < nnz; i++) {
            int row = rows.get(i);
            int col = cols.get(i);
            float val = vals.get(i);

            y.set(row, y.get(row) + val * x.get(col));
        }
    }

    public static void main(String[] args) {
        // Define sparse matrix in COO format
        // Example: 4×4 matrix with 7 non-zeros
        final int numRows = 4;
        final int numCols = 4;
        final int nnz = 7;
        final int blockSize = 256;

        // COO representation
        IntArray rows = new IntArray(nnz);
        rows.set(0, 0); rows.set(1, 0); rows.set(2, 1); rows.set(3, 2); rows.set(4, 2); rows.set(5, 2); rows.set(6, 3);

        IntArray cols = new IntArray(nnz);
        cols.set(0, 0); cols.set(1, 2); cols.set(2, 1); cols.set(3, 0); cols.set(4, 2); cols.set(5, 3); cols.set(6, 3);

        FloatArray vals = new FloatArray(nnz);
        vals.set(0, 1.0f); vals.set(1, 2.0f); vals.set(2, 3.0f); vals.set(3, 4.0f); vals.set(4, 5.0f); vals.set(5, 6.0f); vals.set(6, 7.0f);

        // Input vector
        FloatArray x = new FloatArray(numCols);
        x.set(0, 1.0f); x.set(1, 2.0f); x.set(2, 3.0f); x.set(3, 4.0f);

        // Output vectors
        FloatArray y = new FloatArray(numRows);
        FloatArray expected = new FloatArray(numRows);

        // Initialize output to zero
        for (int i = 0; i < numRows; i++) {
            y.set(i, 0.0f);
        }

        // Compute expected result on CPU
        spMV_COO_CPU(rows, cols, vals, x, expected, nnz, numRows);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid1D(nnz);
        worker.setGlobalWork(((nnz + blockSize - 1) / blockSize) * blockSize, 1, 1);
        worker.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, rows, cols, vals, x, y) //
                .task("t0", Pattern22_SpMV_COO::spMV_COO, context, rows, cols, vals, x, y, nnz) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, y);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < numRows; i++) {
            if (Math.abs(y.get(i) - expected.get(i)) > 0.001f) {
                correct = false;
                System.err.println("Error at row " + i + ": expected " + expected.get(i) + " but got " + y.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 22 - SpMV (COO Format): PASSED");
            System.out.println("Successfully computed sparse matrix-vector product");
            System.out.println("Matrix: " + numRows + "×" + numCols + " with " + nnz + " non-zeros");
            System.out.print("Result: [");
            for (int i = 0; i < numRows; i++) {
                System.out.print(y.get(i));
                if (i < numRows - 1)
                    System.out.print(", ");
            }
            System.out.println("]");
        } else {
            System.out.println("Pattern 22 - SpMV (COO Format): FAILED");
        }
    }
}
