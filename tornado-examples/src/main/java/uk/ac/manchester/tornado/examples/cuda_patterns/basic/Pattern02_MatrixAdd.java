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
import uk.ac.manchester.tornado.api.TornadoExecutionPlanException;

/**
 * CUDA Pattern 02: Matrix Addition
 *
 * Demonstrates:
 * - 2D grid and block configuration
 * - 2D thread indexing (x and y dimensions)
 * - Row-major matrix layout
 * - Efficient memory coalescing
 *
 * CUDA equivalent:
 * __global__ void matrixAdd(float *a, float *b, float *c, int width, int height) {
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *     if (row < height && col < width) {
 *         int idx = row * width + col;
 *         c[idx] = a[idx] + b[idx];
 *     }
 * }
 */
public class Pattern02_MatrixAdd {

    /**
     * Matrix addition kernel using 2D indexing
     *
     * @param context Kernel execution context with 2D thread indexing
     * @param a First input matrix (flattened)
     * @param b Second input matrix (flattened)
     * @param c Output matrix (c = a + b)
     * @param width Matrix width
     * @param height Matrix height
     */
    public static void matrixAdd(KernelContext context, FloatArray a, FloatArray b, FloatArray c, int width, int height) {
        // Get 2D thread coordinates
        int col = context.globalIdx;  // blockIdx.x * blockDim.x + threadIdx.x
        int row = context.globalIdy;  // blockIdx.y * blockDim.y + threadIdx.y

        // Bounds check
        if (row < height && col < width) {
            // Convert 2D coordinates to 1D index (row-major order)
            int idx = row * width + col;
            c.set(idx, a.get(idx) + b.get(idx));
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void matrixAddCPU(FloatArray a, FloatArray b, FloatArray c, int width, int height) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = row * width + col;
                c.set(idx, a.get(idx) + b.get(idx));
            }
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        final int width = 1024;
        final int height = 1024;
        final int size = width * height;
        final int blockSizeX = 16;  // Block dimensions
        final int blockSizeY = 16;

        // Initialize matrices
        FloatArray a = new FloatArray(size);
        FloatArray b = new FloatArray(size);
        FloatArray c = new FloatArray(size);
        FloatArray expected = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            a.set(i, i % 100);
            b.set(i, (i % 100) * 2.0f);
        }

        // Compute expected result on CPU
        matrixAddCPU(a, b, expected, width, height);

        // Create KernelContext and configure 2D grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid2D(width, height);
        worker.setGlobalWork(width, height, 1);
        worker.setLocalWork(blockSizeX, blockSizeY, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Pattern02_MatrixAdd::matrixAdd, context, a, b, c, width, height) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(c.get(i) - expected.get(i)) > 0.01f) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + c.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 02 - Matrix Addition: PASSED");
            System.out.println("Successfully added " + width + "x" + height + " matrices");
        } else {
            System.out.println("Pattern 02 - Matrix Addition: FAILED");
        }
    }
}
