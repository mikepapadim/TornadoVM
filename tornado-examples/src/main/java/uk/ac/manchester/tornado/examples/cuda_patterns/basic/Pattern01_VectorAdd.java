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
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.TornadoExecutionPlanException;

/**
 * CUDA Pattern 01: Vector Addition
 *
 * Demonstrates:
 * - Basic CUDA kernel structure
 * - 1D grid and block configuration
 * - Global thread indexing
 * - Memory access patterns
 *
 * CUDA equivalent:
 * __global__ void vectorAdd(float *a, float *b, float *c, int n) {
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (idx < n) c[idx] = a[idx] + b[idx];
 * }
 */
public class Pattern01_VectorAdd {

    /**
     * Vector addition kernel using KernelContext API
     *
     * @param context Kernel execution context with thread indexing
     * @param a First input vector
     * @param b Second input vector
     * @param c Output vector (c = a + b)
     */
    public static void vectorAdd(KernelContext context, FloatArray a, FloatArray b, FloatArray c) {
        // Get global thread ID (equivalent to: blockIdx.x * blockDim.x + threadIdx.x)
        int idx = context.globalIdx;

        // Perform vector addition
        if (idx < a.getSize()) {
            c.set(idx, a.get(idx) + b.get(idx));
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void vectorAddCPU(FloatArray a, FloatArray b, FloatArray c) {
        for (int i = 0; i < a.getSize(); i++) {
            c.set(i, a.get(i) + b.get(i));
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        final int size = 8192;
        final int localSize = 256; // Block size (threads per block)

        // Initialize data
        FloatArray a = new FloatArray(size);
        FloatArray b = new FloatArray(size);
        FloatArray c = new FloatArray(size);
        FloatArray expected = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            a.set(i, i);
            b.set(i, i * 2.0f);
        }

        // Compute expected result on CPU
        vectorAddCPU(a, b, expected);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid1D(size);
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t0", Pattern01_VectorAdd::vectorAdd, context, a, b, c) //
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
            System.out.println("Pattern 01 - Vector Addition: PASSED");
            System.out.println("Successfully added " + size + " elements");
        } else {
            System.out.println("Pattern 01 - Vector Addition: FAILED");
        }
    }
}
