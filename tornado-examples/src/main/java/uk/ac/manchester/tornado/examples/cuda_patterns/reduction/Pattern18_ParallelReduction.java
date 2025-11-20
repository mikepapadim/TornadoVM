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
package uk.ac.manchester.tornado.examples.cuda_patterns.reduction;

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
 * CUDA Pattern 18: Parallel Reduction (Sum)
 *
 * Demonstrates:
 * - Tree-based parallel reduction algorithm
 * - Logarithmic depth computation
 * - Local/shared memory for intermediate results
 * - Barrier synchronization between reduction steps
 *
 * Parallel reduction is a fundamental pattern that combines all elements
 * of an array into a single value (sum, min, max, etc.) efficiently.
 *
 * This implementation uses a tree-based approach where threads cooperatively
 * reduce values in shared memory with O(log n) steps.
 *
 * CUDA equivalent:
 * __global__ void parallelReduction(float *input, float *output, int n) {
 *     __shared__ float partialSum[BLOCK_SIZE];
 *
 *     int tid = threadIdx.x;
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     // Load data into shared memory
 *     partialSum[tid] = (idx < n) ? input[idx] : 0.0f;
 *     __syncthreads();
 *
 *     // Tree reduction in shared memory
 *     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
 *         if (tid < stride)
 *             partialSum[tid] += partialSum[tid + stride];
 *         __syncthreads();
 *     }
 *
 *     // Write block result
 *     if (tid == 0)
 *         output[blockIdx.x] = partialSum[0];
 * }
 */
public class Pattern18_ParallelReduction {

    /**
     * Parallel reduction kernel (first pass - block level)
     *
     * @param context Kernel execution context
     * @param input Input array
     * @param output Output array (one element per block)
     * @param n Input array size
     */
    public static void parallelReduction(KernelContext context, FloatArray input, FloatArray output, int n) {
        int tid = context.localIdx;
        int globalIdx = context.globalIdx;
        int blockId = context.groupIdx;
        int blockSize = context.localGroupSizeX;

        // Allocate shared memory for this block
        float[] partialSum = context.allocateFloatLocalArray(blockSize);

        // Load data into shared memory (with bounds checking)
        if (globalIdx < n) {
            partialSum[tid] = input.get(globalIdx);
        } else {
            partialSum[tid] = 0.0f;
        }
        context.localBarrier();

        // Tree-based reduction in shared memory
        for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }

        // First thread writes the block's result
        if (tid == 0) {
            output.set(blockId, partialSum[0]);
        }
    }

    /**
     * Final reduction on CPU (for simplicity)
     */
    public static float finalReduction(FloatArray blockSums, int numBlocks) {
        float sum = 0.0f;
        for (int i = 0; i < numBlocks; i++) {
            sum += blockSums.get(i);
        }
        return sum;
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static float reduceCPU(FloatArray input, int n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += input.get(i);
        }
        return sum;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        final int n = 1048576;  // 1M elements
        final int blockSize = 256;
        final int numBlocks = (n + blockSize - 1) / blockSize;

        // Initialize input data
        FloatArray input = new FloatArray(n);
        FloatArray blockSums = new FloatArray(numBlocks);

        for (int i = 0; i < n; i++) {
            input.set(i, 1.0f);  // Simple test: all ones
        }

        // Compute expected result on CPU
        float expected = reduceCPU(input, n);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid1D(n);
        worker.setGlobalWork(numBlocks * blockSize, 1, 1);
        worker.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph (first pass)
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", Pattern18_ParallelReduction::parallelReduction, context, input, blockSums, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, blockSums);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Final reduction on CPU
        float result = finalReduction(blockSums, numBlocks);

        // Validate result
        if (Math.abs(result - expected) < 0.1f) {
            System.out.println("Pattern 18 - Parallel Reduction: PASSED");
            System.out.println("Successfully reduced " + n + " elements");
            System.out.println("Result: " + result + " (expected: " + expected + ")");
            System.out.println("Used " + numBlocks + " blocks of " + blockSize + " threads");
        } else {
            System.out.println("Pattern 18 - Parallel Reduction: FAILED");
            System.err.println("Expected: " + expected + " but got: " + result);
        }
    }
}
