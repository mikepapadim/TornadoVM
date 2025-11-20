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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * CUDA Pattern 20: Prefix Sum / Scan (Single Block, Kogge-Stone)
 *
 * Demonstrates:
 * - Parallel prefix sum computation
 * - Kogge-Stone algorithm (work-efficient scan)
 * - Shared memory and double buffering
 * - O(log n) parallel steps
 *
 * Prefix sum (also called scan) is a fundamental parallel pattern
 * that computes cumulative sums: output[i] = sum(input[0:i]).
 *
 * This implementation uses the Kogge-Stone algorithm which is
 * simple but performs O(n log n) operations.
 *
 * Example: [1, 2, 3, 4] -> [1, 3, 6, 10]
 *
 * CUDA equivalent (simplified Kogge-Stone):
 * __global__ void prefixSum(int *input, int *output, int n) {
 *     __shared__ int temp[BLOCK_SIZE * 2];
 *     int tid = threadIdx.x;
 *     int pout = 0, pin = 1;
 *
 *     // Load input into shared memory
 *     temp[tid] = (tid < n) ? input[tid] : 0;
 *     __syncthreads();
 *
 *     // Kogge-Stone scan
 *     for (int offset = 1; offset < n; offset *= 2) {
 *         pout = 1 - pout;
 *         pin = 1 - pout;
 *
 *         if (tid >= offset)
 *             temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
 *         else
 *             temp[pout * n + tid] = temp[pin * n + tid];
 *         __syncthreads();
 *     }
 *
 *     output[tid] = temp[pout * n + tid];
 * }
 */
public class Pattern20_PrefixSumSingleBlock {

    /**
     * Prefix sum kernel (Kogge-Stone algorithm, single block)
     *
     * @param context Kernel execution context
     * @param input Input array
     * @param output Output array (prefix sums)
     * @param n Array size (must be power of 2 and <= block size)
     */
    public static void prefixSum(KernelContext context, IntArray input, IntArray output, int n) {
        int tid = context.localIdx;
        int blockSize = context.localGroupSizeX;

        // Allocate shared memory for double buffering
        int[] temp = context.allocateIntLocalArray(blockSize * 2);

        // Load input into shared memory (first buffer)
        if (tid < n) {
            temp[tid] = input.get(tid);
        } else {
            temp[tid] = 0;
        }
        context.localBarrier();

        // Kogge-Stone scan with double buffering
        int pout = 0;
        int pin = 1;

        for (int offset = 1; offset < n; offset *= 2) {
            // Swap buffers
            pout = 1 - pout;
            pin = 1 - pout;

            if (tid >= offset && tid < n) {
                // Add element 'offset' positions to the left
                temp[pout * blockSize + tid] = temp[pin * blockSize + tid] + temp[pin * blockSize + tid - offset];
            } else if (tid < n) {
                // Copy unchanged
                temp[pout * blockSize + tid] = temp[pin * blockSize + tid];
            }

            context.localBarrier();
        }

        // Write result to output
        if (tid < n) {
            output.set(tid, temp[pout * blockSize + tid]);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void prefixSumCPU(IntArray input, IntArray output, int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += input.get(i);
            output.set(i, sum);
        }
    }

    public static void main(String[] args) {
        final int n = 256;  // Single block size
        final int blockSize = 256;

        // Initialize input data
        IntArray input = new IntArray(n);
        IntArray output = new IntArray(n);
        IntArray expected = new IntArray(n);

        for (int i = 0; i < n; i++) {
            input.set(i, 1);  // Simple test: all ones -> [1, 2, 3, ..., n]
        }

        // Compute expected result on CPU
        prefixSumCPU(input, expected, n);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid1D(blockSize);
        worker.setGlobalWork(blockSize, 1, 1);
        worker.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", Pattern20_PrefixSumSingleBlock::prefixSum, context, input, output, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < n; i++) {
            if (output.get(i) != expected.get(i)) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + output.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 20 - Prefix Sum (Single Block): PASSED");
            System.out.println("Successfully computed prefix sum of " + n + " elements");
            System.out.print("First 10 elements: ");
            for (int i = 0; i < 10; i++) {
                System.out.print(output.get(i) + " ");
            }
            System.out.println();
            System.out.println("Last element (total sum): " + output.get(n - 1));
        } else {
            System.out.println("Pattern 20 - Prefix Sum (Single Block): FAILED");
        }
    }
}
