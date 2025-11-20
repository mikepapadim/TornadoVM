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
package uk.ac.manchester.tornado.examples.cuda_patterns.histogram;

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
 * CUDA Pattern 13: Histogram Computation (Atomic Operations)
 *
 * Demonstrates:
 * - Atomic operations for concurrent memory updates
 * - Histogram computation pattern
 * - Handling write conflicts in parallel
 * - Global memory atomics performance
 *
 * Computes a histogram by having each thread atomically increment
 * the appropriate bin for its input element. This is the simplest
 * histogram implementation but suffers from atomic contention.
 *
 * CUDA equivalent:
 * __global__ void histogramAtomic(int *input, int *histogram, int numBins, int numElements) {
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     if (idx < numElements) {
 *         int binIndex = input[idx] % numBins;
 *         atomicAdd(&histogram[binIndex], 1);
 *     }
 * }
 */
public class Pattern13_HistogramAtomic {

    /**
     * Histogram kernel using global atomic operations
     *
     * @param context Kernel execution context
     * @param input Input data array
     * @param histogram Output histogram array (initialized to 0)
     * @param numBins Number of histogram bins
     */
    public static void histogramAtomic(KernelContext context, IntArray input, IntArray histogram, int numBins) {
        int idx = context.globalIdx;

        if (idx < input.getSize()) {
            // Compute bin index (modulo to ensure valid range)
            int binIndex = input.get(idx) % numBins;
            if (binIndex < 0) {
                binIndex += numBins; // Handle negative values
            }

            // Atomically increment the histogram bin
            context.atomicAdd(histogram, binIndex, 1);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void histogramCPU(IntArray input, IntArray histogram, int numBins) {
        // Initialize histogram
        for (int i = 0; i < numBins; i++) {
            histogram.set(i, 0);
        }

        // Count occurrences
        for (int i = 0; i < input.getSize(); i++) {
            int binIndex = input.get(i) % numBins;
            if (binIndex < 0) {
                binIndex += numBins;
            }
            histogram.set(binIndex, histogram.get(binIndex) + 1);
        }
    }

    public static void main(String[] args) {
        final int numElements = 1000000;  // 1M elements
        final int numBins = 256;          // 256 bins (e.g., byte values)
        final int localSize = 256;

        // Initialize input data
        IntArray input = new IntArray(numElements);
        IntArray histogram = new IntArray(numBins);
        IntArray expected = new IntArray(numBins);

        // Generate random-like input data
        for (int i = 0; i < numElements; i++) {
            input.set(i, (i * 13 + 7) % numBins); // Pseudo-random values
        }

        // Initialize histogram to zero
        for (int i = 0; i < numBins; i++) {
            histogram.set(i, 0);
        }

        // Compute expected result on CPU
        histogramCPU(input, expected, numBins);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid1D(numElements);
        worker.setGlobalWork(numElements, 1, 1);
        worker.setLocalWork(localSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, histogram) //
                .task("t0", Pattern13_HistogramAtomic::histogramAtomic, context, input, histogram, numBins) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, histogram);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < numBins; i++) {
            if (histogram.get(i) != expected.get(i)) {
                correct = false;
                System.err.println("Error at bin " + i + ": expected " + expected.get(i) + " but got " + histogram.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 13 - Histogram (Atomic): PASSED");
            System.out.println("Successfully computed histogram of " + numElements + " elements into " + numBins + " bins");

            // Print sample of histogram
            System.out.print("Sample bins [0-9]: ");
            for (int i = 0; i < 10; i++) {
                System.out.print(histogram.get(i) + " ");
            }
            System.out.println();
        } else {
            System.out.println("Pattern 13 - Histogram (Atomic): FAILED");
        }
    }
}
