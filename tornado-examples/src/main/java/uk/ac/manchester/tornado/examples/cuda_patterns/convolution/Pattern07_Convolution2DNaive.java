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
package uk.ac.manchester.tornado.examples.cuda_patterns.convolution;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * CUDA Pattern 07: 2D Convolution (Naive)
 *
 * Demonstrates:
 * - 2D convolution operation
 * - Stencil pattern with constant kernel
 * - Boundary condition handling
 * - Redundant global memory reads
 *
 * Applies a 2D convolution kernel to an input image. Each output pixel
 * is computed by applying the convolution kernel centered at that position.
 *
 * CUDA equivalent:
 * __global__ void conv2D(float *input, float *output, float *kernel,
 *                        int width, int height, int kernelSize) {
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *
 *     if (row < height && col < width) {
 *         float sum = 0.0f;
 *         int halfKernel = kernelSize / 2;
 *
 *         for (int i = 0; i < kernelSize; i++) {
 *             for (int j = 0; j < kernelSize; j++) {
 *                 int curRow = row + i - halfKernel;
 *                 int curCol = col + j - halfKernel;
 *
 *                 if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
 *                     sum += input[curRow * width + curCol] *
 *                            kernel[i * kernelSize + j];
 *                 }
 *             }
 *         }
 *         output[row * width + col] = sum;
 *     }
 * }
 */
public class Pattern07_Convolution2DNaive {

    /**
     * Naive 2D convolution kernel
     *
     * @param context Kernel execution context
     * @param input Input image (flattened)
     * @param output Output image (flattened)
     * @param kernel Convolution kernel (flattened)
     * @param width Image width
     * @param height Image height
     * @param kernelSize Kernel size (must be odd, e.g., 3, 5, 7)
     */
    public static void conv2DNaive(KernelContext context, FloatArray input, FloatArray output, FloatArray kernel, int width, int height, int kernelSize) {
        int col = context.globalIdx;
        int row = context.globalIdy;

        if (row < height && col < width) {
            float sum = 0.0f;
            int halfKernel = kernelSize / 2;

            // Apply convolution kernel
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    int curRow = row + i - halfKernel;
                    int curCol = col + j - halfKernel;

                    // Boundary check (zero-padding)
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        float inputVal = input.get(curRow * width + curCol);
                        float kernelVal = kernel.get(i * kernelSize + j);
                        sum += inputVal * kernelVal;
                    }
                }
            }

            output.set(row * width + col, sum);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void conv2DCPU(FloatArray input, FloatArray output, FloatArray kernel, int width, int height, int kernelSize) {
        int halfKernel = kernelSize / 2;

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                float sum = 0.0f;

                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        int curRow = row + i - halfKernel;
                        int curCol = col + j - halfKernel;

                        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                            sum += input.get(curRow * width + curCol) * kernel.get(i * kernelSize + j);
                        }
                    }
                }

                output.set(row * width + col, sum);
            }
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        final int width = 512;
        final int height = 512;
        final int size = width * height;
        final int kernelSize = 5;  // 5×5 kernel
        final int blockSize = 16;

        // Initialize input image
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size);
        FloatArray expected = new FloatArray(size);

        // Initialize convolution kernel (Gaussian-like)
        FloatArray convKernel = new FloatArray(kernelSize * kernelSize);
        float[] gaussianKernel = { //
                1, 4, 6, 4, 1, //
                4, 16, 24, 16, 4, //
                6, 24, 36, 24, 6, //
                4, 16, 24, 16, 4, //
                1, 4, 6, 4, 1 //
        };
        float kernelSum = 256.0f; // Sum of kernel values
        for (int i = 0; i < kernelSize * kernelSize; i++) {
            convKernel.set(i, gaussianKernel[i] / kernelSum); // Normalize
        }

        // Create test input image
        for (int i = 0; i < size; i++) {
            input.set(i, (i % width) / (float) width); // Gradient pattern
        }

        // Compute expected result on CPU
        conv2DCPU(input, expected, convKernel, width, height, kernelSize);

        // Create KernelContext and configure 2D grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid2D(width, height);
        worker.setGlobalWork(width, height, 1);
        worker.setLocalWork(blockSize, blockSize, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, convKernel) //
                .task("t0", Pattern07_Convolution2DNaive::conv2DNaive, context, input, output, convKernel, width, height, kernelSize) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(output.get(i) - expected.get(i)) > 0.001f) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + output.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 07 - 2D Convolution (Naive): PASSED");
            System.out.println("Successfully convolved " + width + "×" + height + " image with " + kernelSize + "×" + kernelSize + " kernel");
        } else {
            System.out.println("Pattern 07 - 2D Convolution (Naive): FAILED");
        }
    }
}
