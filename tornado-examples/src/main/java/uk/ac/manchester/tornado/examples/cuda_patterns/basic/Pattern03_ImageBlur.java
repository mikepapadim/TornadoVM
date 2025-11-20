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
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/**
 * CUDA Pattern 03: Image Blur (Box Blur)
 *
 * Demonstrates:
 * - 2D stencil operations
 * - Boundary handling
 * - Neighborhood access patterns
 * - Image processing fundamentals
 *
 * Applies a simple box blur filter where each output pixel is the average
 * of its surrounding pixels in a square kernel.
 *
 * CUDA equivalent:
 * __global__ void imageBlur(float *input, float *output, int width, int height, int blurRadius) {
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *
 *     if (row < height && col < width) {
 *         float sum = 0.0f;
 *         int count = 0;
 *
 *         for (int r = -blurRadius; r <= blurRadius; r++) {
 *             for (int c = -blurRadius; c <= blurRadius; c++) {
 *                 int curRow = row + r;
 *                 int curCol = col + c;
 *                 if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
 *                     sum += input[curRow * width + curCol];
 *                     count++;
 *                 }
 *             }
 *         }
 *         output[row * width + col] = sum / count;
 *     }
 * }
 */
public class Pattern03_ImageBlur {

    /**
     * Image blur kernel using 2D stencil operation
     *
     * @param context Kernel execution context
     * @param input Input image (flattened grayscale)
     * @param output Output blurred image
     * @param width Image width
     * @param height Image height
     * @param blurRadius Blur kernel radius (kernel size = 2*radius+1)
     */
    public static void imageBlur(KernelContext context, FloatArray input, FloatArray output, int width, int height, int blurRadius) {
        int col = context.globalIdx;
        int row = context.globalIdy;

        if (row < height && col < width) {
            float sum = 0.0f;
            int count = 0;

            // Iterate over blur kernel neighborhood
            for (int r = -blurRadius; r <= blurRadius; r++) {
                for (int c = -blurRadius; c <= blurRadius; c++) {
                    int curRow = row + r;
                    int curCol = col + c;

                    // Boundary check (clamp to edges)
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        sum += input.get(curRow * width + curCol);
                        count++;
                    }
                }
            }

            // Average the values
            output.set(row * width + col, sum / count);
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void imageBlurCPU(FloatArray input, FloatArray output, int width, int height, int blurRadius) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                float sum = 0.0f;
                int count = 0;

                for (int r = -blurRadius; r <= blurRadius; r++) {
                    for (int c = -blurRadius; c <= blurRadius; c++) {
                        int curRow = row + r;
                        int curCol = col + c;

                        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                            sum += input.get(curRow * width + curCol);
                            count++;
                        }
                    }
                }

                output.set(row * width + col, sum / count);
            }
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        final int width = 512;
        final int height = 512;
        final int size = width * height;
        final int blurRadius = 3;  // 7x7 kernel
        final int blockSizeX = 16;
        final int blockSizeY = 16;

        // Initialize image with test pattern (gradient + noise)
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size);
        FloatArray expected = new FloatArray(size);

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = row * width + col;
                // Create gradient pattern
                float value = (float) (row + col) / (width + height);
                // Add some "noise"
                value += ((row * col) % 10) / 100.0f;
                input.set(idx, value);
            }
        }

        // Compute expected result on CPU
        imageBlurCPU(input, expected, width, height, blurRadius);

        // Create KernelContext and configure 2D grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid2D(width, height);
        worker.setGlobalWork(width, height, 1);
        worker.setLocalWork(blockSizeX, blockSizeY, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("t0", Pattern03_ImageBlur::imageBlur, context, input, output, width, height, blurRadius) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(output.get(i) - expected.get(i)) > 0.01f) {
                correct = false;
                System.err.println("Error at index " + i + ": expected " + expected.get(i) + " but got " + output.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 03 - Image Blur: PASSED");
            System.out.println("Successfully blurred " + width + "x" + height + " image with radius " + blurRadius);
        } else {
            System.out.println("Pattern 03 - Image Blur: FAILED");
        }
    }
}
