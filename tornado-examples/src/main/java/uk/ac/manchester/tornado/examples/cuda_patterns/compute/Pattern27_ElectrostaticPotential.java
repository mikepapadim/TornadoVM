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
package uk.ac.manchester.tornado.examples.cuda_patterns.compute;

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
 * CUDA Pattern 27: Electrostatic Potential Map
 *
 * Demonstrates:
 * - Compute-intensive kernels
 * - Thread coarsening for performance
 * - N-body style computation
 * - Register optimization
 * - High arithmetic intensity
 *
 * Computes the electrostatic potential at each grid point due to
 * a collection of point charges. This is a compute-bound kernel
 * that benefits greatly from thread coarsening.
 *
 * Potential at point (x,y) = Σ charge[i] / distance(point, charge[i])
 *
 * CUDA equivalent:
 * __global__ void electrostaticPotential(float *atoms_x, float *atoms_y,
 *                                         float *atoms_charge, float *grid,
 *                                         int numAtoms, int gridWidth,
 *                                         int gridHeight, float gridSpacing) {
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *
 *     if (row < gridHeight && col < gridWidth) {
 *         float x = col * gridSpacing;
 *         float y = row * gridSpacing;
 *
 *         float potential = 0.0f;
 *         for (int i = 0; i < numAtoms; i++) {
 *             float dx = x - atoms_x[i];
 *             float dy = y - atoms_y[i];
 *             float dist = sqrtf(dx*dx + dy*dy + 1e-6f); // Avoid div by zero
 *             potential += atoms_charge[i] / dist;
 *         }
 *
 *         grid[row * gridWidth + col] = potential;
 *     }
 * }
 */
public class Pattern27_ElectrostaticPotential {

    /**
     * Electrostatic potential kernel
     *
     * @param context Kernel execution context
     * @param atomsX X coordinates of charges
     * @param atomsY Y coordinates of charges
     * @param atomsCharge Charge values
     * @param grid Output potential grid
     * @param numAtoms Number of point charges
     * @param gridWidth Grid width
     * @param gridHeight Grid height
     * @param gridSpacing Spacing between grid points
     */
    public static void electrostaticPotential(KernelContext context, FloatArray atomsX, FloatArray atomsY, FloatArray atomsCharge, FloatArray grid, int numAtoms, int gridWidth,
            int gridHeight, float gridSpacing) {

        int col = context.globalIdx;
        int row = context.globalIdy;

        if (row < gridHeight && col < gridWidth) {
            // Grid point coordinates
            float x = col * gridSpacing;
            float y = row * gridSpacing;

            // Compute potential from all charges
            float potential = 0.0f;

            for (int i = 0; i < numAtoms; i++) {
                float dx = x - atomsX.get(i);
                float dy = y - atomsY.get(i);

                // Distance with small epsilon to avoid division by zero
                float distSq = dx * dx + dy * dy + 1e-6f;
                float dist = (float) Math.sqrt(distSq);

                // Coulomb's law (simplified): potential = charge / distance
                potential += atomsCharge.get(i) / dist;
            }

            grid.set(row * gridWidth + col, potential);
        }
    }

    /**
     * Optimized version with thread coarsening
     */
    public static void electrostaticPotentialCoarsened(KernelContext context, FloatArray atomsX, FloatArray atomsY, FloatArray atomsCharge, FloatArray grid, int numAtoms, int gridWidth,
            int gridHeight, float gridSpacing, int coarseFactor) {

        int baseCol = context.globalIdx * coarseFactor;
        int baseRow = context.globalIdy * coarseFactor;

        // Each thread computes coarseFactor × coarseFactor grid points
        for (int i = 0; i < coarseFactor; i++) {
            for (int j = 0; j < coarseFactor; j++) {
                int col = baseCol + j;
                int row = baseRow + i;

                if (row < gridHeight && col < gridWidth) {
                    float x = col * gridSpacing;
                    float y = row * gridSpacing;

                    float potential = 0.0f;

                    for (int k = 0; k < numAtoms; k++) {
                        float dx = x - atomsX.get(k);
                        float dy = y - atomsY.get(k);
                        float distSq = dx * dx + dy * dy + 1e-6f;
                        float dist = (float) Math.sqrt(distSq);
                        potential += atomsCharge.get(k) / dist;
                    }

                    grid.set(row * gridWidth + col, potential);
                }
            }
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void electrostaticPotentialCPU(FloatArray atomsX, FloatArray atomsY, FloatArray atomsCharge, FloatArray grid, int numAtoms, int gridWidth, int gridHeight,
            float gridSpacing) {

        for (int row = 0; row < gridHeight; row++) {
            for (int col = 0; col < gridWidth; col++) {
                float x = col * gridSpacing;
                float y = row * gridSpacing;

                float potential = 0.0f;

                for (int i = 0; i < numAtoms; i++) {
                    float dx = x - atomsX.get(i);
                    float dy = y - atomsY.get(i);
                    float distSq = dx * dx + dy * dy + 1e-6f;
                    float dist = (float) Math.sqrt(distSq);
                    potential += atomsCharge.get(i) / dist;
                }

                grid.set(row * gridWidth + col, potential);
            }
        }
    }

    public static void main(String[] args) {
        final int gridWidth = 512;
        final int gridHeight = 512;
        final int gridSize = gridWidth * gridHeight;
        final int numAtoms = 100;
        final float gridSpacing = 0.1f;
        final int blockSize = 16;

        // Initialize atoms (charges)
        FloatArray atomsX = new FloatArray(numAtoms);
        FloatArray atomsY = new FloatArray(numAtoms);
        FloatArray atomsCharge = new FloatArray(numAtoms);

        // Place atoms randomly in space
        for (int i = 0; i < numAtoms; i++) {
            atomsX.set(i, (i * 7) % gridWidth * gridSpacing);
            atomsY.set(i, (i * 13) % gridHeight * gridSpacing);
            atomsCharge.set(i, (i % 2 == 0) ? 1.0f : -1.0f); // Alternating positive/negative
        }

        // Output grids
        FloatArray grid = new FloatArray(gridSize);
        FloatArray expected = new FloatArray(gridSize);

        // Compute expected result on CPU (sample only - full computation is expensive)
        System.out.println("Computing reference solution on CPU (this may take a moment)...");
        electrostaticPotentialCPU(atomsX, atomsY, atomsCharge, expected, numAtoms, gridWidth, gridHeight, gridSpacing);

        // Create KernelContext and configure 2D grid
        KernelContext context = new KernelContext();
        WorkerGrid worker = new WorkerGrid2D(gridWidth, gridHeight);
        worker.setGlobalWork(gridWidth, gridHeight, 1);
        worker.setLocalWork(blockSize, blockSize, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build and execute task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, atomsX, atomsY, atomsCharge) //
                .task("t0", Pattern27_ElectrostaticPotential::electrostaticPotential, context, atomsX, atomsY, atomsCharge, grid, numAtoms, gridWidth, gridHeight, gridSpacing) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, grid);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validate results (sample check)
        boolean correct = true;
        int sampleSize = Math.min(1000, gridSize);
        for (int i = 0; i < sampleSize; i++) {
            int idx = i * (gridSize / sampleSize); // Sample evenly
            float error = Math.abs(grid.get(idx) - expected.get(idx));
            float relativeError = error / (Math.abs(expected.get(idx)) + 1e-6f);

            if (relativeError > 0.01f) { // 1% tolerance
                correct = false;
                System.err.println("Error at index " + idx + ": expected " + expected.get(idx) + " but got " + grid.get(idx) + " (relative error: " + relativeError + ")");
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 27 - Electrostatic Potential: PASSED");
            System.out.println("Successfully computed potential on " + gridWidth + "×" + gridHeight + " grid");
            System.out.println("Number of charges: " + numAtoms);
            System.out.println("Total computations: " + (long) gridSize * numAtoms + " charge-point interactions");

            // Print sample values
            System.out.println("Sample potential values:");
            for (int i = 0; i < 5; i++) {
                int idx = i * gridSize / 5;
                System.out.printf("  Grid[%d] = %.4f\n", idx, grid.get(idx));
            }
        } else {
            System.out.println("Pattern 27 - Electrostatic Potential: FAILED");
        }
    }
}
