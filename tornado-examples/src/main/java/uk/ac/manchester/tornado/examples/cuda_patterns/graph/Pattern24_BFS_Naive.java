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
package uk.ac.manchester.tornado.examples.cuda_patterns.graph;

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
 * CUDA Pattern 24: Breadth-First Search (Naive)
 *
 * Demonstrates:
 * - Graph algorithms on GPU
 * - Level-synchronous BFS traversal
 * - CSR (Compressed Sparse Row) graph representation
 * - Iterative parallel traversal
 *
 * BFS explores a graph level by level from a source node.
 * This naive version processes all vertices at each level.
 *
 * Graph representation (CSR format):
 * - rowPtr: Starting index of neighbors for each vertex
 * - colIdx: Neighbor vertex IDs
 *
 * Example graph:
 *   0 → 1, 2
 *   1 → 2, 3
 *   2 → 3
 *   3 → (none)
 *
 * CSR: rowPtr=[0,2,4,5,5], colIdx=[1,2,2,3,3]
 *
 * CUDA equivalent:
 * __global__ void bfsNaive(int *rowPtr, int *colIdx, int *levels,
 *                          int *visited, int currentLevel, int numVertices) {
 *     int v = blockIdx.x * blockDim.x + threadIdx.x;
 *
 *     if (v < numVertices && levels[v] == currentLevel) {
 *         // Explore neighbors
 *         for (int i = rowPtr[v]; i < rowPtr[v+1]; i++) {
 *             int neighbor = colIdx[i];
 *             if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
 *                 levels[neighbor] = currentLevel + 1;
 *             }
 *         }
 *     }
 * }
 */
public class Pattern24_BFS_Naive {

    private static final int UNVISITED = -1;

    /**
     * BFS kernel (single level)
     *
     * @param context Kernel execution context
     * @param rowPtr CSR row pointer array
     * @param colIdx CSR column index array (neighbors)
     * @param levels Level/distance from source for each vertex
     * @param currentLevel Current BFS level to process
     * @param numVertices Number of vertices in graph
     * @param updated Flag indicating if any vertex was updated
     */
    public static void bfsNaive(KernelContext context, IntArray rowPtr, IntArray colIdx, IntArray levels, int currentLevel, int numVertices, IntArray updated) {
        int v = context.globalIdx;

        if (v < numVertices && levels.get(v) == currentLevel) {
            // This vertex is at the current level - explore its neighbors
            int start = rowPtr.get(v);
            int end = rowPtr.get(v + 1);

            for (int i = start; i < end; i++) {
                int neighbor = colIdx.get(i);

                // If neighbor hasn't been visited, mark it for next level
                if (levels.get(neighbor) == UNVISITED) {
                    levels.set(neighbor, currentLevel + 1);
                    updated.set(0, 1); // Mark that we updated something
                }
            }
        }
    }

    /**
     * Sequential CPU implementation for validation
     */
    public static void bfsCPU(IntArray rowPtr, IntArray colIdx, IntArray levels, int source, int numVertices) {
        // Initialize levels
        for (int i = 0; i < numVertices; i++) {
            levels.set(i, UNVISITED);
        }
        levels.set(source, 0);

        // BFS traversal
        boolean updated = true;
        int currentLevel = 0;

        while (updated) {
            updated = false;

            for (int v = 0; v < numVertices; v++) {
                if (levels.get(v) == currentLevel) {
                    int start = rowPtr.get(v);
                    int end = rowPtr.get(v + 1);

                    for (int i = start; i < end; i++) {
                        int neighbor = colIdx.get(i);
                        if (levels.get(neighbor) == UNVISITED) {
                            levels.set(neighbor, currentLevel + 1);
                            updated = true;
                        }
                    }
                }
            }

            currentLevel++;
        }
    }

    public static void main(String[] args) {
        // Define graph in CSR format
        // Example: 6-vertex graph
        final int numVertices = 6;
        final int numEdges = 8;
        final int source = 0;
        final int blockSize = 256;

        // CSR representation
        // Graph: 0→1,2  1→2,3  2→3  3→4,5  4→5  5→(none)
        IntArray rowPtr = new IntArray(new int[] { 0, 2, 4, 5, 7, 8, 8 }); // Note: size = numVertices + 1
        IntArray colIdx = new IntArray(new int[] { 1, 2, 2, 3, 3, 4, 5, 5 });

        // Output arrays
        IntArray levels = new IntArray(numVertices);
        IntArray expected = new IntArray(numVertices);
        IntArray updated = new IntArray(1);

        // Initialize levels
        for (int i = 0; i < numVertices; i++) {
            levels.set(i, UNVISITED);
        }
        levels.set(source, 0);

        // Compute expected result on CPU
        bfsCPU(rowPtr, colIdx, expected, source, numVertices);

        // Create KernelContext and configure grid
        KernelContext context = new KernelContext();
        int gridSize = ((numVertices + blockSize - 1) / blockSize) * blockSize;
        WorkerGrid worker = new WorkerGrid1D(numVertices);
        worker.setGlobalWork(gridSize, 1, 1);
        worker.setLocalWork(blockSize, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        // Build task graph
        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, rowPtr, colIdx) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, levels, updated) //
                .task("t0", Pattern24_BFS_Naive::bfsNaive, context, rowPtr, colIdx, levels, 0, numVertices, updated) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, levels, updated);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        // Execute BFS level by level
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            int currentLevel = 0;
            int maxLevels = numVertices; // Prevent infinite loop

            while (currentLevel < maxLevels) {
                updated.set(0, 0);

                // Update task with current level
                executionPlan.withGridScheduler(gridScheduler).execute();

                // Check if any updates were made
                if (updated.get(0) == 0) {
                    break; // No more vertices to explore
                }

                currentLevel++;

                // Update levels for next iteration
                for (int v = 0; v < numVertices; v++) {
                    if (levels.get(v) == currentLevel) {
                        // Vertex found at this level
                    }
                }
            }
        }

        // Validate results
        boolean correct = true;
        for (int i = 0; i < numVertices; i++) {
            if (levels.get(i) != expected.get(i)) {
                correct = false;
                System.err.println("Error at vertex " + i + ": expected level " + expected.get(i) + " but got " + levels.get(i));
                break;
            }
        }

        if (correct) {
            System.out.println("Pattern 24 - BFS (Naive): PASSED");
            System.out.println("Successfully traversed graph with " + numVertices + " vertices and " + numEdges + " edges");
            System.out.print("Levels from source " + source + ": ");
            for (int i = 0; i < numVertices; i++) {
                System.out.print(levels.get(i) + " ");
            }
            System.out.println();
        } else {
            System.out.println("Pattern 24 - BFS (Naive): FAILED");
        }
    }
}
