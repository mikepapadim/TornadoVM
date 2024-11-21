/*
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.unittests.compute;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

import static org.junit.Assert.assertFalse;

public class MMwithBytes extends TornadoTestBase {

    @Test
    public void testMatrixMultiplicationWithBytes() throws TornadoExecutionPlanException {

        // Define matrix/vector dimensions
        final int dim = 128;  // example dimension, adjust as necessary
        final int numRows = 1024; // example number of rows for ByteArray

        // Initialize input data
        ByteArray byteArrayWeights = new ByteArray(numRows * (2 + 32)); // dim + 2 bytes for scale per row
        FloatArray inputVector = new FloatArray(dim);
        FloatArray outputVector = new FloatArray(numRows);

        // Populate the ByteArray with dummy quantized weights and scales
        for (int i = 0; i < numRows; i++) {
            int offset = i * (2 + 32);
            byteArrayWeights.set(offset, (byte) 0);          // scale byte 1
            byteArrayWeights.set(offset + 1, (byte) 0);      // scale byte 2
            for (int j = 2; j < 2 + 32; j++) {
                byteArrayWeights.set(offset + j, (byte) (Math.random() * 255 - 128)); // random quantized values
            }
        }

        // Populate the input vector with random floats
        for (int i = 0; i < dim; i++) {
            inputVector.set(i, (float) Math.random());
        }

        // Expected output should be initialized to zero
        outputVector.init(0.0f);

        // Create the execution plan
        WorkerGrid workerGrid = new WorkerGrid1D(numRows);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", workerGrid);
        workerGrid.setGlobalWork(numRows, 1, 1);
        workerGrid.setLocalWork(32, 1, 1);

        // Define the TaskGraph for matrix multiplication
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, byteArrayWeights, inputVector)
                .task("t0", MMwithBytes::matmulTornado, new KernelContext(), byteArrayWeights, inputVector, outputVector, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputVector);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        // Validation: Manually check output values or use precomputed expected output
        // For simplicity, check that no NaNs or Infinity are in the result
        for (int i = 0; i < numRows; i++) {
            float result = outputVector.get(i);
            assertFalse("Output contains NaN at index " + i, Float.isNaN(result));
            assertFalse("Output contains Infinity at index " + i, Float.isInfinite(result));
        }

        // Optionally, add further checks against a precomputed result array if available
    }

    public static void matmulTornado(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Assuming this is the block size used in quantization
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values

        int idx = context.globalIdx;

        float result = 0f;
        int thisOffset = idx * dim1;

        for (int j = 0; j < dim1; j++) {
            int index = thisOffset + j;
            // Calculate block position
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Read scale (float16) for this block
            int scaleByte1 = thisx.get(blockOffset) & 0xFF;
            int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
            short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
            float scale = decodeFloat16(scaleFloat16);

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and multiply
            result += (quantized * scale) * that.get(j);
        }

        out.set(idx, result);

    }

    private static float decodeFloat16(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Handle special cases
        if (exp == 0x1F) return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        if (exp == 0) {
            if (frac == 0) return sign == 0 ? 0.0f : -0.0f;
            float result = frac * pow2(-24);
            return sign == 0 ? result : -result;
        }

        float result = 1.0f + (frac / 1024.0f);
        result *= pow2(exp - 15);
        return sign == 0 ? result : -result;
    }

    private static float pow2(int n) {
        if (n >= 0) {
            if (n < 31) {
                return (float)(1 << n);
            }
            return Float.POSITIVE_INFINITY;
        }
        if (n > -150) {
            return 1.0f / (1 << -n);
        }
        return 0.0f;
    }
}
