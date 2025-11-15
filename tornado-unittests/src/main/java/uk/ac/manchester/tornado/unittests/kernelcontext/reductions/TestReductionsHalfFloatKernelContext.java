/*
 * Copyright (c) 2021-2025, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.unittests.kernelcontext.reductions;

import static org.junit.Assert.assertEquals;

import java.util.stream.IntStream;
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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * The unit-tests in this class implement some Reduction operations for HalfFloat type using local memory arrays.
 * These tests validate that HalfFloat local arrays can be allocated and used in kernel computations.
 */
public class TestReductionsHalfFloatKernelContext extends TornadoTestBase {

    public static void halfFloatReductionAddLocalMemory(KernelContext context, FloatArray a, FloatArray b) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx;

        // Allocate HalfFloat local array
        HalfFloatArray localA = context.allocateHalfLocalArray(256);
        
        // Load from global memory into local memory
        localA.set(localIdx, new HalfFloat(a.get(globalIdx)));
        context.localBarrier();

        // Perform reduction
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            if (localIdx < stride) {
                HalfFloat current = localA.get(localIdx);
                HalfFloat next = localA.get(localIdx + stride);
                float sum = current.getFloat32() + next.getFloat32();
                localA.set(localIdx, new HalfFloat(sum));
            }
            context.localBarrier();
        }
        
        // Write result back to global memory
        if (localIdx == 0) {
            b.set(groupID, localA.get(0).getFloat32());
        }
    }

    public static void halfFloatBasicLocalArrayAccess(KernelContext context, FloatArray a, FloatArray b) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx;
        int idx = localGroupSize * groupID + localIdx;

        HalfFloatArray localA = context.allocateHalfLocalArray(256);
        localA.set(localIdx, new HalfFloat(a.get(idx)));
        b.set(idx, localA.get(localIdx).getFloat32());
    }

    @Test
    public void test01HalfFloatBasicLocalArrayAccess() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            input.set(i, i);
        }

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatKernelContext::halfFloatBasicLocalArrayAccess, context, input, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        for (int i = 0; i < size; i++) {
            assertEquals(input.get(i), output.get(i), 0.1f);
        }
    }

    @Test
    public void test02HalfFloatReductionAddLocalMemory() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        FloatArray input = new FloatArray(size);
        FloatArray output = new FloatArray(size / localSize);

        for (int i = 0; i < size; i++) {
            input.set(i, 1.0f);
        }

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatKernelContext::halfFloatReductionAddLocalMemory, context, input, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        for (int i = 0; i < output.getSize(); i++) {
            assertEquals(localSize, output.get(i), 1.0f);
        }
    }
}
