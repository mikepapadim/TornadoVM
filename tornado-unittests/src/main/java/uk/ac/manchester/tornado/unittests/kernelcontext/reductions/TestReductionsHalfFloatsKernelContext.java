/*
 * Copyright (c) 2021-2022 APT Group, Department of Computer Science,
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
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * The unit-tests in this class implement some Reduction operations (add, max, min) for {@link HalfFloat} type. These unit-tests check the functional operation of some {@link KernelContext} features, such
 * as global thread identifiers, local thread identifiers, the local group size of the associated WorkerGrid, barriers and allocation of local memory.
 * <p>
 * How to run?
 * </p>
 * <code>
 * tornado-test -V uk.ac.manchester.tornado.unittests.kernelcontext.reductions.TestReductionsHalfFloatsKernelContext
 * </code>
 */
public class TestReductionsHalfFloatsKernelContext extends TornadoTestBase {

    public static HalfFloat computeAddSequential(HalfFloatArray input) {
        HalfFloat acc = new HalfFloat(0.0f);
        for (int i = 0; i < input.getSize(); i++) {
            acc = new HalfFloat(acc.getFloat32() + input.get(i).getFloat32());
        }
        return acc;
    }

    public static void halfFloatReductionAddGlobalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID
        int id = context.globalIdx;

        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                a.set(id, new HalfFloat(a.get(id).getFloat32() + a.get(id + stride).getFloat32()));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, a.get(id));
        }
    }

    public static void halfFloatReductionAddLocalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        HalfFloatArray localA = context.allocateHalfFloatLocalArray(256);
        localA.set(localIdx, a.get(globalIdx));
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localA.set(localIdx, new HalfFloat(localA.get(localIdx).getFloat32() + localA.get(localIdx + stride).getFloat32()));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, localA.get(0));
        }
    }

    public static void halfFloatReductionAddLocalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b, int blockDim) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        HalfFloatArray localA = context.allocateHalfFloatLocalArray(blockDim);
        localA.set(localIdx, a.get(globalIdx));
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                HalfFloat tmp = HalfFloat.add(localA.get(localIdx) , localA.get(localIdx + stride));
                localA.set(localIdx, tmp);
            }
        }
        if (localIdx == 0) {
            b.set(groupID, localA.get(0));
        }
    }

    public static HalfFloat computeMaxSequential(HalfFloatArray input) {
        HalfFloat acc = new HalfFloat(0.0f);
        for (int i = 0; i < input.getSize(); i++) {
            acc = new HalfFloat(TornadoMath.max(acc.getFloat32(), input.get(i).getFloat32()));
        }
        return acc;
    }

    private static void halfFloatReductionMaxGlobalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID
        int id = localGroupSize * groupID + localIdx;

        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                a.set(id, new HalfFloat(TornadoMath.max(a.get(id).getFloat32(), a.get(id + stride).getFloat32())));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, a.get(id));
        }
    }

    public static void halfFloatReductionMaxLocalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        HalfFloatArray localA = context.allocateHalfFloatLocalArray(256);
        localA.set(localIdx, a.get(globalIdx));
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localA.set(localIdx, new HalfFloat(TornadoMath.max(localA.get(localIdx).getFloat32(), localA.get(localIdx + stride).getFloat32())));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, localA.get(0));
        }
    }

    public static HalfFloat computeMinSequential(HalfFloatArray input) {
        HalfFloat acc = new HalfFloat(0.0f);
        for (int i = 0; i < input.getSize(); i++) {
            acc = new HalfFloat(TornadoMath.min(acc.getFloat32(), input.get(i).getFloat32()));
        }
        return acc;
    }

    private static void halfFloatReductionMinGlobalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID
        int id = localGroupSize * groupID + localIdx;

        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                a.set(id, new HalfFloat(TornadoMath.min(a.get(id).getFloat32(), a.get(id + stride).getFloat32())));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, a.get(id));
        }
    }

    public static void halfFloatReductionMinLocalMemory(KernelContext context, HalfFloatArray a, HalfFloatArray b) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        HalfFloatArray localA = context.allocateHalfFloatLocalArray(256);
        localA.set(localIdx, a.get(globalIdx));
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localA.set(localIdx, new HalfFloat(TornadoMath.min(localA.get(localIdx).getFloat32(), localA.get(localIdx + stride).getFloat32())));
            }
        }
        if (localIdx == 0) {
            b.set(groupID, localA.get(0));
        }
    }

    @Test
    public void testHalfFloatReductionsAddGlobalMemory() throws TornadoExecutionPlanException {
        final int size = 512;
        final int localSize = 32;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeAddSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionAddGlobalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(finalSum.getFloat32() + reduce.get(i).getFloat32());
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsAddLocalMemory01() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeAddSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionAddLocalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(finalSum.getFloat32() + reduce.get(i).getFloat32());
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsAddLocalMemory02() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeAddSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionAddLocalMemory, context, input, reduce, localSize) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(finalSum.getFloat32() + reduce.get(i).getFloat32());
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsMaxGlobalMemory() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeMaxSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionMaxGlobalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(TornadoMath.max(finalSum.getFloat32(), reduce.get(i).getFloat32()));
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsMaxLocalMemory() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeMaxSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionMaxLocalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }
        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(TornadoMath.max(finalSum.getFloat32(), reduce.get(i).getFloat32()));
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsMinGlobalMemory() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeMinSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionMinGlobalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(TornadoMath.min(finalSum.getFloat32(), reduce.get(i).getFloat32()));
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }

    @Test
    public void testHalfFloatReductionsMinLocalMemory() throws TornadoExecutionPlanException {
        final int size = 1024;
        final int localSize = 256;
        HalfFloatArray input = new HalfFloatArray(size);
        HalfFloatArray reduce = new HalfFloatArray(size / localSize);
        IntStream.range(0, input.getSize()).sequential().forEach(i -> input.set(i, new HalfFloat((float) i)));
        HalfFloat sequential = computeMinSequential(input);

        WorkerGrid worker = new WorkerGrid1D(size);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        KernelContext context = new KernelContext();

        TaskGraph taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input, localSize) //
                .task("t0", TestReductionsHalfFloatsKernelContext::halfFloatReductionMinLocalMemory, context, input, reduce) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, reduce);
        // Change the Grid
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler) //
                    .execute();
        }

        // Final SUM
        HalfFloat finalSum = new HalfFloat(0.0f);
        for (int i = 0; i < reduce.getSize(); i++) {
            finalSum = new HalfFloat(TornadoMath.min(finalSum.getFloat32(), reduce.get(i).getFloat32()));
        }

        assertEquals(sequential.getFloat32(), finalSum.getFloat32(), 0.1f);
    }
}