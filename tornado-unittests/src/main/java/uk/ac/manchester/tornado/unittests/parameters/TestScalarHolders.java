/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.unittests.parameters;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.types.scalars.DoubleScalar;
import uk.ac.manchester.tornado.api.types.scalars.FloatScalar;
import uk.ac.manchester.tornado.api.types.scalars.IntScalar;
import uk.ac.manchester.tornado.api.types.scalars.LongScalar;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * Scalar holders ({@link IntScalar}, ...): task arguments whose value changes between executions of
 * one plan, without rebuilding the graph or recompiling the kernel.
 *
 * <p>
 * How to test?
 * <code>
 *     tornado-test -V uk.ac.manchester.tornado.unittests.parameters.TestScalarHolders
 * </code>
 * </p>
 */
public class TestScalarHolders extends TornadoTestBase {

    private static final int N = 1024;

    public static void addOffset(IntArray in, IntArray out, IntScalar offset) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) + offset.get());
        }
    }

    public static void saxpy(FloatArray x, FloatArray y, FloatScalar alpha) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            y.set(i, alpha.get() * x.get(i) + y.get(i));
        }
    }

    public static void mixed(LongArray out, LongScalar base, DoubleScalar scale, IntScalar step) {
        for (@Parallel int i = 0; i < out.getSize(); i++) {
            out.set(i, base.get() + (long) (scale.get() * i) + step.get());
        }
    }

    /** The loop bound is a runtime value: each launch is sized for the holder's value at that launch. */
    public static void addToPrefix(IntArray out, IntScalar length, IntScalar value) {
        for (@Parallel int i = 0; i < length.get(); i++) {
            out.set(i, out.get(i) + value.get());
        }
    }

    public static void addToPrefixInt(IntArray out, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            out.set(i, out.get(i) + length + 1);
        }
    }

    /** The active length is a runtime value: the grid covers the capacity, the guard the length. */
    public static void prefixFill(KernelContext context, IntArray out, IntScalar length, IntScalar value) {
        int i = context.globalIdx;
        if (i < length.get()) {
            out.set(i, value.get());
        }
    }

    @Test
    public void testValueChangesBetweenExecutions() throws TornadoExecutionPlanException {
        IntArray in = new IntArray(N);
        IntArray out = new IntArray(N);
        in.init(10);
        IntScalar offset = new IntScalar(1);

        TaskGraph graph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("t0", TestScalarHolders::addOffset, in, out, offset) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (int k : new int[] { 1, 7, -3, 1000 }) {
                offset.set(k);
                plan.execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(10 + k, out.get(i), "offset " + k + " at " + i);
                }
            }
        }
    }

    @Test
    public void testFloatScalar() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N);
        x.init(2.0f);
        FloatScalar alpha = new FloatScalar(0.5f);

        TaskGraph graph = new TaskGraph("s1") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                .task("t0", TestScalarHolders::saxpy, x, y, alpha) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, y);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (float a : new float[] { 0.5f, 3.0f, -1.25f }) {
                y.init(1.0f);
                alpha.set(a);
                plan.execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(a * 2.0f + 1.0f, y.get(i), 1e-6f);
                }
            }
        }
    }

    @Test
    public void testMixedHolderTypes() throws TornadoExecutionPlanException {
        LongArray out = new LongArray(N);
        LongScalar base = new LongScalar(1L << 40);
        DoubleScalar scale = new DoubleScalar(2.0);
        IntScalar step = new IntScalar(3);

        TaskGraph graph = new TaskGraph("s2") //
                .task("t0", TestScalarHolders::mixed, out, base, scale, step) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
            for (int i = 0; i < N; i++) {
                assertEquals((1L << 40) + 2L * i + 3, out.get(i));
            }
            base.set(-5);
            scale.set(0.5);
            step.set(0);
            plan.execute();
            for (int i = 0; i < N; i++) {
                assertEquals(-5 + (long) (0.5 * i), out.get(i));
            }
        }
    }

    @Test
    public void testRuntimeLengthWithKernelContext() throws TornadoExecutionPlanException {
        IntArray out = new IntArray(N);
        IntScalar length = new IntScalar(N);
        IntScalar value = new IntScalar(0);

        WorkerGrid1D worker = new WorkerGrid1D(N);
        worker.setLocalWork(128, 1, 1);
        GridScheduler grid = new GridScheduler("s3.t0", worker);
        TaskGraph graph = new TaskGraph("s3") //
                .task("t0", TestScalarHolders::prefixFill, new KernelContext(), out, length, value) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid).execute();
            length.set(100);
            value.set(42);
            plan.withGridScheduler(grid).execute();
            for (int i = 0; i < N; i++) {
                assertEquals(i < 100 ? 42 : 0, out.get(i), "at " + i);
            }
        }
    }

    @Test
    public void testRuntimeParallelBound() throws TornadoExecutionPlanException {
        IntArray out = new IntArray(N);
        IntScalar length = new IntScalar(300);
        IntScalar value = new IntScalar(1);

        TaskGraph graph = new TaskGraph("s5") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, out) //
                .task("t0", TestScalarHolders::addToPrefix, out, length, value) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            // Shorter, then longer than the first launch: the grid follows the bound.
            for (int n : new int[] { 300, 17, N, 0, 513 }) {
                out.init(-1);
                length.set(n);
                value.set(n + 1);
                plan.execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(i < n ? n : -1, out.get(i), "length " + n + " at " + i);
                }
            }
        }
    }

    /**
     * A plain int bound that takes several values across task graphs: once the compiler treats it as a
     * runtime value (tornado.tc.varyingScalars), each launch is still sized for its own value.
     */
    @Test
    public void testVaryingIntParallelBound() throws TornadoExecutionPlanException {
        for (int n : new int[] { 10, 20, 30, 40, N, 7 }) {
            IntArray out = new IntArray(N);
            out.init(-1);
            TaskGraph graph = new TaskGraph("s6_" + n) //
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, out) //
                    .task("t0", TestScalarHolders::addToPrefixInt, out, n) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.execute();
            }
            for (int i = 0; i < N; i++) {
                assertEquals(i < n ? n : -1, out.get(i), "length " + n + " at " + i);
            }
        }
    }

    /** A holder is read-only on the device, so it cannot be an output. */
    @Test
    public void testHolderAsOutputIsRejected() {
        IntScalar offset = new IntScalar(1);
        assertThrows(TornadoRuntimeException.class, () -> new TaskGraph("s4").transferToHost(DataTransferMode.EVERY_EXECUTION, offset));
    }
}
