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

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

/**
 * {@code TornadoExecutionPlan.withScalar}: a primitive task argument changed between executions of one
 * plan. The kernel keeps its {@code int}; the task is compiled again for a new value (a value seen
 * before reuses its kernel, and a scalar that keeps changing becomes a runtime value).
 *
 * <p>
 * How to test?
 * <code>
 *     tornado-test -V uk.ac.manchester.tornado.unittests.parameters.TestWithScalar
 * </code>
 * </p>
 */
public class TestWithScalar extends TornadoTestBase {

    private static final int N = 1024;

    public static void addOffset(IntArray in, IntArray out, int offset) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) + offset);
        }
    }

    public static void scale(FloatArray x, FloatArray y, float alpha) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            y.set(i, alpha * x.get(i));
        }
    }

    public static void encode(IntArray out, int a, int b) {
        for (@Parallel int i = 0; i < out.getSize(); i++) {
            out.set(i, a * 1000 + b);
        }
    }

    public static void addToPrefix(IntArray out, int length) {
        for (@Parallel int i = 0; i < length; i++) {
            out.set(i, out.get(i) + length + 1);
        }
    }

    @Test
    public void testIntChangesBetweenExecutions() throws TornadoExecutionPlanException {
        IntArray in = new IntArray(N);
        IntArray out = new IntArray(N);
        in.init(10);
        TaskGraph graph = new TaskGraph("w0") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("t0", TestWithScalar::addOffset, in, out, 1) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            // 1 is the value given to task(...); 7 and -3 are new; 7 again reuses its kernel.
            for (int k : new int[] { 1, 7, -3, 7, 1000, 5, 6 }) {
                plan.withScalar("w0.t0", 2, k).execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(10 + k, out.get(i), "offset " + k + " at " + i);
                }
            }
        }
    }

    @Test
    public void testFloat() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(N);
        FloatArray y = new FloatArray(N);
        x.init(2.0f);
        TaskGraph graph = new TaskGraph("w1") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x) //
                .task("t0", TestWithScalar::scale, x, y, 0.5f) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, y);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (float a : new float[] { 0.5f, 3.0f, -1.25f }) {
                plan.withScalar("w1.t0", 2, a).execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(2.0f * a, y.get(i), 1e-6f);
                }
            }
        }
    }

    /** Two arguments with equal values: changing one leaves the other alone. */
    @Test
    public void testEqualArgumentsAreIndependent() throws TornadoExecutionPlanException {
        IntArray out = new IntArray(N);
        TaskGraph graph = new TaskGraph("w2") //
                .task("t0", TestWithScalar::encode, out, 5, 5) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
            assertEquals(5005, out.get(0));
            plan.withScalar("w2.t0", 2, 9).execute();
            assertEquals(5009, out.get(0));
            plan.withScalar("w2.t0", 1, 3).execute();
            assertEquals(3009, out.get(N - 1));
        }
    }

    /** A varying @Parallel bound: each execution covers exactly its own length. */
    @Test
    public void testParallelBound() throws TornadoExecutionPlanException {
        IntArray out = new IntArray(N);
        TaskGraph graph = new TaskGraph("w3") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, out) //
                .task("t0", TestWithScalar::addToPrefix, out, 300) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            for (int n : new int[] { 300, 17, N, 513, 40, 300 }) {
                out.init(-1);
                plan.withScalar("w3.t0", 1, n).execute();
                for (int i = 0; i < N; i++) {
                    assertEquals(i < n ? n : -1, out.get(i), "length " + n + " at " + i);
                }
            }
        }
    }

    @Test
    public void testMisuseIsRejected() {
        IntArray in = new IntArray(N);
        IntArray out = new IntArray(N);
        TaskGraph graph = new TaskGraph("w4") //
                .task("t0", TestWithScalar::addOffset, in, out, 1) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            assertThrows(TornadoRuntimeException.class, () -> plan.withScalar("w4.nope", 2, 3), "unknown task");
            assertThrows(TornadoRuntimeException.class, () -> plan.withScalar("w4.t0", 0, 3), "argument 0 is an array");
            assertThrows(TornadoRuntimeException.class, () -> plan.withScalar("w4.t0", 2, 3L), "a long for an int");
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
    }
}
