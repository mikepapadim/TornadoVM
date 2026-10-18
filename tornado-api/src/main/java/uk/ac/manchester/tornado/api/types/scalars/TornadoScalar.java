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
package uk.ac.manchester.tornado.api.types.scalars;

/**
 * A scalar task argument whose value can change between executions of a plan.
 *
 * <p>
 * A primitive task argument ({@code int n}) is a constant: its value is read once, when the task is
 * created, and the compiler may fold it into the kernel. A scalar holder is a runtime value: the kernel
 * declares the holder type ({@code IntScalar n}) and reads it with {@code n.get()}; the host changes it
 * with {@code n.set(v)}, and every launch passes the value the holder has at that moment, by value, as
 * the kernel parameter. No buffer, transfer or recompilation is involved.
 * </p>
 *
 * <p>
 * A holder may bound a {@code @Parallel} loop ({@code i < n.get()}): each launch is then sized for the
 * holder's value at that launch. Holders are read-only on the device: a kernel cannot {@code set} them.
 * </p>
 */
public sealed interface TornadoScalar permits IntScalar, LongScalar, FloatScalar, DoubleScalar {

    /** The current value, boxed: what the next launch passes. */
    Number value();
}
