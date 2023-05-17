/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2023, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *
 */
package uk.ac.manchester.tornado.runtime.common;

public class BatchMetaData {

    private static java.util.HashMap<Class<?>, Byte> dataTypesSize = new java.util.HashMap<>();

    static {
        dataTypesSize.put(byte.class, (byte) 1);
        dataTypesSize.put(char.class, (byte) 2);
        dataTypesSize.put(short.class, (byte) 2);
        dataTypesSize.put(int.class, (byte) 4);
        dataTypesSize.put(float.class, (byte) 4);
        dataTypesSize.put(long.class, (byte) 8);
        dataTypesSize.put(double.class, (byte) 8);
    }

    private final int totalChunks;
    private final int remainingChunkSize;
    private final short numBytesType;

    /**
     * Constructs a BatchMetaData object with the specified parameters.
     *
     * @param totalChunks        The total number of chunks in the batch.
     * @param remainingChunkSize The size of the remaining chunk (last chunk) in the batch.
     * @param numBytesType       The number of bytes required to represent the data type.
     */
    public BatchMetaData(int totalChunks, int remainingChunkSize, short numBytesType) {
        this.totalChunks = totalChunks;
        this.remainingChunkSize = remainingChunkSize;
        this.numBytesType = numBytesType;
    }

    public int getTotalChunks() {
        return totalChunks;
    }

    public int getRemainingChunkSize() {
        return remainingChunkSize;
    }

    public short getNumBytesType() {
        return numBytesType;
    }

    /**
     * Computes the chunk sizes for batching based on the provided TornadoExecutionContext and batch size.
     *
     * @param context   The TornadoExecutionContext containing the input objects.
     * @param batchSize The desired batch size for processing.
     * @return An instance of BatchMetaData containing the computed chunk sizes.
     * @throws uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException if unsupported data types or input objects with different data types/sizes are encountered.
     */
    public static BatchMetaData computeChunkSizes(uk.ac.manchester.tornado.runtime.graph.TornadoExecutionContext context, long batchSize) {
        // Get the size of the batch
        java.util.List<Object> inputObjects = context.getObjects();
        long totalSize = 0;
        byte typeSize = 1;

        java.util.HashSet<Class<?>> classObjects = new java.util.HashSet<>();
        java.util.HashSet<Long> inputSizes = new java.util.HashSet<>();

        // XXX: Get a list for all objects
        for (Object o : inputObjects) {
            if (o.getClass().isArray()) {
                Class<?> componentType = o.getClass().getComponentType();
                if (dataTypesSize.get(componentType) == null) {
                    throw new uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException("[UNSUPPORTED] Data type not supported for processing in batches");
                }
                long size = java.lang.reflect.Array.getLength(o);
                typeSize = dataTypesSize.get(componentType);
                totalSize = size * typeSize;

                classObjects.add(componentType);
                inputSizes.add(totalSize);
                if (classObjects.size() > 1) {
                    throw new uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException("[UNSUPPORTED] Input objects with different data types not currently supported");
                }
                if (inputSizes.size() > 1) {
                    throw new uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException("[UNSUPPORTED] Input objects with different sizes not currently supported");
                }
            }
        }

        int totalChunks = (int) (totalSize / batchSize);
        int remainingChunkSize = (int) (totalSize % batchSize);

        if (Tornado.DEBUG) {
            System.out.println("Batch Size: " + batchSize);
            System.out.println("Total chunks: " + totalChunks);
            System.out.println("remainingChunkSize: " + remainingChunkSize);
        }
        return new BatchMetaData(totalChunks, remainingChunkSize, typeSize);
    }
}
