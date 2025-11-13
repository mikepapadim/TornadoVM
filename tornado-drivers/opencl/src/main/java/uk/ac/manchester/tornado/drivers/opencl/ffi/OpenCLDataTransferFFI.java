/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 */
package uk.ac.manchester.tornado.drivers.opencl.ffi;

import java.lang.foreign.*;

import static java.lang.foreign.ValueLayout.*;

/**
 * FFI helper for OpenCL array data transfers.
 * Handles conversion between Java arrays and native memory for OpenCL operations.
 */
public class OpenCLDataTransferFFI {

    /**
     * Write byte array to OpenCL device buffer
     */
    public static long writeByteArrayToDevice(long queueId, byte[] array, long hostOffset,
                                               boolean blocking, long offset, long bytes,
                                               long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_BYTE, 0, (int)bytes);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write char array to OpenCL device buffer
     */
    public static long writeCharArrayToDevice(long queueId, char[] array, long hostOffset,
                                               boolean blocking, long offset, long bytes,
                                               long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 2; // chars are 2 bytes
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_CHAR, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write short array to OpenCL device buffer
     */
    public static long writeShortArrayToDevice(long queueId, short[] array, long hostOffset,
                                                boolean blocking, long offset, long bytes,
                                                long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 2;
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_SHORT, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write int array to OpenCL device buffer
     */
    public static long writeIntArrayToDevice(long queueId, int[] array, long hostOffset,
                                              boolean blocking, long offset, long bytes,
                                              long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 4;
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_INT, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write long array to OpenCL device buffer
     */
    public static long writeLongArrayToDevice(long queueId, long[] array, long hostOffset,
                                               boolean blocking, long offset, long bytes,
                                               long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 8;
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_LONG, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write float array to OpenCL device buffer
     */
    public static long writeFloatArrayToDevice(long queueId, float[] array, long hostOffset,
                                                boolean blocking, long offset, long bytes,
                                                long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 4;
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_FLOAT, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write double array to OpenCL device buffer
     */
    public static long writeDoubleArrayToDevice(long queueId, double[] array, long hostOffset,
                                                 boolean blocking, long offset, long bytes,
                                                 long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 8;
            MemorySegment dataSegment = arena.allocate(bytes);
            MemorySegment.copy(array, (int)hostOffset, dataSegment, JAVA_DOUBLE, 0, (int)numElements);
            return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
        }
    }

    /**
     * Write from off-heap memory (MemorySegment) to OpenCL device buffer
     */
    public static long writeMemorySegmentToDevice(long queueId, long hostPointer, long hostOffset,
                                                   boolean blocking, long offset, long bytes,
                                                   long devicePtr, long[] events) throws Throwable {
        MemorySegment sourceSegment = MemorySegment.ofAddress(hostPointer).reinterpret(hostOffset + bytes);
        MemorySegment offsetSegment = sourceSegment.asSlice(hostOffset, bytes);
        return OpenCLFFI.enqueueWriteBuffer(queueId, devicePtr, blocking, offset, bytes, offsetSegment, events);
    }

    /**
     * Read byte array from OpenCL device buffer
     */
    public static long readByteArrayFromDevice(long queueId, byte[] array, long hostOffset,
                                                boolean blocking, long offset, long bytes,
                                                long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_BYTE, 0, array, (int)hostOffset, (int)bytes);
            return eventId;
        }
    }

    /**
     * Read char array from OpenCL device buffer
     */
    public static long readCharArrayFromDevice(long queueId, char[] array, long hostOffset,
                                                boolean blocking, long offset, long bytes,
                                                long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 2;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_CHAR, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read short array from OpenCL device buffer
     */
    public static long readShortArrayFromDevice(long queueId, short[] array, long hostOffset,
                                                 boolean blocking, long offset, long bytes,
                                                 long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 2;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_SHORT, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read int array from OpenCL device buffer
     */
    public static long readIntArrayFromDevice(long queueId, int[] array, long hostOffset,
                                               boolean blocking, long offset, long bytes,
                                               long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 4;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_INT, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read long array from OpenCL device buffer
     */
    public static long readLongArrayFromDevice(long queueId, long[] array, long hostOffset,
                                                boolean blocking, long offset, long bytes,
                                                long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 8;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_LONG, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read float array from OpenCL device buffer
     */
    public static long readFloatArrayFromDevice(long queueId, float[] array, long hostOffset,
                                                 boolean blocking, long offset, long bytes,
                                                 long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 4;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_FLOAT, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read double array from OpenCL device buffer
     */
    public static long readDoubleArrayFromDevice(long queueId, double[] array, long hostOffset,
                                                  boolean blocking, long offset, long bytes,
                                                  long devicePtr, long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            long numElements = bytes / 8;
            MemorySegment dataSegment = arena.allocate(bytes);
            long eventId = OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, dataSegment, events);
            MemorySegment.copy(dataSegment, JAVA_DOUBLE, 0, array, (int)hostOffset, (int)numElements);
            return eventId;
        }
    }

    /**
     * Read to off-heap memory (MemorySegment) from OpenCL device buffer
     */
    public static long readMemorySegmentFromDevice(long queueId, long hostPointer, long hostOffset,
                                                    boolean blocking, long offset, long bytes,
                                                    long devicePtr, long[] events) throws Throwable {
        MemorySegment destSegment = MemorySegment.ofAddress(hostPointer).reinterpret(hostOffset + bytes);
        MemorySegment offsetSegment = destSegment.asSlice(hostOffset, bytes);
        return OpenCLFFI.enqueueReadBuffer(queueId, devicePtr, blocking, offset, bytes, offsetSegment, events);
    }
}
