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
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * OpenCL FFI (Foreign Function Interface) bindings using Java's Panama API.
 * This class replaces JNI calls with direct FFI calls to the OpenCL library.
 */
public class OpenCLFFI {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup OPENCL_LIB;

    // OpenCL function handles
    private static final MethodHandle clGetPlatformIDs;
    private static final MethodHandle clGetDeviceIDs;
    private static final MethodHandle clCreateContext;
    private static final MethodHandle clReleaseContext;
    private static final MethodHandle clCreateCommandQueue;
    private static final MethodHandle clCreateCommandQueueWithProperties;
    private static final MethodHandle clReleaseCommandQueue;
    private static final MethodHandle clCreateProgramWithSource;
    private static final MethodHandle clBuildProgram;
    private static final MethodHandle clGetProgramInfo;
    private static final MethodHandle clGetProgramBuildInfo;
    private static final MethodHandle clReleaseProgram;
    private static final MethodHandle clCreateKernel;
    private static final MethodHandle clSetKernelArg;
    private static final MethodHandle clReleaseKernel;
    private static final MethodHandle clCreateBuffer;
    private static final MethodHandle clReleaseMemObject;
    private static final MethodHandle clEnqueueNDRangeKernel;
    private static final MethodHandle clEnqueueReadBuffer;
    private static final MethodHandle clEnqueueWriteBuffer;
    private static final MethodHandle clFlush;
    private static final MethodHandle clFinish;
    private static final MethodHandle clGetEventInfo;
    private static final MethodHandle clGetEventProfilingInfo;
    private static final MethodHandle clReleaseEvent;
    private static final MethodHandle clWaitForEvents;
    private static final MethodHandle clEnqueueMarkerWithWaitList;
    private static final MethodHandle clEnqueueBarrierWithWaitList;
    private static final MethodHandle clGetPlatformInfo;
    private static final MethodHandle clGetDeviceInfo;
    private static final MethodHandle clGetCommandQueueInfo;
    private static final MethodHandle clGetContextInfo;
    private static final MethodHandle clGetKernelInfo;
    private static final MethodHandle clGetKernelWorkGroupInfo;

    // OpenCL error codes
    public static final int CL_SUCCESS = 0;
    public static final int CL_DEVICE_NOT_FOUND = -1;
    public static final int CL_DEVICE_NOT_AVAILABLE = -2;
    public static final int CL_OUT_OF_HOST_MEMORY = -6;

    static {
        try {
            // Load OpenCL library
            String os = System.getProperty("os.name").toLowerCase();
            String libName;
            if (os.contains("win")) {
                libName = "OpenCL";
            } else if (os.contains("mac")) {
                libName = "OpenCL";
            } else {
                libName = "OpenCL";
            }

            OPENCL_LIB = SymbolLookup.libraryLookup(libName, Arena.global());

            // Initialize function handles
            clGetPlatformIDs = downcallHandle("clGetPlatformIDs",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));

            clGetDeviceIDs = downcallHandle("clGetDeviceIDs",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS));

            clCreateContext = downcallHandle("clCreateContext",
                FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

            clReleaseContext = downcallHandle("clReleaseContext",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clCreateCommandQueue = downcallHandle("clCreateCommandQueue",
                FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

            clCreateCommandQueueWithProperties = downcallHandle("clCreateCommandQueueWithProperties",
                FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

            clReleaseCommandQueue = downcallHandle("clReleaseCommandQueue",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clCreateProgramWithSource = downcallHandle("clCreateProgramWithSource",
                FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

            clBuildProgram = downcallHandle("clBuildProgram",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));

            clGetProgramInfo = downcallHandle("clGetProgramInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetProgramBuildInfo = downcallHandle("clGetProgramBuildInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clReleaseProgram = downcallHandle("clReleaseProgram",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clCreateKernel = downcallHandle("clCreateKernel",
                FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

            clSetKernelArg = downcallHandle("clSetKernelArg",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS));

            clReleaseKernel = downcallHandle("clReleaseKernel",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clCreateBuffer = downcallHandle("clCreateBuffer",
                FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS));

            clReleaseMemObject = downcallHandle("clReleaseMemObject",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clEnqueueNDRangeKernel = downcallHandle("clEnqueueNDRangeKernel",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            clEnqueueReadBuffer = downcallHandle("clEnqueueReadBuffer",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            clEnqueueWriteBuffer = downcallHandle("clEnqueueWriteBuffer",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            clFlush = downcallHandle("clFlush",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clFinish = downcallHandle("clFinish",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clGetEventInfo = downcallHandle("clGetEventInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetEventProfilingInfo = downcallHandle("clGetEventProfilingInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clReleaseEvent = downcallHandle("clReleaseEvent",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            clWaitForEvents = downcallHandle("clWaitForEvents",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS));

            clEnqueueMarkerWithWaitList = downcallHandle("clEnqueueMarkerWithWaitList",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            clEnqueueBarrierWithWaitList = downcallHandle("clEnqueueBarrierWithWaitList",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            clGetPlatformInfo = downcallHandle("clGetPlatformInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetDeviceInfo = downcallHandle("clGetDeviceInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetCommandQueueInfo = downcallHandle("clGetCommandQueueInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetContextInfo = downcallHandle("clGetContextInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetKernelInfo = downcallHandle("clGetKernelInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

            clGetKernelWorkGroupInfo = downcallHandle("clGetKernelWorkGroupInfo",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS));

        } catch (Exception e) {
            throw new ExceptionInInitializerError("Failed to initialize OpenCL FFI: " + e.getMessage());
        }
    }

    private static MethodHandle downcallHandle(String name, FunctionDescriptor descriptor) {
        return OPENCL_LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, descriptor))
            .orElseThrow(() -> new UnsatisfiedLinkError("Failed to find OpenCL function: " + name));
    }

    // Public API methods

    public static int getPlatformCount() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment numPlatforms = arena.allocate(JAVA_INT);
            int status = (int) clGetPlatformIDs.invoke(0, MemorySegment.NULL, numPlatforms);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetPlatformIDs failed with error: " + status);
            }
            return numPlatforms.get(JAVA_INT, 0);
        }
    }

    public static int getPlatformIDs(long[] platformIds) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment platforms = arena.allocate(ADDRESS, platformIds.length);
            MemorySegment numPlatforms = arena.allocate(JAVA_INT);

            int status = (int) clGetPlatformIDs.invoke(platformIds.length, platforms, numPlatforms);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetPlatformIDs failed with error: " + status);
            }

            for (int i = 0; i < platformIds.length; i++) {
                platformIds[i] = platforms.getAtIndex(ADDRESS, i).address();
            }

            return numPlatforms.get(JAVA_INT, 0);
        }
    }

    public static int getDeviceIDs(long platformId, long deviceType, long[] deviceIds) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment platform = MemorySegment.ofAddress(platformId);
            MemorySegment devices = arena.allocate(ADDRESS, deviceIds.length);
            MemorySegment numDevices = arena.allocate(JAVA_INT);

            int status = (int) clGetDeviceIDs.invoke(platform, deviceType, deviceIds.length, devices, numDevices);
            if (status != CL_SUCCESS) {
                return status;
            }

            for (int i = 0; i < deviceIds.length; i++) {
                deviceIds[i] = devices.getAtIndex(ADDRESS, i).address();
            }

            return numDevices.get(JAVA_INT, 0);
        }
    }

    public static long createContext(long[] properties, long[] devices) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment propsSegment = MemorySegment.NULL;
            if (properties != null && properties.length > 0) {
                propsSegment = arena.allocate(JAVA_LONG, properties.length);
                for (int i = 0; i < properties.length; i++) {
                    propsSegment.setAtIndex(JAVA_LONG, i, properties[i]);
                }
            }

            MemorySegment devicesSegment = arena.allocate(ADDRESS, devices.length);
            for (int i = 0; i < devices.length; i++) {
                devicesSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(devices[i]));
            }

            MemorySegment errcode = arena.allocate(JAVA_INT);
            MemorySegment context = (MemorySegment) clCreateContext.invoke(
                propsSegment, devices.length, devicesSegment,
                MemorySegment.NULL, MemorySegment.NULL, errcode);

            int status = errcode.get(JAVA_INT, 0);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clCreateContext failed with error: " + status);
            }

            return context.address();
        }
    }

    public static int releaseContext(long contextId) throws Throwable {
        MemorySegment context = MemorySegment.ofAddress(contextId);
        return (int) clReleaseContext.invoke(context);
    }

    public static long createCommandQueue(long contextId, long deviceId, long properties) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = MemorySegment.ofAddress(contextId);
            MemorySegment device = MemorySegment.ofAddress(deviceId);
            MemorySegment errcode = arena.allocate(JAVA_INT);

            MemorySegment queue = (MemorySegment) clCreateCommandQueue.invoke(
                context, device, properties, errcode);

            int status = errcode.get(JAVA_INT, 0);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clCreateCommandQueue failed with error: " + status);
            }

            return queue.address();
        }
    }

    public static int releaseCommandQueue(long queueId) throws Throwable {
        MemorySegment queue = MemorySegment.ofAddress(queueId);
        return (int) clReleaseCommandQueue.invoke(queue);
    }

    public static long createProgramWithSource(long contextId, String source) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = MemorySegment.ofAddress(contextId);
            MemorySegment sourceStr = arena.allocateUtf8String(source);
            MemorySegment sourcePtr = arena.allocate(ADDRESS);
            sourcePtr.set(ADDRESS, 0, sourceStr);
            MemorySegment errcode = arena.allocate(JAVA_INT);

            MemorySegment program = (MemorySegment) clCreateProgramWithSource.invoke(
                context, 1, sourcePtr, MemorySegment.NULL, errcode);

            int status = errcode.get(JAVA_INT, 0);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clCreateProgramWithSource failed with error: " + status);
            }

            return program.address();
        }
    }

    public static int buildProgram(long programId, long[] devices, String options) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment program = MemorySegment.ofAddress(programId);
            MemorySegment devicesSegment = MemorySegment.NULL;

            if (devices != null && devices.length > 0) {
                devicesSegment = arena.allocate(ADDRESS, devices.length);
                for (int i = 0; i < devices.length; i++) {
                    devicesSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(devices[i]));
                }
            }

            MemorySegment optionsStr = options != null ? arena.allocateUtf8String(options) : MemorySegment.NULL;

            return (int) clBuildProgram.invoke(program,
                devices != null ? devices.length : 0,
                devicesSegment, optionsStr, MemorySegment.NULL, MemorySegment.NULL);
        }
    }

    public static int releaseProgram(long programId) throws Throwable {
        MemorySegment program = MemorySegment.ofAddress(programId);
        return (int) clReleaseProgram.invoke(program);
    }

    public static long createKernel(long programId, String kernelName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment program = MemorySegment.ofAddress(programId);
            MemorySegment nameStr = arena.allocateUtf8String(kernelName);
            MemorySegment errcode = arena.allocate(JAVA_INT);

            MemorySegment kernel = (MemorySegment) clCreateKernel.invoke(program, nameStr, errcode);

            int status = errcode.get(JAVA_INT, 0);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clCreateKernel failed with error: " + status);
            }

            return kernel.address();
        }
    }

    public static int setKernelArg(long kernelId, int argIndex, long argSize, MemorySegment argValue) throws Throwable {
        MemorySegment kernel = MemorySegment.ofAddress(kernelId);
        return (int) clSetKernelArg.invoke(kernel, argIndex, argSize, argValue);
    }

    public static int releaseKernel(long kernelId) throws Throwable {
        MemorySegment kernel = MemorySegment.ofAddress(kernelId);
        return (int) clReleaseKernel.invoke(kernel);
    }

    public static long createBuffer(long contextId, long flags, long size, MemorySegment hostPtr) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = MemorySegment.ofAddress(contextId);
            MemorySegment errcode = arena.allocate(JAVA_INT);

            MemorySegment buffer = (MemorySegment) clCreateBuffer.invoke(
                context, flags, size, hostPtr, errcode);

            int status = errcode.get(JAVA_INT, 0);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clCreateBuffer failed with error: " + status);
            }

            return buffer.address();
        }
    }

    public static int releaseMemObject(long memId) throws Throwable {
        MemorySegment mem = MemorySegment.ofAddress(memId);
        return (int) clReleaseMemObject.invoke(mem);
    }

    public static long enqueueNDRangeKernel(long queueId, long kernelId, int workDim,
                                            long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize,
                                            long[] waitList) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);
            MemorySegment kernel = MemorySegment.ofAddress(kernelId);

            MemorySegment offsetSegment = globalWorkOffset != null ?
                arena.allocate(JAVA_LONG, globalWorkOffset.length) : MemorySegment.NULL;
            if (globalWorkOffset != null) {
                for (int i = 0; i < globalWorkOffset.length; i++) {
                    offsetSegment.setAtIndex(JAVA_LONG, i, globalWorkOffset[i]);
                }
            }

            MemorySegment globalSegment = arena.allocate(JAVA_LONG, globalWorkSize.length);
            for (int i = 0; i < globalWorkSize.length; i++) {
                globalSegment.setAtIndex(JAVA_LONG, i, globalWorkSize[i]);
            }

            MemorySegment localSegment = localWorkSize != null ?
                arena.allocate(JAVA_LONG, localWorkSize.length) : MemorySegment.NULL;
            if (localWorkSize != null) {
                for (int i = 0; i < localWorkSize.length; i++) {
                    localSegment.setAtIndex(JAVA_LONG, i, localWorkSize[i]);
                }
            }

            MemorySegment waitSegment = MemorySegment.NULL;
            int numEvents = 0;
            if (waitList != null && waitList.length > 0) {
                waitSegment = arena.allocate(ADDRESS, waitList.length);
                for (int i = 0; i < waitList.length; i++) {
                    waitSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(waitList[i]));
                }
                numEvents = waitList.length;
            }

            MemorySegment event = arena.allocate(ADDRESS);

            int status = (int) clEnqueueNDRangeKernel.invoke(queue, kernel, workDim,
                offsetSegment, globalSegment, localSegment, numEvents, waitSegment, event);

            if (status != CL_SUCCESS) {
                throw new RuntimeException("clEnqueueNDRangeKernel failed with error: " + status);
            }

            return event.get(ADDRESS, 0).address();
        }
    }

    public static long enqueueWriteBuffer(long queueId, long bufferId, boolean blocking,
                                          long offset, long size, MemorySegment ptr, long[] waitList) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);
            MemorySegment buffer = MemorySegment.ofAddress(bufferId);

            MemorySegment waitSegment = MemorySegment.NULL;
            int numEvents = 0;
            if (waitList != null && waitList.length > 0) {
                waitSegment = arena.allocate(ADDRESS, waitList.length);
                for (int i = 0; i < waitList.length; i++) {
                    waitSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(waitList[i]));
                }
                numEvents = waitList.length;
            }

            MemorySegment event = arena.allocate(ADDRESS);

            int status = (int) clEnqueueWriteBuffer.invoke(queue, buffer, blocking ? 1 : 0,
                offset, size, ptr, numEvents, waitSegment, event);

            if (status != CL_SUCCESS) {
                throw new RuntimeException("clEnqueueWriteBuffer failed with error: " + status);
            }

            return event.get(ADDRESS, 0).address();
        }
    }

    public static long enqueueReadBuffer(long queueId, long bufferId, boolean blocking,
                                         long offset, long size, MemorySegment ptr, long[] waitList) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);
            MemorySegment buffer = MemorySegment.ofAddress(bufferId);

            MemorySegment waitSegment = MemorySegment.NULL;
            int numEvents = 0;
            if (waitList != null && waitList.length > 0) {
                waitSegment = arena.allocate(ADDRESS, waitList.length);
                for (int i = 0; i < waitList.length; i++) {
                    waitSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(waitList[i]));
                }
                numEvents = waitList.length;
            }

            MemorySegment event = arena.allocate(ADDRESS);

            int status = (int) clEnqueueReadBuffer.invoke(queue, buffer, blocking ? 1 : 0,
                offset, size, ptr, numEvents, waitSegment, event);

            if (status != CL_SUCCESS) {
                throw new RuntimeException("clEnqueueReadBuffer failed with error: " + status);
            }

            return event.get(ADDRESS, 0).address();
        }
    }

    public static int flush(long queueId) throws Throwable {
        MemorySegment queue = MemorySegment.ofAddress(queueId);
        return (int) clFlush.invoke(queue);
    }

    public static int finish(long queueId) throws Throwable {
        MemorySegment queue = MemorySegment.ofAddress(queueId);
        return (int) clFinish.invoke(queue);
    }

    public static int releaseEvent(long eventId) throws Throwable {
        MemorySegment event = MemorySegment.ofAddress(eventId);
        return (int) clReleaseEvent.invoke(event);
    }

    public static int waitForEvents(long[] events) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment eventsSegment = arena.allocate(ADDRESS, events.length);
            for (int i = 0; i < events.length; i++) {
                eventsSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(events[i]));
            }
            return (int) clWaitForEvents.invoke(events.length, eventsSegment);
        }
    }

    public static long enqueueMarkerWithWaitList(long queueId, long[] waitList) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);

            MemorySegment waitSegment = MemorySegment.NULL;
            int numEvents = 0;
            if (waitList != null && waitList.length > 0) {
                waitSegment = arena.allocate(ADDRESS, waitList.length);
                for (int i = 0; i < waitList.length; i++) {
                    waitSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(waitList[i]));
                }
                numEvents = waitList.length;
            }

            MemorySegment event = arena.allocate(ADDRESS);

            int status = (int) clEnqueueMarkerWithWaitList.invoke(queue, numEvents, waitSegment, event);

            if (status != CL_SUCCESS) {
                throw new RuntimeException("clEnqueueMarkerWithWaitList failed with error: " + status);
            }

            return event.get(ADDRESS, 0).address();
        }
    }

    public static long enqueueBarrierWithWaitList(long queueId, long[] waitList) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);

            MemorySegment waitSegment = MemorySegment.NULL;
            int numEvents = 0;
            if (waitList != null && waitList.length > 0) {
                waitSegment = arena.allocate(ADDRESS, waitList.length);
                for (int i = 0; i < waitList.length; i++) {
                    waitSegment.setAtIndex(ADDRESS, i, MemorySegment.ofAddress(waitList[i]));
                }
                numEvents = waitList.length;
            }

            MemorySegment event = arena.allocate(ADDRESS);

            int status = (int) clEnqueueBarrierWithWaitList.invoke(queue, numEvents, waitSegment, event);

            if (status != CL_SUCCESS) {
                throw new RuntimeException("clEnqueueBarrierWithWaitList failed with error: " + status);
            }

            return event.get(ADDRESS, 0).address();
        }
    }

    public static String getPlatformInfo(long platformId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment platform = MemorySegment.ofAddress(platformId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetPlatformInfo.invoke(platform, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetPlatformInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetPlatformInfo.invoke(platform, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetPlatformInfo failed with error: " + status);
            }

            return buffer.getUtf8String(0);
        }
    }

    public static byte[] getDeviceInfo(long deviceId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment device = MemorySegment.ofAddress(deviceId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetDeviceInfo.invoke(device, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetDeviceInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetDeviceInfo.invoke(device, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetDeviceInfo failed with error: " + status);
            }

            return buffer.toArray(JAVA_BYTE);
        }
    }

    public static byte[] getCommandQueueInfo(long queueId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment queue = MemorySegment.ofAddress(queueId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetCommandQueueInfo.invoke(queue, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetCommandQueueInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetCommandQueueInfo.invoke(queue, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetCommandQueueInfo failed with error: " + status);
            }

            return buffer.toArray(JAVA_BYTE);
        }
    }

    public static byte[] getProgramBuildInfo(long programId, long deviceId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment program = MemorySegment.ofAddress(programId);
            MemorySegment device = MemorySegment.ofAddress(deviceId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetProgramBuildInfo.invoke(program, device, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetProgramBuildInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetProgramBuildInfo.invoke(program, device, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetProgramBuildInfo failed with error: " + status);
            }

            return buffer.toArray(JAVA_BYTE);
        }
    }

    public static byte[] getEventProfilingInfo(long eventId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment event = MemorySegment.ofAddress(eventId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetEventProfilingInfo.invoke(event, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetEventProfilingInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetEventProfilingInfo.invoke(event, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetEventProfilingInfo failed with error: " + status);
            }

            return buffer.toArray(JAVA_BYTE);
        }
    }

    public static byte[] getKernelWorkGroupInfo(long kernelId, long deviceId, int paramName) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment kernel = MemorySegment.ofAddress(kernelId);
            MemorySegment device = MemorySegment.ofAddress(deviceId);
            MemorySegment sizeRet = arena.allocate(JAVA_LONG);

            // First call to get size
            int status = (int) clGetKernelWorkGroupInfo.invoke(kernel, device, paramName, 0L, MemorySegment.NULL, sizeRet);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetKernelWorkGroupInfo failed with error: " + status);
            }

            long size = sizeRet.get(JAVA_LONG, 0);
            MemorySegment buffer = arena.allocate(size);

            // Second call to get data
            status = (int) clGetKernelWorkGroupInfo.invoke(kernel, device, paramName, size, buffer, MemorySegment.NULL);
            if (status != CL_SUCCESS) {
                throw new RuntimeException("clGetKernelWorkGroupInfo failed with error: " + status);
            }

            return buffer.toArray(JAVA_BYTE);
        }
    }
}
