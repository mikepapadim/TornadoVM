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
package uk.ac.manchester.tornado.drivers.ptx.ffi;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * PTX/CUDA FFI (Foreign Function Interface) bindings using Java's Panama API.
 * This class replaces JNI calls with direct FFI calls to the CUDA Driver API library.
 */
public class PTXFFI {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUDA_LIB;

    // CUDA function handles
    private static final MethodHandle cuInit;
    private static final MethodHandle cuDeviceGetCount;
    private static final MethodHandle cuDeviceGet;
    private static final MethodHandle cuDeviceGetName;
    private static final MethodHandle cuDeviceGetAttribute;
    private static final MethodHandle cuDeviceTotalMem;
    private static final MethodHandle cuCtxCreate;
    private static final MethodHandle cuCtxDestroy;
    private static final MethodHandle cuCtxSynchronize;
    private static final MethodHandle cuCtxGetCurrent;
    private static final MethodHandle cuCtxSetCurrent;
    private static final MethodHandle cuModuleLoad;
    private static final MethodHandle cuModuleLoadData;
    private static final MethodHandle cuModuleLoadDataEx;
    private static final MethodHandle cuModuleUnload;
    private static final MethodHandle cuModuleGetFunction;
    private static final MethodHandle cuMemAlloc;
    private static final MethodHandle cuMemFree;
    private static final MethodHandle cuMemcpyHtoD;
    private static final MethodHandle cuMemcpyDtoH;
    private static final MethodHandle cuMemcpyHtoDAsync;
    private static final MethodHandle cuMemcpyDtoHAsync;
    private static final MethodHandle cuLaunchKernel;
    private static final MethodHandle cuStreamCreate;
    private static final MethodHandle cuStreamCreateWithPriority;
    private static final MethodHandle cuStreamDestroy;
    private static final MethodHandle cuStreamSynchronize;
    private static final MethodHandle cuEventCreate;
    private static final MethodHandle cuEventRecord;
    private static final MethodHandle cuEventDestroy;
    private static final MethodHandle cuEventSynchronize;
    private static final MethodHandle cuEventElapsedTime;
    private static final MethodHandle cuCtxGetStreamPriorityRange;

    // CUDA error codes
    public static final int CUDA_SUCCESS = 0;
    public static final int CUDA_ERROR_INVALID_VALUE = 1;
    public static final int CUDA_ERROR_OUT_OF_MEMORY = 2;
    public static final int CUDA_ERROR_NOT_INITIALIZED = 3;

    static {
        try {
            // Load CUDA library
            String os = System.getProperty("os.name").toLowerCase();
            String libName;
            if (os.contains("win")) {
                libName = "nvcuda";
            } else {
                libName = "cuda";
            }

            CUDA_LIB = SymbolLookup.libraryLookup(libName, Arena.global());

            // Initialize function handles
            cuInit = downcallHandle("cuInit",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT));

            cuDeviceGetCount = downcallHandle("cuDeviceGetCount",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuDeviceGet = downcallHandle("cuDeviceGet",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

            cuDeviceGetName = downcallHandle("cuDeviceGetName",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));

            cuDeviceGetAttribute = downcallHandle("cuDeviceGetAttribute",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));

            cuDeviceTotalMem = downcallHandle("cuDeviceTotalMem_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

            cuCtxCreate = downcallHandle("cuCtxCreate_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));

            cuCtxDestroy = downcallHandle("cuCtxDestroy_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuCtxSynchronize = downcallHandle("cuCtxSynchronize",
                FunctionDescriptor.of(JAVA_INT));

            cuCtxGetCurrent = downcallHandle("cuCtxGetCurrent",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuCtxSetCurrent = downcallHandle("cuCtxSetCurrent",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuModuleLoad = downcallHandle("cuModuleLoad",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            cuModuleLoadData = downcallHandle("cuModuleLoadData",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            cuModuleLoadDataEx = downcallHandle("cuModuleLoadDataEx",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

            cuModuleUnload = downcallHandle("cuModuleUnload",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuModuleGetFunction = downcallHandle("cuModuleGetFunction",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

            cuMemAlloc = downcallHandle("cuMemAlloc_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));

            cuMemFree = downcallHandle("cuMemFree_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuMemcpyHtoD = downcallHandle("cuMemcpyHtoD_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));

            cuMemcpyDtoH = downcallHandle("cuMemcpyDtoH_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));

            cuMemcpyHtoDAsync = downcallHandle("cuMemcpyHtoDAsync_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

            cuMemcpyDtoHAsync = downcallHandle("cuMemcpyDtoHAsync_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

            cuLaunchKernel = downcallHandle("cuLaunchKernel",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
                    JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

            cuStreamCreate = downcallHandle("cuStreamCreate",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

            cuStreamCreateWithPriority = downcallHandle("cuStreamCreateWithPriority",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));

            cuStreamDestroy = downcallHandle("cuStreamDestroy_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuStreamSynchronize = downcallHandle("cuStreamSynchronize",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuEventCreate = downcallHandle("cuEventCreate",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));

            cuEventRecord = downcallHandle("cuEventRecord",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

            cuEventDestroy = downcallHandle("cuEventDestroy_v2",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuEventSynchronize = downcallHandle("cuEventSynchronize",
                FunctionDescriptor.of(JAVA_INT, ADDRESS));

            cuEventElapsedTime = downcallHandle("cuEventElapsedTime",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

            cuCtxGetStreamPriorityRange = downcallHandle("cuCtxGetStreamPriorityRange",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

        } catch (Exception e) {
            throw new ExceptionInInitializerError("Failed to initialize CUDA FFI: " + e.getMessage());
        }
    }

    private static MethodHandle downcallHandle(String name, FunctionDescriptor descriptor) {
        return CUDA_LIB.find(name)
            .map(addr -> LINKER.downcallHandle(addr, descriptor))
            .orElseThrow(() -> new UnsatisfiedLinkError("Failed to find CUDA function: " + name));
    }

    // Public API methods

    public static int init(int flags) throws Throwable {
        return (int) cuInit.invoke(flags);
    }

    public static int getDeviceCount() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment count = arena.allocate(JAVA_INT);
            int status = (int) cuDeviceGetCount.invoke(count);
            if (status != CUDA_SUCCESS) {
                return 0;
            }
            return count.get(JAVA_INT, 0);
        }
    }

    public static int getDevice(int ordinal) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment device = arena.allocate(JAVA_INT);
            int status = (int) cuDeviceGet.invoke(device, ordinal);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuDeviceGet failed with error: " + status);
            }
            return device.get(JAVA_INT, 0);
        }
    }

    public static String getDeviceName(int device) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            int nameSize = 256;
            MemorySegment name = arena.allocate(nameSize);
            int status = (int) cuDeviceGetName.invoke(name, nameSize, device);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuDeviceGetName failed with error: " + status);
            }
            return name.getUtf8String(0);
        }
    }

    public static int getDeviceAttribute(int attribute, int device) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment value = arena.allocate(JAVA_INT);
            int status = (int) cuDeviceGetAttribute.invoke(value, attribute, device);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuDeviceGetAttribute failed with error: " + status);
            }
            return value.get(JAVA_INT, 0);
        }
    }

    public static long getDeviceTotalMemory(int device) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bytes = arena.allocate(JAVA_LONG);
            int status = (int) cuDeviceTotalMem.invoke(bytes, device);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuDeviceTotalMem failed with error: " + status);
            }
            return bytes.get(JAVA_LONG, 0);
        }
    }

    public static long createContext(int flags, int device) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = arena.allocate(ADDRESS);
            int status = (int) cuCtxCreate.invoke(context, flags, device);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuCtxCreate failed with error: " + status);
            }
            return context.get(ADDRESS, 0).address();
        }
    }

    public static int destroyContext(long context) throws Throwable {
        MemorySegment ctxSegment = MemorySegment.ofAddress(context);
        return (int) cuCtxDestroy.invoke(ctxSegment);
    }

    public static int synchronizeContext() throws Throwable {
        return (int) cuCtxSynchronize.invoke();
    }

    public static long getCurrentContext() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment context = arena.allocate(ADDRESS);
            int status = (int) cuCtxGetCurrent.invoke(context);
            if (status != CUDA_SUCCESS) {
                return 0;
            }
            return context.get(ADDRESS, 0).address();
        }
    }

    public static int setCurrentContext(long context) throws Throwable {
        MemorySegment ctxSegment = context != 0 ? MemorySegment.ofAddress(context) : MemorySegment.NULL;
        return (int) cuCtxSetCurrent.invoke(ctxSegment);
    }

    public static long loadModule(String filename) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment module = arena.allocate(ADDRESS);
            MemorySegment filenameStr = arena.allocateUtf8String(filename);
            int status = (int) cuModuleLoad.invoke(module, filenameStr);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuModuleLoad failed with error: " + status);
            }
            return module.get(ADDRESS, 0).address();
        }
    }

    public static long loadModuleData(byte[] ptxSource) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment module = arena.allocate(ADDRESS);
            MemorySegment sourceSegment = arena.allocate(ptxSource.length + 1);
            sourceSegment.copyFrom(MemorySegment.ofArray(ptxSource));
            sourceSegment.set(JAVA_BYTE, ptxSource.length, (byte) 0); // null terminator
            int status = (int) cuModuleLoadData.invoke(module, sourceSegment);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuModuleLoadData failed with error: " + status);
            }
            return module.get(ADDRESS, 0).address();
        }
    }

    public static int unloadModule(long module) throws Throwable {
        MemorySegment moduleSegment = MemorySegment.ofAddress(module);
        return (int) cuModuleUnload.invoke(moduleSegment);
    }

    public static long getFunction(long module, String name) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment function = arena.allocate(ADDRESS);
            MemorySegment moduleSegment = MemorySegment.ofAddress(module);
            MemorySegment nameStr = arena.allocateUtf8String(name);
            int status = (int) cuModuleGetFunction.invoke(function, moduleSegment, nameStr);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuModuleGetFunction failed with error: " + status);
            }
            return function.get(ADDRESS, 0).address();
        }
    }

    public static long memAlloc(long size) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment devicePtr = arena.allocate(ADDRESS);
            int status = (int) cuMemAlloc.invoke(devicePtr, size);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuMemAlloc failed with error: " + status);
            }
            return devicePtr.get(ADDRESS, 0).address();
        }
    }

    public static int memFree(long devicePtr) throws Throwable {
        MemorySegment ptr = MemorySegment.ofAddress(devicePtr);
        return (int) cuMemFree.invoke(ptr);
    }

    public static int memcpyHtoD(long devicePtr, MemorySegment hostPtr, long size) throws Throwable {
        MemorySegment devPtr = MemorySegment.ofAddress(devicePtr);
        return (int) cuMemcpyHtoD.invoke(devPtr, hostPtr, size);
    }

    public static int memcpyDtoH(MemorySegment hostPtr, long devicePtr, long size) throws Throwable {
        MemorySegment devPtr = MemorySegment.ofAddress(devicePtr);
        return (int) cuMemcpyDtoH.invoke(hostPtr, devPtr, size);
    }

    public static int memcpyHtoDAsync(long devicePtr, MemorySegment hostPtr, long size, long stream) throws Throwable {
        MemorySegment devPtr = MemorySegment.ofAddress(devicePtr);
        MemorySegment streamSegment = stream != 0 ? MemorySegment.ofAddress(stream) : MemorySegment.NULL;
        return (int) cuMemcpyHtoDAsync.invoke(devPtr, hostPtr, size, streamSegment);
    }

    public static int memcpyDtoHAsync(MemorySegment hostPtr, long devicePtr, long size, long stream) throws Throwable {
        MemorySegment devPtr = MemorySegment.ofAddress(devicePtr);
        MemorySegment streamSegment = stream != 0 ? MemorySegment.ofAddress(stream) : MemorySegment.NULL;
        return (int) cuMemcpyDtoHAsync.invoke(hostPtr, devPtr, size, streamSegment);
    }

    public static int launchKernel(long function, int gridDimX, int gridDimY, int gridDimZ,
                                    int blockDimX, int blockDimY, int blockDimZ,
                                    int sharedMemBytes, long stream, MemorySegment kernelParams) throws Throwable {
        MemorySegment funcSegment = MemorySegment.ofAddress(function);
        MemorySegment streamSegment = stream != 0 ? MemorySegment.ofAddress(stream) : MemorySegment.NULL;
        return (int) cuLaunchKernel.invoke(funcSegment, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ, sharedMemBytes, streamSegment, kernelParams, MemorySegment.NULL);
    }

    public static long createStream(int flags) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment stream = arena.allocate(ADDRESS);
            int status = (int) cuStreamCreate.invoke(stream, flags);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuStreamCreate failed with error: " + status);
            }
            return stream.get(ADDRESS, 0).address();
        }
    }

    public static long createStreamWithPriority(int flags, int priority) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment stream = arena.allocate(ADDRESS);
            int status = (int) cuStreamCreateWithPriority.invoke(stream, flags, priority);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuStreamCreateWithPriority failed with error: " + status);
            }
            return stream.get(ADDRESS, 0).address();
        }
    }

    public static int destroyStream(long stream) throws Throwable {
        MemorySegment streamSegment = MemorySegment.ofAddress(stream);
        return (int) cuStreamDestroy.invoke(streamSegment);
    }

    public static int synchronizeStream(long stream) throws Throwable {
        MemorySegment streamSegment = MemorySegment.ofAddress(stream);
        return (int) cuStreamSynchronize.invoke(streamSegment);
    }

    public static long createEvent(int flags) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment event = arena.allocate(ADDRESS);
            int status = (int) cuEventCreate.invoke(event, flags);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuEventCreate failed with error: " + status);
            }
            return event.get(ADDRESS, 0).address();
        }
    }

    public static int recordEvent(long event, long stream) throws Throwable {
        MemorySegment eventSegment = MemorySegment.ofAddress(event);
        MemorySegment streamSegment = stream != 0 ? MemorySegment.ofAddress(stream) : MemorySegment.NULL;
        return (int) cuEventRecord.invoke(eventSegment, streamSegment);
    }

    public static int destroyEvent(long event) throws Throwable {
        MemorySegment eventSegment = MemorySegment.ofAddress(event);
        return (int) cuEventDestroy.invoke(eventSegment);
    }

    public static int synchronizeEvent(long event) throws Throwable {
        MemorySegment eventSegment = MemorySegment.ofAddress(event);
        return (int) cuEventSynchronize.invoke(eventSegment);
    }

    public static float getElapsedTime(long startEvent, long endEvent) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment milliseconds = arena.allocate(JAVA_FLOAT);
            MemorySegment start = MemorySegment.ofAddress(startEvent);
            MemorySegment end = MemorySegment.ofAddress(endEvent);
            int status = (int) cuEventElapsedTime.invoke(milliseconds, start, end);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuEventElapsedTime failed with error: " + status);
            }
            return milliseconds.get(JAVA_FLOAT, 0);
        }
    }

    public static int[] getStreamPriorityRange() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment leastPriority = arena.allocate(JAVA_INT);
            MemorySegment greatestPriority = arena.allocate(JAVA_INT);
            int status = (int) cuCtxGetStreamPriorityRange.invoke(leastPriority, greatestPriority);
            if (status != CUDA_SUCCESS) {
                throw new RuntimeException("cuCtxGetStreamPriorityRange failed with error: " + status);
            }
            return new int[] { leastPriority.get(JAVA_INT, 0), greatestPriority.get(JAVA_INT, 0) };
        }
    }
}
