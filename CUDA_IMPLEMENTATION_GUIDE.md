# CUDA C++ Code Generation: Detailed Implementation Guide

## Overview

This is a **step-by-step, class-by-class implementation guide** with complete code examples for migrating TornadoVM from PTX assembly generation to CUDA C++ code generation.

**Total Work**: ~2,850 lines of new code + ~100 lines of modifications
**Timeline**: 8 weeks (detailed in phases below)
**Approach**: Incremental - each phase is testable independently

---

## Phase 1: NVRTC Infrastructure (Week 1-2)

**Goal**: Set up the NVRTC compilation layer so we can compile CUDA C++ strings into GPU modules.

### Step 1.1: Create NVRTCModule.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/NVRTCModule.java`

**Lines**: ~85 lines

**Complete Code**:

```java
/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 */
package uk.ac.manchester.tornado.drivers.ptx;

/**
 * Wrapper for NVRTC (NVIDIA Runtime Compilation) compiled modules.
 * Compiles CUDA C++ source code at runtime using NVRTC API.
 */
public class NVRTCModule {
    public final byte[] moduleWrapper;      // CUmodule handle
    public final String kernelFunctionName;
    private int maxBlockSize;
    public final String javaName;
    private final String source;            // CUDA C++ source
    private String compilationLog;          // NVRTC compilation log

    /**
     * Compile CUDA C++ source to a GPU module using NVRTC.
     *
     * @param name Java method name
     * @param cudaSource CUDA C++ source code (string)
     * @param kernelFunctionName Name of the __global__ kernel function
     * @param compileOptions NVRTC compiler options (e.g., "-arch=compute_75", "-O3")
     */
    public NVRTCModule(String name, String cudaSource, String kernelFunctionName, String[] compileOptions) {
        this.source = cudaSource;
        this.kernelFunctionName = kernelFunctionName;
        this.javaName = name;
        this.maxBlockSize = -1;

        // Step 1: Compile CUDA C++ to PTX using NVRTC
        NVRTCCompilationResult result = nvrtcCompile(cudaSource, compileOptions);
        this.compilationLog = result.log;

        if (result.ptxCode == null || result.ptxCode.length == 0) {
            // Compilation failed
            System.err.println("NVRTC compilation failed for kernel: " + kernelFunctionName);
            System.err.println("Compilation log:\n" + compilationLog);
            this.moduleWrapper = new byte[0];
        } else {
            // Step 2: Load compiled PTX into CUDA module
            this.moduleWrapper = cuModuleLoadData(result.ptxCode);
        }
    }

    // Native method declarations
    private static native NVRTCCompilationResult nvrtcCompile(String source, String[] options);
    private static native byte[] cuModuleLoadData(byte[] binary);
    private static native long cuModuleUnload(byte[] module);
    private static native int cuOccupancyMaxPotentialBlockSize(byte[] module, String funcName);

    public int getPotentialBlockSizeMaxOccupancy() {
        if (maxBlockSize < 0) {
            maxBlockSize = cuOccupancyMaxPotentialBlockSize(moduleWrapper, kernelFunctionName);
        }
        return maxBlockSize;
    }

    public String getSource() {
        return source;
    }

    public String getCompilationLog() {
        return compilationLog;
    }

    public boolean isCompilationSuccess() {
        return moduleWrapper.length != 0;
    }

    public void unload() {
        cuModuleUnload(moduleWrapper);
    }

    /**
     * Result of NVRTC compilation.
     */
    public static class NVRTCCompilationResult {
        public final byte[] ptxCode;
        public final String log;

        public NVRTCCompilationResult(byte[] ptxCode, String log) {
            this.ptxCode = ptxCode;
            this.log = log;
        }
    }
}
```

**Key Points**:
- Takes CUDA C++ source as `String` (not `byte[]` like PTXModule)
- Uses NVRTC to compile to PTX, then loads PTX into module
- Stores compilation log for debugging
- Same interface as PTXModule for compatibility

---

### Step 1.2: Create NVRTCModule.cpp

**Location**: `/tornado-drivers/ptx-jni/src/main/cpp/source/NVRTCModule.cpp`

**Lines**: ~220 lines

**Complete Code**:

```cpp
/*
 * MIT License
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * The University of Manchester.
 */

#include <jni.h>
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "NVRTCModule.h"
#include "ptx_log.h"

// Helper: Convert CUmodule to jbyteArray
jbyteArray from_module(JNIEnv *env, CUmodule *module) {
    jbyteArray array = env->NewByteArray(sizeof(CUmodule));
    env->SetByteArrayRegion(array, 0, sizeof(CUmodule),
                           static_cast<const jbyte *>((void *) module));
    return array;
}

// Helper: Convert jbyteArray to CUmodule
void array_to_module(JNIEnv *env, CUmodule *module_ptr, jbyteArray javaWrapper) {
    env->GetByteArrayRegion(javaWrapper, 0, sizeof(CUmodule),
                           static_cast<jbyte *>((void *) module_ptr));
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    nvrtcCompile
 * Signature: (Ljava/lang/String;[Ljava/lang/String;)Luk/ac/manchester/tornado/drivers/ptx/NVRTCModule$NVRTCCompilationResult;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_nvrtcCompile
  (JNIEnv *env, jclass clazz, jstring source, jobjectArray options) {

    // Convert Java String to C string
    const char *cuda_source = env->GetStringUTFChars(source, nullptr);
    if (cuda_source == nullptr) {
        return nullptr; // OutOfMemoryError already thrown
    }

    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(
        &prog,              // program handle
        cuda_source,        // CUDA source code
        "kernel.cu",        // program name (for error messages)
        0,                  // number of headers
        nullptr,            // header sources
        nullptr             // header names
    );

    if (result != NVRTC_SUCCESS) {
        std::cerr << "nvrtcCreateProgram failed: " << nvrtcGetErrorString(result) << std::endl;
        env->ReleaseStringUTFChars(source, cuda_source);

        // Return empty result
        jclass resultClass = env->FindClass("uk/ac/manchester/tornado/drivers/ptx/NVRTCModule$NVRTCCompilationResult");
        jmethodID constructor = env->GetMethodID(resultClass, "<init>", "([BLjava/lang/String;)V");
        jbyteArray emptyArray = env->NewByteArray(0);
        jstring errorMsg = env->NewStringUTF("Failed to create NVRTC program");
        return env->NewObject(resultClass, constructor, emptyArray, errorMsg);
    }

    // Convert Java String[] to C char**
    jsize numOptions = env->GetArrayLength(options);
    std::vector<const char*> opts(numOptions);
    std::vector<jstring> jstrings(numOptions);

    for (jsize i = 0; i < numOptions; i++) {
        jstrings[i] = (jstring)env->GetObjectArrayElement(options, i);
        opts[i] = env->GetStringUTFChars(jstrings[i], nullptr);
    }

    // Compile the program
    result = nvrtcCompileProgram(
        prog,           // program handle
        numOptions,     // number of options
        opts.data()     // options array
    );

    // Release Java strings
    for (jsize i = 0; i < numOptions; i++) {
        env->ReleaseStringUTFChars(jstrings[i], opts[i]);
    }
    env->ReleaseStringUTFChars(source, cuda_source);

    // Get compilation log (regardless of success/failure)
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::vector<char> log(log_size);
    nvrtcGetProgramLog(prog, log.data());

    jstring logString = env->NewStringUTF(log.data());

    if (result != NVRTC_SUCCESS) {
        std::cerr << "NVRTC Compilation failed:" << std::endl;
        std::cerr << log.data() << std::endl;

        nvrtcDestroyProgram(&prog);

        // Return result with empty PTX but with log
        jclass resultClass = env->FindClass("uk/ac/manchester/tornado/drivers/ptx/NVRTCModule$NVRTCCompilationResult");
        jmethodID constructor = env->GetMethodID(resultClass, "<init>", "([BLjava/lang/String;)V");
        jbyteArray emptyArray = env->NewByteArray(0);
        return env->NewObject(resultClass, constructor, emptyArray, logString);
    }

    // Get compiled PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Convert to Java byte array
    jbyteArray ptxArray = env->NewByteArray(ptx_size);
    env->SetByteArrayRegion(ptxArray, 0, ptx_size,
                           reinterpret_cast<jbyte*>(ptx.data()));

    nvrtcDestroyProgram(&prog);

    // Create and return NVRTCCompilationResult object
    jclass resultClass = env->FindClass("uk/ac/manchester/tornado/drivers/ptx/NVRTCModule$NVRTCCompilationResult");
    jmethodID constructor = env->GetMethodID(resultClass, "<init>", "([BLjava/lang/String;)V");
    return env->NewObject(resultClass, constructor, ptxArray, logString);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleLoadData
 * Signature: ([B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleLoadData
  (JNIEnv *env, jclass clazz, jbyteArray binary) {

    size_t data_length = env->GetArrayLength(binary);
    char *data = new char[data_length + 1];
    env->GetByteArrayRegion(binary, 0, data_length, reinterpret_cast<jbyte *>(data));
    data[data_length] = 0; // Null-terminate

    CUmodule module;
    CUresult result = cuModuleLoadData(&module, data);
    delete[] data;

    LOG_PTX_AND_VALIDATE("cuModuleLoadData", result);

    if (result != CUDA_SUCCESS) {
        printf("Module load failed! (%d)\n", result);
        fflush(stdout);
        return env->NewByteArray(0);
    }

    return from_module(env, &module);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleUnload
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleUnload
  (JNIEnv *env, jclass clazz, jbyteArray module_wrapper) {
    CUmodule module;
    array_to_module(env, &module, module_wrapper);
    CUresult result = cuModuleUnload(module);
    LOG_PTX_AND_VALIDATE("cuModuleUnload", result);
    return (jlong) result;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuOccupancyMaxPotentialBlockSize
 * Signature: ([BLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuOccupancyMaxPotentialBlockSize
  (JNIEnv *env, jclass clazz, jbyteArray module_wrapper, jstring func_name) {
    CUmodule module;
    array_to_module(env, &module, module_wrapper);

    const char *native_function_name = env->GetStringUTFChars(func_name, 0);
    CUfunction kernel;
    CUresult result = cuModuleGetFunction(&kernel, module, native_function_name);
    LOG_PTX_AND_VALIDATE("cuModuleGetFunction", result);
    env->ReleaseStringUTFChars(func_name, native_function_name);

    int min_grid_size;
    int block_size;
    result = cuOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0, 0);
    LOG_PTX_AND_VALIDATE("cuOccupancyMaxPotentialBlockSize", result);
    return block_size;
}
```

**Key Points**:
- Uses NVRTC API: `nvrtcCreateProgram()`, `nvrtcCompileProgram()`, `nvrtcGetPTX()`
- Returns both PTX code and compilation log (for debugging)
- Handles errors gracefully with detailed messages
- Reuses existing `cuModuleLoadData()` to load compiled PTX

---

### Step 1.3: Create NVRTCModule.h

**Location**: `/tornado-drivers/ptx-jni/src/main/cpp/source/NVRTCModule.h`

**Lines**: ~35 lines

**Complete Code**:

```cpp
/*
 * MIT License
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * The University of Manchester.
 */

#ifndef _Included_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
#define _Included_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    nvrtcCompile
 * Signature: (Ljava/lang/String;[Ljava/lang/String;)Luk/ac/manchester/tornado/drivers/ptx/NVRTCModule$NVRTCCompilationResult;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_nvrtcCompile
  (JNIEnv *, jclass, jstring, jobjectArray);

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleLoadData
 * Signature: ([B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleLoadData
  (JNIEnv *, jclass, jbyteArray);

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleUnload
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleUnload
  (JNIEnv *, jclass, jbyteArray);

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuOccupancyMaxPotentialBlockSize
 * Signature: ([BLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuOccupancyMaxPotentialBlockSize
  (JNIEnv *, jclass, jbyteArray, jstring);

#ifdef __cplusplus
}
#endif
#endif
```

---

### Step 1.4: Update CMakeLists.txt

**Location**: `/tornado-drivers/ptx-jni/CMakeLists.txt`

**Change**: Add NVRTC library linking

**Before** (find this section):
```cmake
find_library(CUDA_LIB cuda HINTS ${CUDA_LIB_PATH})

target_link_libraries(tornado-ptx ${CUDA_LIB})
```

**After** (add NVRTC):
```cmake
find_library(CUDA_LIB cuda HINTS ${CUDA_LIB_PATH})
find_library(NVRTC_LIB nvrtc HINTS ${CUDA_LIB_PATH})

# Add new source file
target_sources(tornado-ptx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/source/PTXModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/source/NVRTCModule.cpp  # NEW
    # ... other sources
)

target_link_libraries(tornado-ptx ${CUDA_LIB} ${NVRTC_LIB})
```

**Key Points**:
- Finds NVRTC library (ships with CUDA Toolkit)
- Links both `libcuda.so` and `libnvrtc.so`
- Adds new NVRTCModule.cpp to build

---

### Step 1.5: Create NVRTCCompilerOptions.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/NVRTCCompilerOptions.java`

**Lines**: ~120 lines

**Complete Code**:

```java
/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 */
package uk.ac.manchester.tornado.drivers.ptx;

import java.util.ArrayList;
import java.util.List;

/**
 * Maps PTX JIT compiler options to NVRTC compiler options.
 */
public class NVRTCCompilerOptions {

    /**
     * Convert PTX JIT options to NVRTC command-line options.
     *
     * @param jitOptions CUjitOption values
     * @param jitValues Option values
     * @return Array of NVRTC option strings
     */
    public static String[] fromJitOptions(int[] jitOptions, long[] jitValues) {
        List<String> opts = new ArrayList<>();

        for (int i = 0; i < jitOptions.length; i++) {
            CUjitOption option = CUjitOption.fromValue(jitOptions[i]);

            switch (option) {
                case CU_JIT_OPTIMIZATION_LEVEL:
                    // PTX JIT: 0-4, NVRTC: -O0 to -O3
                    int level = (int) jitValues[i];
                    if (level > 3) level = 3; // Cap at -O3
                    opts.add("-O" + level);
                    break;

                case CU_JIT_MAX_REGISTERS:
                    // Limit register usage per thread
                    opts.add("-maxrregcount=" + jitValues[i]);
                    break;

                case CU_JIT_TARGET:
                    // Target GPU architecture
                    int computeCapability = (int) jitValues[i];
                    opts.add("-arch=compute_" + computeCapability);
                    break;

                case CU_JIT_GENERATE_DEBUG_INFO:
                    // Generate debug info
                    if (jitValues[i] != 0) {
                        opts.add("-G");
                    }
                    break;

                case CU_JIT_GENERATE_LINE_INFO:
                    // Generate line number information
                    if (jitValues[i] != 0) {
                        opts.add("-lineinfo");
                    }
                    break;

                case CU_JIT_LOG_VERBOSE:
                    // Verbose output (not directly supported, but enable warnings)
                    if (jitValues[i] != 0) {
                        opts.add("-Xptxas=-v");
                    }
                    break;

                case CU_JIT_CACHE_MODE:
                    // NVRTC doesn't have direct cache mode equivalent
                    // Ignore for now
                    break;

                default:
                    System.err.println("Warning: Unsupported JIT option for NVRTC: " + option);
                    break;
            }
        }

        // Add default includes for CUDA runtime
        opts.add("-default-device");

        return opts.toArray(new String[0]);
    }

    /**
     * Get default NVRTC options for a given compute capability.
     *
     * @param computeCapability e.g., 75 for sm_75
     * @return Default NVRTC options
     */
    public static String[] getDefaultOptions(int computeCapability) {
        return new String[] {
            "-arch=compute_" + computeCapability,
            "-O3",
            "-default-device"
        };
    }

    /**
     * Create NVRTC options from compiler flags string.
     * Example: "CU_JIT_OPTIMIZATION_LEVEL 3 CU_JIT_MAX_REGISTERS 32"
     *
     * @param compilerFlags Compiler flags string
     * @return NVRTC option strings
     */
    public static String[] fromCompilerFlagsString(String compilerFlags) {
        if (compilerFlags == null || compilerFlags.trim().isEmpty()) {
            return new String[0];
        }

        String[] parts = compilerFlags.trim().split("\\s+");
        if (parts.length % 2 != 0) {
            throw new IllegalArgumentException(
                "Malformed compilerFlags: expected pairs of <flag> <value>. Got: " + compilerFlags
            );
        }

        int[] jitOptions = new int[parts.length / 2];
        long[] jitValues = new long[parts.length / 2];

        for (int i = 0; i < parts.length; i += 2) {
            CUjitOption option = CUjitOption.valueOf(parts[i]);
            jitOptions[i / 2] = option.getValue();
            jitValues[i / 2] = Long.parseLong(parts[i + 1]);
        }

        return fromJitOptions(jitOptions, jitValues);
    }
}
```

**Key Points**:
- Maps CUjitOption enums to NVRTC string options
- Handles optimization levels, register limits, debug info
- Provides defaults for common cases

---

### Step 1.6: Update CUjitOption.java (if needed)

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/CUjitOption.java`

**Change**: Add `fromValue()` method if not present

**Add this method**:
```java
public static CUjitOption fromValue(int value) {
    for (CUjitOption option : values()) {
        if (option.getValue() == value) {
            return option;
        }
    }
    throw new IllegalArgumentException("Unknown CUjitOption value: " + value);
}
```

---

### Step 1.7: Test NVRTC Infrastructure

**Location**: Create test file `/tornado-drivers/ptx/src/test/java/uk/ac/manchester/tornado/drivers/ptx/tests/NVRTCModuleTest.java`

**Complete Test**:

```java
package uk.ac.manchester.tornado.drivers.ptx.tests;

import org.junit.Test;
import static org.junit.Assert.*;
import uk.ac.manchester.tornado.drivers.ptx.NVRTCModule;

public class NVRTCModuleTest {

    @Test
    public void testSimpleKernelCompilation() {
        String cudaSource =
            "__global__ void add(int* a, int* b, int* c, int n) {\n" +
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    if (i < n) {\n" +
            "        c[i] = a[i] + b[i];\n" +
            "    }\n" +
            "}\n";

        String[] options = {"-arch=compute_75", "-O3"};

        NVRTCModule module = new NVRTCModule("add_test", cudaSource, "add", options);

        assertTrue("Compilation should succeed", module.isCompilationSuccess());
        assertNotNull("Module wrapper should not be null", module.moduleWrapper);
        assertTrue("Module wrapper should not be empty", module.moduleWrapper.length > 0);

        System.out.println("Compilation log:");
        System.out.println(module.getCompilationLog());
    }

    @Test
    public void testCompilationError() {
        String badSource =
            "__global__ void bad_kernel() {\n" +
            "    undefined_variable++;\n" +  // Error: undefined variable
            "}\n";

        String[] options = {"-arch=compute_75"};

        NVRTCModule module = new NVRTCModule("bad", badSource, "bad_kernel", options);

        assertFalse("Compilation should fail", module.isCompilationSuccess());
        assertNotNull("Log should contain error message", module.getCompilationLog());
        assertTrue("Log should mention undefined variable",
                  module.getCompilationLog().contains("undefined"));
    }

    @Test
    public void testCompilerOptions() {
        String cudaSource = "__global__ void test() { }";

        // Test different optimization levels
        String[] optsO0 = {"-arch=compute_75", "-O0"};
        String[] optsO3 = {"-arch=compute_75", "-O3"};

        NVRTCModule moduleO0 = new NVRTCModule("test", cudaSource, "test", optsO0);
        NVRTCModule moduleO3 = new NVRTCModule("test", cudaSource, "test", optsO3);

        assertTrue(moduleO0.isCompilationSuccess());
        assertTrue(moduleO3.isCompilationSuccess());
    }
}
```

**Run Test**:
```bash
cd tornado-drivers/ptx
mvn test -Dtest=NVRTCModuleTest
```

**Expected Output**:
```
[INFO] Running uk.ac.manchester.tornado.drivers.ptx.tests.NVRTCModuleTest
Compilation log:

[INFO] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0
```

---

## ✅ Phase 1 Complete

At this point, you should be able to:
- ✅ Compile CUDA C++ strings using NVRTC
- ✅ Load compiled modules into CUDA
- ✅ Get detailed error messages for compilation failures
- ✅ Test the infrastructure independently

**Next**: Phase 2 - Code Generation Layer

---

## Phase 2: Code Generation Layer (Week 3-5)

**Goal**: Create `CUDAAssembler` and `CUDALIRStmt` to generate CUDA C++ instead of PTX assembly.

### Step 2.1: Create CodeGenMode Enum

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/CodeGenMode.java`

**Lines**: ~15 lines

**Complete Code**:

```java
package uk.ac.manchester.tornado.drivers.ptx;

/**
 * Code generation mode for PTX backend.
 */
public enum CodeGenMode {
    /** Generate PTX assembly (default, current behavior) */
    PTX,

    /** Generate CUDA C++ source code */
    CUDA;

    public static CodeGenMode fromString(String mode) {
        if (mode == null) {
            return PTX; // Default
        }
        try {
            return valueOf(mode.toUpperCase());
        } catch (IllegalArgumentException e) {
            System.err.println("Warning: Unknown code generation mode '" + mode + "', using PTX");
            return PTX;
        }
    }
}
```

---

### Step 2.2: Create CUDAAssembler.java (Part 1 - Basic Structure)

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/asm/CUDAAssembler.java`

**Lines**: ~800 total (we'll build incrementally)

**Part 1: Class structure and basics (~200 lines)**

```java
/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 */
package uk.ac.manchester.tornado.drivers.ptx.graal.asm;

import org.graalvm.compiler.asm.Assembler;
import jdk.vm.ci.code.TargetDescription;
import jdk.vm.ci.code.Register;
import jdk.vm.ci.meta.Value;
import jdk.vm.ci.meta.Constant;
import org.graalvm.compiler.lir.Variable;
import org.graalvm.compiler.lir.ConstantValue;

import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXKind;
import uk.ac.manchester.tornado.drivers.ptx.graal.compiler.PTXLIRGenerationResult;

import java.util.*;

/**
 * Assembler for generating CUDA C++ source code (instead of PTX assembly).
 *
 * This class generates readable CUDA C++ code that will be compiled using NVRTC.
 */
public class CUDAAssembler extends Assembler {

    private final PTXLIRGenerationResult lirGenRes;
    private final StringBuilder codeBuffer;
    private int indentLevel;
    private final Map<Value, String> variableNames;
    private final Set<String> declaredVariables;
    private int tempVarCounter;

    public CUDAAssembler(TargetDescription target, PTXLIRGenerationResult lirGenRes) {
        super(target, null);
        this.lirGenRes = lirGenRes;
        this.codeBuffer = new StringBuilder(4096);
        this.indentLevel = 0;
        this.variableNames = new HashMap<>();
        this.declaredVariables = new HashSet<>();
        this.tempVarCounter = 0;
    }

    // ========================================================================
    // Core Emission Methods
    // ========================================================================

    /**
     * Emit a line of CUDA C++ code with proper indentation.
     */
    public void emitLine(String code) {
        emitIndent();
        codeBuffer.append(code);
        codeBuffer.append("\n");
    }

    /**
     * Emit code without newline or indentation.
     */
    public void emit(String code) {
        codeBuffer.append(code);
    }

    /**
     * Emit indentation based on current level.
     */
    private void emitIndent() {
        for (int i = 0; i < indentLevel; i++) {
            codeBuffer.append("    "); // 4 spaces per level
        }
    }

    /**
     * Increase indentation level.
     */
    public void indent() {
        indentLevel++;
    }

    /**
     * Decrease indentation level.
     */
    public void dedent() {
        if (indentLevel > 0) {
            indentLevel--;
        }
    }

    // ========================================================================
    // Type Conversion Methods
    // ========================================================================

    /**
     * Convert PTXKind to CUDA C++ type string.
     */
    public static String toCType(PTXKind kind) {
        switch (kind) {
            case S8:  return "char";
            case U8:  return "unsigned char";
            case S16: return "short";
            case U16: return "unsigned short";
            case S32: return "int";
            case U32: return "unsigned int";
            case S64: return "long long";
            case U64: return "unsigned long long";
            case F32: return "float";
            case F64: return "double";
            case PRED: return "bool";

            // Vector types
            case INT2:   return "int2";
            case INT3:   return "int3";
            case INT4:   return "int4";
            case UINT2:  return "uint2";
            case UINT3:  return "uint3";
            case UINT4:  return "uint4";
            case FLOAT2: return "float2";
            case FLOAT3: return "float3";
            case FLOAT4: return "float4";
            case DOUBLE2: return "double2";

            default:
                throw new RuntimeException("Unsupported PTXKind for CUDA: " + kind);
        }
    }

    // ========================================================================
    // Variable Management
    // ========================================================================

    /**
     * Get or create a variable name for a Value.
     */
    public String getVariableName(Value value) {
        if (variableNames.containsKey(value)) {
            return variableNames.get(value);
        }

        String name;
        if (value instanceof Variable) {
            Variable var = (Variable) value;
            // Use Variable's index to create unique name
            name = "v" + var.index;
        } else if (value instanceof Register) {
            Register reg = (Register) value;
            name = "r" + reg.number;
        } else if (value instanceof ConstantValue) {
            // Constants don't need names, return literal
            return formatConstant((ConstantValue) value);
        } else {
            // Generate temp name
            name = "tmp" + (tempVarCounter++);
        }

        variableNames.put(value, name);
        return name;
    }

    /**
     * Declare a variable if not already declared.
     */
    public void declareVariable(Value value, PTXKind kind) {
        String name = getVariableName(value);

        if (!declaredVariables.contains(name)) {
            String cType = toCType(kind);
            emitLine(cType + " " + name + ";");
            declaredVariables.add(name);
        }
    }

    /**
     * Format a constant value.
     */
    public static String formatConstant(ConstantValue cv) {
        Constant constant = cv.getConstant();
        if (constant instanceof JavaConstant) {
            JavaConstant jc = (JavaConstant) constant;
            switch (jc.getJavaKind()) {
                case Int:
                    return String.valueOf(jc.asInt());
                case Long:
                    return String.valueOf(jc.asLong()) + "LL";
                case Float:
                    float f = jc.asFloat();
                    if (Float.isNaN(f)) return "NAN";
                    if (Float.isInfinite(f)) return f > 0 ? "INFINITY" : "-INFINITY";
                    return String.format("%ff", f);
                case Double:
                    double d = jc.asDouble();
                    if (Double.isNaN(d)) return "NAN";
                    if (Double.isInfinite(d)) return d > 0 ? "INFINITY" : "-INFINITY";
                    return String.format("%f", d);
                case Boolean:
                    return jc.asBoolean() ? "true" : "false";
                default:
                    return constant.toValueString();
            }
        }
        return constant.toValueString();
    }

    // ========================================================================
    // Kernel Structure Methods
    // ========================================================================

    /**
     * Emit kernel function declaration.
     */
    public void emitKernelStart(String kernelName, List<Parameter> parameters) {
        emit("__global__ void ");
        emit(kernelName);
        emit("(");

        for (int i = 0; i < parameters.size(); i++) {
            Parameter param = parameters.get(i);
            if (i > 0) emit(", ");
            emit(param.type + " " + param.name);
        }

        emitLine(") {");
        indent();
    }

    /**
     * Emit kernel function end.
     */
    public void emitKernelEnd() {
        dedent();
        emitLine("}");
    }

    /**
     * Emit thread index initialization boilerplate.
     */
    public void emitThreadIndexInit() {
        emitLine("// Thread index calculations");
        emitLine("int blockSize = blockDim.x;");
        emitLine("int gridSize = gridDim.x;");
        emitLine("int tid = threadIdx.x;");
        emitLine("int blockId = blockIdx.x;");
        emitLine("int globalId = blockIdx.x * blockDim.x + threadIdx.x;");
        emitLine("int totalThreads = gridDim.x * blockDim.x;");
        emitLine("");
    }

    /**
     * Simple parameter holder.
     */
    public static class Parameter {
        public final String type;
        public final String name;

        public Parameter(String type, String name) {
            this.type = type;
            this.name = name;
        }
    }

    // ========================================================================
    // Get Generated Code
    // ========================================================================

    /**
     * Get the generated CUDA C++ code as byte array.
     */
    public byte[] getCode() {
        return codeBuffer.toString().getBytes();
    }

    /**
     * Get the generated CUDA C++ code as string.
     */
    public String getCodeString() {
        return codeBuffer.toString();
    }

    @Override
    public void reset() {
        super.reset();
        codeBuffer.setLength(0);
        indentLevel = 0;
        variableNames.clear();
        declaredVariables.clear();
        tempVarCounter = 0;
    }
}
```

**Key Points**:
- Extends `Assembler` (like PTXAssembler)
- Uses `StringBuilder` to accumulate C++ code
- Manages indentation for readable output
- Tracks variable declarations to avoid duplicates
- Maps PTXKind to C++ types

---

### Step 2.3: Create CUDAAssembler.java (Part 2 - Statement Emission)

**Add these methods to CUDAAssembler.java** (~200 more lines):

```java
    // ========================================================================
    // Statement Emission Methods
    // ========================================================================

    /**
     * Emit a variable assignment.
     * Example: v5 = v3 + v4;
     */
    public void emitAssignment(Value dest, String expression) {
        String destName = getVariableName(dest);
        emitLine(destName + " = " + expression + ";");
    }

    /**
     * Emit a binary operation.
     * Example: v2 = v0 + v1;
     */
    public void emitBinaryOp(Value dest, Value left, Value right, String operator) {
        String destName = getVariableName(dest);
        String leftName = getVariableName(left);
        String rightName = getVariableName(right);
        emitLine(destName + " = " + leftName + " " + operator + " " + rightName + ";");
    }

    /**
     * Emit a unary operation.
     * Example: v1 = -v0;
     */
    public void emitUnaryOp(Value dest, Value operand, String operator) {
        String destName = getVariableName(dest);
        String operandName = getVariableName(operand);
        emitLine(destName + " = " + operator + operandName + ";");
    }

    /**
     * Emit array load.
     * Example: v3 = array[v2];
     */
    public void emitArrayLoad(Value dest, Value base, Value index, PTXKind kind) {
        String destName = getVariableName(dest);
        String baseName = getVariableName(base);
        String indexName = getVariableName(index);
        String cType = toCType(kind);

        emitLine(destName + " = ((" + cType + "*)" + baseName + ")[" + indexName + "];");
    }

    /**
     * Emit array store.
     * Example: array[v2] = v3;
     */
    public void emitArrayStore(Value base, Value index, Value value, PTXKind kind) {
        String baseName = getVariableName(base);
        String indexName = getVariableName(index);
        String valueName = getVariableName(value);
        String cType = toCType(kind);

        emitLine("((" + cType + "*)" + baseName + ")[" + indexName + "] = " + valueName + ";");
    }

    /**
     * Emit pointer dereference load.
     * Example: v1 = *((int*)v0);
     */
    public void emitPointerLoad(Value dest, Value ptr, PTXKind kind) {
        String destName = getVariableName(dest);
        String ptrName = getVariableName(ptr);
        String cType = toCType(kind);

        emitLine(destName + " = *((" + cType + "*)" + ptrName + ");");
    }

    /**
     * Emit pointer dereference store.
     * Example: *((int*)v0) = v1;
     */
    public void emitPointerStore(Value ptr, Value value, PTXKind kind) {
        String ptrName = getVariableName(ptr);
        String valueName = getVariableName(value);
        String cType = toCType(kind);

        emitLine("*((" + cType + "*)" + ptrName + ") = " + valueName + ";");
    }

    /**
     * Emit type conversion (cast).
     * Example: v1 = (float)v0;
     */
    public void emitConvert(Value dest, Value source, PTXKind destKind, PTXKind sourceKind) {
        String destName = getVariableName(dest);
        String sourceName = getVariableName(source);
        String destType = toCType(destKind);

        emitLine(destName + " = (" + destType + ")" + sourceName + ";");
    }

    // ========================================================================
    // Control Flow Methods
    // ========================================================================

    /**
     * Emit label.
     * Example: LABEL_5:
     */
    public void emitLabel(String label) {
        dedent(); // Labels are outdented
        emitLine(label + ":");
        indent();
    }

    /**
     * Emit unconditional branch (goto).
     * Example: goto LABEL_5;
     */
    public void emitGoto(String label) {
        emitLine("goto " + label + ";");
    }

    /**
     * Emit conditional branch.
     * Example: if (v0) goto LABEL_5;
     */
    public void emitConditionalGoto(Value condition, String label) {
        String condName = getVariableName(condition);
        emitLine("if (" + condName + ") goto " + label + ";");
    }

    /**
     * Emit negated conditional branch.
     * Example: if (!v0) goto LABEL_5;
     */
    public void emitNegatedConditionalGoto(Value condition, String label) {
        String condName = getVariableName(condition);
        emitLine("if (!" + condName + ") goto " + label + ";");
    }

    /**
     * Emit if statement start.
     */
    public void emitIfStart(Value condition) {
        String condName = getVariableName(condition);
        emitLine("if (" + condName + ") {");
        indent();
    }

    /**
     * Emit else clause.
     */
    public void emitElse() {
        dedent();
        emitLine("} else {");
        indent();
    }

    /**
     * Emit if/else end.
     */
    public void emitIfEnd() {
        dedent();
        emitLine("}");
    }

    /**
     * Emit while loop start.
     */
    public void emitWhileStart(Value condition) {
        String condName = getVariableName(condition);
        emitLine("while (" + condName + ") {");
        indent();
    }

    /**
     * Emit while loop end.
     */
    public void emitWhileEnd() {
        dedent();
        emitLine("}");
    }

    /**
     * Emit for loop start.
     */
    public void emitForLoop(String init, String condition, String increment) {
        emitLine("for (" + init + "; " + condition + "; " + increment + ") {");
        indent();
    }

    /**
     * Emit for loop end.
     */
    public void emitForEnd() {
        dedent();
        emitLine("}");
    }

    /**
     * Emit return statement.
     */
    public void emitReturn() {
        emitLine("return;");
    }

    /**
     * Emit break statement.
     */
    public void emitBreak() {
        emitLine("break;");
    }

    /**
     * Emit continue statement.
     */
    public void emitContinue() {
        emitLine("continue;");
    }

    // ========================================================================
    // CUDA-Specific Methods
    // ========================================================================

    /**
     * Emit __syncthreads() barrier.
     */
    public void emitSyncThreads() {
        emitLine("__syncthreads();");
    }

    /**
     * Emit shared memory declaration.
     * Example: __shared__ int sharedMem[256];
     */
    public void emitSharedMemoryDeclaration(String name, PTXKind kind, int size) {
        String cType = toCType(kind);
        emitLine("__shared__ " + cType + " " + name + "[" + size + "];");
    }

    /**
     * Emit atomic operation.
     * Example: atomicAdd(&array[i], value);
     */
    public void emitAtomicOp(String operation, Value address, Value value) {
        String addrName = getVariableName(address);
        String valueName = getVariableName(value);
        emitLine(operation + "(" + addrName + ", " + valueName + ");");
    }

    /**
     * Emit math function call.
     * Example: v1 = sqrtf(v0);
     */
    public void emitMathFunction(Value dest, String function, Value... args) {
        String destName = getVariableName(dest);
        StringBuilder call = new StringBuilder(function + "(");
        for (int i = 0; i < args.length; i++) {
            if (i > 0) call.append(", ");
            call.append(getVariableName(args[i]));
        }
        call.append(")");

        emitLine(destName + " = " + call.toString() + ";");
    }

    /**
     * Emit comment.
     */
    public void emitComment(String comment) {
        emitLine("// " + comment);
    }

    /**
     * Emit blank line.
     */
    public void emitBlankLine() {
        codeBuffer.append("\n");
    }
}
```

**Key Methods Added**:
- Assignment and arithmetic operations
- Array/pointer loads and stores
- Control flow (if, while, for, goto, labels)
- CUDA-specific constructs (__syncthreads, __shared__, atomics)
- Math functions and type conversions

---

### Step 2.4: Begin CUDALIRStmt.java

This is a large file (~1500 lines). I'll show the structure and key statement types. The full file mirrors PTXLIRStmt but with CUDA emission.

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/lir/CUDALIRStmt.java`

**Structure** (you'll expand this with all statement types):

```java
/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 */
package uk.ac.manchester.tornado.drivers.ptx.graal.lir;

import org.graalvm.compiler.lir.LIRInstruction;
import org.graalvm.compiler.lir.asm.CompilationResultBuilder;
import jdk.vm.ci.meta.Value;
import jdk.vm.ci.meta.AllocatableValue;
import org.graalvm.compiler.lir.Variable;

import uk.ac.manchester.tornado.drivers.ptx.graal.asm.CUDAAssembler;
import uk.ac.manchester.tornado.drivers.ptx.graal.compiler.PTXCompilationResultBuilder;

/**
 * CUDA C++ LIR statements.
 *
 * This file contains all LIR instruction types for CUDA C++ code generation.
 * Each statement knows how to emit itself as CUDA C++ code.
 */
public class CUDALIRStmt {

    // ========================================================================
    // Base Classes
    // ========================================================================

    /**
     * Abstract base for all CUDA LIR instructions.
     */
    public abstract static class AbstractInstruction extends LIRInstruction {
        public AbstractInstruction(LIRInstructionClass<? extends LIRInstruction> c) {
            super(c);
        }

        /**
         * Emit this instruction as CUDA C++ code.
         */
        public abstract void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm);

        @Override
        public void emitCode(CompilationResultBuilder crb) {
            emitCode((PTXCompilationResultBuilder) crb, (CUDAAssembler) crb.asm);
        }
    }

    // ========================================================================
    // Assignment Statements
    // ========================================================================

    /**
     * Simple assignment: dest = source
     */
    public static class AssignStmt extends AbstractInstruction {
        @Def protected AllocatableValue dest;
        @Use protected Value source;
        private final PTXKind kind;

        public AssignStmt(AllocatableValue dest, Value source, PTXKind kind) {
            super(TYPE);
            this.dest = dest;
            this.source = source;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            // Ensure variable is declared
            asm.declareVariable(dest, kind);

            // Emit assignment
            String destName = asm.getVariableName(dest);
            String sourceName = asm.getVariableName(source);
            asm.emitLine(destName + " = " + sourceName + ";");
        }

        public static final LIRInstructionClass<AssignStmt> TYPE =
            LIRInstructionClass.create(AssignStmt.class);
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    /**
     * Binary arithmetic: dest = left OP right
     * Operations: +, -, *, /, %, &, |, ^, <<, >>
     */
    public static class BinaryExprStmt extends AbstractInstruction {
        @Def protected AllocatableValue dest;
        @Use protected Value left;
        @Use protected Value right;
        private final PTXKind kind;
        private final BinaryOp operation;

        public enum BinaryOp {
            ADD("+"), SUB("-"), MUL("*"), DIV("/"), REM("%"),
            AND("&"), OR("|"), XOR("^"),
            SHL("<<"), SHR(">>"), USHR(">>"),
            // Comparisons
            EQ("=="), NE("!="), LT("<"), LE("<="), GT(">"), GE(">=");

            private final String symbol;
            BinaryOp(String symbol) { this.symbol = symbol; }
            public String getSymbol() { return symbol; }
        }

        public BinaryExprStmt(AllocatableValue dest, Value left, Value right,
                             BinaryOp operation, PTXKind kind) {
            super(TYPE);
            this.dest = dest;
            this.left = left;
            this.right = right;
            this.operation = operation;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.declareVariable(dest, kind);
            asm.emitBinaryOp(dest, left, right, operation.getSymbol());
        }

        public static final LIRInstructionClass<BinaryExprStmt> TYPE =
            LIRInstructionClass.create(BinaryExprStmt.class);
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /**
     * Load from memory: dest = *ptr
     */
    public static class LoadStmt extends AbstractInstruction {
        @Def protected AllocatableValue dest;
        @Use protected Value address;
        private final PTXKind kind;

        public LoadStmt(AllocatableValue dest, Value address, PTXKind kind) {
            super(TYPE);
            this.dest = dest;
            this.address = address;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.declareVariable(dest, kind);
            asm.emitPointerLoad(dest, address, kind);
        }

        public static final LIRInstructionClass<LoadStmt> TYPE =
            LIRInstructionClass.create(LoadStmt.class);
    }

    /**
     * Store to memory: *ptr = value
     */
    public static class StoreStmt extends AbstractInstruction {
        @Use protected Value address;
        @Use protected Value value;
        private final PTXKind kind;

        public StoreStmt(Value address, Value value, PTXKind kind) {
            super(TYPE);
            this.address = address;
            this.value = value;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitPointerStore(address, value, kind);
        }

        public static final LIRInstructionClass<StoreStmt> TYPE =
            LIRInstructionClass.create(StoreStmt.class);
    }

    /**
     * Array load: dest = array[index]
     */
    public static class ArrayLoadStmt extends AbstractInstruction {
        @Def protected AllocatableValue dest;
        @Use protected Value base;
        @Use protected Value index;
        private final PTXKind kind;

        public ArrayLoadStmt(AllocatableValue dest, Value base, Value index, PTXKind kind) {
            super(TYPE);
            this.dest = dest;
            this.base = base;
            this.index = index;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.declareVariable(dest, kind);
            asm.emitArrayLoad(dest, base, index, kind);
        }

        public static final LIRInstructionClass<ArrayLoadStmt> TYPE =
            LIRInstructionClass.create(ArrayLoadStmt.class);
    }

    /**
     * Array store: array[index] = value
     */
    public static class ArrayStoreStmt extends AbstractInstruction {
        @Use protected Value base;
        @Use protected Value index;
        @Use protected Value value;
        private final PTXKind kind;

        public ArrayStoreStmt(Value base, Value index, Value value, PTXKind kind) {
            super(TYPE);
            this.base = base;
            this.index = index;
            this.value = value;
            this.kind = kind;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitArrayStore(base, index, value, kind);
        }

        public static final LIRInstructionClass<ArrayStoreStmt> TYPE =
            LIRInstructionClass.create(ArrayStoreStmt.class);
    }

    // ========================================================================
    // Control Flow Statements
    // ========================================================================

    /**
     * Label definition.
     */
    public static class LabelStmt extends AbstractInstruction {
        private final String label;

        public LabelStmt(String label) {
            super(TYPE);
            this.label = label;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitLabel(label);
        }

        public static final LIRInstructionClass<LabelStmt> TYPE =
            LIRInstructionClass.create(LabelStmt.class);
    }

    /**
     * Unconditional branch: goto label
     */
    public static class GotoStmt extends AbstractInstruction {
        private final String targetLabel;

        public GotoStmt(String targetLabel) {
            super(TYPE);
            this.targetLabel = targetLabel;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitGoto(targetLabel);
        }

        public static final LIRInstructionClass<GotoStmt> TYPE =
            LIRInstructionClass.create(GotoStmt.class);
    }

    /**
     * Conditional branch: if (condition) goto label
     */
    public static class ConditionalBranchStmt extends AbstractInstruction {
        @Use protected Value condition;
        private final String targetLabel;
        private final boolean negated;

        public ConditionalBranchStmt(Value condition, String targetLabel, boolean negated) {
            super(TYPE);
            this.condition = condition;
            this.targetLabel = targetLabel;
            this.negated = negated;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            if (negated) {
                asm.emitNegatedConditionalGoto(condition, targetLabel);
            } else {
                asm.emitConditionalGoto(condition, targetLabel);
            }
        }

        public static final LIRInstructionClass<ConditionalBranchStmt> TYPE =
            LIRInstructionClass.create(ConditionalBranchStmt.class);
    }

    /**
     * Return from kernel.
     */
    public static class ReturnStmt extends AbstractInstruction {
        public ReturnStmt() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitReturn();
        }

        public static final LIRInstructionClass<ReturnStmt> TYPE =
            LIRInstructionClass.create(ReturnStmt.class);
    }

    // ========================================================================
    // CUDA-Specific Statements
    // ========================================================================

    /**
     * Barrier synchronization: __syncthreads()
     */
    public static class SyncThreadsStmt extends AbstractInstruction {
        public SyncThreadsStmt() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitSyncThreads();
        }

        public static final LIRInstructionClass<SyncThreadsStmt> TYPE =
            LIRInstructionClass.create(SyncThreadsStmt.class);
    }

    /**
     * Atomic operation.
     */
    public static class AtomicStmt extends AbstractInstruction {
        @Use protected Value address;
        @Use protected Value value;
        private final String operation; // "atomicAdd", "atomicCAS", etc.

        public AtomicStmt(Value address, Value value, String operation) {
            super(TYPE);
            this.address = address;
            this.value = value;
            this.operation = operation;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.emitAtomicOp(operation, address, value);
        }

        public static final LIRInstructionClass<AtomicStmt> TYPE =
            LIRInstructionClass.create(AtomicStmt.class);
    }

    // ========================================================================
    // Math Operations
    // ========================================================================

    /**
     * Math function call.
     */
    public static class MathFunctionStmt extends AbstractInstruction {
        @Def protected AllocatableValue dest;
        @Use protected Value[] args;
        private final String function;
        private final PTXKind kind;

        public MathFunctionStmt(AllocatableValue dest, String function,
                               PTXKind kind, Value... args) {
            super(TYPE);
            this.dest = dest;
            this.function = function;
            this.kind = kind;
            this.args = args;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
            asm.declareVariable(dest, kind);
            asm.emitMathFunction(dest, function, args);
        }

        public static final LIRInstructionClass<MathFunctionStmt> TYPE =
            LIRInstructionClass.create(MathFunctionStmt.class);
    }

    // TODO: Add remaining statement types
    // - ConvertStmt (type conversions)
    // - SelectStmt (ternary operator)
    // - VectorLoadStmt, VectorStoreStmt
    // - SharedMemoryDeclStmt
    // - More as needed
}
```

**Note**: This is a simplified version showing the pattern. You'll need to add all statement types from PTXLIRStmt.java, adapting each `emitCode()` method to generate CUDA C++ instead of PTX assembly.

---

Due to length constraints, I'll continue with the remaining phases in the next section. Would you like me to:
1. Continue with the rest of Phase 2 and remaining phases?
2. Focus on a specific part in more detail?
3. Show the integration steps (PTXCodeCache, PTXBackend modifications)?

Let me know and I'll continue with the detailed class-by-class guide!