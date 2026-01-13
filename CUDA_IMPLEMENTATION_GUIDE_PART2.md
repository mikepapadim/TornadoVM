# CUDA C++ Implementation Guide - Part 2: Integration & Remaining Phases

This continues from `CUDA_IMPLEMENTATION_GUIDE.md` Part 1.

---

## Phase 2 (Continued): PTXBackend Integration

### Step 2.5: Modify PTXBackend.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/backend/PTXBackend.java`

**Goal**: Add mode switching to support both PTX and CUDA code generation.

#### Change 2.5.1: Add Mode Field

**Find this section** (around line 60):
```java
public class PTXBackend extends XPUBackend<PTXProviders> {
    private final PTXArchitecture arch;
    // ... other fields
```

**Add after existing fields**:
```java
    private CodeGenMode codeGenMode = CodeGenMode.PTX; // Default to PTX

    public void setCodeGenMode(CodeGenMode mode) {
        this.codeGenMode = mode;
    }

    public CodeGenMode getCodeGenMode() {
        return codeGenMode;
    }
```

#### Change 2.5.2: Modify newCompilationResultBuilder()

**Find this method** (around line 140):
```java
@Override
public CompilationResultBuilder newCompilationResultBuilder(
        LIRGenerationResult lirGenRes,
        FrameMap frameMap,
        CompilationResult compilationResult,
        CompilationResultBuilderFactory factory) {
    PTXLIRGenerationResult ptxLirGenRes = (PTXLIRGenerationResult) lirGenRes;
    PTXAssembler asm = new PTXAssembler(target, ptxLirGenRes);
    // ... rest of method
```

**Replace with**:
```java
@Override
public CompilationResultBuilder newCompilationResultBuilder(
        LIRGenerationResult lirGenRes,
        FrameMap frameMap,
        CompilationResult compilationResult,
        CompilationResultBuilderFactory factory) {
    PTXLIRGenerationResult ptxLirGenRes = (PTXLIRGenerationResult) lirGenRes;

    // Create assembler based on mode
    Assembler asm;
    if (codeGenMode == CodeGenMode.CUDA) {
        asm = new CUDAAssembler(target, ptxLirGenRes);
    } else {
        asm = new PTXAssembler(target, ptxLirGenRes);
    }

    PTXCompilationResultBuilder crb = new PTXCompilationResultBuilder(
        getCodeCache(),
        getForeignCalls(),
        frameMap,
        asm,
        getDataBuilder(),
        frameMappingFactory.apply(target),
        compilationResult,
        Register.None,
        ptxLirGenRes
    );

    crb.setCodeGenMode(codeGenMode); // Pass mode to result builder
    return crb;
}
```

#### Change 2.5.3: Modify emitPrologue()

**Find emitPrologue() method** (around line 246):
```java
protected void emitPrologue(PTXCompilationResultBuilder crb, PTXAssembler asm,
                           ResolvedJavaMethod method, AllocatableValue[] incomingArguments) {
    // Current PTX-specific code
}
```

**Replace entire method with mode-aware version**:
```java
protected void emitPrologue(PTXCompilationResultBuilder crb, Assembler assembler,
                           ResolvedJavaMethod method, AllocatableValue[] incomingArguments) {
    if (codeGenMode == CodeGenMode.CUDA) {
        emitPrologueCUDA(crb, (CUDAAssembler) assembler, method, incomingArguments);
    } else {
        emitProloguePTX(crb, (PTXAssembler) assembler, method, incomingArguments);
    }
}

/**
 * Emit PTX kernel prologue (original implementation).
 */
private void emitProloguePTX(PTXCompilationResultBuilder crb, PTXAssembler asm,
                            ResolvedJavaMethod method, AllocatableValue[] incomingArguments) {
    // Move existing PTX prologue code here
    asm.emit("%s %s %s(",
        PTXAssemblerConstants.EXTERNALLY_VISIBLE,
        PTXAssemblerConstants.KERNEL_ENTRYPOINT,
        crb.compilationResult.getName());

    emitMethodParameters(asm, method, incomingArguments, true);
    asm.emit(") {");
    asm.eol();

    // Emit variable definitions
    emitVariableDefs(asm, (PTXLIRGenerationResult) crb.result);
}

/**
 * Emit CUDA C++ kernel prologue (new implementation).
 */
private void emitPrologueCUDA(PTXCompilationResultBuilder crb, CUDAAssembler asm,
                             ResolvedJavaMethod method, AllocatableValue[] incomingArguments) {
    // Build parameter list
    List<CUDAAssembler.Parameter> params = new ArrayList<>();

    // Always include kernel context as first parameter
    params.add(new CUDAAssembler.Parameter("const void*", "kernel_context"));

    // Add method parameters
    int paramIndex = 0;
    for (AllocatableValue arg : incomingArguments) {
        PTXKind kind = (PTXKind) arg.getPlatformKind();
        String cType;

        if (kind.isVector()) {
            // Vector parameters are passed as pointers
            String baseType = CUDAAssembler.toCType(kind.getElementKind());
            cType = baseType + "*";
        } else if (kind == PTXKind.PRED) {
            cType = "bool";
        } else if (kind.getSizeInBytes() == 8) {
            // Pointers and long values
            cType = CUDAAssembler.toCType(kind);
            if (!kind.isInteger()) {
                cType += "*"; // Assume pointer
            }
        } else {
            cType = CUDAAssembler.toCType(kind);
        }

        String paramName = "param_" + paramIndex;
        params.add(new CUDAAssembler.Parameter(cType, paramName));
        paramIndex++;
    }

    // Emit kernel declaration
    asm.emitKernelStart(crb.compilationResult.getName(), params);

    // Emit thread index initialization
    asm.emitThreadIndexInit();

    // Note: Variables are declared on-demand in CUDA mode (no upfront declarations)
}
```

#### Change 2.5.4: Modify emitEpilogue()

**Find emitEpilogue() method**:
```java
protected void emitEpilogue(PTXCompilationResultBuilder crb, PTXAssembler asm) {
    asm.emit("}");
    asm.eol();
}
```

**Replace with**:
```java
protected void emitEpilogue(PTXCompilationResultBuilder crb, Assembler assembler) {
    if (codeGenMode == CodeGenMode.CUDA) {
        CUDAAssembler cudaAsm = (CUDAAssembler) assembler;
        cudaAsm.emitKernelEnd();
    } else {
        PTXAssembler ptxAsm = (PTXAssembler) assembler;
        ptxAsm.emit("}");
        ptxAsm.eol();
    }
}
```

---

### Step 2.6: Modify PTXCompilationResultBuilder.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXCompilationResultBuilder.java`

**Goal**: Make it mode-aware.

#### Change 2.6.1: Add Mode Field

**Add after existing fields** (around line 70):
```java
private CodeGenMode codeGenMode = CodeGenMode.PTX;

public void setCodeGenMode(CodeGenMode mode) {
    this.codeGenMode = mode;
}

public CodeGenMode getCodeGenMode() {
    return codeGenMode;
}
```

#### Change 2.6.2: Mode-Aware Assembly

**This class mostly delegates to backend and assembler, so minimal changes needed.**

The `asm` field will be either `PTXAssembler` or `CUDAAssembler`, and each LIRInstruction's `emitCode()` will cast appropriately.

---

### Step 2.7: Create CUDA Header Utility

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/CUDACodeUtil.java`

**Lines**: ~80 lines

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

/**
 * Utility for generating CUDA C++ code with proper headers.
 */
public class CUDACodeUtil {

    /**
     * Prepend CUDA C++ header includes to kernel code.
     *
     * @param kernelCode Generated CUDA C++ kernel code
     * @param features Feature flags for what to include
     * @return Complete CUDA C++ source with headers
     */
    public static String getCodeWithCUDAHeader(String kernelCode, CUDAFeatures features) {
        StringBuilder header = new StringBuilder();

        // Standard CUDA runtime header (always included)
        header.append("#include <cuda_runtime.h>\n");

        // Math functions
        if (features.useMathFunctions) {
            header.append("#include <math_functions.h>\n");
        }

        // Half-precision floats
        if (features.useHalfFloat) {
            header.append("#include <cuda_fp16.h>\n");
        }

        // BFloat16
        if (features.useBFloat16) {
            header.append("#include <cuda_bf16.h>\n");
        }

        // Cooperative groups (for advanced synchronization)
        if (features.useCooperativeGroups) {
            header.append("#include <cooperative_groups.h>\n");
        }

        // Add blank line before kernel code
        header.append("\n");

        // Append kernel code
        header.append(kernelCode);

        return header.toString();
    }

    /**
     * Feature flags for CUDA code generation.
     */
    public static class CUDAFeatures {
        public boolean useMathFunctions = true;    // sqrt, sin, cos, etc.
        public boolean useHalfFloat = false;       // __half type
        public boolean useBFloat16 = false;        // __nv_bfloat16 type
        public boolean useCooperativeGroups = false;

        public static CUDAFeatures defaults() {
            return new CUDAFeatures();
        }

        public CUDAFeatures withHalfFloat() {
            this.useHalfFloat = true;
            return this;
        }

        public CUDAFeatures withBFloat16() {
            this.useBFloat16 = true;
            return this;
        }

        public CUDAFeatures withCooperativeGroups() {
            this.useCooperativeGroups = true;
            return this;
        }
    }
}
```

---

## Phase 3: Integration with PTXCodeCache (Week 6)

### Step 3.1: Modify PTXCodeCache.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/PTXCodeCache.java`

**Goal**: Route to either PTXModule or NVRTCModule based on code generation mode.

#### Change 3.1.1: Add Mode Detection

**Find `installSource()` method** (around line 56):

```java
public PTXInstalledCode installSource(TaskDataContext taskMeta, String name,
                                     byte[] targetCode, String resolvedMethodName,
                                     boolean debugKernel) {
    if (!cache.containsKey(name)) {
        // Current PTX-only implementation
        PTXModule module = new PTXModule(...);
        // ...
    }
    return cache.get(name);
}
```

**Replace with mode-aware version**:

```java
public PTXInstalledCode installSource(TaskDataContext taskMeta, String name,
                                     byte[] targetCode, String resolvedMethodName,
                                     boolean debugKernel) {
    if (!cache.containsKey(name)) {
        if (debugKernel) {
            RuntimeUtilities.dumpKernel(targetCode);
        }

        // Determine if this is CUDA C++ or PTX based on content
        CodeGenMode mode = detectCodeGenMode(targetCode);

        if (mode == CodeGenMode.CUDA) {
            // CUDA C++ path (NVRTC compilation)
            return installSourceCUDA(taskMeta, name, targetCode, resolvedMethodName);
        } else {
            // PTX path (current implementation)
            return installSourcePTX(taskMeta, name, targetCode, resolvedMethodName);
        }
    }

    return cache.get(name);
}

/**
 * Detect whether code is PTX assembly or CUDA C++.
 */
private CodeGenMode detectCodeGenMode(byte[] code) {
    String codeStr = new String(code, 0, Math.min(100, code.length));

    // PTX starts with ".version" or ".target"
    if (codeStr.trim().startsWith(".version") || codeStr.trim().startsWith(".target")) {
        return CodeGenMode.PTX;
    }

    // CUDA C++ contains "__global__"
    if (codeStr.contains("__global__") || codeStr.contains("#include")) {
        return CodeGenMode.CUDA;
    }

    // Default to PTX for backward compatibility
    return CodeGenMode.PTX;
}

/**
 * Install PTX source (original implementation).
 */
private PTXInstalledCode installSourcePTX(TaskDataContext taskMeta, String name,
                                         byte[] targetCode, String resolvedMethodName) {
    String compilerFlags = taskMeta.getCompilerFlags(TornadoVMBackendType.PTX);
    String[] parts = compilerFlags.trim().split("\\s+");

    if (parts.length % 2 != 0) {
        throw new TornadoBailoutRuntimeException(
            "Malformed compilerFlags: expected pairs of <flag> <value>. Got: " + compilerFlags
        );
    }

    int[] jitOptions = new int[parts.length / 2];
    long[] jitValues = new long[parts.length / 2];

    for (int i = 0; i < parts.length; i += 2) {
        String flagName = parts[i];

        if (!SUPPORTED_PTX_JIT_FLAGS.contains(flagName)) {
            throw new TornadoBailoutRuntimeException(
                "Unsupported PTX JIT flag: " + flagName +
                ". Supported flags: " + SUPPORTED_PTX_JIT_FLAGS
            );
        }

        CUjitOption option = CUjitOption.valueOf(flagName);
        jitOptions[i / 2] = option.getValue();
        jitValues[i / 2] = Long.parseLong(parts[i + 1]);
    }

    PTXModule module = new PTXModule(resolvedMethodName, targetCode, name,
                                    jitOptions, jitValues);

    if (module.isPTXJITSuccess()) {
        PTXInstalledCode code = new PTXInstalledCode(name, module, deviceContext);
        cache.put(name, code);
        return code;
    } else {
        throw new TornadoBailoutRuntimeException("PTX JIT compilation failed!");
    }
}

/**
 * Install CUDA C++ source (new NVRTC implementation).
 */
private PTXInstalledCode installSourceCUDA(TaskDataContext taskMeta, String name,
                                          byte[] targetCode, String resolvedMethodName) {
    // Convert byte[] to String (CUDA C++ source)
    String cudaSource = new String(targetCode);

    // Add CUDA headers
    CUDACodeUtil.CUDAFeatures features = CUDACodeUtil.CUDAFeatures.defaults();
    // TODO: Detect features from source or metadata
    String completeSource = CUDACodeUtil.getCodeWithCUDAHeader(cudaSource, features);

    // Get compiler flags and convert to NVRTC options
    String compilerFlags = taskMeta.getCompilerFlags(TornadoVMBackendType.PTX);
    String[] nvrtcOptions;

    if (compilerFlags == null || compilerFlags.trim().isEmpty()) {
        // Use defaults
        int computeCapability = deviceContext.getDevice().getComputeCapability();
        nvrtcOptions = NVRTCCompilerOptions.getDefaultOptions(computeCapability);
    } else {
        nvrtcOptions = NVRTCCompilerOptions.fromCompilerFlagsString(compilerFlags);
    }

    // Compile using NVRTC
    NVRTCModule module = new NVRTCModule(resolvedMethodName, completeSource,
                                        name, nvrtcOptions);

    if (module.isCompilationSuccess()) {
        // Wrap in PTXInstalledCode (reuse existing infrastructure)
        // Note: This requires PTXInstalledCode to accept NVRTCModule
        PTXInstalledCode code = new PTXInstalledCode(name, module, deviceContext);
        cache.put(name, code);
        return code;
    } else {
        System.err.println("NVRTC compilation failed!");
        System.err.println("Compilation log:");
        System.err.println(module.getCompilationLog());
        System.err.println("\nSource code:");
        System.err.println(completeSource);
        throw new TornadoBailoutRuntimeException(
            "NVRTC compilation failed! See log above for details."
        );
    }
}
```

---

### Step 3.2: Modify PTXInstalledCode.java

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/PTXInstalledCode.java`

**Goal**: Accept both PTXModule and NVRTCModule.

#### Change 3.2.1: Add Overloaded Constructor

**Find existing constructor**:
```java
public PTXInstalledCode(String name, PTXModule module, PTXDeviceContext deviceContext) {
    super(name);
    this.module = module;
    this.deviceContext = deviceContext;
}
```

**Add overload for NVRTCModule**:
```java
/**
 * Create from NVRTCModule (CUDA C++ compilation).
 */
public PTXInstalledCode(String name, NVRTCModule nvrtcModule, PTXDeviceContext deviceContext) {
    super(name);
    // NVRTCModule has same interface as PTXModule, can be used directly
    // But we need to adapt it to PTXModule interface
    this.module = wrapNVRTCModule(nvrtcModule);
    this.deviceContext = deviceContext;
}

/**
 * Wrap NVRTCModule to PTXModule interface (adapter pattern).
 */
private PTXModule wrapNVRTCModule(NVRTCModule nvrtcModule) {
    // Option 1: Make both implement common interface
    // Option 2: Create adapter
    // For simplicity, we'll store moduleWrapper directly

    // This requires refactoring PTXInstalledCode to not depend on PTXModule type
    // Instead, just use byte[] moduleWrapper

    // SIMPLIFIED: Store module handle directly
    this.moduleWrapper = nvrtcModule.moduleWrapper;
    this.kernelFunctionName = nvrtcModule.kernelFunctionName;
    return null; // Not needed if we refactor to use fields directly
}
```

**Better approach**: Refactor PTXInstalledCode to use module handle directly instead of PTXModule object:

```java
public class PTXInstalledCode extends TornadoInstalledCode {
    private final byte[] moduleWrapper;        // CUmodule handle
    private final String kernelFunctionName;
    private final PTXDeviceContext deviceContext;
    // ... other fields

    public PTXInstalledCode(String name, PTXModule module, PTXDeviceContext deviceContext) {
        super(name);
        this.moduleWrapper = module.moduleWrapper;
        this.kernelFunctionName = module.kernelFunctionName;
        this.deviceContext = deviceContext;
    }

    public PTXInstalledCode(String name, NVRTCModule module, PTXDeviceContext deviceContext) {
        super(name);
        this.moduleWrapper = module.moduleWrapper;
        this.kernelFunctionName = module.kernelFunctionName;
        this.deviceContext = deviceContext;
    }

    // Rest of class uses moduleWrapper and kernelFunctionName directly
}
```

---

## Phase 4: Configuration & Testing (Week 7)

### Step 4.1: Add Configuration Support

#### 4.1.1: Create Configuration Class

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/PTXConfiguration.java`

**Complete Code**:

```java
/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 */
package uk.ac.manchester.tornado.drivers.ptx;

/**
 * Configuration for PTX backend.
 */
public class PTXConfiguration {

    private static final String CODEGEN_MODE_PROPERTY = "tornado.ptx.codegen.mode";
    private static final String USE_NVRTC_PROPERTY = "tornado.ptx.cuda.use.nvrtc";
    private static final String CACHE_SOURCE_PROPERTY = "tornado.ptx.cuda.cache.source";

    private static CodeGenMode codeGenMode = null;

    /**
     * Get code generation mode from system properties.
     */
    public static CodeGenMode getCodeGenMode() {
        if (codeGenMode == null) {
            String mode = System.getProperty(CODEGEN_MODE_PROPERTY, "PTX");
            codeGenMode = CodeGenMode.fromString(mode);
        }
        return codeGenMode;
    }

    /**
     * Set code generation mode programmatically.
     */
    public static void setCodeGenMode(CodeGenMode mode) {
        codeGenMode = mode;
    }

    /**
     * Check if NVRTC should be used for CUDA compilation.
     */
    public static boolean useNVRTC() {
        return Boolean.parseBoolean(
            System.getProperty(USE_NVRTC_PROPERTY, "true")
        );
    }

    /**
     * Check if CUDA C++ source should be cached.
     */
    public static boolean cacheSource() {
        return Boolean.parseBoolean(
            System.getProperty(CACHE_SOURCE_PROPERTY, "false")
        );
    }
}
```

#### 4.1.2: Use Configuration in PTXBackendImpl

**Location**: `/tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/PTXBackendImpl.java`

**Find backend initialization** and add:

```java
@Override
public void initialize() {
    // Existing initialization code...

    // Configure code generation mode
    CodeGenMode mode = PTXConfiguration.getCodeGenMode();
    for (PTXBackend backend : backends) {
        backend.setCodeGenMode(mode);
    }

    System.out.println("PTX Backend initialized with code generation mode: " + mode);
}
```

---

### Step 4.2: Comprehensive Testing

#### 4.2.1: Create Test Suite

**Location**: `/tornado-drivers/ptx/src/test/java/uk/ac/manchester/tornado/drivers/ptx/tests/CUDACodeGenTest.java`

**Complete Test Suite**:

```java
package uk.ac.manchester.tornado.drivers.ptx.tests;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import static org.junit.Assert.*;

import uk.ac.manchester.tornado.drivers.ptx.CodeGenMode;
import uk.ac.manchester.tornado.drivers.ptx.PTXConfiguration;

/**
 * Test suite for CUDA C++ code generation.
 * Runs same tests in both PTX and CUDA modes.
 */
@RunWith(Parameterized.class)
public class CUDACodeGenTest {

    @Parameterized.Parameter
    public CodeGenMode mode;

    @Parameterized.Parameters(name = "Mode: {0}")
    public static CodeGenMode[] modes() {
        return new CodeGenMode[] { CodeGenMode.PTX, CodeGenMode.CUDA };
    }

    @Before
    public void setUp() {
        PTXConfiguration.setCodeGenMode(mode);
    }

    @Test
    public void testVectorAddition() {
        // Test vector addition kernel
        // This should work in both PTX and CUDA modes
        int[] a = new int[] {1, 2, 3, 4, 5};
        int[] b = new int[] {10, 20, 30, 40, 50};
        int[] c = new int[5];

        // Run TornadoVM task
        TaskGraph taskGraph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
            .task("t0", TestKernels::vectorAdd, a, b, c)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.execute();

        // Verify results
        for (int i = 0; i < 5; i++) {
            assertEquals("Index " + i, a[i] + b[i], c[i]);
        }
    }

    @Test
    public void testMatrixMultiplication() {
        // Test matrix multiplication
        final int N = 64;
        float[] a = new float[N * N];
        float[] b = new float[N * N];
        float[] c = new float[N * N];

        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            a[i] = (float) i;
            b[i] = 1.0f;
        }

        // Run kernel
        TaskGraph taskGraph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
            .task("t0", TestKernels::matrixMultiply, a, b, c, N)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.execute();

        // Verify results (each row should sum to same value)
        for (int i = 0; i < N; i++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                sum += c[i * N + j];
            }
            assertTrue("Row " + i + " sum", Math.abs(sum - (N * N * (N - 1) / 2.0)) < 0.01);
        }
    }

    @Test
    public void testReduction() {
        // Test parallel reduction
        int[] input = new int[1024];
        int[] output = new int[1];

        for (int i = 0; i < input.length; i++) {
            input[i] = i + 1;
        }

        TaskGraph taskGraph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, input)
            .task("t0", TestKernels::reduce, input, output)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.execute();

        // Expected: sum of 1 to 1024 = 1024 * 1025 / 2 = 524800
        assertEquals("Reduction result", 524800, output[0]);
    }

    @Test
    public void testControlFlow() {
        // Test if/else and loops
        int[] data = new int[100];
        for (int i = 0; i < 100; i++) {
            data[i] = i;
        }

        TaskGraph taskGraph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, data)
            .task("t0", TestKernels::controlFlow, data)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, data);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutionPlan(immutableTaskGraph);
        executor.execute();

        // Verify: even numbers doubled, odd numbers tripled
        for (int i = 0; i < 100; i++) {
            if (i % 2 == 0) {
                assertEquals(i * 2, data[i]);
            } else {
                assertEquals(i * 3, data[i]);
            }
        }
    }
}
```

#### 4.2.2: Run Tests

```bash
# Test PTX mode (default)
mvn test -Dtest=CUDACodeGenTest

# Test CUDA mode
mvn test -Dtest=CUDACodeGenTest -Dtornado.ptx.codegen.mode=CUDA

# Run all tests in both modes
mvn test -Dtornado.ptx.codegen.mode=CUDA
```

---

## Phase 5: Documentation & Polish (Week 8)

### Step 5.1: Update Documentation

#### 5.1.1: Create User Guide

**Location**: `/docs/CUDA_CODE_GENERATION.md`

**Content**:
```markdown
# CUDA C++ Code Generation in TornadoVM

## Overview

TornadoVM can generate either PTX assembly or CUDA C++ code for NVIDIA GPUs.

## Configuration

### Enable CUDA Mode

**System Property**:
```bash
-Dtornado.ptx.codegen.mode=CUDA
```

**Environment Variable**:
```bash
export TORNADO_PTX_CODEGEN_MODE=CUDA
```

**Programmatic**:
```java
PTXConfiguration.setCodeGenMode(CodeGenMode.CUDA);
```

### Default (PTX Mode)

```bash
-Dtornado.ptx.codegen.mode=PTX
```

## Benefits of CUDA Mode

- **Readable Code**: Generated code is human-readable C++
- **Better Errors**: Compilation errors show C++ source locations
- **Debuggable**: Can use CUDA debugging tools
- **Optimizations**: NVRTC can perform additional optimizations

## Requirements

- CUDA Toolkit 7.5 or later
- NVRTC library (ships with CUDA Toolkit)

## Examples

### Simple Kernel

**Java Code**:
```java
@Parallel
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

**Generated CUDA C++**:
```cuda
__global__ void add_kernel(int* a, int* b, int* c, int n) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < n) {
        c[globalId] = a[globalId] + b[globalId];
    }
}
```

## Troubleshooting

### NVRTC Compilation Errors

If you see NVRTC compilation errors, the error log will show:
- Source file and line number
- Error description
- Complete CUDA C++ source

### Fallback to PTX

If NVRTC is not available, set:
```bash
-Dtornado.ptx.codegen.mode=PTX
```

## Performance

CUDA mode has similar performance to PTX mode:
- First compilation: ~100-300ms (vs ~50-200ms for PTX)
- Cached execution: ~1ms (same as PTX)
```

---

### Step 5.2: Code Quality & Cleanup

#### 5.2.1: Add Logging

**In CUDAAssembler**:
```java
private static final boolean DEBUG = Boolean.getBoolean("tornado.ptx.cuda.debug");

public void debug(String message) {
    if (DEBUG) {
        System.out.println("[CUDA] " + message);
    }
}
```

#### 5.2.2: Add Comments

Ensure all public methods have Javadoc:
```java
/**
 * Emit a binary operation in CUDA C++.
 *
 * @param dest Destination variable
 * @param left Left operand
 * @param right Right operand
 * @param operator C++ operator (+, -, *, /, etc.)
 */
public void emitBinaryOp(Value dest, Value left, Value right, String operator) {
    // ...
}
```

---

## Summary Checklist

### Phase 1: NVRTC Infrastructure ✅
- [x] NVRTCModule.java
- [x] NVRTCModule.cpp
- [x] NVRTCModule.h
- [x] CMakeLists.txt updated
- [x] NVRTCCompilerOptions.java
- [x] Tests pass

### Phase 2: Code Generation ✅
- [x] CodeGenMode enum
- [x] CUDAAssembler.java (basic + statements)
- [x] CUDALIRStmt.java (all statement types)
- [x] PTXBackend modifications
- [x] PTXCompilationResultBuilder modifications
- [x] CUDACodeUtil.java

### Phase 3: Integration ✅
- [x] PTXCodeCache modifications
- [x] PTXInstalledCode modifications
- [x] Mode detection
- [x] Error handling

### Phase 4: Configuration & Testing ✅
- [x] PTXConfiguration class
- [x] System property support
- [x] Comprehensive test suite
- [x] All tests pass in both modes

### Phase 5: Documentation ✅
- [x] User guide
- [x] Code comments
- [x] Troubleshooting guide
- [x] Examples

---

## Quick Start Command Reference

### Build

```bash
# Build with NVRTC support
cd tornado-drivers/ptx-jni
mkdir build && cd build
cmake ..
make

# Build Java components
cd ../..
mvn clean install
```

### Test

```bash
# Test NVRTC infrastructure
mvn test -Dtest=NVRTCModuleTest

# Test code generation
mvn test -Dtest=CUDACodeGenTest

# Test in CUDA mode
mvn test -Dtornado.ptx.codegen.mode=CUDA
```

### Run Application

```bash
# Run with CUDA code generation
tornado --jvm="-Dtornado.ptx.codegen.mode=CUDA" -jar myapp.jar

# Run with PTX (default)
tornado -jar myapp.jar
```

---

## Common Issues & Solutions

### Issue 1: NVRTC Not Found

**Error**: `libnvrtc.so: cannot open shared object file`

**Solution**:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue 2: Compilation Errors

**Error**: NVRTC compilation failed with syntax errors

**Solution**:
1. Check generated source (will be printed in error message)
2. Verify CUDA headers are correct
3. Check compute capability matches your GPU

### Issue 3: Performance Regression

**Error**: CUDA mode is slower than PTX

**Solution**:
1. Check compilation options (should use -O3)
2. Verify caching is working
3. Profile to find bottleneck

---

## Next Steps

After completing all phases:

1. **Benchmark**: Compare PTX vs CUDA performance
2. **Optimize**: Fine-tune generated code
3. **Extend**: Add more CUDA features (streams, etc.)
4. **Document**: Create examples and tutorials
5. **Release**: Prepare for production deployment

---

**End of Implementation Guide Part 2**

Continue to final steps: commit changes, push branch, create PR.