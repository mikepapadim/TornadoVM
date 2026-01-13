# CUDA C++ Compilation Strategy: JNI Layer Analysis

## Executive Summary

This document extends the CUDA code generation plan with detailed analysis of the **JNI compilation layer** - the critical piece where generated code is compiled to GPU binaries.

**Current**: PTX Assembly → `cuModuleLoadDataEx()` JIT compilation
**Proposed**: CUDA C++ → **NVRTC** Runtime Compilation (recommended)

---

## Current PTX JIT Compilation Flow

### 1. Code Generation to Compilation

```
┌─────────────────────────────────────────────────────────────┐
│ Java Layer (TornadoVM)                                       │
├─────────────────────────────────────────────────────────────┤
│ PTXCompiler.compileSketchForDevice()                         │
│   → PTXAssembler generates PTX text → byte[]                │
│   → PTXCodeUtil.getCodeWithAttachedPTXHeader()              │
│   → PTXCodeCache.installSource(byte[] targetCode)           │
│       → PTXModule constructor                                │
│           → cuModuleLoadDataEx(source, options, values)     │
└────────────────────────┬────────────────────────────────────┘
                         │ JNI Boundary
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Native Layer (C++ JNI)                                       │
├─────────────────────────────────────────────────────────────┤
│ PTXModule.cpp:                                              │
│   Java_uk_..._PTXModule_cuModuleLoadDataEx()               │
│     → Parse jbyteArray to char* ptx                         │
│     → cuModuleLoadDataEx(&module, ptx, options, values)    │
│     → Returns CUmodule as jbyteArray                        │
└────────────────────────┬────────────────────────────────────┘
                         │ CUDA Driver API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ CUDA Driver (libcuda.so)                                     │
├─────────────────────────────────────────────────────────────┤
│ cuModuleLoadDataEx()                                        │
│   → PTX Parser                                               │
│   → PTX to SASS JIT Compiler                                │
│   → Returns compiled CUmodule                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Key Files in Current Architecture

#### **Java Layer**
- **PTXModule.java** (67 lines)
  - Line 34: `cuModuleLoadDataEx(source, jitOptions, jitValues)`
  - Native method declarations (lines 41-47)
  - Wraps CUmodule pointer as byte[]

- **PTXCodeCache.java** (133 lines)
  - Line 56: `installSource()` - entry point for code caching
  - Lines 72-103: Parse JIT compiler flags
  - Line 105: Create PTXModule (triggers compilation)
  - Caches compiled modules by name

- **CUjitOption.java** - Enum for CUDA JIT compiler options
  ```java
  CU_JIT_MAX_REGISTERS(0),
  CU_JIT_OPTIMIZATION_LEVEL(7),
  CU_JIT_TARGET(9),
  CU_JIT_GENERATE_DEBUG_INFO(11),
  CU_JIT_LOG_VERBOSE(12),
  CU_JIT_GENERATE_LINE_INFO(13),
  CU_JIT_CACHE_MODE(14)
  ```

#### **Native Layer**
- **PTXModule.cpp** (186 lines)
  - Lines 89-146: `cuModuleLoadDataEx()` implementation
  - Line 99: Null-terminate PTX string
  - Lines 103-120: Parse JIT options/values
  - Line 123: **THE KEY CALL**: `cuModuleLoadDataEx(&module, ptx, ...)`
  - Lines 139-143: Error handling (returns empty array on failure)

### 3. PTX Header Generation

**PTXCodeUtil.java** (`getCodeWithAttachedPTXHeader()`):
```java
// Prepends PTX directives to kernel code
StringBuilder ptxHeader = new StringBuilder();
ptxHeader.append(".version ").append(computeCapability).append("\n");
ptxHeader.append(".target sm_").append(smVersion).append("\n");
ptxHeader.append(".address_size ").append(addressSize).append("\n\n");

byte[] headerBytes = ptxHeader.toString().getBytes();
byte[] combined = new byte[headerBytes.length + targetCode.length];
System.arraycopy(headerBytes, 0, combined, 0, headerBytes.length);
System.arraycopy(targetCode, 0, combined, headerBytes.length, targetCode.length);
return combined;
```

**Example PTX Header**:
```ptx
.version 7.5
.target sm_75
.address_size 64

.visible .entry kernel_name(...) {
    ...
}
```

---

## Three Options for CUDA C++ Compilation

### Option A: NVRTC (NVIDIA Runtime Compilation) ⭐ **RECOMMENDED**

**What is NVRTC?**
- NVIDIA's official runtime compilation library for CUDA C++
- Ships with CUDA Toolkit (no external dependencies)
- API: `libnvrtc.so` (similar to current `libcuda.so` usage)
- Designed exactly for this use case: runtime CUDA C++ compilation

**Compilation Flow**:
```
CUDA C++ Source (string)
    ↓
nvrtcCreateProgram(source)
    ↓
nvrtcCompileProgram(options)
    ↓
nvrtcGetPTX() or nvrtcGetCUBIN()
    ↓
cuModuleLoadData(ptx/cubin)
    ↓
Executable CUmodule
```

**Advantages**:
✅ Runtime compilation (no external compiler needed)
✅ Same deployment model as current PTX JIT
✅ No NVCC installation required
✅ Better error messages than PTX JIT
✅ Can cache compiled PTX/CUBIN like current system
✅ Supports all CUDA C++ features
✅ Low latency (similar to PTX JIT)

**Disadvantages**:
⚠️ Requires CUDA 7.5+ (widely available)
⚠️ New JNI bindings needed (~200 lines)
⚠️ Slightly different compiler options than PTX JIT

---

### Option B: NVCC Offline Compilation

**What is NVCC?**
- NVIDIA's CUDA C++ compiler (command-line tool)
- Requires CUDA Toolkit installation
- Compiles CUDA C++ → PTX or cubin files

**Compilation Flow**:
```
CUDA C++ Source → Write to temp file
    ↓
System.exec("nvcc -ptx source.cu -o output.ptx")
    ↓
Read output.ptx
    ↓
cuModuleLoadData(ptx) (existing path)
    ↓
Executable CUmodule
```

**Advantages**:
✅ No new JNI bindings (use existing PTX path)
✅ Full CUDA compiler with all optimizations
✅ Can generate PTX and use existing compilation path
✅ Familiar toolchain

**Disadvantages**:
❌ Requires NVCC installation (deployment complexity)
❌ Slow compilation (fork/exec overhead)
❌ Filesystem I/O (temp files)
❌ Platform-specific (path to nvcc)
❌ Not suitable for runtime JIT

---

### Option C: Hybrid (CUDA C++ → NVCC → PTX → CUDA JIT)

**Flow**:
```
Generate CUDA C++ → NVCC compile to PTX → Current PTX path
```

**Advantages**:
✅ No JNI changes needed
✅ Reuses entire existing compilation infrastructure
✅ CUDA C++ for readability, PTX for deployment

**Disadvantages**:
❌ Requires NVCC (same as Option B)
❌ Two compilation steps (slow)
❌ Loses benefits of direct CUDA C++ compilation

---

## Recommended Approach: NVRTC (Option A)

### Why NVRTC is Best

1. **Matches Current Architecture**: Runtime compilation, no external dependencies
2. **Production-Ready**: Used by many GPU frameworks (PyTorch, TensorFlow)
3. **Future-Proof**: NVIDIA's official path for runtime compilation
4. **Better Debugging**: C++ source-level errors vs. PTX assembly errors
5. **Performance**: Can optimize at C++ level before generating PTX/SASS

---

## NVRTC Implementation Details

### 1. New JNI Methods Needed

**Create: NVRTCModule.java**
```java
package uk.ac.manchester.tornado.drivers.ptx;

public class NVRTCModule {
    public final byte[] moduleWrapper;
    public final String kernelFunctionName;
    private final String source;

    public NVRTCModule(String name, String cudaSource, String kernelName,
                       String[] compileOptions) {
        // Compile CUDA C++ to PTX/CUBIN via NVRTC
        byte[] compiledCode = nvrtcCompile(cudaSource, compileOptions);

        // Load compiled code into CUDA module (existing path)
        this.moduleWrapper = cuModuleLoadData(compiledCode);
        this.source = cudaSource;
        this.kernelFunctionName = kernelName;
    }

    // New native methods
    private static native byte[] nvrtcCompile(String source, String[] options);
    private static native String nvrtcGetErrorLog(long program);

    // Existing methods
    private static native byte[] cuModuleLoadData(byte[] binary);
    private static native long cuModuleUnload(byte[] module);
    // ... rest same as PTXModule
}
```

### 2. New C++ JNI Implementation

**Create: NVRTCModule.cpp** (~200 lines)

```cpp
#include <jni.h>
#include <nvrtc.h>
#include <cuda.h>
#include <vector>
#include <string>

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    nvrtcCompile
 * Signature: (Ljava/lang/String;[Ljava/lang/String;)[B
 */
JNIEXPORT jbyteArray JNICALL
Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_nvrtcCompile
  (JNIEnv *env, jclass clazz, jstring source, jobjectArray options) {

    // Convert Java string to C++ string
    const char *cuda_source = env->GetStringUTFChars(source, nullptr);

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
        // Error handling
        env->ReleaseStringUTFChars(source, cuda_source);
        return env->NewByteArray(0);  // Empty array = error
    }

    // Convert Java String[] to C++ char**
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

    if (result != NVRTC_SUCCESS) {
        // Get compilation log
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(prog, log.data());

        printf("NVRTC Compilation failed:\n%s\n", log.data());
        fflush(stdout);

        nvrtcDestroyProgram(&prog);
        return env->NewByteArray(0);
    }

    // Get compiled PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Convert to Java byte array
    jbyteArray result_array = env->NewByteArray(ptx_size);
    env->SetByteArrayRegion(result_array, 0, ptx_size,
                           reinterpret_cast<jbyte*>(ptx.data()));

    nvrtcDestroyProgram(&prog);
    return result_array;
}
```

### 3. Compiler Options Mapping

**PTX JIT Options → NVRTC Options**

| PTX JIT Option | NVRTC Equivalent |
|----------------|------------------|
| `CU_JIT_OPTIMIZATION_LEVEL 4` | `"-O4"` |
| `CU_JIT_MAX_REGISTERS 32` | `"-maxrregcount=32"` |
| `CU_JIT_TARGET sm_75` | `"-arch=compute_75"` |
| `CU_JIT_GENERATE_DEBUG_INFO` | `"-G"` |
| `CU_JIT_GENERATE_LINE_INFO` | `"-lineinfo"` |

**New: NVRTCCompilerOptions.java**
```java
public class NVRTCCompilerOptions {
    public static String[] fromJitOptions(int[] jitOptions, long[] jitValues) {
        List<String> opts = new ArrayList<>();

        for (int i = 0; i < jitOptions.length; i++) {
            CUjitOption option = CUjitOption.fromValue(jitOptions[i]);
            switch (option) {
                case CU_JIT_OPTIMIZATION_LEVEL:
                    opts.add("-O" + jitValues[i]);
                    break;
                case CU_JIT_MAX_REGISTERS:
                    opts.add("-maxrregcount=" + jitValues[i]);
                    break;
                case CU_JIT_TARGET:
                    int sm = (int) jitValues[i];
                    opts.add("-arch=compute_" + sm);
                    break;
                case CU_JIT_GENERATE_DEBUG_INFO:
                    opts.add("-G");
                    break;
                case CU_JIT_GENERATE_LINE_INFO:
                    opts.add("-lineinfo");
                    break;
                // ... other options
            }
        }

        return opts.toArray(new String[0]);
    }
}
```

### 4. CUDA C++ Header Template

Unlike PTX which needs `.version` and `.target`, CUDA C++ needs proper includes:

```cuda
// Auto-generated header
#include <cuda_runtime.h>

// Optional: math intrinsics
#include <cuda_fp16.h>      // for half-float
#include <cuda_bf16.h>      // for bfloat16

// Kernel code follows
__global__ void kernel_name(...) {
    // Generated code
}
```

**Implementation**: `CUDACodeUtil.java`
```java
public static String getCodeWithCUDAHeader(String kernelCode,
                                          boolean useHalfFloat,
                                          boolean useBFloat16) {
    StringBuilder header = new StringBuilder();
    header.append("#include <cuda_runtime.h>\n");

    if (useHalfFloat) {
        header.append("#include <cuda_fp16.h>\n");
    }
    if (useBFloat16) {
        header.append("#include <cuda_bf16.h>\n");
    }

    header.append("\n");
    header.append(kernelCode);

    return header.toString();
}
```

---

## Modified Architecture with NVRTC

### Compilation Flow Changes

**Before (PTX)**:
```
CUDAAssembler → PTX byte[] → PTXCodeCache → PTXModule
    → cuModuleLoadDataEx → CUmodule
```

**After (CUDA C++ with NVRTC)**:
```
CUDAAssembler → CUDA C++ string → CUDACodeCache → NVRTCModule
    → nvrtcCompile → PTX/CUBIN → cuModuleLoadData → CUmodule
```

### Files to Modify/Create

#### **Modify**:
1. **PTXCodeCache.java** (+30 lines)
   - Add mode detection: Check if source is PTX or CUDA C++
   - Route to PTXModule or NVRTCModule accordingly

   ```java
   public PTXInstalledCode installSource(..., byte[] targetCode, ...) {
       if (isGeneratingCUDA()) {
           String cudaSource = new String(targetCode);
           String[] nvrtcOpts = NVRTCCompilerOptions.fromJitOptions(...);
           NVRTCModule module = new NVRTCModule(name, cudaSource,
                                               kernelName, nvrtcOpts);
           // ... rest same
       } else {
           PTXModule module = new PTXModule(name, targetCode,
                                           kernelName, jitOpts, jitVals);
           // ... existing code
       }
   }
   ```

#### **Create**:
1. **NVRTCModule.java** (~80 lines)
   - Mirror PTXModule structure
   - Use NVRTC for compilation instead of direct JIT

2. **NVRTCModule.cpp** (~200 lines)
   - JNI bindings for NVRTC API
   - Methods: `nvrtcCompile()`, `nvrtcGetErrorLog()`

3. **NVRTCCompilerOptions.java** (~100 lines)
   - Map CUjitOption enums to NVRTC string options
   - Handle architecture flags, optimization levels, etc.

4. **CUDACodeUtil.java** (~50 lines)
   - Generate CUDA C++ header with includes
   - Similar to PTXCodeUtil but for C++

#### **Build System**:
1. **CMakeLists.txt** for ptx-jni
   - Add NVRTC library: `find_library(NVRTC_LIB nvrtc)`
   - Link: `target_link_libraries(tornado-ptx ${CUDA_LIB} ${NVRTC_LIB})`

---

## Error Handling Comparison

### PTX JIT Errors (Current)
```
PTX to cubin JIT compilation using cuModuleLoadDataEx failed! (700)
```
- Error codes are cryptic (700 = CUDA_ERROR_ILLEGAL_ADDRESS)
- No source location information
- Difficult to debug

### NVRTC Errors (New)
```
NVRTC Compilation failed:
kernel.cu(15): error: identifier "threadIdz" is undefined
kernel.cu(17): error: no operator "+" matches these operands
            operand types are: float + int *
2 errors detected in the compilation of "kernel.cu".
```
- Source file and line numbers
- Clear error messages
- Shows operator mismatches, undefined identifiers, etc.

---

## Caching Strategy

### Current PTX Caching
```java
ConcurrentHashMap<String, PTXInstalledCode> cache;
// Key: kernel name
// Value: Compiled CUmodule wrapper
```

### NVRTC Caching (Same + Optional Intermediate)
```java
// Level 1: Source code cache (avoid regeneration)
ConcurrentHashMap<String, String> cudaSourceCache;

// Level 2: Compiled module cache (current)
ConcurrentHashMap<String, PTXInstalledCode> moduleCache;

// Optional Level 3: Intermediate PTX cache (for debugging)
ConcurrentHashMap<String, byte[]> ptxCache;
```

---

## Performance Considerations

### Compilation Time

| Method | First Compilation | Cached Execution |
|--------|------------------|------------------|
| PTX JIT | ~50-200ms | ~1ms (cache hit) |
| NVRTC → PTX → JIT | ~100-300ms | ~1ms (cache hit) |
| NVRTC → CUBIN direct | ~150-400ms | ~0.5ms (no JIT) |

**Impact**: Slightly slower first compilation, same cached performance

### Memory Overhead

| Method | Storage per Kernel |
|--------|-------------------|
| PTX JIT | PTX text (~10-50 KB) + CUmodule |
| NVRTC | CUDA C++ (~5-20 KB) + PTX (~10-50 KB) + CUmodule |

**Impact**: ~2x source storage if caching CUDA C++ and PTX

---

## Migration Path

### Phase 1: Parallel Infrastructure (Week 1-2)
- Create NVRTC JNI bindings (NVRTCModule.cpp)
- Create Java wrapper (NVRTCModule.java)
- Add NVRTC option mapping (NVRTCCompilerOptions.java)
- Update build system (CMakeLists.txt)
- Test basic CUDA C++ → module compilation

### Phase 2: Integration (Week 3)
- Modify PTXCodeCache to detect source type
- Route CUDA C++ through NVRTC path
- Keep PTX path unchanged for backward compatibility
- Add configuration flag: `tornado.ptx.use.nvrtc=true`

### Phase 3: Code Generation (Week 4-6)
- Implement CUDAAssembler (from main plan)
- Implement CUDALIRStmt emission (from main plan)
- Generate valid CUDA C++ output

### Phase 4: Testing & Optimization (Week 7-8)
- Test full pipeline with NVRTC
- Benchmark vs PTX mode
- Error handling and logging
- Documentation

---

## Configuration

### Environment Variables
```bash
# Use CUDA C++ generation + NVRTC compilation
export TORNADO_CODEGEN_MODE=CUDA
export TORNADO_CUDA_USE_NVRTC=true

# Fallback: Use PTX generation (default)
export TORNADO_CODEGEN_MODE=PTX
```

### Java Properties
```java
// In tornado.properties
tornado.ptx.codegen.mode=CUDA        # Generate CUDA C++ instead of PTX
tornado.ptx.cuda.use.nvrtc=true      # Use NVRTC for compilation
tornado.ptx.cuda.cache.source=true   # Cache CUDA C++ source
```

---

## Testing Strategy

### Unit Tests (New)
```java
@Test
public void testNVRTCCompilation() {
    String cudaSource = "__global__ void test(int* a) { a[0] = 42; }";
    String[] opts = {"-arch=compute_75", "-O3"};

    NVRTCModule module = new NVRTCModule("test", cudaSource, "test", opts);
    assertTrue(module.isCompilationSuccess());
}

@Test
public void testNVRTCErrorHandling() {
    String badSource = "__global__ void test() { undefined_var++; }";
    // Should fail gracefully with error message
}
```

### Integration Tests (Existing + New Mode)
```java
// Run all existing tests with CUDA mode enabled
@ParameterizedTest
@ValueSource(strings = {"PTX", "CUDA"})
public void testVectorAddition(String mode) {
    System.setProperty("tornado.ptx.codegen.mode", mode);
    // ... existing test code
}
```

---

## Summary: JNI Layer Changes

### Minimal Changes Needed (Option A - NVRTC)

**New Files** (3 files, ~350 lines):
1. `NVRTCModule.java` (~80 lines)
2. `NVRTCModule.cpp` (~200 lines)
3. `NVRTCCompilerOptions.java` (~70 lines)

**Modified Files** (2 files, ~50 lines total):
1. `PTXCodeCache.java` (~30 lines added for routing)
2. `CMakeLists.txt` (ptx-jni) (~20 lines for NVRTC linking)

**Unchanged Files**:
- `PTXModule.java` - Keep for backward compatibility
- `PTXModule.cpp` - Keep for PTX mode
- All other compilation infrastructure

### Alternative: Minimal Changes (Option C - Hybrid NVCC)

**New Files** (1 file, ~100 lines):
1. `NVCCCompiler.java` - Process wrapper for nvcc

**Modified Files** (1 file, ~30 lines):
1. `PTXCodeCache.java` - Call NVCC before PTXModule

**Trade-offs**: No JNI changes, but requires NVCC installation and slower

---

## Recommendation

**Use NVRTC (Option A)** for the following reasons:

1. ✅ **Minimal JNI Changes**: Only 3 new files (~350 lines)
2. ✅ **Runtime Compilation**: No external dependencies (nvcc)
3. ✅ **Better Errors**: C++ source-level error messages
4. ✅ **Future-Proof**: NVIDIA's recommended approach
5. ✅ **Performance**: Similar to current PTX JIT
6. ✅ **Production Ready**: Used by PyTorch, TensorFlow, etc.

Combined with the main CUDA generation plan, total effort:
- Code generation layer: ~2,500 lines
- Compilation layer: ~350 lines
- **Total new code**: ~2,850 lines
- **Timeline**: 6-8 weeks for complete implementation

---

**Document Version**: 1.0
**Date**: 2026-01-13
**Companion to**: CUDA_CODE_GENERATION_PLAN.md
