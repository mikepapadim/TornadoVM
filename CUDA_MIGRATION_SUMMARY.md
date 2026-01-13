# PTX to CUDA C++ Migration: Complete Summary

## Quick Overview

This document summarizes the complete plan for migrating TornadoVM from PTX assembly generation to CUDA C++ code generation.

**Read this first**, then dive into detailed documents as needed.

---

## The Big Picture

### What We Have Now
```
Java Code
  ↓ [Graal IR + Optimizations - 70% of pipeline]
  ↓
PTX Assembly Generation (PTXAssembler)
  ↓
PTX Text (assembly language)
  ↓ [JNI → cuModuleLoadDataEx]
  ↓
CUDA JIT Compiler
  ↓
GPU Binary
```

### What We Want
```
Java Code
  ↓ [Graal IR + Optimizations - SAME 70% pipeline]
  ↓
CUDA C++ Generation (CUDAAssembler)
  ↓
CUDA C++ Source Code
  ↓ [JNI → NVRTC]
  ↓
NVRTC Compiler (runtime)
  ↓
GPU Binary
```

### Why This is Relatively Easy
**70% of the pipeline is reusable!** Only the final code emission layer needs changes.

---

## Two Main Changes Required

### 1. Code Generation Layer (30% of backend)
**Creates CUDA C++ instead of PTX assembly**

**Details**: See `CUDA_CODE_GENERATION_PLAN.md`

**Files to Create**:
- `CUDAAssembler.java` (~800 lines) - Emits C++ syntax
- `CUDALIRStmt.java` (~1,500 lines) - C++ statement emission

**Files to Modify**:
- `PTXBackend.java` (~50 lines) - Add mode switch

**Key Transformations**:
```
PTX:   ld.global.s32 %r1, [%r0];
CUDA:  int r1 = ptr[index];

PTX:   add.s32 %r2, %r0, %r1;
CUDA:  int r2 = r0 + r1;

PTX:   setp.lt.s32 rpb0, rsi2, 8; @!rpb0 bra BLOCK_3;
CUDA:  if (i >= 8) break;
```

---

### 2. Compilation Layer (JNI)
**Uses NVRTC to compile CUDA C++ at runtime**

**Details**: See `CUDA_COMPILATION_STRATEGY.md`

**Files to Create**:
- `NVRTCModule.java` (~80 lines) - Java wrapper
- `NVRTCModule.cpp` (~200 lines) - JNI bindings
- `NVRTCCompilerOptions.java` (~70 lines) - Option mapping

**Files to Modify**:
- `PTXCodeCache.java` (~30 lines) - Route to NVRTC
- `CMakeLists.txt` (~20 lines) - Link NVRTC library

**What is NVRTC?**
- NVIDIA Runtime Compilation library
- Compiles CUDA C++ to PTX/CUBIN at runtime
- Ships with CUDA Toolkit (no external dependencies)
- Better error messages than PTX JIT

---

## Side-by-Side Comparison

| Aspect | Current (PTX) | New (CUDA C++) |
|--------|---------------|----------------|
| **Output Format** | PTX Assembly | CUDA C++ Source |
| **Readability** | Low (assembly) | High (C++ code) |
| **Debugging** | Difficult | Easy (source-level) |
| **Compilation** | CUDA JIT | NVRTC (runtime) |
| **Error Messages** | Cryptic codes | C++ compiler errors |
| **Code Size** | ~10-50 KB | ~5-20 KB |
| **First Compile** | ~50-200ms | ~100-300ms |
| **Cached Execute** | ~1ms | ~1ms (same) |
| **External Deps** | None | None (NVRTC in CUDA) |

---

## Concrete Example

### Input: Java Code
```java
@Parallel
for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
}
```

### Current Output: PTX Assembly
```ptx
.visible .entry add_kernel(
    .param .u64 .ptr .global .align 8 a,
    .param .u64 .ptr .global .align 8 b,
    .param .u64 .ptr .global .align 8 c) {
    .reg .s32 rsi<5>;
    .reg .u64 rud<7>;

    mov.u32 rui0, %tid.x;
    mov.u32 rui1, %ctaid.x;
    mov.u32 rui2, %ntid.x;
    mad.lo.s32 rsi1, rui1, rui2, rui0;

    cvt.s64.s32 rsd0, rsi1;
    shl.b64 rsd1, rsd0, 2;
    add.u64 rud5, rud1, rsd1;
    ld.global.s32 rsi4, [rud5];
    add.u64 rud6, rud2, rsd1;
    ld.global.s32 rsi5, [rud6];
    add.s32 rsi6, rsi4, rsi5;
    add.u64 rud7, rud3, rsd1;
    st.global.s32 [rud7], rsi6;
    ret;
}
```

### New Output: CUDA C++
```cuda
__global__ void add_kernel(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

**Much more readable!**

---

## Files Summary

### Complete Change List

**Code Generation Layer** (from `CUDA_CODE_GENERATION_PLAN.md`):
```
NEW:    CUDAAssembler.java               (~800 lines)
NEW:    CUDALIRStmt.java                 (~1,500 lines)
MODIFY: PTXBackend.java                  (~50 lines changed)
```

**Compilation Layer** (from `CUDA_COMPILATION_STRATEGY.md`):
```
NEW:    NVRTCModule.java                 (~80 lines)
NEW:    NVRTCModule.cpp                  (~200 lines)
NEW:    NVRTCCompilerOptions.java        (~70 lines)
MODIFY: PTXCodeCache.java                (~30 lines added)
MODIFY: CMakeLists.txt                   (~20 lines for NVRTC)
```

**Total New Code**: ~2,850 lines
**Total Modified Code**: ~100 lines
**Reused Existing Code**: ~15,000+ lines (70%)

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- Set up NVRTC JNI bindings
- Test basic CUDA C++ → module compilation
- Verify NVRTC integration works
- Create CUDAAssembler skeleton

**Milestone**: Can compile simple CUDA C++ kernel via NVRTC

---

### Phase 2: Code Generation (Weeks 3-4)
- Implement CUDAAssembler core methods
- Implement basic CUDALIRStmt statements
- Generate kernel prologue/epilogue
- Handle basic operations (load, store, arithmetic)

**Milestone**: Vector addition kernel works end-to-end

---

### Phase 3: Control Flow (Week 5)
- Implement branch/loop emission
- Convert PTX labels to structured control flow
- Handle predicates and conditional execution

**Milestone**: Matrix multiplication works

---

### Phase 4: Advanced Features (Week 6)
- Shared memory (`__shared__`)
- Synchronization (`__syncthreads()`)
- Vector operations
- Math intrinsics
- Atomic operations

**Milestone**: Complex kernels (reductions, FFT) work

---

### Phase 5: Testing & Polish (Weeks 7-8)
- Run full TornadoVM test suite
- Performance benchmarking (vs PTX mode)
- Error handling and debugging
- Documentation
- Code cleanup

**Milestone**: Production-ready CUDA mode

---

## Configuration

### How to Enable CUDA Mode

**Option 1: Environment Variables**
```bash
export TORNADO_CODEGEN_MODE=CUDA
export TORNADO_CUDA_USE_NVRTC=true
```

**Option 2: Java Properties**
```properties
# tornado.properties
tornado.ptx.codegen.mode=CUDA
tornado.ptx.cuda.use.nvrtc=true
```

**Option 3: Programmatic**
```java
TornadoVMBackend.setCodeGenMode(CodeGenMode.CUDA);
```

**Default**: PTX mode (backward compatible)

---

## Testing Strategy

### Unit Tests
- Test each CUDALIRStmt type individually
- Test NVRTC compilation with various options
- Test error handling (bad syntax, undefined vars)

### Integration Tests
- Run existing TornadoVM test suite in CUDA mode
- Compare outputs: PTX mode vs CUDA mode
- Verify numerical correctness (floating-point)

### Performance Tests
- Benchmark compilation time (first run)
- Benchmark execution time (PTX vs CUDA)
- Measure memory overhead (cache size)

**Target**: Within 5% performance of PTX mode

---

## Risks & Mitigations

### Risk 1: NVRTC Compilation Slower
**Mitigation**: Cache compiled modules aggressively (same as PTX)

### Risk 2: NVRTC Not Available
**Mitigation**: Fall back to PTX mode automatically

### Risk 3: Generated C++ Has Syntax Errors
**Mitigation**:
- Comprehensive unit tests
- Log generated code for debugging
- NVRTC provides detailed error messages

### Risk 4: Performance Regression
**Mitigation**:
- Benchmark thoroughly
- Keep PTX mode as option
- Profile and optimize hot paths

---

## Benefits

### Short-term (Development)
✅ **Easier Debugging**: C++ errors are much clearer than PTX errors
✅ **Faster Iteration**: Easier to understand generated code
✅ **Better Testing**: Can compile and test CUDA C++ independently

### Long-term (Production)
✅ **Maintainability**: C++ code is easier to read and modify
✅ **Debuggability**: Source-level debugging in CUDA debuggers
✅ **Optimization**: NVRTC can do higher-level optimizations
✅ **Future-Proof**: NVIDIA may deprecate PTX JIT eventually

### Documentation
✅ **User Understanding**: Users can see actual CUDA code
✅ **Examples**: Generated C++ is self-documenting
✅ **Education**: Helps users learn CUDA from TornadoVM

---

## Decision Points

### Architecture Decisions Made

✅ **Code Generation**: Create new CUDAAssembler (not modify PTXAssembler)
  - Cleaner separation
  - Easier testing
  - No risk to PTX mode

✅ **Compilation**: Use NVRTC (not NVCC offline)
  - Runtime compilation (no external tools)
  - Better fits TornadoVM architecture
  - Faster than fork/exec NVCC

✅ **Backward Compatibility**: Keep PTX mode as default
  - Gradual migration
  - Safety net
  - Users can choose

---

## Open Questions

### For Team Discussion

**Q1**: Should CUDA mode be opt-in or opt-out?
- **Option A**: Opt-in (PTX default) - safer, gradual adoption
- **Option B**: Opt-out (CUDA default) - faster migration
- **Recommendation**: Start opt-in, switch to opt-out after validation

**Q2**: How to handle control flow?
- **Option A**: Always use `goto` (simplest, works everywhere)
- **Option B**: Detect loops, generate `while`/`for` when possible
- **Recommendation**: Start with goto, optimize later

**Q3**: Cache intermediate PTX from NVRTC?
- **Option A**: No caching (compile CUDA → binary each time)
- **Option B**: Cache PTX (compile CUDA → PTX → binary)
- **Recommendation**: Start without caching, add if needed

**Q4**: Support both modes long-term or deprecate PTX?
- **Option A**: Support both modes indefinitely
- **Option B**: Deprecate PTX after 1-2 releases
- **Recommendation**: Support both for at least 2-3 releases

---

## Next Steps

### Immediate Actions (Week 1)

1. **Review documents** with team
   - `CUDA_CODE_GENERATION_PLAN.md` (code generation)
   - `CUDA_COMPILATION_STRATEGY.md` (NVRTC integration)
   - `CUDA_MIGRATION_SUMMARY.md` (this document)

2. **Answer open questions** and finalize decisions

3. **Set up development branch**
   ```bash
   git checkout -b cuda-codegen-implementation
   ```

4. **Create tracking issues**
   - Issue #1: NVRTC JNI bindings
   - Issue #2: CUDAAssembler implementation
   - Issue #3: CUDALIRStmt implementation
   - Issue #4: Integration and testing

5. **Start with Phase 1**: NVRTC infrastructure

---

## Success Metrics

### Definition of Done

✅ Can generate valid CUDA C++ code for all kernel types
✅ NVRTC compilation succeeds with proper error handling
✅ Full TornadoVM test suite passes in CUDA mode
✅ Performance within 5% of PTX mode (up or down)
✅ Documentation updated with examples
✅ Configuration options work correctly
✅ Error messages are helpful and actionable

### Quality Criteria

✅ Generated CUDA C++ is readable (passes manual inspection)
✅ No memory leaks in JNI layer
✅ Proper error handling (no crashes on bad input)
✅ Code follows TornadoVM style guidelines
✅ Adequate test coverage (>80% for new code)

---

## Related Documents

📄 **CUDA_CODE_GENERATION_PLAN.md** - Detailed code generation architecture
  - PTX vs CUDA C++ comparison
  - Classes to modify
  - Statement emission logic
  - 5-phase implementation plan

📄 **CUDA_COMPILATION_STRATEGY.md** - NVRTC integration details
  - JNI layer analysis
  - NVRTC API usage
  - Compiler options mapping
  - Error handling strategy

📄 **CUDA_MIGRATION_SUMMARY.md** - This document
  - High-level overview
  - Quick reference
  - Timeline and metrics

---

## Questions?

For technical questions:
- Code generation: See `CUDA_CODE_GENERATION_PLAN.md`
- Compilation: See `CUDA_COMPILATION_STRATEGY.md`
- Architecture: Review `PTXBackend.java`, `PTXAssembler.java`

For project planning:
- Timeline: See "Implementation Timeline" section above
- Effort: ~2,850 lines new code, 6-8 weeks
- Risk: Low (70% of code reused, PTX mode remains available)

---

**Status**: Planning Complete ✅
**Next Phase**: Implementation
**Target Completion**: 6-8 weeks from start
**Confidence**: High (architecture is well-understood, changes are localized)
