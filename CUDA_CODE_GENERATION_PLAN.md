# TornadoVM: PTX to CUDA C/C++ Code Generation Plan

## Executive Summary

TornadoVM currently generates **PTX assembly code** which is then JIT-compiled by CUDA at runtime. This plan outlines the minimal changes needed to generate **CUDA C/C++ source code** instead, enabling better readability, debuggability, and potentially better optimization by the CUDA compiler.

**Key Insight**: The entire compilation pipeline from Java → Graal IR → LIR can be **reused completely**. Only the final code emission stage (approximately 30% of the backend) needs modification.

---

## Current Architecture Overview

### Compilation Pipeline

```
Java Code
    ↓
Graal IR (Graph-based High-Level IR)
    ↓
PTXHighTier (inlining, DCE, intrinsics replacement)
    ↓
PTXMidTier (guard lowering, exception elimination)
    ↓
PTXLowTier (address lowering, FMA optimization)
    ↓
LIR Generation (PTXLIRGenerator - platform-independent)
    ↓
LIR Statements (PTXLIRStmt - platform-specific)
    ↓
PTXAssembler.emit() - GENERATES PTX ASSEMBLY TEXT
    ↓
PTX Assembly (.ptx file)
    ↓
CUDA JIT Compiler (cuModuleLoadDataEx)
    ↓
Executable GPU Binary
```

### Key Components (Current)

| Component | Location | Responsibility | Lines | Reusable? |
|-----------|----------|----------------|-------|-----------|
| PTXCompiler | compiler/PTXCompiler.java | Orchestrates 3-phase compilation | ~800 | ✅ 100% |
| PTXHighTier | compiler/PTXHighTier.java | High-level optimizations | ~200 | ✅ 100% |
| PTXMidTier | compiler/PTXMidTier.java | Mid-level optimizations | ~150 | ✅ 100% |
| PTXLowTier | compiler/PTXLowTier.java | Low-level optimizations | ~150 | ✅ 100% |
| PTXLIRGenerator | compiler/PTXLIRGenerator.java | Graal IR → LIR conversion | ~250 | ✅ 100% |
| PTXBackend | backend/PTXBackend.java | Backend orchestration | ~300 | ⚠️ 70% (prologue/epilogue need changes) |
| PTXCompilationResultBuilder | compiler/PTXCompilationResultBuilder.java | LIR iteration | ~200 | ✅ 95% |
| **PTXAssembler** | **asm/PTXAssembler.java** | **PTX text emission** | **~825** | **❌ 0% - NEEDS REPLACEMENT** |
| **PTXLIRStmt** | **lir/PTXLIRStmt.java** | **Statement emission logic** | **~1669** | **❌ 0% - NEEDS REPLACEMENT** |

**Total reusable code: ~70%**
**Code requiring changes: ~30%**

---

## PTX Assembly vs CUDA C/C++

### Concrete Example

#### Current PTX Output (`add.ptx`)

```ptx
.visible .entry s0_t0_add_arrays_intarray_arrays_intarray_arrays_intarray(
    .param .u64 .ptr .global .align 8 kernel_context,
    .param .u64 .ptr .global .align 8 a,
    .param .u64 .ptr .global .align 8 b,
    .param .u64 .ptr .global .align 8 c) {
    .reg .s64 rsd<3>;
    .reg .pred rpb<2>;
    .reg .u32 rui<5>;
    .reg .s32 rsi<9>;
    .reg .u64 rud<9>;

BLOCK_0:
    ld.param.u64	rud0, [kernel_context];
    ld.param.u64	rud1, [a];
    ld.param.u64	rud2, [b];
    ld.param.u64	rud3, [c];
    mov.u32	rui0, %nctaid.x;
    mov.u32	rui1, %ntid.x;
    mul.wide.u32	rud4, rui0, rui1;
    cvt.s32.u64	rsi0, rud4;
    mov.u32	rui2, %tid.x;
    mov.u32	rui3, %ctaid.x;
    mad.lo.s32	rsi1, rui3, rui1, rui2;

BLOCK_1:
    mov.s32	rsi2, rsi1;
LOOP_COND_1:
    setp.lt.s32	rpb0, rsi2, 8;
    @!rpb0 bra	BLOCK_3;

BLOCK_2:
    add.s32	rsi3, rsi2, 4;
    cvt.s64.s32	rsd0, rsi3;
    shl.b64	rsd1, rsd0, 2;
    add.u64	rud5, rud1, rsd1;
    ld.global.s32	rsi4, [rud5];
    add.u64	rud6, rud2, rsd1;
    ld.global.s32	rsi5, [rud6];
    add.u64	rud7, rud3, rsd1;
    add.s32	rsi6, rsi4, rsi5;
    st.global.s32	[rud7], rsi6;
    add.s32	rsi7, rsi0, rsi2;
    mov.s32	rsi2, rsi7;
    bra.uni	LOOP_COND_1;

BLOCK_3:
    ret;
}
```

#### Desired CUDA C/C++ Output

```cuda
__global__ void s0_t0_add_arrays_intarray_arrays_intarray(
    const void* kernel_context,
    int* a,
    int* b,
    int* c) {

    // Thread index calculation
    int blockSize = blockDim.x;
    int totalSize = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over elements
    for (int i = tid; i < 8; i += totalSize) {
        int offset = i + 4;
        int val_a = a[offset];
        int val_b = b[offset];
        c[offset] = val_a + val_b;
    }
}
```

### Key Differences

| Aspect | PTX Assembly | CUDA C/C++ |
|--------|--------------|-----------|
| **Function declaration** | `.visible .entry` | `__global__ void` |
| **Parameters** | `.param .u64 .ptr .global` | `int* a` (standard C/C++ pointers) |
| **Register allocation** | Explicit `.reg .s32 rsi<9>` | Automatic by compiler |
| **Thread indexing** | `%tid.x`, `%ctaid.x`, `%ntid.x` | `threadIdx.x`, `blockIdx.x`, `blockDim.x` |
| **Memory access** | `ld.global.s32 rsi4, [rud5]` | `val = a[index]` (C/C++ array access) |
| **Control flow** | `setp`, `@predicate bra LABEL` | `if`, `while`, `for` (structured) |
| **Arithmetic** | `add.s32 rsi3, rsi2, 4` | `offset = i + 4` (C++ operators) |
| **Type annotations** | Explicit `.s32`, `.f64`, `.u64` | Implicit C/C++ types |

---

## Minimal Changes Approach

### Strategy: "Replace the Tip, Keep the Pipeline"

The entire compilation stack up to LIR generation is **format-agnostic**. We only need to replace the final emission layer.

```
┌────────────────────────────────────────────────────────┐
│          COMPLETELY REUSABLE (70%)                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Java → Graal IR → Optimization Tiers → LIR      │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬────────────────────────────────┘
                        │ (LIR = abstract operations)
                        ▼
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────────┐       ┌──────────────────────┐
│  PTXAssembler     │       │  CUDAAssembler       │
│  PTXLIRStmt       │       │  CUDALIRStmt         │
│  (Emits PTX)      │       │  (Emits CUDA C++)    │
└───────────────────┘       └──────────────────────┘
```

### What Changes (30% of Backend)

1. **PTXAssembler → CUDAAssembler** (~825 lines)
   - Replace `emit()` methods to output C++ syntax instead of PTX
   - Remove PTX-specific constants (`.visible`, `.entry`, `.reg`)
   - Add CUDA-specific keywords (`__global__`, `__shared__`, `threadIdx.x`)

2. **PTXLIRStmt → CUDALIRStmt** (~1669 lines)
   - Each statement's `emitCode()` method generates C++ instead of PTX
   - Example transformations:
     - `ld.global.s32 %r1, [%r0]` → `int val = ptr[index];`
     - `st.global.s32 [%r0], %r1` → `ptr[index] = val;`
     - `setp.lt.s32 rpb0, rsi2, 8` → `bool cond = (i < 8);`
     - `@!rpb0 bra BLOCK_3` → `if (!cond) break;`

3. **PTXBackend.emitPrologue/Epilogue** (~50 lines)
   - Change from `.visible .entry name(...)` to `__global__ void name(...)`
   - Skip register declarations (C++ compiler handles this)
   - Add thread index initialization boilerplate

---

## Classes to Touch

### Critical Files (Must Modify/Create)

#### 1. **Create: CUDAAssembler.java** (New file, ~800 lines)
**Location**: `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/asm/CUDAAssembler.java`

**Purpose**: Replace PTXAssembler for CUDA C++ emission

**Key Methods to Implement**:
```java
public class CUDAAssembler extends Assembler {
    // Core emission
    public void emitKernelStart(String name, List<Value> params);
    public void emitKernelEnd();
    public void emitThreadIndexInit();

    // Variable operations
    public void emitVariableDeclaration(Variable var, PTXKind kind);
    public void emitAssignment(Variable lhs, Value rhs);

    // Memory operations
    public void emitLoad(Variable dest, AbstractAddress addr);
    public void emitStore(AbstractAddress addr, Value value);

    // Arithmetic
    public void emitBinaryOp(Variable dest, String op, Value left, Value right);
    public void emitUnaryOp(Variable dest, String op, Value value);

    // Control flow
    public void emitLabel(String label);  // Convert to structured control
    public void emitBranch(String target); // Convert to if/while/break
    public void emitConditionalBranch(Variable predicate, boolean negate, String target);

    // Helpers
    public String toCType(PTXKind kind);  // .s32 → int, .f64 → double
    public String toVariableName(Value value);  // %rsi4 → rsi4 (drop %)
}
```

#### 2. **Create: CUDALIRStmt.java** (New file, ~1500 lines)
**Location**: `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/lir/CUDALIRStmt.java`

**Purpose**: Mirror PTXLIRStmt but with CUDA C++ emission

**Key Statement Classes to Implement**:
```java
// Copy structure from PTXLIRStmt.java, but change emitCode() implementations
public class LoadDataStmt extends AbstractInstruction {
    @Override
    public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
        // PTX: ld.global.s32 %r1, [%r0];
        // CUDA: int r1 = *((int*)r0);
        asm.emit(dest.toString() + " = ");
        address.emitAsPointerDeref(asm);
        asm.emit(";\n");
    }
}

public class StoreDataStmt extends AbstractInstruction {
    @Override
    public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
        // PTX: st.global.s32 [%r0], %r1;
        // CUDA: *((int*)addr) = value;
        address.emitAsPointerDeref(asm);
        asm.emit(" = " + value.toString() + ";\n");
    }
}

public class BinaryExprStmt extends AbstractInstruction {
    @Override
    public void emitCode(PTXCompilationResultBuilder crb, CUDAAssembler asm) {
        // PTX: add.s32 %r2, %r0, %r1;
        // CUDA: int r2 = r0 + r1;
        asm.emit(dest.toString() + " = ");
        asm.emit(left.toString() + " " + operatorToCSymbol(op) + " " + right.toString());
        asm.emit(";\n");
    }
}

// Similar transformations for all 100+ statement types
```

#### 3. **Modify: PTXBackend.java** (~300 lines, modify ~50 lines)
**Location**: `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/backend/PTXBackend.java`

**Changes Needed**:
- Lines 246-259: `emitPrologue()` - Add mode switch for CUDA vs PTX
- Lines 254-257: `emitVariableDefs()` - Skip for CUDA mode (auto allocation)
- Add new method: `emitPrologueCUDA()`

**Example**:
```java
public class PTXBackend extends XPUBackend<PTXProviders> {
    private enum CodeGenMode { PTX, CUDA }
    private CodeGenMode mode = CodeGenMode.PTX;  // Default to PTX

    public void setCodeGenMode(CodeGenMode mode) {
        this.mode = mode;
    }

    @Override
    protected void emitPrologue(...) {
        if (mode == CodeGenMode.CUDA) {
            emitPrologueCUDA(crb, asm, method, incomingArguments);
        } else {
            emitProloguePTX(crb, asm, method, incomingArguments);
        }
    }

    private void emitPrologueCUDA(PTXCompilationResultBuilder crb,
                                   CUDAAssembler asm,
                                   ResolvedJavaMethod method,
                                   AllocatableValue[] incomingArguments) {
        // Emit: __global__ void kernel_name(params...) {
        asm.emit("__global__ void ");
        asm.emit(crb.compilationResult.getName());
        asm.emit("(");
        emitParametersCUDA(asm, method, incomingArguments);
        asm.emit(") {\n");

        // Emit thread index initialization
        asm.emit("    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");
        asm.emit("    int blockSize = blockDim.x;\n");
        asm.emit("    int totalSize = gridDim.x * blockDim.x;\n");
    }
}
```

#### 4. **Optional: Create CUDABackend.java** (New file, ~300 lines)
**Location**: `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/backend/CUDABackend.java`

**Alternative approach**: Extend PTXBackend, override only emission methods

```java
public class CUDABackend extends PTXBackend {
    @Override
    public CompilationResultBuilder newCompilationResultBuilder(...) {
        return new PTXCompilationResultBuilder(
            new CUDAAssembler(target, lirGenRes),  // Use CUDA assembler
            // ... rest same as PTXBackend
        );
    }

    @Override
    protected void emitPrologue(...) {
        emitPrologueCUDA(...);
    }
}
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal**: Set up infrastructure for dual-mode compilation

1. **Create CUDAAssembler skeleton**
   - Extend `org.graalvm.compiler.asm.Assembler`
   - Implement basic `emit()`, `emitLine()`, `emitValue()` methods
   - Add helper: `toCType()` to map PTXKind → C++ types
   - Add helper: `toVariableName()` to format variable names

2. **Add backend mode configuration**
   - Add `CodeGenMode` enum to PTXBackend
   - Add flag/property to switch modes: `tornado.ptx.codegen.mode=CUDA`
   - Wire configuration through PTXBackendImpl

3. **Create test infrastructure**
   - Add simple test case (vector add kernel)
   - Verify CUDA C++ output formatting
   - Ensure compilation still works in PTX mode

**Deliverable**: Can generate empty CUDA kernel skeleton

---

### Phase 2: Basic Statements (Week 2)

**Goal**: Implement core statement types

1. **Implement CUDALIRStmt base classes**
   - `LoadDataStmt` - Array/pointer loads
   - `StoreDataStmt` - Array/pointer stores
   - `AssignStmt` - Variable assignments
   - `BinaryExprStmt` - Arithmetic operations
   - `UnaryExprStmt` - Unary operations

2. **Implement prologue/epilogue**
   - `emitPrologueCUDA()` in PTXBackend
   - Kernel signature with parameters
   - Thread index initialization
   - Closing brace

3. **Test simple kernels**
   - Vector addition
   - Scalar operations
   - Array access patterns

**Deliverable**: Can generate functional CUDA kernel for simple operations

---

### Phase 3: Control Flow (Week 3)

**Goal**: Handle branches, loops, and predicates

1. **Structured control flow**
   - Convert `@predicate bra LABEL` to `if (!cond) goto`
   - Detect loop patterns (back-edges in CFG)
   - Generate `while` loops where possible
   - Fallback to `goto` for complex control flow

2. **Implement control flow statements**
   - `ConditionalBranchStmt` → if/goto
   - `SetPredicateStmt` → boolean assignments
   - `LabelStmt` → keep labels for gotos
   - Loop detection heuristic

3. **Test complex control flow**
   - Nested loops
   - Conditional branches
   - Early exits (break/continue)

**Deliverable**: Can generate CUDA code with loops and conditionals

---

### Phase 4: Advanced Features (Week 4)

**Goal**: Handle remaining PTX features

1. **Memory spaces**
   - Global memory (already handled)
   - Shared memory (`__shared__` arrays)
   - Local memory (stack variables)
   - Constant memory (`__constant__`)

2. **Vector operations**
   - Vector loads/stores
   - SIMD operations
   - Packed data types

3. **Synchronization**
   - `bar.sync` → `__syncthreads()`
   - Atomic operations → CUDA atomics
   - Memory fences

4. **Special operations**
   - Math intrinsics (sin, cos, sqrt) → CUDA math functions
   - `dp4a` → `__dp4a()` intrinsic
   - Half-float operations → `__half` type

**Deliverable**: Feature-complete CUDA code generation

---

### Phase 5: Optimization & Testing (Week 5)

**Goal**: Optimize output, comprehensive testing

1. **Code quality improvements**
   - Dead code elimination (unused variables)
   - Common subexpression elimination
   - Better variable naming
   - Pretty-printing with indentation

2. **Comprehensive testing**
   - Run full TornadoVM test suite
   - Compare PTX vs CUDA performance
   - Verify numerical correctness
   - Test edge cases

3. **Documentation**
   - Update developer docs
   - Add examples
   - Document configuration

**Deliverable**: Production-ready CUDA code generation

---

## Detailed Class Mapping

### Files That Stay Unchanged (70%)

✅ **All files remain as-is:**
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXCompiler.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXHighTier.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXMidTier.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXLowTier.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXLIRGenerator.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXLIRGenerationResult.java`
- `tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/compiler/PTXCompilationResultBuilder.java` (95% unchanged)
- All optimization phases
- All graph builders

### Files That Need New Variants (25%)

❌ **Require CUDA equivalents:**
- `PTXAssembler.java` (825 lines) → **CUDAAssembler.java**
- `PTXLIRStmt.java` (1669 lines) → **CUDALIRStmt.java**

### Files That Need Minor Modifications (5%)

⚠️ **Add conditional logic:**
- `PTXBackend.java` (~50 lines changed out of 300)
  - Add mode switching
  - Add `emitPrologueCUDA()`
  - Conditional variable declaration

---

## Technical Challenges & Solutions

### Challenge 1: Label-based vs Structured Control Flow

**Problem**: PTX uses explicit labels and branches. C++ prefers structured control (if/while/for).

**Solution**:
- Phase 1: Emit `goto` statements (valid C++) for all branches
- Phase 2: Pattern matching to detect loops (back-edges in CFG)
- Phase 3: Convert simple loops to `while`/`for`
- Keep `goto` for complex control flow (break to outer loop, etc.)

**Example**:
```c
// Phase 1 (goto-based)
BLOCK_1:
    if (i < 8) goto BLOCK_2;
    goto BLOCK_3;
BLOCK_2:
    // loop body
    i += stride;
    goto BLOCK_1;
BLOCK_3:
    return;

// Phase 2 (structured)
while (i < 8) {
    // loop body
    i += stride;
}
```

### Challenge 2: Register Allocation

**Problem**: PTX requires explicit register declarations. C++ has automatic allocation.

**Solution**:
- Skip `.reg` declarations entirely in CUDA mode
- Just emit variable assignments: `int rsi4 = rsi2 + 4;`
- Let CUDA compiler handle register allocation
- Result: Simpler code, potentially better optimization

### Challenge 3: Type System Mapping

**Problem**: PTX has explicit type suffixes (`.s32`, `.f64`). C++ is implicit.

**Solution**:
Create mapping function:
```java
public String toCType(PTXKind kind) {
    switch (kind) {
        case S8: case U8: return "char";
        case S16: case U16: return "short";
        case S32: case U32: return "int";
        case S64: case U64: return "long long";
        case F32: return "float";
        case F64: return "double";
        case PRED: return "bool";
        // Vectors
        case INT2: return "int2";
        case FLOAT4: return "float4";
        // ...
    }
}
```

### Challenge 4: Memory Address Calculations

**Problem**: PTX uses pointer arithmetic with explicit shifts. C++ uses array indexing.

**Solution**:
```java
// PTX: add.u64 rud5, rud1, rsd1; ld.global.s32 rsi4, [rud5];
// Becomes: int rsi4 = ((int*)rud1)[rsd1 / 4];

public void emitArrayAccess(Variable dest, Variable base, Variable offset, PTXKind kind) {
    String typecast = "((" + toCType(kind) + "*)" + base + ")";
    String index = "[" + offset + " / " + kind.getSizeInBytes() + "]";
    emit(dest + " = " + typecast + index + ";\n");
}
```

---

## Configuration & Deployment

### Configuration Options

Add property to enable CUDA mode:
```properties
# tornado.properties
tornado.ptx.codegen.mode=PTX    # Default (current behavior)
tornado.ptx.codegen.mode=CUDA   # New mode (CUDA C++ generation)
```

### Compilation Pipeline Changes

**PTX Mode** (unchanged):
```
TornadoVM → PTX Assembly → CUDA JIT Compiler → GPU Binary
```

**CUDA Mode** (new):
```
TornadoVM → CUDA C++ Source → NVCC Compiler → GPU Binary
```

### Backward Compatibility

- Default mode remains PTX (no breaking changes)
- Users opt-in to CUDA mode via configuration
- Both modes can coexist in same build

---

## Performance Considerations

### Expected Benefits of CUDA Mode

1. **Better Optimization**: NVCC can perform whole-program optimizations
2. **Better Readability**: Easier to debug generated code
3. **More Maintainable**: C++ is more familiar than PTX assembly
4. **Future-Proof**: NVIDIA may deprecate PTX JIT in favor of ahead-of-time compilation

### Potential Trade-offs

1. **Compilation Time**: NVCC compilation may be slower than PTX JIT
2. **Binary Size**: Need to include CUDA C++ compiler in build process
3. **Debugging**: More complex toolchain (but better source-level debugging)

---

## Testing Strategy

### Unit Tests

Create parallel test suites:
```
PTXAssemblerTest.java  → CUDAAssemblerTest.java
PTXBackendTest.java    → CUDABackendTest.java
```

### Integration Tests

Run full test suite in both modes:
```bash
# PTX mode (existing)
mvn test -Dtornado.ptx.codegen.mode=PTX

# CUDA mode (new)
mvn test -Dtornado.ptx.codegen.mode=CUDA
```

### Validation Tests

1. **Functional Correctness**: Same numerical results in both modes
2. **Performance Parity**: CUDA should be within 5% of PTX performance
3. **Memory Safety**: No crashes or memory leaks
4. **Edge Cases**: NaN, infinity, overflow, underflow

---

## Success Criteria

✅ **Milestone 1** (Foundation): Generate valid CUDA kernel skeletons
✅ **Milestone 2** (Basic): Vector addition works end-to-end
✅ **Milestone 3** (Control Flow): Matrix multiplication works
✅ **Milestone 4** (Advanced): Full test suite passes
✅ **Milestone 5** (Production): Performance within 5% of PTX mode

---

## Summary of Minimal Changes

### Files to Create (2 files, ~2500 lines)
1. **CUDAAssembler.java** (~800 lines)
2. **CUDALIRStmt.java** (~1500 lines)

### Files to Modify (1 file, ~50 lines changed)
1. **PTXBackend.java** (add mode switch + CUDA prologue)

### Files to Leave Unchanged (~15,000+ lines)
- All compiler phases
- All LIR generation
- All optimization passes
- All graph construction

### Effort Estimate
- **Code to write**: ~2,500 lines (mostly copying PTXLIRStmt and changing emission)
- **Code to modify**: ~50 lines
- **Code to test**: Run existing test suite in new mode
- **Timeline**: 4-5 weeks for full implementation

---

## Appendix: Key Files Reference

### Core Backend Files

```
tornado-drivers/ptx/src/main/java/uk/ac/manchester/tornado/drivers/ptx/graal/
├── asm/
│   ├── PTXAssembler.java (825 lines)           ← REPLACE WITH CUDAAssembler.java
│   └── PTXAssemblerConstants.java (113 lines)  ← MODIFY for CUDA constants
├── backend/
│   └── PTXBackend.java (300+ lines)            ← ADD mode switch (~50 lines)
├── compiler/
│   ├── PTXCompiler.java (800 lines)            ← UNCHANGED
│   ├── PTXLIRGenerator.java (250 lines)        ← UNCHANGED
│   ├── PTXCompilationResultBuilder.java        ← UNCHANGED (95%)
│   ├── PTXHighTier.java                        ← UNCHANGED
│   ├── PTXMidTier.java                         ← UNCHANGED
│   └── PTXLowTier.java                         ← UNCHANGED
└── lir/
    ├── PTXLIRStmt.java (1669 lines)            ← REPLACE WITH CUDALIRStmt.java
    ├── PTXKind.java (300 lines)                ← UNCHANGED (only toString() output changes)
    └── PTXMemorySpace.java (51 lines)          ← UNCHANGED
```

### Reference Output Examples

- **PTX Example**: `tornado-assembly/src/examples/generated/add.ptx`
- **Desired CUDA**: See "Concrete Example" section above

---

## Next Steps

1. **Review this plan** with the team
2. **Approve architecture** (modify PTXBackend vs create CUDABackend)
3. **Set up development branch**
4. **Implement Phase 1** (foundation + skeleton)
5. **Iterate through phases** with regular testing

---

## Questions & Decisions Needed

1. **Architecture**: Modify PTXBackend or create separate CUDABackend?
   - **Recommendation**: Modify PTXBackend with mode switch (simpler maintenance)

2. **Configuration**: Property file or command-line flag?
   - **Recommendation**: Both (property for persistent, flag for testing)

3. **Naming**: Keep "PTX" in class names or rename to generic "GPU"?
   - **Recommendation**: Keep PTX naming (avoid massive refactor)

4. **Control Flow**: Always use `goto` or try to generate structured loops?
   - **Recommendation**: Start with `goto`, optimize later

5. **Testing**: Parallel test suites or unified with mode parameter?
   - **Recommendation**: Unified suite, run in both modes

---

**Document Version**: 1.0
**Date**: 2026-01-13
**Author**: Claude (based on TornadoVM codebase analysis)
