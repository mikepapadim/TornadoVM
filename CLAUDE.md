# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TornadoVM is a plug-in to OpenJDK and GraalVM that allows programmers to automatically run Java programs on heterogeneous hardware. It targets OpenCL, PTX (NVIDIA CUDA), and SPIR-V compatible devices including multi-core CPUs, GPUs (Intel, NVIDIA, AMD), and FPGAs (Intel and Xilinx).

TornadoVM enables transparent JIT compilation from Java to device-specific code (OpenCL C, PTX assembly, SPIR-V binary) using a graph-based execution model and GraalVM's compiler infrastructure.

## Build Commands

### Basic Build
```bash
# Full build with OpenCL backend (default)
make

# Build with specific backend(s)
make BACKEND=opencl
make BACKEND=ptx
make BACKEND=spirv
make BACKEND=opencl,ptx,spirv

# Build for GraalVM with polyglot support
make polyglot
```

### Incremental Builds
```bash
# Fast incremental build (skips assembly if only unittests changed)
# Approximately 2x faster than full build (10s vs 22s for typical changes)
make incremental

# Incremental build that forces assembly rebuild
make incremental-full

# Rebuild with dependency updates
make rebuild-deps-jdk21
```

**Important:** The build system has smart assembly logic and optimizations:
- `make incremental` only rebuilds changed modules and skips assembly for unittest-only changes
- Uses **fast-assembly profile** when rebuilding distribution:
  - Skips tar.gz creation (saves ~5-6s)
  - Skips shade plugin for tornado-assembly (saves ~4-5s)
  - Only creates directory structure needed for development
- Changed modules are detected via `git diff` against staged files
- Assembly optimization reduces assembly time from 12.5s to 1.2s (10x faster!)
- Run `make incremental-full` if you need to force assembly rebuild

### Clean
```bash
make clean
```

## Running Tests

### Unit Tests
```bash
# Run all tests
make tests

# Run fast test subset (uses --quickPass flag)
make fast-tests

# Run specific test
tornado-test --ea -V uk.ac.manchester.tornado.unittests.fails.HeapFail#test03

# Run with specific backend (SPIR-V examples)
make tests-spirv-levelzero
make tests-spirv-opencl
```

### Running Examples
```bash
# Run example with kernel printing and debug output
tornado --printKernel --debug -m tornado.examples/uk.ac.manchester.tornado.examples.VectorAddInt --params="8192"

# General pattern for running examples
java @tornado-argfile -cp tornado-examples/target/tornado-examples-<version>.jar <MainClass>
```

### Test Commands
- `tornado --devices` - List available devices
- `tornado-test` - Main test runner
- `test-native.sh` - Native library tests

## Code Style

TornadoVM uses auto-formatters for Eclipse and IntelliJ:
- **IntelliJ** (recommended by team): Import XML auto-formatter from `scripts/templates/eclipse-settings/Tornado.xml`
  - Settings → Code Style → Java → Import Scheme
  - Enable "Use single class import" (Settings → Code Style → Java → Imports)
  - Enable Save Actions for auto-formatting and import optimization
- **Eclipse**: Run `python3 scripts/eclipseSetup.py`

Run checkstyle before committing:
```bash
make checkstyle
```

## High-Level Architecture

TornadoVM uses a multi-tiered architecture that compiles Java to device code through several abstraction layers:

### 1. Task Graph Model (tornado-runtime/tasks/)

Users define computation using a **task graph** that explicitly represents parallelism and data flow:

```java
TaskGraph taskGraph = new TaskGraph("s0")
    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inputData)
    .task("kernelName", MyClass::myMethod, args...)
    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputData);

ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph);
plan.execute();
```

**Key files:**
- `TornadoTaskGraph.java` - Public API entry point
- `SchedulableTask` interface - Represents executable tasks (CompilableTask, PrebuiltTask)
- `TaskDataContext.java` - Per-task execution metadata

### 2. Graph Compilation (tornado-runtime/graph/)

The task graph is compiled into a **directed acyclic graph (DAG)** of operations:

**TornadoGraph** contains nodes:
- `TaskNode` - Kernel execution
- `CopyInNode` / `CopyOutNode` - Data transfers (host↔device)
- `AllocateNode` / `DeallocateNode` - Memory management
- `ContextOpNode` - Async device operations

**Compilation flow:**
```
TaskGraph → TornadoGraph (DAG) → TornadoVMBytecodes[] → Interpreter execution
```

**Key files:**
- `TornadoGraph.java` - DAG representation with typed nodes
- `TornadoVMGraphCompiler.java` - Converts graph to bytecode instructions
- `TornadoExecutionContext.java` - Manages execution state, memory, and dependencies
- `nodes/AbstractNode.java` - Base class for all graph nodes

### 3. Bytecode Interpretation (tornado-runtime/interpreter/)

The graph is compiled to **device-agnostic bytecodes** interpreted by `TornadoVMInterpreter`:

**Bytecode operations:**
- `CONTEXT` - Device context setup
- `TASK` - Kernel invocation (triggers JIT compilation)
- `TRANSFER_HOST_TO_DEVICE` / `TRANSFER_DEVICE_TO_HOST` - Memory transfers
- `BARRIER` - Synchronization
- `BEGIN` / `END` - Transaction boundaries

Each device gets its own interpreter instance that manages:
- Memory allocation and transfers
- Kernel JIT compilation
- Event tracking and synchronization

**Key files:**
- `TornadoVMInterpreter.java` - Bytecode interpreter and memory manager
- `TornadoVM.java` - Main orchestrator, manages array of interpreters

### 4. JIT Compilation Pipeline (tornado-runtime/graal/)

When a task is first executed, TornadoVM uses **GraalVM's compiler infrastructure** to JIT-compile Java bytecode to device code:

```
Java Bytecode
    ↓ [Graph Builder]
Graal IR (StructuredGraph)
    ↓ [SketchTier - parallel analysis]
    ↓ [HighTier - inlining, canonicalization]
    ↓ [MidTier - optimizations]
    ↓ [LowTier - address lowering]
Low-Level IR (LIR)
    ↓ [Backend code generation]
OpenCL C / PTX / SPIR-V
```

**Key abstractions:**
- `TornadoSuitesProvider` - Defines compilation tiers
- `TornadoCompilerConfiguration` - Configures compilation phases
- Custom IR nodes: `ParallelRangeNode`, `ThreadIdFixedWithNextNode`, reduction nodes

**Parallelism detection:**
- Java `@Parallel` annotations on loops indicate GPU threads
- `KernelContext` API gives explicit thread ID access (CUDA/OpenCL-like style)

**Key files:**
- `TornadoCoreRuntime.java` - GraalVM/JVMCI integration singleton
- `compiler/TornadoSuitesProvider.java` - Compilation tier definitions
- `graal/nodes/` - Custom IR nodes for parallel patterns

### 5. Backend Abstraction (tornado-drivers/)

TornadoVM supports multiple backends through the `XPUBackend` abstraction:

**Backend hierarchy:**
```
XPUBackend<P extends Providers>
    ├── OCLBackend (OpenCL C code generation)
    ├── PTXBackend (NVIDIA PTX assembly)
    └── SPIRVBackend (SPIR-V binary)
```

Each backend implements:
- Device discovery and initialization
- Code generation from LIR
- Memory management
- Kernel launch and synchronization

**Device discovery pattern:**
1. Backend implementations (`OCLBackendImpl`, `PTXBackendImpl`, `SPIRVBackendImpl`) discover platforms/devices at init
2. Create array of `XPUBackend` instances (one per device)
3. Loaded dynamically via Java ServiceLoader in `TornadoCoreRuntime.loadBackends()`

**Key files:**
- `tornado-runtime/graal/backend/XPUBackend.java` - Abstract backend interface
- `tornado-drivers/opencl/OCLBackendImpl.java` - OpenCL platform discovery
- `tornado-drivers/ptx/PTXBackendImpl.java` - NVIDIA CUDA device management
- `tornado-drivers/spirv/SPIRVBackendImpl.java` - SPIR-V runtime (Level Zero, OpenCL)

### 6. Execution Flow Summary

```
User: TaskGraph.execute()
  ↓
TornadoExecutionContext.compile() → TornadoGraph (DAG)
  ↓
TornadoVMGraphCompiler.compile() → TornadoVMBytecodes[]
  ↓
TornadoVM.execute() → launches TornadoVMInterpreter[] (one per device)
  ↓
TornadoVMInterpreter.execute() for each device:
  - Interpret bytecodes
  - JIT compile kernels on first execution (Java → OpenCL/PTX/SPIR-V)
  - Manage device buffers (allocate, transfer)
  - Launch kernels via backend driver
  ↓
Device execution (OpenCL/CUDA/Level Zero runtime)
```

## Module Structure

```
tornado-api/           - Public API (Apache 2 license)
tornado-runtime/       - Core runtime and GraalVM integration (GPL v2 + Classpath Exception)
tornado-drivers/       - Backend implementations (GPL v2 + Classpath Exception)
  ├── drivers-common/  - Shared driver code
  ├── opencl/          - OpenCL backend + JNI bindings
  ├── ptx/             - NVIDIA PTX backend + JNI bindings
  └── spirv/           - SPIR-V backend
tornado-annotation/    - @Parallel and @Reduce annotations (Apache 2)
tornado-matrices/      - Matrix data structures (Apache 2)
tornado-examples/      - Example applications (Apache 2)
tornado-unittests/     - Test suite (Apache 2)
tornado-benchmarks/    - JMH benchmarks (Apache 2)
tornado-assembly/      - Build distribution package
```

## Important Development Notes

### JNI and Native Code

OpenCL and PTX backends include JNI components:
- `tornado-drivers/opencl-jni/` - OpenCL JNI bindings (built with CMake)
- `tornado-drivers/ptx-jni/` - CUDA PTX JNI bindings (built with CMake)

Native libraries are built automatically during Maven build and copied to `dist/tornado-sdk/`.

### GraalVM Dependencies

For standard JDK (non-GraalVM), the build system automatically downloads required GraalVM jars:
- `bin/pull_graal_jars.py` fetches compiler jars to `graalJars/`
- These are included in classpath via `tornado-argfile`

### Distribution Structure

The `make` command produces:
```
dist/tornado-sdk/tornado-sdk-<version>/
  ├── bin/               - tornado, tornado-test, benchmark scripts
  ├── lib/               - JARs and native libraries (.so/.dylib/.dll)
  ├── etc/               - Export lists and configuration
  └── examples/          - Example source code and JARs
```

After building, use:
- `tornado` - Run TornadoVM applications
- `tornado-test` - Run unit tests
- `tornado --devices` - List available accelerators

### Memory Model

TornadoVM transparently manages host↔device memory:
- `DataTransferMode.FIRST_EXECUTION` - Copy only on first invocation
- `DataTransferMode.EVERY_EXECUTION` - Copy on every invocation
- `DataTransferMode.UNDER_DEMAND` - Copy when data dependencies require it

Object state tracked per device:
- `LocalObjectState` - Host-side object metadata
- `DataObjectState` - Device-side buffer state
- `XPUDeviceBufferState` - Buffer lifecycle (ALLOCATED, RESIDENT, etc.)

**Batch processing** for memory-constrained execution:
- Large arrays split into chunks
- `BatchConfiguration` tracks offset/stride per batch
- Iteratively processes batches without exceeding device memory

## Contributing

- Fork and create PRs to the `develop` branch (see CONTRIBUTING.md)
- Sign the Contributor License Agreement (CLA) when prompted
- Ensure code follows auto-formatter rules (run `make checkstyle`)
- PRs reviewed by 2+ TornadoVM team members
- Extensive testing required across OCL/PTX/SPIRV backends

## Debugging and Profiling

Key environment variables and flags (used via `tornado` command):
```bash
# Print generated kernel code
tornado --printKernel ...

# Enable debug output
tornado --debug ...

# Enable profiler
tornado --jvm="-Dtornado.profiler=True" ...

# Dump Graal IR to IGV (Ideal Graph Visualizer)
tornado --jvm="-Dgraal.Dump=*:5 -Dgraal.PrintGraph=Network" ...

# List all devices
tornado --devices
```

## Related Documentation

- Main docs: https://tornadovm.readthedocs.io/
- Installation guide: https://tornadovm.readthedocs.io/en/latest/installation.html
- Programming guide: https://tornadovm.readthedocs.io/en/latest/programming.html
- Execution flags: https://tornadovm.readthedocs.io/en/latest/flags.html
- Benchmarking: https://tornadovm.readthedocs.io/en/latest/benchmarking.html
