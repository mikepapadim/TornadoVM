# CUDA Patterns in TornadoVM

This directory contains TornadoVM implementations of fundamental CUDA programming patterns based on the [cuda-patterns](https://github.com/philipfabianek/cuda-patterns) repository.

## Overview

These examples demonstrate how to implement common GPU programming patterns using TornadoVM's **Kernel Context API**, which provides low-level access to GPU features similar to CUDA. All patterns are portable across OpenCL and CUDA backends.

## Pattern Categories

### 📊 Basic Operations (01-06)

Fundamental GPU programming concepts and memory access patterns.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 01. Vector Addition | `basic/Pattern01_VectorAdd.java` | 1D grid/blocks, global thread indexing | `globalIdx`, basic parallelism |
| 02. Matrix Addition | `basic/Pattern02_MatrixAdd.java` | 2D grid/blocks, row-major layout | `globalIdx`, `globalIdy` |
| 03. Image Blur | `basic/Pattern03_ImageBlur.java` | 2D stencil, boundary handling | 2D indexing, neighborhood access |
| 04. Matrix Multiply (Naive) | `basic/Pattern04_MatrixMultiplyNaive.java` | Basic matmul, global memory | Computational intensity baseline |
| 05. Matrix Multiply (Tiled) | `basic/Pattern05_MatrixMultiplyTiled.java` | Shared memory, tiling | `allocateFloatLocalArray()`, `localBarrier()` |
| 06. Matrix Multiply (Coarsened) | `basic/Pattern06_MatrixMultiplyCoarsened.java` | Thread coarsening, register reuse | Work per thread optimization |

**Performance Notes:**
- Naive → Tiled: ~10-20x speedup via shared memory
- Tiled → Coarsened: Additional ~1.5-2x via thread coarsening

### 🔄 Convolution & Stencil (07-12)

Stencil operations and convolution patterns for image processing.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 07. 2D Convolution (Naive) | `convolution/Pattern07_Convolution2DNaive.java` | 2D stencil, constant kernel | Redundant global memory reads |
| 08-09. 2D Convolution (Tiled) | *(Implementable)* | Halo regions, shared memory | Local memory for input tiles |
| 10-12. 3D Stencil | *(Implementable)* | 3D indexing, plane-sweep | `globalIdz`, 3D work groups |

**Key Optimization:** Tiled convolution uses shared memory to cache input tiles, reducing global memory bandwidth by ~5-10x.

### 📈 Histogram Computation (13-17)

Progressive optimization from atomic operations to privatization.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 13. Histogram (Atomic) | `histogram/Pattern13_HistogramAtomic.java` | Global atomics, contention | `atomicAdd()` on global memory |
| 14. Histogram (Privatized) | `histogram/Pattern14_HistogramPrivatized.java` | Shared memory atomics, merge | `allocateIntLocalArray()`, privatization |
| 15-17. Histogram (Optimized) | *(Implementable)* | Thread coarsening, aggregation | Further contention reduction |

**Performance Notes:**
- Privatized version achieves ~10-100x speedup over naive atomics depending on contention
- Demonstrates classic reduction in atomic contention through privatization

### ➕ Reduction & Scanning (18-21)

Fundamental parallel patterns for array reduction and prefix sums.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 18. Parallel Reduction | `reduction/Pattern18_ParallelReduction.java` | Tree reduction, O(log n) depth | `localBarrier()`, shared memory |
| 19. Reduction (Coarsened) | *(Implementable)* | Load multiple elements per thread | Improved arithmetic intensity |
| 20. Prefix Sum (Single Block) | `reduction/Pattern20_PrefixSumSingleBlock.java` | Kogge-Stone scan | Double buffering in local memory |
| 21. Prefix Sum (Multi-Block) | *(Implementable)* | Hierarchical scan | Block-level coordination |

**Algorithm:** Tree-based reduction achieves O(log n) time complexity vs O(n) sequential.

### 🔢 Sparse Operations (22-23)

Sparse matrix computations with irregular memory access.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 22. SpMV (COO Format) | `sparse/Pattern22_SpMV_COO.java` | Coordinate format, atomics | `atomicAdd()` for row accumulation |
| 23. SpMV (ELL Format) | *(Implementable)* | Regular memory access | Better coalescing than COO |

**Format Comparison:**
- **COO:** Simple but requires atomics (contention issues)
- **ELL:** Regular access pattern but may have padding overhead

### 🌐 Graph Algorithms (24-26)

Parallel graph traversal algorithms.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 24. BFS (Naive) | `graph/Pattern24_BFS_Naive.java` | Level-synchronous, CSR format | Iterative kernel launches |
| 25. BFS (Frontier) | *(Implementable)* | Work queue, dynamic parallelism | Active vertex tracking |
| 26. BFS (Privatized) | *(Implementable)* | Shared memory queues | Block-level frontier |

**CSR Format:** Compressed Sparse Row efficiently represents graph adjacency.

### ⚡ Compute-Intensive Kernels (27-29)

High arithmetic intensity computations.

| Pattern | File | CUDA Concepts | TornadoVM Features |
|---------|------|---------------|-------------------|
| 27. Electrostatic Potential | `compute/Pattern27_ElectrostaticPotential.java` | N-body computation | High compute/memory ratio |
| 28. Electrostatic (Coarsened) | *(See Pattern27)* | Thread coarsening | Work per thread optimization |
| 29. Electrostatic (Optimized) | *(Implementable)* | Micro-optimizations | Register usage, unrolling |

**Performance:** Thread coarsening can achieve ~34% of peak FLOPS with careful tuning.

## TornadoVM Kernel Context API Reference

### Thread Indexing

| TornadoVM API | CUDA Equivalent | OpenCL Equivalent |
|---------------|-----------------|-------------------|
| `context.globalIdx` | `blockIdx.x * blockDim.x + threadIdx.x` | `get_global_id(0)` |
| `context.globalIdy` | `blockIdx.y * blockDim.y + threadIdx.y` | `get_global_id(1)` |
| `context.globalIdz` | `blockIdx.z * blockDim.z + threadIdx.z` | `get_global_id(2)` |
| `context.localIdx` | `threadIdx.x` | `get_local_id(0)` |
| `context.localIdy` | `threadIdx.y` | `get_local_id(1)` |
| `context.localIdz` | `threadIdx.z` | `get_local_id(2)` |
| `context.groupIdx` | `blockIdx.x` | `get_group_id(0)` |
| `context.groupIdy` | `blockIdx.y` | `get_group_id(1)` |
| `context.groupIdz` | `blockIdx.z` | `get_group_id(2)` |

### Grid and Block Dimensions

| TornadoVM API | CUDA Equivalent | OpenCL Equivalent |
|---------------|-----------------|-------------------|
| `context.localGroupSizeX` | `blockDim.x` | `get_local_size(0)` |
| `context.localGroupSizeY` | `blockDim.y` | `get_local_size(1)` |
| `context.localGroupSizeZ` | `blockDim.z` | `get_local_size(2)` |
| `context.globalGroupSizeX` | `gridDim.x * blockDim.x` | `get_global_size(0)` |

### Shared/Local Memory

```java
// Allocate shared memory (equivalent to CUDA __shared__)
float[] sharedMem = context.allocateFloatLocalArray(size);
int[] sharedInt = context.allocateIntLocalArray(size);
double[] sharedDouble = context.allocateDoubleLocalArray(size);
```

**CUDA Equivalent:**
```cuda
__shared__ float sharedMem[SIZE];
```

### Synchronization

```java
// Synchronize threads in a block (local memory fence)
context.localBarrier();  // __syncthreads() in CUDA

// Global memory fence
context.globalBarrier();  // Not commonly used
```

### Atomic Operations

```java
// Atomic add (supports int, long, float, double)
context.atomicAdd(array, index, value);
```

**CUDA Equivalent:**
```cuda
atomicAdd(&array[index], value);
```

## Usage Pattern

### 1. Create Kernel with KernelContext

```java
public static void myKernel(KernelContext context, FloatArray input, FloatArray output) {
    int idx = context.globalIdx;

    // Your kernel logic here
    output.set(idx, input.get(idx) * 2.0f);
}
```

### 2. Configure WorkerGrid

```java
// 1D grid
WorkerGrid worker = new WorkerGrid1D(totalElements);
worker.setGlobalWork(globalSize, 1, 1);
worker.setLocalWork(blockSize, 1, 1);

// 2D grid
WorkerGrid worker = new WorkerGrid2D(width, height);
worker.setGlobalWork(width, height, 1);
worker.setLocalWork(blockSizeX, blockSizeY, 1);

// 3D grid
WorkerGrid worker = new WorkerGrid3D(width, height, depth);
worker.setGlobalWork(width, height, depth);
worker.setLocalWork(blockSizeX, blockSizeY, blockSizeZ);
```

### 3. Create and Execute Task Graph

```java
KernelContext context = new KernelContext();
GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

TaskGraph taskGraph = new TaskGraph("s0")
    .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
    .task("t0", MyClass::myKernel, context, input, output)
    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
    executionPlan.withGridScheduler(gridScheduler).execute();
}
```

## Pattern Summary: CUDA vs TornadoVM

| CUDA Feature | TornadoVM Equivalent | Status |
|--------------|---------------------|--------|
| 1D/2D/3D Grid Indexing | ✅ `globalIdx/y/z`, `localIdx/y/z` | **Fully Supported** |
| Shared Memory (`__shared__`) | ✅ `allocateFloatLocalArray()` | **Fully Supported** |
| Thread Synchronization (`__syncthreads()`) | ✅ `localBarrier()` | **Fully Supported** |
| Atomic Operations | ✅ `atomicAdd()` (int, long, float, double) | **Fully Supported** |
| Work Group Coordination | ✅ `groupIdx/y/z`, sizes | **Fully Supported** |
| Constant Memory | ⚠️ Pass as parameters | **Alternative Available** |
| Dynamic Parallelism | ❌ Not supported | **Not Available** |
| Warp-level Primitives | ❌ Not exposed | **Not Available** |

## Compilation and Execution

### Build Examples

```bash
cd tornado-examples
mvn clean install
```

### Run Individual Patterns

```bash
# Run with PTX/CUDA backend
tornado --jvm="-Ds0.t0.device=0:0" --printKernel \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd

# Run with OpenCL backend
tornado --jvm="-Ds0.t0.device=0:1" --printKernel \
    uk.ac.manchester.tornado.examples.cuda_patterns.histogram.Pattern13_HistogramAtomic

# View generated GPU code
tornado --jvm="-Ds0.t0.device=0:0" --printKernel --debug \
    uk.ac.manchester.tornado.examples.cuda_patterns.reduction.Pattern18_ParallelReduction
```

### Performance Profiling

```bash
# Enable profiling
tornado --enableProfiler console \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled
```

## Pattern Implementation Status

### ✅ Implemented (13 patterns)

1. ✅ Vector Addition
2. ✅ Matrix Addition
3. ✅ Image Blur
4. ✅ Matrix Multiply (Naive)
5. ✅ Matrix Multiply (Tiled)
6. ✅ Matrix Multiply (Coarsened)
7. ✅ 2D Convolution (Naive)
13. ✅ Histogram (Atomic)
14. ✅ Histogram (Privatized)
18. ✅ Parallel Reduction
20. ✅ Prefix Sum (Single Block)
22. ✅ SpMV (COO)
24. ✅ BFS (Naive)
27. ✅ Electrostatic Potential

### 🔄 Implementable (15 patterns)

8-9. 2D Convolution (Tiled) - *Can use local memory for tiles*
10-12. 3D Stencil Operations - *3D indexing available*
15-17. Histogram (Advanced) - *Thread coarsening possible*
19. Parallel Reduction (Coarsened) - *Load balancing optimization*
21. Prefix Sum (Multi-Block) - *Hierarchical coordination*
23. SpMV (ELL Format) - *Regular memory access pattern*
25-26. BFS (Frontier/Privatized) - *Work queue and local memory*
28-29. Electrostatic (Optimized) - *Micro-optimizations*

## Key Insights

### ✅ What Works Well in TornadoVM

1. **Shared Memory Optimization:** Local memory allocation and barriers work seamlessly
2. **Atomic Operations:** Full support for concurrent data structure updates
3. **Multi-dimensional Indexing:** Natural support for 1D/2D/3D grids
4. **Write-once Portability:** Same Java code runs on CUDA and OpenCL
5. **Type Safety:** Java's type system catches errors at compile time

### ⚠️ Limitations vs Native CUDA

1. **No Warp Primitives:** No direct access to warp shuffle, ballot, etc.
2. **No Dynamic Parallelism:** Cannot launch kernels from within kernels
3. **Limited Constant Memory:** Must pass constants as kernel parameters
4. **JVM Overhead:** Initial JIT compilation adds startup latency

### 🎯 Best Use Cases

- **Scientific Computing:** Matrix operations, simulations, molecular dynamics
- **Image Processing:** Filters, convolution, transformations
- **Data Analytics:** Reductions, histograms, aggregations
- **Graph Analytics:** BFS, shortest paths, PageRank
- **Machine Learning:** Matrix multiplications, convolutions (inference)

## Performance Tips

1. **Use Local Memory:** Cache frequently accessed data in shared memory
2. **Coalesce Memory Access:** Ensure threads access contiguous memory
3. **Thread Coarsening:** Increase work per thread for compute-bound kernels
4. **Minimize Atomics:** Use privatization to reduce atomic contention
5. **Tune Block Size:** Experiment with 128, 256, 512 threads per block
6. **Profile:** Use `--enableProfiler` to identify bottlenecks

## Resources

- **Original CUDA Patterns:** [github.com/philipfabianek/cuda-patterns](https://github.com/philipfabianek/cuda-patterns)
- **TornadoVM Docs:** [tornadovm.readthedocs.io](https://tornadovm.readthedocs.io)
- **PMPP Book:** "Programming Massively Parallel Processors" by Hwu et al.
- **TornadoVM Paper:** [TornadoVM: JVMCI/Graal Powered Acceleration](https://dl.acm.org/doi/10.1145/3313808.3313819)

## Contributing

To add new patterns:

1. Create Java file in appropriate subdirectory
2. Follow naming convention: `PatternXX_DescriptiveName.java`
3. Include comprehensive comments explaining CUDA concepts
4. Provide CPU validation code
5. Add performance notes and optimizations
6. Update this README

## License

Apache License 2.0 - Same as TornadoVM project

---

**Summary:** **All 29 CUDA patterns from the original repository can be implemented** using TornadoVM's Kernel Context API. This demonstrates TornadoVM's capability as a comprehensive GPU programming framework for Java, providing CUDA-like control with write-once, run-anywhere portability.
