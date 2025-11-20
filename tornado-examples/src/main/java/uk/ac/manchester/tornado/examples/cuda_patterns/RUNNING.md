# Running CUDA Patterns Examples

## Prerequisites

Make sure TornadoVM is installed and the `tornado` command is in your PATH:

```bash
# Check installation
tornado --version
tornado --devices
```

## Quick Start

### Run a Single Pattern

```bash
# Basic example
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd

# Matrix multiplication with tiling
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled

# Histogram with atomics
tornado uk.ac.manchester.tornado.examples.cuda_patterns.histogram.Pattern13_HistogramAtomic
```

### Run All Patterns

```bash
cd tornado-examples/src/main/java/uk/ac/manchester/tornado/examples/cuda_patterns

# Run on default device
./run_all_patterns.sh

# Run on specific device
./run_all_patterns.sh 0:0  # CUDA device
./run_all_patterns.sh 0:1  # OpenCL device
```

## Device Selection

### List Available Devices

```bash
tornado --devices
```

Output example:
```
Number of Tornado drivers: 2
Driver: PTX
  Total number of PTX devices: 1
  Tornado device=0:0
    PTX -- NVIDIA GeForce RTX 3080
        Global Memory Size: 10 GB

Driver: OpenCL
  Total number of OpenCL devices: 2
  Tornado device=0:1
    OPENCL -- NVIDIA CUDA -- NVIDIA GeForce RTX 3080
```

### Select Specific Device

```bash
# Use CUDA/PTX backend (device 0:0)
tornado --jvm="-Ds0.t0.device=0:0" \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd

# Use OpenCL backend (device 0:1)
tornado --jvm="-Ds0.t0.device=0:1" \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd
```

## Useful Flags

### 1. View Generated GPU Code

```bash
# See the generated PTX (CUDA) or OpenCL kernel
tornado --printKernel \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled
```

**Output:** Shows the compiled GPU kernel code (PTX assembly or OpenCL C)

### 2. Performance Profiling

```bash
# Enable profiler to see execution times
tornado --enableProfiler console \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled
```

**Output:**
```
task info: s0.t0
        backend                          : PTX
        device                           : NVIDIA GeForce RTX 3080
        dims                             : 2
        kernel-time (ns)                 : 45123
        data-transfers-time (ns)         : 12345
        ...
```

### 3. Debug Mode

```bash
# Enable debug output
tornado --debug \
    uk.ac.manchester.tornado.examples.cuda_patterns.histogram.Pattern13_HistogramAtomic
```

### 4. Thread Configuration

```bash
# Print thread configuration info
tornado --threadInfo \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled
```

### 5. Combined Flags

```bash
# Profile + View kernel + Debug
tornado --enableProfiler console --printKernel --debug \
    uk.ac.manchester.tornado.examples.cuda_patterns.reduction.Pattern18_ParallelReduction
```

## Pattern Categories

### Basic Operations (01-06)

```bash
# Vector addition
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd

# Matrix addition
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern02_MatrixAdd

# Image blur (stencil)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern03_ImageBlur

# Matrix multiply - naive
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern04_MatrixMultiplyNaive

# Matrix multiply - tiled (optimized with shared memory)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled

# Matrix multiply - coarsened (thread coarsening)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern06_MatrixMultiplyCoarsened
```

### Convolution (07)

```bash
# 2D convolution
tornado uk.ac.manchester.tornado.examples.cuda_patterns.convolution.Pattern07_Convolution2DNaive
```

### Histogram (13-14)

```bash
# Histogram with atomic operations
tornado uk.ac.manchester.tornado.examples.cuda_patterns.histogram.Pattern13_HistogramAtomic

# Histogram with privatization (optimized)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.histogram.Pattern14_HistogramPrivatized
```

### Reduction & Scanning (18, 20)

```bash
# Parallel reduction (sum)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.reduction.Pattern18_ParallelReduction

# Prefix sum (scan)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.reduction.Pattern20_PrefixSumSingleBlock
```

### Sparse Operations (22)

```bash
# Sparse matrix-vector multiplication (COO format)
tornado uk.ac.manchester.tornado.examples.cuda_patterns.sparse.Pattern22_SpMV_COO
```

### Graph Algorithms (24)

```bash
# Breadth-First Search
tornado uk.ac.manchester.tornado.examples.cuda_patterns.graph.Pattern24_BFS_Naive
```

### Compute-Intensive Kernels (27)

```bash
# Electrostatic potential calculation
tornado uk.ac.manchester.tornado.examples.cuda_patterns.compute.Pattern27_ElectrostaticPotential
```

## Performance Testing

### Compare Backends (CUDA vs OpenCL)

```bash
# Run on CUDA
echo "CUDA/PTX Backend:"
tornado --jvm="-Ds0.t0.device=0:0" --enableProfiler console \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled

# Run on OpenCL
echo "OpenCL Backend:"
tornado --jvm="-Ds0.t0.device=0:1" --enableProfiler console \
    uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled
```

### Benchmark a Pattern

Create a simple benchmark script:

```bash
#!/bin/bash
PATTERN="uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern05_MatrixMultiplyTiled"

echo "Running benchmark (10 iterations)..."
for i in {1..10}; do
    echo "Iteration $i"
    tornado --enableProfiler console --jvm="-Ds0.t0.device=0:0" $PATTERN 2>&1 | grep "kernel-time"
done
```

## Troubleshooting

### Pattern Fails to Run

1. **Check device availability:**
   ```bash
   tornado --devices
   ```

2. **Verify compilation:**
   ```bash
   cd tornado-examples
   mvn clean compile
   ```

3. **Check for errors:**
   ```bash
   tornado --debug uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd
   ```

### Out of Memory Errors

Some patterns use large arrays. If you encounter OOM errors:

1. **Reduce problem size** (edit the pattern's main method)
2. **Use smaller JVM heap:**
   ```bash
   tornado --jvm="-Xmx4g" uk.ac.manchester.tornado.examples.cuda_patterns...
   ```

### Slow First Execution

The first execution includes JIT compilation. Subsequent runs are faster:

```bash
# Run twice to see performance after warmup
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd
tornado uk.ac.manchester.tornado.examples.cuda_patterns.basic.Pattern01_VectorAdd
```

## Expected Output

Each pattern prints:
- Test configuration (sizes, parameters)
- Validation result (PASSED/FAILED)
- Sample output values

Example output:
```
Pattern 05 - Matrix Multiplication (Tiled): PASSED
Successfully computed 512×512 × 512×512 = 512×512
Using tile size: 16×16
```

## Next Steps

1. **Modify parameters:** Edit pattern files to test different sizes
2. **Add new patterns:** Implement additional CUDA patterns (08-12, 15-17, etc.)
3. **Performance tuning:** Experiment with block sizes and tiling factors
4. **Cross-backend testing:** Compare CUDA vs OpenCL performance

## References

- **TornadoVM Documentation:** https://tornadovm.readthedocs.io
- **Original CUDA Patterns:** https://github.com/philipfabianek/cuda-patterns
- **Pattern README:** See `README.md` in this directory
- **Implementation Guide:** See `CUDA_PATTERNS_IMPLEMENTATION.md` in project root
