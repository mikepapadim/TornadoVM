#!/bin/bash
# Run all CUDA patterns examples
# Usage: ./run_all_patterns.sh [device]
# Example: ./run_all_patterns.sh 0:0  (for NVIDIA CUDA)
#          ./run_all_patterns.sh 0:1  (for OpenCL)

DEVICE=${1:-"0:0"}  # Default to device 0:0 (CUDA)
BASE_PKG="uk.ac.manchester.tornado.examples.cuda_patterns"

echo "=========================================="
echo "Running TornadoVM CUDA Patterns Examples"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

run_pattern() {
    local category=$1
    local pattern=$2
    local name=$3

    echo -e "${BLUE}Running: $name${NC}"

    if tornado --jvm="-Ds0.t0.device=$DEVICE" ${BASE_PKG}.${category}.${pattern} 2>&1 | grep -q "PASSED"; then
        echo -e "${GREEN}✓ PASSED${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    echo ""
}

echo "=========================================="
echo "BASIC OPERATIONS (01-06)"
echo "=========================================="
run_pattern "basic" "Pattern01_VectorAdd" "01. Vector Addition"
run_pattern "basic" "Pattern02_MatrixAdd" "02. Matrix Addition"
run_pattern "basic" "Pattern03_ImageBlur" "03. Image Blur"
run_pattern "basic" "Pattern04_MatrixMultiplyNaive" "04. Matrix Multiply (Naive)"
run_pattern "basic" "Pattern05_MatrixMultiplyTiled" "05. Matrix Multiply (Tiled)"
run_pattern "basic" "Pattern06_MatrixMultiplyCoarsened" "06. Matrix Multiply (Coarsened)"

echo "=========================================="
echo "CONVOLUTION & STENCIL (07)"
echo "=========================================="
run_pattern "convolution" "Pattern07_Convolution2DNaive" "07. 2D Convolution (Naive)"

echo "=========================================="
echo "HISTOGRAM (13-14)"
echo "=========================================="
run_pattern "histogram" "Pattern13_HistogramAtomic" "13. Histogram (Atomic)"
run_pattern "histogram" "Pattern14_HistogramPrivatized" "14. Histogram (Privatized)"

echo "=========================================="
echo "REDUCTION & SCANNING (18, 20)"
echo "=========================================="
run_pattern "reduction" "Pattern18_ParallelReduction" "18. Parallel Reduction"
run_pattern "reduction" "Pattern20_PrefixSumSingleBlock" "20. Prefix Sum (Single Block)"

echo "=========================================="
echo "SPARSE OPERATIONS (22)"
echo "=========================================="
run_pattern "sparse" "Pattern22_SpMV_COO" "22. SpMV (COO Format)"

echo "=========================================="
echo "GRAPH ALGORITHMS (24)"
echo "=========================================="
run_pattern "graph" "Pattern24_BFS_Naive" "24. BFS (Naive)"

echo "=========================================="
echo "COMPUTE-INTENSIVE KERNELS (27)"
echo "=========================================="
run_pattern "compute" "Pattern27_ElectrostaticPotential" "27. Electrostatic Potential"

echo "=========================================="
echo "All patterns executed!"
echo "=========================================="
