package uk.ac.manchester.tornado.examples.compute;

import java.util.ArrayList;
import java.util.LongSummaryStatistics;
import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class MMwithBytesBenchmark {

    private static final float DELTA = 1e-4f;
    private static final int WARM_UP_ITERATIONS = 140;
    private static final int BENCHMARK_ITERATIONS = 120;
    private static final int LOCAL_WORK_GROUP_SIZE = 32;
    private static final Random random = new Random(42);

    // blockSize is a fundamental parameter of the quantization scheme
    private static final int QUANTIZATION_BLOCK_SIZE = 32;

    /**
     * Initializes a ByteArray with quantized weights and scales.
     */
    private static void initializeByteArrayWeights(ByteArray byteArrayWeights, int numTotalBlocks) {
        for (int i = 0; i < numTotalBlocks; i++) {
            int offset = i * (2 + QUANTIZATION_BLOCK_SIZE);
            byteArrayWeights.set(offset, (byte) 0);          // scale byte 1
            byteArrayWeights.set(offset + 1, (byte) 0);      // scale byte 2
            for (int j = 2; j < 2 + QUANTIZATION_BLOCK_SIZE; j++) {
                byteArrayWeights.set(offset + j, (byte) ((i * 3f) * 255 - 128)); // quantized values
            }
        }
    }

    private static void fillRandomData(FloatArray array) {
        for (int i = 0; i < array.getSize(); i++) {
            array.set(i, (float) Math.random());
        }
    }

    /**
     * Sequential implementation of matrix multiplication with bytes
     */
    public static void matmulSequential(ByteArray byteArrayWeights, FloatArray inputVector, FloatArray outputVector, int dim) {
        final int numOutputRows = outputVector.getSize();
        final int blockSize = QUANTIZATION_BLOCK_SIZE;
        final int bytesPerBlock = 2 + blockSize;
        final int blocksPerRow = dim / blockSize; // Critical calculation

        for (int idx = 0; idx < numOutputRows; idx++) {
            float result = 0f;

            for (int j = 0; j < dim; j++) {
                // Calculate which block this element belongs to
                int rowBlockStart = idx * blocksPerRow;
                int blockIndexInRow = j / blockSize;
                int blockIndex = rowBlockStart + blockIndexInRow;
                int withinBlockIndex = j % blockSize;
                int blockOffset = blockIndex * bytesPerBlock;

                // Read scale (float16) for this block
                int scaleByte1 = byteArrayWeights.get(blockOffset) & 0xFF;
                int scaleByte2 = byteArrayWeights.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                float scale = decodeFloat16(scaleFloat16);

                // Read quantized value
                byte quantized = byteArrayWeights.get(blockOffset + 2 + withinBlockIndex);

                // Dequantize and multiply
                result += (quantized * scale) * inputVector.get(j);
            }
            outputVector.set(idx, result);
        }
    }

    public static void matmulTornadoX(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int blockSize = 32;        // Number of quantized values per block
        final int bytesPerScale = 2;     // Size of FP16 scale
        // final int bytesPerBlockData = blockSize; // Data part of the block
        final int bytesPerBlock = bytesPerScale + blockSize; // Total size of one block (scale + data)

        // Number of blocks that constitute one full row of the conceptual matrix in 'thisx'.
        // This assumes dim1 is the width of the matrix and 'thisx' is structured accordingly.
        // If dim1 is not a multiple of blockSize, this assumes padding or specific layout handling
        // in the 'thisx' data structure to maintain this fixed blocksPerRow stride.
        final int blocksPerRow = dim1 / blockSize;

        int idx = context.globalIdx; // Current work-item, corresponds to the output row index.
        float result = 0f;

        // Base byte offset in 'thisx' for the start of all blocks belonging to the current row 'idx'.
        // This highlights the row-major access: we are calculating based on the 'idx'-th row.
        int baseOffsetForRowIdxBlocks = idx * blocksPerRow * bytesPerBlock;

        float currentBlockScale = 0f; // Holds the decoded scale for the current block

        for (int j = 0; j < dim1; j++) {
            // Check if we've moved to a new block within the row
            if (j % blockSize == 0) {
                int currentBlockIndexInRow = j / blockSize;
                int currentBlockOverallOffset = baseOffsetForRowIdxBlocks + (currentBlockIndexInRow * bytesPerBlock);

                // Read and decode the FP16 scale for this new block
                int scaleByte1 = thisx.get(currentBlockOverallOffset) & 0xFF;
                int scaleByte2 = thisx.get(currentBlockOverallOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                currentBlockScale = decodeFloat16(scaleFloat16);
            }

            // Calculate the offset for the quantized value within the current block
            // The data part of the block starts after the scale
            int currentBlockIndexInRowForData = j / blockSize; // or use the one from the if block
            int dataSegmentStartOffsetInBlock = baseOffsetForRowIdxBlocks + (currentBlockIndexInRowForData * bytesPerBlock) + bytesPerScale;
            int withinBlockDataIndex = j % blockSize;

            byte quantizedValue = thisx.get(dataSegmentStartOffsetInBlock + withinBlockDataIndex);

            // Dequantize using the current block's scale and accumulate
            result += (quantizedValue * currentBlockScale) * that.get(j);
        }
        out.set(idx, result);
    }

    /**
     * Optimized quantized matrix-vector multiplication combining row-major optimization
     * techniques with quantized byte array processing
     */
    public static void matmulQuantizedOptimized(
            KernelContext context,
            ByteArray quantizedWeights,    // Quantized weights in blocks
            FloatArray inputVector,        // Input vector
            FloatArray outputVector,       // Output vector
            int inputDim,                  // Input dimension
            int outputDim,                 // Output dimension
            int localWorkGroupSize         // Workgroup size
    ) {
        final int BLOCK_SIZE = 32;
        final int BYTES_PER_SCALE = 2;
        final int BYTES_PER_BLOCK = BYTES_PER_SCALE + BLOCK_SIZE;
        final int BLOCKS_PER_ROW = inputDim / BLOCK_SIZE;

        // One workgroup per output row (like optimized row-major)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if beyond output dimension
        if (rowId >= outputDim) {
            return;
        }

        // Allocate local memory for reduction and input caching
        float[] localPartialSums = context.allocateFloatLocalArray(localSize);
        float[] localInputCache = context.allocateFloatLocalArray(BLOCK_SIZE);

        int baseRowOffset = rowId * BLOCKS_PER_ROW * BYTES_PER_BLOCK;
        float totalSum = 0.0f;

        // Process blocks cooperatively within workgroup
        for (int blockIdx = 0; blockIdx < BLOCKS_PER_ROW; blockIdx++) {
            int blockOffset = baseRowOffset + (blockIdx * BYTES_PER_BLOCK);
            int inputBlockStart = blockIdx * BLOCK_SIZE;

            // Thread 0 loads and decodes the scale for this block
            float blockScale = 0.0f;
            if (localId == 0) {
                int scaleByte1 = quantizedWeights.get(blockOffset) & 0xFF;
                int scaleByte2 = quantizedWeights.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                blockScale = decodeFloat16(scaleFloat16);
            }

            // Broadcast scale to all threads in workgroup via local memory
            if (localId == 0) {
                localPartialSums[0] = blockScale; // Reuse array for scale broadcast
            }
            context.localBarrier();
            blockScale = localPartialSums[0];

            // Cooperatively load input vector chunk to local memory
            for (int j = localId; j < BLOCK_SIZE && (inputBlockStart + j) < inputDim; j += localSize) {
                localInputCache[j] = inputVector.get(inputBlockStart + j);
            }
            context.localBarrier();

            // Each thread processes subset of quantized values in this block
            float blockPartialSum = 0.0f;
            for (int j = localId; j < BLOCK_SIZE && (inputBlockStart + j) < inputDim; j += localSize) {
                byte quantizedValue = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + j);
                // Fused dequantization + multiply + accumulate
                blockPartialSum += (quantizedValue * blockScale) * localInputCache[j];
            }

            // Store partial sum in local memory for reduction
            localPartialSums[localId] = blockPartialSum;
            context.localBarrier();

            // Parallel reduction within workgroup for this block
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localPartialSums[localId] += localPartialSums[localId + stride];
                }
                context.localBarrier();
            }

            // Thread 0 accumulates the block result
            if (localId == 0) {
                totalSum += localPartialSums[0];
            }
            context.localBarrier();
        }

        // Thread 0 writes final result
        if (localId == 0) {
            outputVector.set(rowId, totalSum);
        }
    }

    /**
     * Alternative implementation with vectorized processing for better instruction-level parallelism
     */
    public static void matmulQuantizedVectorized(
            KernelContext context,
            ByteArray quantizedWeights,
            FloatArray inputVector,
            FloatArray outputVector,
            int inputDim,
            int outputDim,
            int localWorkGroupSize
    ) {
        final int BLOCK_SIZE = 32;
        final int BYTES_PER_SCALE = 2;
        final int BYTES_PER_BLOCK = BYTES_PER_SCALE + BLOCK_SIZE;
        final int BLOCKS_PER_ROW = inputDim / BLOCK_SIZE;
        final int VECTOR_WIDTH = 4; // Process 4 elements at once

        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        if (rowId >= outputDim) return;

        float[] localPartialSums = context.allocateFloatLocalArray(localSize);
        float[] localInputCache = context.allocateFloatLocalArray(BLOCK_SIZE);

        int baseRowOffset = rowId * BLOCKS_PER_ROW * BYTES_PER_BLOCK;
        float totalSum = 0.0f;

        for (int blockIdx = 0; blockIdx < BLOCKS_PER_ROW; blockIdx++) {
            int blockOffset = baseRowOffset + (blockIdx * BYTES_PER_BLOCK);
            int inputBlockStart = blockIdx * BLOCK_SIZE;

            // Load scale (thread 0 only)
            float blockScale = 0.0f;
            if (localId == 0) {
                int scaleByte1 = quantizedWeights.get(blockOffset) & 0xFF;
                int scaleByte2 = quantizedWeights.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                blockScale = decodeFloat16(scaleFloat16);
                localPartialSums[0] = blockScale;
            }
            context.localBarrier();
            blockScale = localPartialSums[0];

            // Cooperatively load input
            for (int j = localId; j < BLOCK_SIZE && (inputBlockStart + j) < inputDim; j += localSize) {
                localInputCache[j] = inputVector.get(inputBlockStart + j);
            }
            context.localBarrier();

            // Vectorized processing with loop unrolling
            float blockPartialSum = 0.0f;
            int j = localId * VECTOR_WIDTH;

            // Process 4 elements at once where possible
            while (j + VECTOR_WIDTH - 1 < BLOCK_SIZE && (inputBlockStart + j + VECTOR_WIDTH - 1) < inputDim) {
                // Load 4 quantized values
                byte q0 = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + j);
                byte q1 = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + j + 1);
                byte q2 = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + j + 2);
                byte q3 = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + j + 3);

                // Fused dequantize + multiply + accumulate (vectorized)
                blockPartialSum += (q0 * blockScale) * localInputCache[j];
                blockPartialSum += (q1 * blockScale) * localInputCache[j + 1];
                blockPartialSum += (q2 * blockScale) * localInputCache[j + 2];
                blockPartialSum += (q3 * blockScale) * localInputCache[j + 3];

                j += localSize * VECTOR_WIDTH;
            }

            // Handle remaining elements
            for (int k = localId; k < BLOCK_SIZE && (inputBlockStart + k) < inputDim; k += localSize) {
                if (k >= j) { // Skip already processed elements
                    byte quantizedValue = quantizedWeights.get(blockOffset + BYTES_PER_SCALE + k);
                    blockPartialSum += (quantizedValue * blockScale) * localInputCache[k];
                }
            }

            // Reduction
            localPartialSums[localId] = blockPartialSum;
            context.localBarrier();

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localPartialSums[localId] += localPartialSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                totalSum += localPartialSums[0];
            }
            context.localBarrier();
        }

        if (localId == 0) {
            outputVector.set(rowId, totalSum);
        }
    }

    // Helper method (same as before)
    private static float decodeFloat16(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        if (exp == 0x1F) {
            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        if (exp == 0) {
            if (frac == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            float result = frac * 5.9604645E-8f; // 2^-24 precalculated
            return sign == 0 ? result : -result;
        }

        float result = 1.0f + (frac / 1024.0f);
        if (exp < 15) {
            int shift = 15 - exp;
            result = (shift < 31) ? result / (1 << shift) : 0.0f;
        } else {
            int shift = exp - 15;
            result = (shift < 31) ? result * (1 << shift) : Float.POSITIVE_INFINITY;
        }

        return sign == 0 ? result : -result;
    }



    public static void main(String[] args) {
        System.out.println("Matrix Multiplication with Bytes Benchmark");
        System.out.println("=========================================");

        int dim = 128;
        int numRows = 1024;

        if (args.length >= 2) {
            try {
                dim = Integer.parseInt(args[0]);
                numRows = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Error parsing dimensions. Using defaults.");
            }
        }

        System.out.println("Configuration:");
        System.out.println("- Input dimension (dim): " + dim);
        System.out.println("- Number of output rows (numRows): " + numRows);
        System.out.println("- Quantization block size: " + QUANTIZATION_BLOCK_SIZE);
        System.out.println("- Local work group size: " + LOCAL_WORK_GROUP_SIZE);
        System.out.println("- Warmup iterations: " + WARM_UP_ITERATIONS);
        System.out.println("- Benchmark iterations: " + BENCHMARK_ITERATIONS);
        System.out.println();

        // Calculate total blocks correctly
        int blocksPerRow = dim / QUANTIZATION_BLOCK_SIZE;
        int totalBlocks = numRows * blocksPerRow;
        int byteArraySize = totalBlocks * (2 + QUANTIZATION_BLOCK_SIZE);

        System.out.println("Calculations for byteArrayWeights:");
        System.out.println("- Blocks per row: " + blocksPerRow);
        System.out.println("- Total blocks: " + totalBlocks);
        System.out.println("- Total ByteArray size: " + byteArraySize + " bytes");

        // Initialize arrays with correct sizes
        ByteArray byteArrayWeights = new ByteArray(byteArraySize);
        FloatArray inputVector = new FloatArray(dim);
        FloatArray outputSequential = new FloatArray(numRows);
        FloatArray outputTornado = new FloatArray(numRows);

        System.out.println("Initializing data...");
        initializeByteArrayWeights(byteArrayWeights, totalBlocks);
        fillRandomData(inputVector);
        outputSequential.init(0.0f);
        outputTornado.init(0.0f);

        ArrayList<Long> sequentialTimers = new ArrayList<>();
        ArrayList<Long> tornadoTimers = new ArrayList<>();

        System.out.println("Setting up TornadoVM execution...");
        //        WorkerGrid1D worker = new WorkerGrid1D(numRows);
        //        worker.setLocalWork(LOCAL_WORK_GROUP_SIZE, 1, 1);
        //        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        // Adjust these parameters for optimal performance on your GPU
        final int LOCAL_WORK_SIZE = 64;  // Multiple of 32 for NVIDIA (warp size)
        final int ITEMS_PER_THREAD = 2;  // Each thread processes multiple rows

//        // Calculate global size (fewer worker threads, each doing more work)
//        int globalWorkerThreads = (numRows + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
//        // Round up to multiple of local size for the worker grid
//        globalWorkerThreads = ((globalWorkerThreads + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE;
//
//        WorkerGrid1D worker = new WorkerGrid1D(globalWorkerThreads); // This is the number of *launched GPU threads*
//        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
//        // GridScheduler scheduler = new GridScheduler("s0.t0", worker); // This line might not be needed if using default scheduler with worker grid
//
//        worker.setLocalWork(LOCAL_WORK_SIZE, 1, 1);
//        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        // Updated worker grid setup
        int globalWorkerThreads = numRows; // One workgroup per output row
        int localWorkSize = 128; // Or 64, 32 depending on your GPU

        WorkerGrid1D worker = new WorkerGrid1D(globalWorkerThreads * localWorkSize);
        worker.setLocalWork(localWorkSize, 1, 1);
        GridScheduler scheduler = new GridScheduler("s0.t0", worker);

        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, byteArrayWeights, inputVector)
                .task("t0", MMwithBytesBenchmark::matmulQuantizedVectorized,
                        new KernelContext(), byteArrayWeights, inputVector, outputTornado,
                        dim, numRows, localWorkSize)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputTornado);

//        TaskGraph taskGraph = new TaskGraph("s0").transferToDevice(DataTransferMode.FIRST_EXECUTION, byteArrayWeights, inputVector).task("t0", MMwithBytesBenchmark::matmulTornadoX,
//                new KernelContext(), byteArrayWeights, inputVector, outputTornado, dim).transferToHost(DataTransferMode.EVERY_EXECUTION, outputTornado);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph);
        executionPlan.withGridScheduler(scheduler);

        System.out.println("Warming up sequential implementation...");
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            matmulSequential(byteArrayWeights, inputVector, outputSequential, dim);
        }

        System.out.println("Benchmarking sequential implementation...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            matmulSequential(byteArrayWeights, inputVector, outputSequential, dim);
            long end = System.nanoTime();
            sequentialTimers.add(end - start);
        }

        System.out.println("Warming up TornadoVM (KernelContext) implementation...");
        for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
            executionPlan.execute();
        }

        System.out.println("Benchmarking TornadoVM (KernelContext) implementation...");
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            executionPlan.execute();
            long end = System.nanoTime();
            tornadoTimers.add(end - start);
        }
        executionPlan.freeDeviceMemory();

        System.out.println("Validating results...");
        boolean isValid = true;
        float maxErrorTornado = 0.0f;

        for (int i = 0; i < numRows; i++) {
            float errorTornado = Math.abs(outputSequential.get(i) - outputTornado.get(i));
            maxErrorTornado = Math.max(maxErrorTornado, errorTornado);
            if (errorTornado > DELTA) {
                System.out.printf("[ERROR] Index %d: Expected %.6f, TornadoVM %.6f, Diff %.6f\n", i, outputSequential.get(i), outputTornado.get(i), errorTornado);
                isValid = false;
                if (i > 20)
                    break; // Print at most 20 errors
            }
        }

        if (isValid) {
            System.out.println("Validation PASSED ✓ (Max error: " + maxErrorTornado + ")");
        } else {
            System.out.println("Validation FAILED ✗ (Max error: " + maxErrorTornado + ")");
        }

        LongSummaryStatistics statsSeq = sequentialTimers.stream().mapToLong(Long::longValue).summaryStatistics();
        LongSummaryStatistics statsTornado = tornadoTimers.stream().mapToLong(Long::longValue).summaryStatistics();

        long flopsPerOutputElement = 2L * dim;
        long totalFlops = flopsPerOutputElement * numRows;

        double avgTimeSeqMs = statsSeq.getAverage() / 1_000_000.0;
        double avgTimeTornadoMs = statsTornado.getAverage() / 1_000_000.0;

        double seqGFlops = (totalFlops / (avgTimeSeqMs / 1000.0)) / 1_000_000_000.0;
        double tornadoGFlops = (totalFlops / (avgTimeTornadoMs / 1000.0)) / 1_000_000_000.0;

        System.out.println("\nPerformance Results:");
        System.out.println("====================");
        System.out.printf("Matrix (conceptual weights): %d x %d (output rows x dim/input_vector_size)\n", numRows, dim);
        System.out.printf("Total FLOPs per run: %.2f MFLOPs\n", totalFlops / 1_000_000.0);

        System.out.println("\nSequential Implementation:");
        System.out.printf("  Average time: %.3f ms\n", avgTimeSeqMs);
        System.out.printf("  Min time: %.3f ms\n", (double) statsSeq.getMin() / 1_000_000.0);
        System.out.printf("  Max time: %.3f ms\n", (double) statsSeq.getMax() / 1_000_000.0);
        System.out.printf("  Performance: %.2f GFLOP/s\n", seqGFlops);

        System.out.println("\nKernelContext Implementation (TornadoVM):");
        System.out.printf("  Average time: %.3f ms\n", avgTimeTornadoMs);
        System.out.printf("  Min time: %.3f ms\n", (double) statsTornado.getMin() / 1_000_000.0);
        System.out.printf("  Max time: %.3f ms\n", (double) statsTornado.getMax() / 1_000_000.0);
        System.out.printf("  Performance: %.2f GFLOP/s\n", tornadoGFlops);

        if (avgTimeTornadoMs > 0) {
            double speedupTornado = avgTimeSeqMs / avgTimeTornadoMs;
            System.out.printf("\nSpeedup: KernelContext vs Java Sequential %.2fx\n", speedupTornado);
        } else {
            System.out.println("\nSpeedup: N/A (TornadoVM average time is zero or too small)");
        }
    }
}