package uk.ac.manchester.tornado.unittests.llm;

import java.util.Random;

import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

import static org.junit.Assert.assertEquals;

public class TestCodeGenBug extends TornadoTestBase {

    private static final boolean DEBUG = true;

    /**
     * Calculates attention scores for each head and position
     */
    public static void calculateAttentionScores(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray query, FloatArray keyCache, FloatArray attScores, int kvDim, int kvMul,
            int headSize, int loff, int localWorkgroupSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Get the query vector offset for this head
        int queryOffset = h * headSize;

        // Attention scores offset for this head
        int attOffset = h * seqLen;
        int position = positionNlayer.get(0);

        for (int t = threadId; t <= position; t += blockDim) {
            // Get the key vector for this head and at this timestep
            int keyOffset = loff + t * kvDim + (h / kvMul) * headSize;

            // Calculate the attention score as the dot product of query and key
            float score = 0.0f;
            for (int i = 0; i < 8192; i++) {
                score += query.get(queryOffset + i) * keyCache.get(keyOffset + i);
            }

            // Scale by sqrt(head_size)
            score /= TornadoMath.sqrt(headSize);

            // Save the score to the attention buffer
            attScores.set(attOffset + t, score);
        }
    }

    /**
     * Sequential reference implementation of attention score calculation
     */
    public static void calculateAttentionScoresSequential(IntArray positionNlayer, int seqLen, FloatArray query, FloatArray keyCache, FloatArray attScores, int numHeads, int kvDim, int kvMul,
            int headSize, int loff) {
        int position = positionNlayer.get(0);

        for (int h = 0; h < numHeads; h++) {
            // Get the query vector offset for this head
            int queryOffset = h * headSize;

            // Attention scores offset for this head
            int attOffset = h * seqLen;

            for (int t = 0; t <= position; t++) {
                // Get the key vector for this head and at this timestep
                int keyOffset = loff + t * kvDim + (h / kvMul) * headSize;

                // Calculate the attention score as the dot product of query and key
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += query.get(queryOffset + i) * keyCache.get(keyOffset + i);
                }

                // Scale by sqrt(head_size)
                score /= Math.sqrt(headSize);

                // Save the score to the attention buffer
                attScores.set(attOffset + t, score);
            }
        }
    }

    private static GridScheduler getGridScheduler(int numHeads, int localSize) {
        WorkerGrid headWorker = new WorkerGrid1D(numHeads);
        headWorker.setGlobalWork(numHeads, 1, 1);
        headWorker.setLocalWork(localSize, 1, 1);

        // Configure grid scheduler with worker grid for the attention task
        GridScheduler gridScheduler = new GridScheduler();
        gridScheduler.addWorkerGrid("attention.scores", headWorker);
        return gridScheduler;
    }

    @Test
    public void testParallelAttentionScores() throws TornadoExecutionPlanException {
        final int numHeads = 2048;        // Number of attention heads
        final int headSize = 32;       // Size of each head
        final int kvMul = 1;           // KV multiplier (usually 1 for standard attention)
        final int seqLen = 16;         // Maximum sequence length
        final int position = 1024;        // Current position in sequence (0-based)
        final int kvDim = headSize;    // Key/value dimension
        final int loff = 0;            // Layer offset in the KV cache
        final int localSize = 4;       // Local workgroup size

        // Set up position array
        IntArray positionNlayer = new IntArray(1);
        positionNlayer.set(0, position);

        // Allocate arrays
        FloatArray query = new FloatArray(numHeads * headSize);
        FloatArray keyCache = new FloatArray(seqLen * kvDim * (numHeads / kvMul));

        // Set up attention score arrays for both parallel and sequential implementations
        FloatArray attScoresParallel = new FloatArray(numHeads * seqLen);
        FloatArray attScoresSequential = new FloatArray(numHeads * seqLen);

        // Initialize with random data
        Random random = new Random(42);
        for (int i = 0; i < query.getSize(); i++) {
            query.set(i, random.nextFloat() * 2 - 1);
        }

        for (int i = 0; i < keyCache.getSize(); i++) {
            keyCache.set(i, random.nextFloat() * 2 - 1);
        }

        attScoresParallel.init(0);
        attScoresSequential.init(0);

//        calculateAttentionScoresSequential(positionNlayer, seqLen, query, keyCache, attScoresSequential, numHeads, kvDim, kvMul, headSize, loff);

        // Create kernel context
        KernelContext context = new KernelContext();

        // Create task graph for parallel implementation
        TaskGraph taskGraph = new TaskGraph("attention")
                // Transfer all input arrays to device
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, positionNlayer, query, keyCache, attScoresParallel)

                // Calculate attention scores in parallel
                .task("scores", TestCodeGenBug::calculateAttentionScores, context, positionNlayer, seqLen, query, keyCache, attScoresParallel, kvDim, kvMul, headSize, loff, localSize)

                // Transfer result back to host
                .transferToHost(DataTransferMode.EVERY_EXECUTION, attScoresParallel);

        // Create worker grid for attention heads
        GridScheduler gridScheduler = getGridScheduler(numHeads, localSize);

        // Create execution plan
        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        // Execute the task graph
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(gridScheduler).execute();
        }

        for (int h = 0; h < numHeads; h++) {
            for (int t = 0; t <= position; t++) {
                int idx = h * seqLen + t;
                assertEquals("Attention score mismatch at head " + h + ", position " + t, attScoresSequential.get(idx), attScoresParallel.get(idx), Math.abs(attScoresSequential.get(
                        idx) * 0.01f) + 1e-5f);
            }
        }
    }
}
