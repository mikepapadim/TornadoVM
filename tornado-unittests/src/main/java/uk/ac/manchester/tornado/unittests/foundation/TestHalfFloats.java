package uk.ac.manchester.tornado.unittests.foundation;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.unittests.common.TornadoTestBase;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * <p>
 * How to test?
 * </p>
 * <code>
 * tornado-test --igv --threadInfo --printKernel --printBytecodes -V uk.ac.manchester.tornado.unittests.foundation.TestHalfFloats#testConvertFP32toFP16
 * </code>
 */
public class TestHalfFloats extends TornadoTestBase {

    public static void convertFP32toFP16(KernelContext context,  FloatArray wrapX, HalfFloatArray x) {
        int i = context.globalIdx;
        float valInput = wrapX.get(i);
        HalfFloat val = new HalfFloat(valInput);
        x.set(i,val);
    }

    public static void convertFP32toFP32(KernelContext context,  FloatArray wrapX, FloatArray x) {
        int i = context.globalIdx;
        float valInput = wrapX.get(i);
        x.set(i,valInput);
    }


    public static void parallelCopy(FloatArray wrapX, HalfFloatArray x) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            float valInput = wrapX.get(i);
            HalfFloat val = new HalfFloat(valInput);
            x.set(i,val);
        }
    }

    @Test
    public void testConvertFP32toFP16() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(1024);
        HalfFloatArray y = new HalfFloatArray(1024);

//        x.init(new Random().nextFloat());
//        x.init(2f);
        x.init(2f);
        y.clear();

        KernelContext context = new KernelContext();

        TaskGraph tg = new TaskGraph("graph");
        tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, x);
        tg.task("convert", TestHalfFloats::convertFP32toFP16,context,x,y);
//        tg.task("convert", TestHalfFloats::parallelCopy,x,y);
        tg.transferToHost(DataTransferMode.EVERY_EXECUTION,y);

        ImmutableTaskGraph immutableTaskGraph = tg.snapshot();
        WorkerGrid workerGrid  = new WorkerGrid1D(1024);
        workerGrid.setLocalWork(32,1,1);

        GridScheduler scheduler = new GridScheduler("graph.convert", workerGrid);

        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(scheduler).execute();
        }

        for (int i = 0; i < 1024; i++) {
            assertEquals(x.get(i), y.get(i).getFloat32(), 0.001f);
        }
    }

    @Test
    public void testConvertFP32toFP32() throws TornadoExecutionPlanException {
        FloatArray x = new FloatArray(1024);
        FloatArray y = new FloatArray(1024);

        //        x.init(new Random().nextFloat());
        //        x.init(2f);
        x.init(2f);
        y.clear();

        KernelContext context = new KernelContext();

        TaskGraph tg = new TaskGraph("graph");
        tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, x);
        tg.task("convert", TestHalfFloats::convertFP32toFP32,context,x,y);
        //        tg.task("convert", TestHalfFloats::parallelCopy,x,y);
        tg.transferToHost(DataTransferMode.EVERY_EXECUTION,y);

        ImmutableTaskGraph immutableTaskGraph = tg.snapshot();
        WorkerGrid workerGrid  = new WorkerGrid1D(1024);
        workerGrid.setLocalWork(32,1,1);

        GridScheduler scheduler = new GridScheduler("graph.convert", workerGrid);

        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph)) {
            executionPlan.withGridScheduler(scheduler).execute();
        }

        for (int i = 0; i < 1024; i++) {
            assertEquals(x.get(i), y.get(i), 0.001f);
        }
    }

}
