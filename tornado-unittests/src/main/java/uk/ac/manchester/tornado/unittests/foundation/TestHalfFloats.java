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



//    __kernel void convertFP32toFP16(__global long *_kernel_context, __constant uchar *_constant_region, __local uchar *_local_region, __global int *_atomics, __global uchar *wrapX, __global uchar *x)
//    {
//        int i_3, i_2, i_8;
//        long l_5, l_10, l_9, l_4;
//        ulong ul_1, ul_0, ul_6, ul_11;
//        half half_7;
//
//        // BLOCK 0
//        ul_0  =  (ulong) wrapX;
//        ul_1  =  (ulong) x;
//        i_2  =  get_global_id(0);
//        i_3  =  i_2 + 4;
//        l_4  =  (long) i_3;
//        l_5  =  l_4 << 2;
//        ul_6  =  ul_0 + l_5;
//        half_7  =  *((__global half *) ul_6); < ------------- Issue here -> it would be loading a float
 //        i_8  =  i_2 + 8;                   <-- No need -> as read and write are from index (i)
//        l_9  =  (long) i_8;                 <--
//        l_10  =  l_9 << 1;                  <--
//        ul_11  =  ul_1 + l_10;              <-- so it should be here -> ul11 =  ul_1 + l_5
//        *((__global half *) ul_11)  =  half_7;
//        return;
//    }  //  kernel

// TODO: The issue seems to be in TornadoHalfFloatReplacement -> lines 150++

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
