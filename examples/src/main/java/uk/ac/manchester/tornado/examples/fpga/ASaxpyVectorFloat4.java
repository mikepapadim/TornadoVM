package uk.ac.manchester.tornado.examples.fpga;


//import org.junit.Ignore;
//import org.junit.Test;
//import static org.junit.Assert.assertEquals;


import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.collections.types.*;
import uk.ac.manchester.tornado.runtime.api.*;


public class ASaxpyVectorFloat4 {


    public static void saxpy(float alpha, VectorFloat4 x, VectorFloat4 y, VectorFloat4 b) {

        for (@Parallel int i = 0; i < x.getLength(); i++) {
           // y = alpha * x[i] + b[i];
            Float4 temp = Float4.mult(x.get(i),alpha);
            y.set(i, Float4.add(temp,b.get(i)));
        }
    }


//  for (@Parallel int i = 0; i < a.getLength(); i++) {
//            results.set(i, Float3.add(a.get(i), b.get(i)));
//        }

    public static void main(String[] args) {
        int numElements = Integer.parseInt(args[0]);

//        numElements = numElements/4;
        float alpha = 2f;
        VectorFloat4 xx = new VectorFloat4(numElements);
        VectorFloat4 yy = new VectorFloat4(numElements);
        VectorFloat4 bb = new VectorFloat4(numElements);
        VectorFloat4 results = new VectorFloat4(numElements);


        xx.fill(450f);
        yy.fill(0);
        bb.fill(20);


        TaskSchedule s0 = new TaskSchedule("s0").task("t0", ASaxpyVectorFloat4::saxpy, alpha, xx, yy,bb).streamOut(yy);

        for (int idx = 0; idx < 10; idx++) {
            s0.execute();

            long start = System.nanoTime();
            saxpy(alpha,xx, results,bb);
            long stop = System.nanoTime();

            long seqTime = stop -start;
            //System.out.println("Sequential time: " + seqTime + "\n");

            System.out.println("Checking result");
            boolean wrongResult = false;

            for (int i = 0; i < yy.getLength(); i++) {

                if(Math.abs(yy.get(i).getW() - results.get(i).getW()) > 0.1 ){
                    wrongResult = true;
                } else if(Math.abs(yy.get(i).getX() - results.get(i).getX()) > 0.1 ){
                    wrongResult = true;
                } if(Math.abs(yy.get(i).getZ() - results.get(i).getZ()) > 0.1 ){
                    wrongResult = true;
                } if(Math.abs(yy.get(i).getY()- results.get(i).getY()) > 0.1 ){
                    wrongResult = true;
                }
            }

            if (!wrongResult) {
                System.out.println("Test success");
            } else {
                System.out.println("Result is wrong");
            }

        }
    }

}
