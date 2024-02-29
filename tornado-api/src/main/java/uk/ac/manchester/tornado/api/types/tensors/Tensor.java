package uk.ac.manchester.tornado.api.types.tensors;

import java.lang.foreign.MemorySegment;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

public abstract sealed class Tensor extends TornadoNativeArray permits FloatTensor, HalfFloatTensor, LongTensor {
    private final Shape shape;

    private final DType dtype;

    protected Tensor(Shape shape, DType dtype) {
        this.shape = shape;
        this.dtype = dtype;
    }

    public Shape getShape() {
        return shape;
    }

    public DType getDtype() {
        return dtype;
    }

    public abstract int getSize();

    public abstract MemorySegment getSegment();

    public abstract long getNumBytesOfSegment();

    public abstract long getNumBytesWithoutHeader();

    public abstract void clear();

    public abstract int getElementSize();

    public abstract void reshape(Shape newShape);

    public abstract void slice(Shape slice);
}
