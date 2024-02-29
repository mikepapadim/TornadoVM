package uk.ac.manchester.tornado.api.types.tensors;

import java.lang.foreign.MemorySegment;

import uk.ac.manchester.tornado.api.types.arrays.LongArray;

public non-sealed class LongTensor extends Tensor {

    private final LongArray tensorData;

    public LongTensor(Shape shape) {
        super(shape, DType.INT64);
        this.tensorData = new LongArray(shape.getSize());
    }

    public LongArray getTensorData() {
        return tensorData;
    }

    public MemorySegment getAsMemorySegment() {
        return tensorData.getSegment();
    }

    @Override
    public int getSize() {
        return 0;
    }

    @Override
    public MemorySegment getSegment() {
        return null;
    }

    @Override
    public long getNumBytesOfSegment() {
        return 0;
    }

    @Override
    public long getNumBytesWithoutHeader() {
        return 0;
    }

    @Override
    public void clear() {

    }

    @Override
    public int getElementSize() {
        return 0;
    }

    @Override
    public void reshape(Shape newShape) {
    }

    @Override
    public void slice(Shape slice) {
    }
}
