package uk.ac.manchester.tornado.api.types.tensor;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Vector;

public abstract class AbstractTensor<V extends Vector<?>, T extends Number, A> implements AutoCloseable {
    protected final int[] shape;
    protected final DType dType;
    protected final AbstractTensor[] sliceCache;
    protected final int capacity;
    //    private volatile TensorCache originCache = null;

    protected AbstractTensor(DType dType, int[] shape, boolean cacheSlices) {
        //        Preconditions.checkArgument(shape != null && shape.length > 0);
        this.dType = dType;
        this.shape = shape;

        int c = 1;
        for (int i = 0; i < shape.length; i++)
            c *= shape[i];

        this.capacity = c;

        this.sliceCache = cacheSlices ? new AbstractTensor[shape[0]] : null;
    }

    protected abstract AbstractTensor make(int... shape);

    protected abstract AbstractTensor make(int heapOffset, int heapLength, int[] shape, boolean cacheSlices);

    final public int dims() {
        return shape.length;
    }

    final public int[] shape() {
        return shape;
    }

    final public int size() {
        return capacity;
    }

    public abstract float get(int... dims);

    public abstract void set(float v, int... dims);

    public AbstractTensor slice(int... dims) {
        return slice(false, dims);
    }

    public AbstractTensor slice(boolean cacheInnerSlice, int... dims) {
        //        Preconditions.checkArgument(dims.length < shape.length, "Too many dimensions specified for tensor");

        if (dims.length == 1 && sliceCache != null && sliceCache[dims[0]] != null)
            return sliceCache[dims[0]];

        int[] slicedShape = Arrays.copyOfRange(shape, dims.length, shape.length);

        int totalOffset = 0;
        for (int d = 0; d <= dims.length - 1; d++) {
            int offset = 1;
            for (int i = shape.length - 1; i > d; i--) { // factor scaling of each dim shape
                offset *= shape[i];
            }

            totalOffset += dims[d] * offset;
        }

        int length = 1;
        for (int i = 0; i < slicedShape.length; i++)
            length *= slicedShape[i];

        AbstractTensor r = this.make(totalOffset, length, slicedShape, cacheInnerSlice);
        if (dims.length == 1 && sliceCache != null)
            sliceCache[dims[0]] = r;

        return r;
    }

    public AbstractTensor[] split(int numChunks, int dim) {
        AbstractTensor[] chunks = new AbstractTensor[numChunks];
        int innerLength = this.shape[dim] / numChunks;

        if (Integer.bitCount(innerLength) != 1) {
            throw new IllegalStateException("Chunks must be power of 2");
        }

        int[] newShape = Arrays.copyOf(this.shape, shape.length);
        newShape[dim] = innerLength;
        int newCapacity = 1;
        for (int i = 0; i < newShape.length; i++) {
            newCapacity *= newShape[i];
        }

        for (int i = 0; i < numChunks; i++) {
            chunks[i] = this.make(i * newCapacity, newCapacity, newShape, true);
        }

        return chunks;
    }

    final public boolean iterate(int[] cursor) {
        int[] shape = shape();
        //        Preconditions.checkArgument(cursor.length == shape.length);

        for (int i = cursor.length - 1; i >= 0; i--) {
            //            Preconditions.checkArgument(cursor[i] >= 0 && cursor[i] < shape[i]);
            if (cursor[i] + 1 < shape[i]) {
                cursor[i]++;
                break;
            } else {
                cursor[i] = 0;
                if (i == 0)
                    return false;
            }
        }

        return true;
    }

    final public int getOffset(int[] dims) {
        int[] shape = shape();
        //        Preconditions.checkArgument(dims.length == shape.length, "Method requires all dimensions specified");
        int totalOffset = 0;

        for (int d = 0; d < dims.length - 1; d++) { // Stop before last dimension
            int offset = 1;
            for (int i = shape.length - 1; i > d; i--) { // factor scaling of each dim shape
                offset *= shape[i];
            }

            totalOffset += dims[d] * offset;
        }

        return totalOffset + dims[shape.length - 1];
    }

    final public AbstractTensor transpose() {
        int[] shape = shape();
        int[] tshape = new int[dims()];

        for (int i = 0; i < tshape.length; i++)
            tshape[i] = shape[shape.length - i - 1];

        AbstractTensor tt = this.make(tshape);
        int[] cursor = new int[dims()];
        int[] tcursor = new int[dims()];
        do {
            float v = this.get(cursor);

            for (int i = 0; i < tcursor.length; i++)
                tcursor[i] = cursor[cursor.length - i - 1];

            tt.set(v, tcursor);
        } while (iterate(cursor));

        return tt;
    }

    final public DType dType() {
        return dType;
    }

    public abstract A getArray();

    public abstract int getArrayOffset(int offset);

    public abstract V getVector(VectorSpecies<T> species, int offset);

    public abstract void intoTensor(V vector, int offset);

    public void intoTensor(V vector, int offset, VectorMask<T> mask) {
        throw new UnsupportedOperationException();
    }

    public abstract MemorySegment getMemorySegment();

    public abstract int getMemorySegmentOffset(int offset);

    public abstract boolean hasMemorySegment();

    public abstract void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length);

    public abstract void clear();

    public void close() {
        if (originCache != null)
            originCache.release(this);
    }

    void setOwnerCache(TensorCache cache) {
        this.originCache = cache;
    }

    public AbstractTensor quantize(DType dType) {
        if (this.dims() != 2 || this.dType == dType)
            return this;

        return switch (dType) {
            //            case Q4 -> new Q4ByteBufferTensor(this);
            //            case I8 -> new Q8ByteBufferTensor(this);
            //            case F32 -> new FloatBufferTensor(this);
            //            case BF16 -> new BFloat16BufferTensor(this);
            default -> this;
        };
    }

    //    public TensorInfo save(FileChannel out) throws IOException {
    //        ByteBuffer bb = getMemorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
    //
    //        long startOffset = out.position();
    //
    //        out.write(bb);
    //
    //        long[] lshape = new long[shape.length];
    //        for (int i = 0; i < shape.length; i++)
    //            lshape[i] = shape[i];
    //
    //        return new TensorInfo(dType, lshape, new long[] { startOffset, out.position() });
    //    }

    public void debug(String id) {
        if (false) {
            double tmp = 0.0;
            for (int i = 0; i < size(); i++) {
                tmp += get(i);
            }
            System.out.println(String.format("%s = %.5f", id, tmp));
        }
    }
}