package uk.ac.manchester.tornado.drivers.cuda.mm;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.api.exceptions.TornadoMemoryException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.mm.ObjectBuffer;
import uk.ac.manchester.tornado.drivers.cuda.CUDADeviceContext;
import uk.ac.manchester.tornado.runtime.common.Tornado;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.shouldNotReachHere;
import static uk.ac.manchester.tornado.runtime.TornadoCoreRuntime.getVMConfig;
import static uk.ac.manchester.tornado.runtime.common.RuntimeUtilities.humanReadableByteCount;
import static uk.ac.manchester.tornado.runtime.common.Tornado.*;

public abstract class CUDAArrayWrapper<T> implements ObjectBuffer {
    private static final int ARRAY_ALIGNMENT = Integer.parseInt(getProperty("tornado.cuda.array.align", "128"));

    private int arrayHeaderSize;
    private long bytesToAllocate;
    private long bufferOffset;
    protected CUDADeviceContext deviceContext;
    private JavaKind kind;
    private int arrayLengthOffset;
    private boolean onDevice;
    private boolean isFinal;

    public CUDAArrayWrapper(CUDADeviceContext deviceContext, JavaKind kind, boolean isFinal) {
        this.deviceContext = deviceContext;
        this.kind = kind;
        this.isFinal = isFinal;

        bufferOffset = -1;
        arrayHeaderSize = getVMConfig().getArrayBaseOffset(kind);
        arrayLengthOffset = getVMConfig().arrayOopDescLengthOffset();
    }

    @SuppressWarnings("unchecked")
    private T cast(Object array) {
        try {
            return (T) array;
        } catch (Exception | Error e) {
            shouldNotReachHere("[ERROR] Unable to cast object: " + e.getMessage());
        }
        return null;
    }

    @Override
    public long toBuffer() {
        return deviceContext.getMemoryManager().toBuffer();
    }

    @Override
    public long getBufferOffset() { return bufferOffset; }

    @Override
    public long toAbsoluteAddress() {
        return deviceContext.getMemoryManager().toAbsoluteDeviceAddress(bufferOffset);
    }

    @Override
    public long toRelativeAddress() { return bufferOffset; }

    @Override
    public void read(Object reference) {
        read(reference, 0, null, false);
    }

    @Override
    public int read(Object reference, long hostOffset, int[] events, boolean useDeps) {
        T array = cast(reference);
        if (array == null) throw new TornadoRuntimeException("[ERROR] output data is NULL");

        if (VALIDATE_ARRAY_HEADERS) {
            if (validateArrayHeader(array)) {
                return readArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
            } else {
                shouldNotReachHere("Array header is invalid");
            }
        } else {
            return readArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
        }
        return -1;
    }

    private boolean validateArrayHeader(T array) {
        final CUDAByteBuffer header = prepareArrayHeader();
        header.read();
        final int numElements = header.getInt(arrayLengthOffset);
        final boolean valid = numElements == Array.getLength(array);
        if (!valid) {
            fatal("Array: expected=%d, got=%d", Array.getLength(array), numElements);
            header.dump(8);
        }
        return valid;
    }

    private CUDAByteBuffer prepareArrayHeader() {
        final CUDAByteBuffer header = deviceContext.getMemoryManager().getSubBuffer((int) bufferOffset, arrayHeaderSize);
        header.buffer.clear();
        header.buffer.position(header.buffer.capacity());
        return header;
    }

    @Override
    public void write(Object reference) {
        final T array = cast(reference);
        if (array == null) {
            throw new TornadoRuntimeException("[ERROR] data is NULL");
        }
        buildArrayHeader(Array.getLength(array)).write();
        // TODO: Writing with offset != 0
        writeArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, 0, null);
        onDevice = true;
    }

    private CUDAByteBuffer buildArrayHeader(int arraySize) {
        final CUDAByteBuffer header = deviceContext.getMemoryManager().getSubBuffer((int) bufferOffset, arrayHeaderSize);
        header.buffer.clear();
        int index = 0;
        while (index < arrayLengthOffset) {
            header.buffer.put((byte) 0);
            index++;
        }
        header.buffer.putInt(arraySize);
        return header;
    }

    @Override
    public int enqueueRead(final Object value, long hostOffset, final int[] events, boolean useDeps) {
        final T array = cast(value);
        if (array == null) {
            throw new TornadoRuntimeException("[ERROR] output data is NULL");
        }
        final int returnEvent;
        if (isFinal) {
            returnEvent = enqueueReadArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
        } else {
            returnEvent = enqueueReadArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
        }
        return useDeps ? returnEvent : -1;
    }

    @Override
    public List<Integer> enqueueWrite(Object reference, long batchSize, long hostOffset, int[] events, boolean useDeps) {
        final T array = cast(reference);
        ArrayList<Integer> listEvents = new ArrayList<>();

        if (array == null) {
            throw new TornadoRuntimeException("ERROR] Data to be copied is NULL");
        }
        final int returnEvent;
        if (isFinal && onDevice) {
            returnEvent = enqueueWriteArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
        } else {
            // We first write the header for the object and then we write actual
            // buffer
            final int headerEvent;
            if (batchSize <= 0) {
                headerEvent = buildArrayHeader(Array.getLength(array)).enqueueWrite((useDeps) ? events : null);
            } else {
                headerEvent = buildArrayHeaderBatch(batchSize).enqueueWrite((useDeps) ? events : null);
            }
            returnEvent = enqueueWriteArrayData(toBuffer() + bufferOffset + arrayHeaderSize, bytesToAllocate - arrayHeaderSize, array, hostOffset, (useDeps) ? events : null);
            onDevice = true;
            // returnEvent = deviceContext.enqueueMarker(internalEvents);

            listEvents.add(headerEvent);
            listEvents.add(returnEvent);
        }
        return useDeps ? listEvents : null;
    }

    private CUDAByteBuffer buildArrayHeaderBatch(long arraySize) {
        final CUDAByteBuffer header = deviceContext.getMemoryManager().getSubBuffer((int) bufferOffset, arrayHeaderSize);
        header.buffer.clear();
        int index = 0;
        while (index < arrayLengthOffset) {
            header.buffer.put((byte) 0);
            index++;
        }
        header.buffer.putLong(arraySize);
        return header;
    }

    @Override
    public void allocate(Object value, long batchSize) {
        long newBufferSize = 0;
        long sizeOfBatch = arrayHeaderSize + batchSize;
        if (batchSize > 0) {
            newBufferSize = sizeOfBatch;
        }

        if ((batchSize > 0) && (bufferOffset != -1) && (newBufferSize < bytesToAllocate)) {
            bytesToAllocate = newBufferSize;
        }

        if (bufferOffset == -1) {
            final T hostArray = cast(value);
            if (batchSize <= 0) {
                bytesToAllocate = sizeOf(hostArray);
            } else {
                bytesToAllocate = sizeOfBatch;
            }

            if (bytesToAllocate <= 0) {
                throw new TornadoMemoryException("[ERROR] Bytes Allocated <= 0: " + bytesToAllocate);
            }
            assert hostArray != null;
            bufferOffset = deviceContext.getMemoryManager().tryAllocate(bytesToAllocate, arrayHeaderSize, getAlignment());

            if (Tornado.FULL_DEBUG) {
                info("allocated: array kind=%s, size=%s, length offset=%d, header size=%d, bo=0x%x",
                        kind.getJavaName(),
                        humanReadableByteCount(bytesToAllocate, true),
                        arrayLengthOffset,
                        arrayHeaderSize,
                        bufferOffset
                );

                info("allocated: %s", toString());
            }
        }
    }

    private long sizeOf(final T array) {
        return (long) arrayHeaderSize + ((long) Array.getLength(array) * (long) kind.getByteCount());
    }

    @Override
    public int getAlignment() {
        return ARRAY_ALIGNMENT;
    }

    @Override
    public boolean isValid() {
        return onDevice;
    }

    @Override
    public void invalidate() {
        onDevice = false;
    }

    @Override
    public void printHeapTrace() {
        System.out.printf("0x%x\ttype=%s\n", toAbsoluteAddress(), kind.getJavaName());
    }

    @Override
    public long size() {
        return bytesToAllocate;
    }

    /**
     * Copy data from the device to the main host.
     *
     * @param address
     *            Device Buffer address
     * @param bytes
     *            Bytes to be copied back to the host
     * @param value
     *            Host array that resides the final data
     * @param waitEvents
     *            List of events to wait for.
     * @return Event information
     */
    protected abstract int enqueueReadArrayData(long address, long bytes, T value, long hostOffset, int[] waitEvents);

    protected abstract int readArrayData(long address, long bytes, T value, long hostOffset, int[] waitEvents);

    /**
     * Copy data that resides in the host to the target device.
     *
     * @param address
     *            Device Buffer address
     * @param bytes
     *            Bytes to be copied
     * @param value
     *            Host array to be copied
     *
     * @param waitEvents
     *            List of events to wait for.
     * @return Event information
     */
    protected abstract int enqueueWriteArrayData(long address, long bytes, T value, long hostOffset, int[] waitEvents);

    protected abstract void writeArrayData(long address, long bytes, T value, int hostOffset, int[] waitEvents);
}
