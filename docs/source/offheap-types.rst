Use TornadoVM Data Types
==================================
Starting from v1.0, TornadoVM is providing its custom off-heap data types. Below is a list of the new types, with an arrow pointing from the on-heap primitive array types to their off-heap equivalent.

* ``int[]`` -> ``IntArray``
* ``float[]`` -> ``FloatArray``
* ``double[]`` -> ``DoubleArray``
* ``long[]`` -> ``LongArray``
* ``char[]`` -> ``CharArray``
* ``short[]`` -> ``ShortArray``
* ``byte[]`` -> ``ByteArray``

The existing Matrix and Vector collection types that TornadoVM offers (e.g., ``VectorFloat``, ``Matrix2DDouble``, etc.)  have been refactored to use internally these off-heap data types instead of primitive arrays.

1. Off-heap Data Types
-------------------------
The new off-heap types encapsulate a `Memory Segment <https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/foreign/MemorySegment.html>`_, a contiguous region of memory outside the Java heap. To allocate off-heap memory using the TornadoVM API, each type offers a constructor with one argument that indicates the number of elements that the Memory Segment will contain.

E.g.:

.. code:: java

   // allocate an off-heap memory segment that will contain 16 int values
   IntArray intArray = new IntArray(16);

Additionally, developers can create an instance of a TornadoVM native array by invoking factory methods for different data representations. In the following examples we will demonstrate the API functions for the ``FloatArray`` type, but the same methods apply for all support native array types. 

.. code:: java

   // from on-heap array to TornadoVM native array
   public static FloatArray fromArray(float[] values);
   // from individual items to TornadoVM native array
   public static FloatArray fromElements(float... values);
   // from Memory Segment to TornadoVM native array
   public static FloatArray fromSegment(MemorySegment segment); 

The main methods that the off-heap types expose to manage the Memory Segment of each type are presented in the list below. 

.. code:: java

   public MemorySegment getSegment() // returns the memory segment without the Tornado Array Header as slice
   public MemorySegment getSegmentWithHeader() //  returns the Memory Segment with the Tornado Array Header
   public void set(int index, float value) // sets a value at a specific index
      E.g.:
          FloatArray floatArray = new FloatArray(16);
          floatArray.set(0, 10.0f); // at index 0 the value is 10.0f
   public float get(int index) // returns the value of a specific index
      E.g.:
          FloatArray floatArray = FloatArray.fromArray(new float[] {2.0f, 1.0f, 2.0f, 5.0f});
          float floatValue = floatArray.get(3); // returns 5.0f
   public void clear() // sets the values of the segment to 0
      E.g.:
          FloatArray floatArray = new FloatArray(1024);
          floatArray.clear(); // the floatArray contains 1024 zeros
   public void init(float value) // initializes the segment with a specific value
      E.g.:
   	  FloatArray floatArray = new FloatArray(1024);
          floatArray.init(1.0f); // the floatArray contains 1024 ones
   public int getSize() // returns the number of elements in the segment
      E.g.:
          FloatArray floatArray = new FloatArray(16);
          int size = floatArray.getSize(); // returns 16
   public float[] toHeapArray(); // Converts the data from off-heap to on-heap
   public long getNumBytesOfSegmentWithHeader(); // Returns the total number of bytes the underlying Memory Segment occupies, including the header bytes
   public long getNumBytesOfSegment(); // Returns the total number of bytes the underlying Memory Segment occupies, excluding the header bytes
   
**NOTE:** The methods ``init()`` and ``clear()`` are essential because, contrary to their counterpart primitive arrays which are initialized by default with 0, the new types contain garbage values when first created.

Arrays with more than ``Integer.MAX_VALUE`` elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every native array type can hold more than ``Integer.MAX_VALUE`` (2^31 - 1) elements, e.g., a ``HalfFloatArray`` of 10 billion elements (20 GB).
Such arrays are created with the ``long`` constructor and accessed with the ``long`` overloads, on the host and inside kernels:

.. code:: java

   public FloatArray(long numberOfElements) // allocates an array that can exceed Integer.MAX_VALUE elements
   public void set(long index, float value)
   public float get(long index)
   public long getSizeLong() // returns the number of elements as a long
   public FloatArray slice(long offset, long length)

``getSize()`` keeps returning an ``int``. On the host it throws an ``IllegalStateException`` for arrays whose size does not fit in an ``int``; use ``getSizeLong()`` for those.
``toHeapArray()`` also requires the size to fit in a Java array.

Thread indices (e.g., ``KernelContext.globalIdx``) are ``int``, so a kernel that reaches elements beyond ``Integer.MAX_VALUE`` builds a ``long`` index from them:

.. code:: java

   public static void scaleRows(KernelContext context, HalfFloatArray data, int rowLength) {
       int row = context.globalIdx;
       long start = (long) row * rowLength;
       if (start < data.getSizeLong()) {
           for (int j = 0; j < rowLength; j++) {
               data.set(start + j, HalfFloat.mult(data.get(start + j), new HalfFloat(0.5f)));
           }
       }
   }

The device must allow an allocation of the array's size (e.g., ``CL_DEVICE_MAX_MEM_ALLOC_SIZE`` on OpenCL), and the TornadoVM device heap must be large enough, which can be set with ``-Dtornado.device.memory=<size>``.

2. Example: Migrating from On-heap to Off-heap Data Types
-------------------------------------------------------------------

Below is an example of a TornadoVM program that uses primitive arrays:

.. code:: java

    public static void main(String[] args) {
        int[] input = new int[numElements];

        Arrays.fill(input, 10);

        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .task("t", Example::add, input, 1)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, input);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutor(immutableTaskGraph);
        executor.execute();
    }

    public static void add(int[] input, int value) {
        for (@Parallel int i = 0; i < input.length; i++) {
            input[i] = input[i] + value;
        }
    }

Here is how the code above would need to be transformed to use the new data types (the changes are highlighted):

.. code-block:: java
   :emphasize-lines: 2,4,16,18

    public static void main(String[] args) {
        IntArray input = new IntArray(numElements); // create a new off heap segment of int values

        input.init(10); // initialize all the values of the input to be 10

        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input)
                .task("t", Example::add, input, 1)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, input);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executor = new TornadoExecutor(immutableTaskGraph);
        executor.execute();
    }

    public static void acc(IntArray input, int value) { // Pass the IntArray as a parameter
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            input.set(i, input.get(i) + value);  // Use the set and get functions access data
        }
    }

3. Scalar Parameters: Constants and Runtime Values
-------------------------------------------------------------------

A primitive task argument (``int``, ``long``, ``float``, ``double``) is a **constant**. Its value is read once, when the task is created, and the compiler may fold it into the kernel. That is the fastest code, but changing the value means building a new task graph, and each new value compiles a new kernel.

When a value changes from one execution to the next (a step counter, a scale factor, an active length), pass a **scalar holder** instead: ``IntScalar``, ``LongScalar``, ``FloatScalar`` or ``DoubleScalar`` from ``uk.ac.manchester.tornado.api.types.scalars``.
The kernel declares the holder type and reads it with ``get()``. The host changes it with ``set()`` between executions, and every launch passes the holder's value at that moment, by value, as the kernel parameter. No buffer, data transfer, graph rebuild or recompilation is involved:

.. code-block:: java

    public static void addOffset(IntArray in, IntArray out, IntScalar offset) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) + offset.get());
        }
    }

    IntScalar offset = new IntScalar(1);
    TaskGraph taskGraph = new TaskGraph("s0")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, in)
            .task("t0", Example::addOffset, in, out, offset)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

    try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
        for (int step = 0; step < 100; step++) {
            offset.set(step);    // the next launch passes the new value
            plan.execute();      // same kernel, no recompilation
        }
    }

Holders are read-only on the device: a kernel cannot ``set`` them, and a holder cannot be transferred to the host. To return a scalar, use a one-element array.

A holder can also bound a ``@Parallel`` loop. The compiler records the bound as a runtime launch domain, and each launch is sized for the holder's value at that moment, still from one compiled kernel:

.. code-block:: java

    public static void addToPrefix(IntArray out, IntScalar length, IntScalar value) {
        for (@Parallel int i = 0; i < length.get(); i++) {
            out.set(i, out.get(i) + value.get());
        }
    }

    length.set(17);     // the next launch runs 17 threads
    plan.execute();
    length.set(1024);   // and this one 1024
    plan.execute();

To keep a plain primitive in the kernel instead, change it through the execution plan with ``withScalar``. Give the task as ``"graphName.taskName"``, the argument's position in ``task(...)``, and the new value boxed as the original (``Integer`` for an ``int``):

.. code-block:: java

    TaskGraph taskGraph = new TaskGraph("s0")
            .task("t0", Example::addOffset, in, out, 1)     // int offset
            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
    try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
        for (int step = 0; step < 100; step++) {
            plan.withScalar("s0.t0", 2, step).execute();
        }
    }

The compiler treats a primitive as a constant and may fold it into the kernel, so the first new values each compile the task again. A value seen before reuses its kernel. Once an argument has taken a few different values (``-Dtornado.tc.varyingScalars``, default 3), the kernel is compiled once with the argument as a runtime value, and later changes need no compilation. A primitive that bounds a ``@Parallel`` loop works the same way: each launch is sized for its value. Use a holder when the value changes from the first execution on. Use ``withScalar`` when the kernel should keep its primitive signature, or when the value changes rarely and its constant should stay folded.
