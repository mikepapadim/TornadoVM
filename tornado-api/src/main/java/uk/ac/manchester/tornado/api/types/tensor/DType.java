package uk.ac.manchester.tornado.api.types.tensor;

/**
 * Represents different data types supported by the system and machine learning frameworks.
 *
 * <p>Supported Data Types:</p>
 * <ul>
 * <li>{@code BOOL}: Boolean type: returns True if the value is greater than 0, otherwise False. (Size: 1 byte)</li>
 * <li>{@code U8}: Unsigned byte type. (Size: 1 byte)</li>
 * <li>{@code I8}: Signed byte type. (Size: 1 byte)</li>
 * <li>{@code I16}: 16-bit signed integer type. (Size: 2 bytes)</li>
 * <li>{@code U16}: 16-bit unsigned integer type. (Size: 2 bytes)</li>
 * <li>{@code F16}: Half-precision (16-bit) floating point type. (Size: 2 bytes)</li>
 * <li>{@code BF16}: Brain (16-bit) floating point type. (Size: 2 bytes)</li>
 * <li>{@code I32}: 32-bit signed integer type. (Size: 4 bytes)</li>
 * <li>{@code U32}: 32-bit unsigned integer type. (Size: 4 bytes)</li>
 * <li>{@code F32}: 32-bit floating point type. (Size: 4 bytes)</li>
 * <li>{@code F64}: 64-bit floating point type. (Size: 8 bytes)</li>
 * <li>{@code I64}: 64-bit signed integer type. (Size: 8 bytes)</li>
 * <li>{@code U64}: 64-bit unsigned integer type. (Size: 8 bytes)</li>
 * <li>{@code Q4}: Represents a custom data type. (Size: 1 byte)</li>
 * <li>{@code Q5}: Represents a custom data type. (Size: 1 byte)</li>
 * </ul>
 *
 * <p>Supported Machine Learning Frameworks:</p>
 * <ul>
 * <li>TensorFlow: An open-source machine learning framework developed by Google.</li>
 * <li>PyTorch: An open-source machine learning library developed by Facebook.</li>
 * <li>Apache MXNet: An open-source deep learning framework.</li>
 * <li>NumPy: A fundamental package for scientific computing with Python.</li>
 * <li>JAX: A library for numerical computing and machine learning research.</li>
 * <li>TensorRT: NVIDIA's inference optimizer and runtime for deploying deep learning models.</li>
 * </ul>
 *
 * <p>These frameworks provide extensive support for various data types to accommodate different types of machine learning tasks and models.</p>
 */

public enum DType {
    // BOOL represents a boolean type.
    BOOL(1),
    // U8 represents an unsigned byte type.
    U8(1),
    // I8 represents a signed byte type.
    I8(1),
    // I16 represents a 16-bit signed integer type.
    I16(2),
    // U16 represents a 16-bit unsigned integer type.
    U16(2),
    // F16 represents a half-precision (16-bit) floating point type.
    F16(2),
    // BF16 represents a brain (16-bit) floating point type.
    BF16(2),
    // I32 represents a 32-bit signed integer type.
    I32(4),
    // U32 represents a 32-bit unsigned integer type.
    U32(4),
    // F32 represents a 32-bit floating point type.
    F32(4),
    // F64 represents a 64-bit floating point type.
    F64(8),
    // I64 represents a 64-bit signed integer type.
    I64(8),
    // U64 represents a 64-bit unsigned integer type.
    U64(8);

    //    Q4(1), Q5(1);

    private final int size;

    private DType(int size) {
        this.size = size;
    }

    public int size() {
        return size;
    }
}