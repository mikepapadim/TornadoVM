package uk.ac.manchester.tornado.runtime.types;

/**
 * TornadoVM's representation of Java primitive types and object references.
 * This is a replacement for jdk.vm.ci.meta.JavaKind with NO JVMCI dependency.
 */
public enum TornadoJavaKind {
    Boolean(boolean.class, "Z", 1),
    Byte(byte.class, "B", 1),
    Short(short.class, "S", 2),
    Char(char.class, "C", 2),
    Int(int.class, "I", 4),
    Long(long.class, "J", 8),
    Float(float.class, "F", 4),
    Double(double.class, "D", 8),
    Object(Object.class, "L", 4),  // Object reference size (compressed oops)
    Void(void.class, "V", 0),
    Illegal(null, "?", 0);

    private final Class<?> javaClass;
    private final String typeChar;
    private final int byteCount;

    TornadoJavaKind(Class<?> javaClass, String typeChar, int byteCount) {
        this.javaClass = javaClass;
        this.typeChar = typeChar;
        this.byteCount = byteCount;
    }

    public Class<?> toJavaClass() {
        return javaClass;
    }

    public String getTypeChar() {
        return typeChar;
    }

    public int getByteCount() {
        return byteCount;
    }

    public boolean isPrimitive() {
        return this != Object && this != Void && this != Illegal;
    }

    public boolean isNumericInteger() {
        return this == Byte || this == Short || this == Int || this == Long || this == Char;
    }

    public boolean isNumericFloat() {
        return this == Float || this == Double;
    }

    public boolean isObject() {
        return this == Object;
    }

    public static TornadoJavaKind fromJavaClass(Class<?> clazz) {
        if (clazz == boolean.class) return Boolean;
        if (clazz == byte.class) return Byte;
        if (clazz == short.class) return Short;
        if (clazz == char.class) return Char;
        if (clazz == int.class) return Int;
        if (clazz == long.class) return Long;
        if (clazz == float.class) return Float;
        if (clazz == double.class) return Double;
        if (clazz == void.class) return Void;
        return Object;
    }

    public static TornadoJavaKind fromTypeString(String typeChar) {
        return switch (typeChar) {
            case "Z" -> Boolean;
            case "B" -> Byte;
            case "S" -> Short;
            case "C" -> Char;
            case "I" -> Int;
            case "J" -> Long;
            case "F" -> Float;
            case "D" -> Double;
            case "V" -> Void;
            case "L", "[" -> Object;
            default -> Illegal;
        };
    }
}
