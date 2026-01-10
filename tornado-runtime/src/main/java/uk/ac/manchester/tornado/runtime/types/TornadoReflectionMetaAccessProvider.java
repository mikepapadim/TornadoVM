package uk.ac.manchester.tornado.runtime.types;

import java.lang.reflect.Executable;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

/**
 * Reflection-based implementation of TornadoMetaAccessProvider.
 * Uses pure Java reflection with NO JVMCI dependency.
 */
public class TornadoReflectionMetaAccessProvider implements TornadoMetaAccessProvider {
    private final Map<Class<?>, TornadoResolvedType> typeCache = new HashMap<>();

    @Override
    public synchronized TornadoResolvedType lookupJavaType(Class<?> clazz) {
        return typeCache.computeIfAbsent(clazz, c -> new TornadoReflectionResolvedType(c, this));
    }

    @Override
    public TornadoResolvedMethod lookupJavaMethod(Executable executable) {
        return new TornadoReflectionResolvedMethod(executable, this);
    }

    @Override
    public TornadoResolvedField lookupJavaField(Field field) {
        return new TornadoReflectionResolvedField(field, this);
    }

    @Override
    public int getArrayBaseOffset(TornadoJavaKind kind) {
        // Standard HotSpot array base offset (object header + array length)
        return 16;
    }

    @Override
    public int getArrayIndexScale(TornadoJavaKind kind) {
        return kind.getByteCount();
    }
}
