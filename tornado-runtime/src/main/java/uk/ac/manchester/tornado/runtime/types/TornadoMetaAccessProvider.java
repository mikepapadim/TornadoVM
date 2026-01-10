package uk.ac.manchester.tornado.runtime.types;

import java.lang.reflect.Executable;
import java.lang.reflect.Field;

/**
 * TornadoVM's metadata access provider.
 * This is a replacement for jdk.vm.ci.meta.MetaAccessProvider with NO JVMCI dependency.
 */
public interface TornadoMetaAccessProvider {

    // ========== Core lookup methods ==========

    TornadoResolvedType lookupJavaType(Class<?> clazz);

    TornadoResolvedMethod lookupJavaMethod(Executable executable);

    TornadoResolvedField lookupJavaField(Field field);

    // ========== Array metadata ==========

    int getArrayBaseOffset(TornadoJavaKind kind);

    int getArrayIndexScale(TornadoJavaKind kind);
}
