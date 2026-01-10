package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;

/**
 * TornadoVM's representation of a resolved Java type (class, interface, array, or primitive).
 * This is a replacement for jdk.vm.ci.meta.ResolvedJavaType with NO JVMCI dependency.
 */
public interface TornadoResolvedType {

    // ========== Basic type information ==========

    String getName();

    String toJavaName();

    TornadoJavaKind getJavaKind();

    boolean isPrimitive();

    boolean isInterface();

    boolean isInstanceClass();

    boolean isArray();

    boolean isEnum();

    int getModifiers();

    // ========== Type hierarchy ==========

    TornadoResolvedType getSuperclass();

    TornadoResolvedType[] getInterfaces();

    boolean isAssignableFrom(TornadoResolvedType other);

    TornadoResolvedType getEnclosingType();

    boolean isLocal();

    boolean isMember();

    // ========== Array operations ==========

    TornadoResolvedType getComponentType();

    TornadoResolvedType getArrayClass();

    // ========== Fields and methods ==========

    TornadoResolvedField[] getInstanceFields(boolean includeSuperclasses);

    TornadoResolvedField[] getStaticFields();

    TornadoResolvedMethod[] getDeclaredMethods();

    TornadoResolvedMethod[] getDeclaredConstructors();

    TornadoResolvedMethod resolveMethod(TornadoResolvedMethod method);

    // ========== Annotations ==========

    <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    Annotation[] getAnnotations();

    Annotation[] getDeclaredAnnotations();

    // ========== Initialization state ==========

    boolean isInitialized();

    void initialize();

    boolean isLinked();

    // ========== Underlying Java class ==========

    Class<?> getJavaClass();
}
