package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;

/**
 * TornadoVM's representation of a resolved Java field.
 * This is a replacement for jdk.vm.ci.meta.ResolvedJavaField with NO JVMCI dependency.
 */
public interface TornadoResolvedField {

    // ========== Basic field information ==========

    String getName();

    TornadoResolvedType getType();

    TornadoResolvedType getDeclaringClass();

    int getModifiers();

    boolean isStatic();

    boolean isFinal();

    boolean isSynthetic();

    int getOffset();

    TornadoJavaKind getJavaKind();

    // ========== Annotations ==========

    <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    Annotation[] getAnnotations();

    Annotation[] getDeclaredAnnotations();

    // ========== Underlying Java field ==========

    Field getJavaField();
}
