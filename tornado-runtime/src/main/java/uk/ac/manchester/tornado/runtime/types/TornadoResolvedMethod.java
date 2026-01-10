package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;
import java.lang.reflect.Executable;

/**
 * TornadoVM's representation of a resolved Java method or constructor.
 * This is a replacement for jdk.vm.ci.meta.ResolvedJavaMethod with NO JVMCI dependency.
 */
public interface TornadoResolvedMethod {

    // ========== Basic method information ==========

    String getName();

    TornadoResolvedType getDeclaringClass();

    TornadoSignature getSignature();

    int getModifiers();

    boolean isConstructor();

    boolean isStatic();

    boolean isFinal();

    boolean isSynchronized();

    boolean isBridge();

    boolean isVarArgs();

    boolean isSynthetic();

    // ========== Annotations ==========

    <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    Annotation[] getAnnotations();

    Annotation[] getDeclaredAnnotations();

    Annotation[][] getParameterAnnotations();

    // ========== Code access ==========

    byte[] getCode();

    int getCodeSize();

    int getMaxLocals();

    int getMaxStackSize();

    // ========== Underlying Java method ==========

    Executable getExecutable();
}
