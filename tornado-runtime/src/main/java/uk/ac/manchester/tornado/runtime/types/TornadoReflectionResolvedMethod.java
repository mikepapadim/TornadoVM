package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

/**
 * Reflection-based implementation of TornadoResolvedMethod.
 * Uses pure Java reflection with NO JVMCI dependency.
 */
public class TornadoReflectionResolvedMethod implements TornadoResolvedMethod {
    private final Executable executable;
    private final TornadoReflectionMetaAccessProvider metaAccess;
    private TornadoSignature signature;

    public TornadoReflectionResolvedMethod(Executable executable, TornadoReflectionMetaAccessProvider metaAccess) {
        this.executable = executable;
        this.metaAccess = metaAccess;
    }

    @Override
    public Executable getExecutable() {
        return executable;
    }

    // ========== Basic method information ==========

    @Override
    public String getName() {
        if (executable instanceof Constructor) {
            return "<init>";
        }
        return executable.getName();
    }

    @Override
    public TornadoResolvedType getDeclaringClass() {
        return metaAccess.lookupJavaType(executable.getDeclaringClass());
    }

    @Override
    public TornadoSignature getSignature() {
        if (signature == null) {
            signature = new TornadoReflectionSignature(executable, metaAccess);
        }
        return signature;
    }

    @Override
    public int getModifiers() {
        return executable.getModifiers();
    }

    @Override
    public boolean isConstructor() {
        return executable instanceof Constructor;
    }

    @Override
    public boolean isStatic() {
        return Modifier.isStatic(executable.getModifiers());
    }

    @Override
    public boolean isFinal() {
        return Modifier.isFinal(executable.getModifiers());
    }

    @Override
    public boolean isSynchronized() {
        return Modifier.isSynchronized(executable.getModifiers());
    }

    @Override
    public boolean isBridge() {
        if (executable instanceof Method method) {
            return method.isBridge();
        }
        return false;
    }

    @Override
    public boolean isVarArgs() {
        return executable.isVarArgs();
    }

    @Override
    public boolean isSynthetic() {
        return executable.isSynthetic();
    }

    // ========== Annotations ==========

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return executable.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return executable.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return executable.getDeclaredAnnotations();
    }

    @Override
    public Annotation[][] getParameterAnnotations() {
        return executable.getParameterAnnotations();
    }

    // ========== Code access (stubs - cannot access bytecode without JVMCI) ==========

    @Override
    public byte[] getCode() {
        return null;  // Cannot access bytecode without JVMCI
    }

    @Override
    public int getCodeSize() {
        return 0;  // Cannot determine without bytecode
    }

    @Override
    public int getMaxLocals() {
        return 0;  // Cannot determine without bytecode
    }

    @Override
    public int getMaxStackSize() {
        return 0;  // Cannot determine without bytecode
    }

    // ========== Object methods ==========

    @Override
    public String toString() {
        return "TornadoReflectionResolvedMethod<" + executable + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof TornadoReflectionResolvedMethod other) {
            return executable.equals(other.executable);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return executable.hashCode();
    }
}
