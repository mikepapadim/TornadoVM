package uk.ac.manchester.tornado.runtime.types;

import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;

/**
 * Reflection-based implementation of TornadoSignature.
 * Uses pure Java reflection with NO JVMCI dependency.
 */
public class TornadoReflectionSignature implements TornadoSignature {
    private final Executable executable;
    private final TornadoReflectionMetaAccessProvider metaAccess;

    public TornadoReflectionSignature(Executable executable, TornadoReflectionMetaAccessProvider metaAccess) {
        this.executable = executable;
        this.metaAccess = metaAccess;
    }

    @Override
    public int getParameterCount(boolean receiver) {
        int count = executable.getParameterCount();
        return receiver ? count + 1 : count;
    }

    @Override
    public TornadoResolvedType getParameterType(int index) {
        Class<?>[] parameterTypes = executable.getParameterTypes();
        if (index < 0 || index >= parameterTypes.length) {
            throw new IndexOutOfBoundsException("Parameter index " + index + " out of bounds");
        }
        return metaAccess.lookupJavaType(parameterTypes[index]);
    }

    @Override
    public TornadoResolvedType getReturnType() {
        if (executable instanceof Method method) {
            return metaAccess.lookupJavaType(method.getReturnType());
        } else if (executable instanceof Constructor) {
            return metaAccess.lookupJavaType(void.class);
        }
        throw new IllegalStateException("Unknown executable type: " + executable.getClass());
    }

    @Override
    public TornadoJavaKind getParameterKind(int index) {
        Class<?>[] parameterTypes = executable.getParameterTypes();
        if (index < 0 || index >= parameterTypes.length) {
            throw new IndexOutOfBoundsException("Parameter index " + index + " out of bounds");
        }
        return TornadoJavaKind.fromJavaClass(parameterTypes[index]);
    }

    @Override
    public TornadoJavaKind getReturnKind() {
        if (executable instanceof Method method) {
            return TornadoJavaKind.fromJavaClass(method.getReturnType());
        } else if (executable instanceof Constructor) {
            return TornadoJavaKind.Void;
        }
        throw new IllegalStateException("Unknown executable type: " + executable.getClass());
    }

    @Override
    public String toMethodDescriptor() {
        var builder = new StringBuilder("(");
        for (Class<?> paramType : executable.getParameterTypes()) {
            builder.append(getTypeDescriptor(paramType));
        }
        builder.append(")");

        if (executable instanceof Method method) {
            builder.append(getTypeDescriptor(method.getReturnType()));
        } else {
            builder.append("V");  // Constructor returns void
        }

        return builder.toString();
    }

    private String getTypeDescriptor(Class<?> clazz) {
        if (clazz.isPrimitive()) {
            return TornadoJavaKind.fromJavaClass(clazz).getTypeChar();
        } else if (clazz.isArray()) {
            return clazz.getName().replace('.', '/');
        } else {
            return "L" + clazz.getName().replace('.', '/') + ";";
        }
    }

    @Override
    public String toString() {
        return "TornadoReflectionSignature<" + toMethodDescriptor() + ">";
    }
}
