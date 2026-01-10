package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Reflection-based implementation of TornadoResolvedField.
 * Uses pure Java reflection with NO JVMCI dependency.
 */
public class TornadoReflectionResolvedField implements TornadoResolvedField {
    private final Field field;
    private final TornadoReflectionMetaAccessProvider metaAccess;

    public TornadoReflectionResolvedField(Field field, TornadoReflectionMetaAccessProvider metaAccess) {
        this.field = field;
        this.metaAccess = metaAccess;
    }

    @Override
    public Field getJavaField() {
        return field;
    }

    // ========== Basic field information ==========

    @Override
    public String getName() {
        return field.getName();
    }

    @Override
    public TornadoResolvedType getType() {
        return metaAccess.lookupJavaType(field.getType());
    }

    @Override
    public TornadoResolvedType getDeclaringClass() {
        return metaAccess.lookupJavaType(field.getDeclaringClass());
    }

    @Override
    public int getModifiers() {
        return field.getModifiers();
    }

    @Override
    public boolean isStatic() {
        return Modifier.isStatic(field.getModifiers());
    }

    @Override
    public boolean isFinal() {
        return Modifier.isFinal(field.getModifiers());
    }

    @Override
    public boolean isSynthetic() {
        return field.isSynthetic();
    }

    @Override
    public int getOffset() {
        try {
            // Use sun.misc.Unsafe to get field offset
            java.lang.reflect.Field unsafeField = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            unsafeField.setAccessible(true);
            sun.misc.Unsafe unsafe = (sun.misc.Unsafe) unsafeField.get(null);
            if (isStatic()) {
                return (int) unsafe.staticFieldOffset(field);
            } else {
                return (int) unsafe.objectFieldOffset(field);
            }
        } catch (Exception e) {
            throw new RuntimeException("Cannot get field offset for: " + field, e);
        }
    }

    @Override
    public TornadoJavaKind getJavaKind() {
        return TornadoJavaKind.fromJavaClass(field.getType());
    }

    // ========== Annotations ==========

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return field.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return field.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return field.getDeclaredAnnotations();
    }

    // ========== Object methods ==========

    @Override
    public String toString() {
        return "TornadoReflectionResolvedField<" + field + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof TornadoReflectionResolvedField other) {
            return field.equals(other.field);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return field.hashCode();
    }
}
