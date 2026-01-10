package uk.ac.manchester.tornado.runtime.types;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.stream.Stream;

/**
 * Reflection-based implementation of TornadoResolvedType.
 * Uses pure Java reflection with NO JVMCI dependency.
 */
public class TornadoReflectionResolvedType implements TornadoResolvedType {
    private final Class<?> javaClass;
    private final TornadoReflectionMetaAccessProvider metaAccess;

    public TornadoReflectionResolvedType(Class<?> javaClass, TornadoReflectionMetaAccessProvider metaAccess) {
        this.javaClass = javaClass;
        this.metaAccess = metaAccess;
    }

    @Override
    public Class<?> getJavaClass() {
        return javaClass;
    }

    // ========== Basic type information ==========

    @Override
    public String getName() {
        return javaClass.getName();
    }

    @Override
    public String toJavaName() {
        // Return Java-formatted class name (same as getName() for our implementation)
        return javaClass.getName();
    }

    @Override
    public TornadoJavaKind getJavaKind() {
        return TornadoJavaKind.fromJavaClass(javaClass);
    }

    @Override
    public boolean isPrimitive() {
        return javaClass.isPrimitive();
    }

    @Override
    public boolean isInterface() {
        return javaClass.isInterface();
    }

    @Override
    public boolean isInstanceClass() {
        return !javaClass.isArray() && !javaClass.isPrimitive() && !javaClass.isInterface();
    }

    @Override
    public boolean isArray() {
        return javaClass.isArray();
    }

    @Override
    public boolean isEnum() {
        return javaClass.isEnum();
    }

    @Override
    public int getModifiers() {
        return javaClass.getModifiers();
    }

    // ========== Type hierarchy ==========

    @Override
    public TornadoResolvedType getSuperclass() {
        Class<?> superclass = javaClass.getSuperclass();
        if (superclass == null) {
            return null;
        }
        return metaAccess.lookupJavaType(superclass);
    }

    @Override
    public TornadoResolvedType[] getInterfaces() {
        Class<?>[] interfaces = javaClass.getInterfaces();
        return Arrays.stream(interfaces)
                .map(metaAccess::lookupJavaType)
                .toArray(TornadoResolvedType[]::new);
    }

    @Override
    public boolean isAssignableFrom(TornadoResolvedType other) {
        if (other instanceof TornadoReflectionResolvedType otherReflection) {
            return javaClass.isAssignableFrom(otherReflection.javaClass);
        }
        return false;
    }

    @Override
    public TornadoResolvedType getEnclosingType() {
        Class<?> enclosingClass = javaClass.getEnclosingClass();
        if (enclosingClass == null) {
            return null;
        }
        return metaAccess.lookupJavaType(enclosingClass);
    }

    @Override
    public boolean isLocal() {
        return javaClass.isLocalClass();
    }

    @Override
    public boolean isMember() {
        return javaClass.isMemberClass();
    }

    // ========== Array operations ==========

    @Override
    public TornadoResolvedType getComponentType() {
        if (!javaClass.isArray()) {
            return null;
        }
        return metaAccess.lookupJavaType(javaClass.getComponentType());
    }

    @Override
    public TornadoResolvedType getArrayClass() {
        return metaAccess.lookupJavaType(java.lang.reflect.Array.newInstance(javaClass, 0).getClass());
    }

    // ========== Fields and methods ==========

    @Override
    public TornadoResolvedField[] getInstanceFields(boolean includeSuperclasses) {
        if (includeSuperclasses) {
            return getAllFields(false);
        } else {
            return getDeclaredFields(false);
        }
    }

    @Override
    public TornadoResolvedField[] getStaticFields() {
        return getDeclaredFields(true);
    }

    private TornadoResolvedField[] getDeclaredFields(boolean staticOnly) {
        Field[] fields = javaClass.getDeclaredFields();
        return Arrays.stream(fields)
                .filter(f -> staticOnly == Modifier.isStatic(f.getModifiers()))
                .map(metaAccess::lookupJavaField)
                .toArray(TornadoResolvedField[]::new);
    }

    private TornadoResolvedField[] getAllFields(boolean staticOnly) {
        return getAllFieldsStream(javaClass, staticOnly)
                .map(metaAccess::lookupJavaField)
                .toArray(TornadoResolvedField[]::new);
    }

    private Stream<Field> getAllFieldsStream(Class<?> clazz, boolean staticOnly) {
        if (clazz == null) {
            return Stream.empty();
        }
        Stream<Field> fields = Arrays.stream(clazz.getDeclaredFields())
                .filter(f -> staticOnly == Modifier.isStatic(f.getModifiers()));
        return Stream.concat(fields, getAllFieldsStream(clazz.getSuperclass(), staticOnly));
    }

    @Override
    public TornadoResolvedMethod[] getDeclaredMethods() {
        return Arrays.stream(javaClass.getDeclaredMethods())
                .map(metaAccess::lookupJavaMethod)
                .toArray(TornadoResolvedMethod[]::new);
    }

    @Override
    public TornadoResolvedMethod[] getDeclaredConstructors() {
        return Arrays.stream(javaClass.getDeclaredConstructors())
                .map(metaAccess::lookupJavaMethod)
                .toArray(TornadoResolvedMethod[]::new);
    }

    @Override
    public TornadoResolvedMethod resolveMethod(TornadoResolvedMethod method) {
        // Simple resolution - just return the method
        return method;
    }

    // ========== Annotations ==========

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return javaClass.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return javaClass.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return javaClass.getDeclaredAnnotations();
    }

    // ========== Initialization state ==========

    @Override
    public boolean isInitialized() {
        return true;  // Classes are already initialized through normal class loading
    }

    @Override
    public void initialize() {
        // Classes are already initialized through normal class loading
    }

    @Override
    public boolean isLinked() {
        return true;  // Classes are already linked
    }

    // ========== Object methods ==========

    @Override
    public String toString() {
        return "TornadoReflectionResolvedType<" + javaClass.getName() + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof TornadoReflectionResolvedType other) {
            return javaClass.equals(other.javaClass);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return javaClass.hashCode();
    }
}
