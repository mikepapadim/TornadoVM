package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.meta.*;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedField;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedMethod;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedType;

import java.lang.annotation.Annotation;
import java.util.Arrays;

/**
 * Adapter that implements JVMCI ResolvedJavaType interface by delegating to TornadoResolvedType.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 */
public class JVMCIResolvedJavaTypeAdapter implements ResolvedJavaType {
    private final TornadoResolvedType tornadoType;
    private final JVMCIMetaAccessAdapter metaAccess;

    public JVMCIResolvedJavaTypeAdapter(TornadoResolvedType tornadoType, JVMCIMetaAccessAdapter metaAccess) {
        this.tornadoType = tornadoType;
        this.metaAccess = metaAccess;
    }

    public TornadoResolvedType getTornadoType() {
        return tornadoType;
    }

    // ========== JavaType interface ==========

    @Override
    public String getName() {
        // Convert Java class name to JVM internal format
        // e.g., "java.lang.String" -> "Ljava/lang/String;"
        String javaName = tornadoType.getName();
        if (tornadoType.isPrimitive()) {
            // Primitives don't have L and ; wrapper
            return javaName;
        }
        if (tornadoType.isArray()) {
            // Arrays already have the correct format from Class.getName()
            return javaName.replace('.', '/');
        }
        // Regular classes: add L prefix, replace dots with slashes, add ; suffix
        return "L" + javaName.replace('.', '/') + ";";
    }

    @Override
    public JavaKind getJavaKind() {
        return JVMCIMetaAccessAdapter.convertTornadoJavaKind(tornadoType.getJavaKind());
    }

    @Override
    public ResolvedJavaType resolve(ResolvedJavaType accessingClass) {
        return this;
    }

    // ========== ResolvedJavaType interface ==========

    @Override
    public boolean isPrimitive() {
        return tornadoType.isPrimitive();
    }

    @Override
    public boolean isInterface() {
        return tornadoType.isInterface();
    }

    @Override
    public boolean isInstanceClass() {
        return tornadoType.isInstanceClass();
    }

    @Override
    public boolean isArray() {
        return tornadoType.isArray();
    }

    @Override
    public boolean isEnum() {
        return tornadoType.isEnum();
    }

    @Override
    public int getModifiers() {
        return tornadoType.getModifiers();
    }

    @Override
    public boolean isInitialized() {
        return tornadoType.isInitialized();
    }

    @Override
    public void initialize() {
        tornadoType.initialize();
    }

    @Override
    public boolean isLinked() {
        return tornadoType.isLinked();
    }

    @Override
    public void link() {
        // Ensure the class is initialized/linked
        // In our reflection-based implementation, we just ensure initialization
        tornadoType.initialize();
    }

    @Override
    public boolean isAssignableFrom(ResolvedJavaType other) {
        if (other instanceof JVMCIResolvedJavaTypeAdapter adapter) {
            return tornadoType.isAssignableFrom(adapter.tornadoType);
        }
        return false;
    }

    @Override
    public boolean isInstance(JavaConstant obj) {
        return false;  // Cannot determine without HotSpot internals
    }

    @Override
    public ResolvedJavaType getSuperclass() {
        TornadoResolvedType superclass = tornadoType.getSuperclass();
        if (superclass == null) {
            return null;
        }
        return new JVMCIResolvedJavaTypeAdapter(superclass, metaAccess);
    }

    @Override
    public ResolvedJavaType[] getInterfaces() {
        TornadoResolvedType[] interfaces = tornadoType.getInterfaces();
        return Arrays.stream(interfaces)
                .map(t -> new JVMCIResolvedJavaTypeAdapter(t, metaAccess))
                .toArray(ResolvedJavaType[]::new);
    }

    @Override
    public ResolvedJavaType getComponentType() {
        TornadoResolvedType componentType = tornadoType.getComponentType();
        if (componentType == null) {
            return null;
        }
        return new JVMCIResolvedJavaTypeAdapter(componentType, metaAccess);
    }

    @Override
    public ResolvedJavaType getArrayClass() {
        TornadoResolvedType arrayClass = tornadoType.getArrayClass();
        return new JVMCIResolvedJavaTypeAdapter(arrayClass, metaAccess);
    }

    @Override
    public ResolvedJavaField[] getInstanceFields(boolean includeSuperclasses) {
        TornadoResolvedField[] fields = tornadoType.getInstanceFields(includeSuperclasses);
        return Arrays.stream(fields)
                .map(f -> new JVMCIResolvedJavaFieldAdapter(f, metaAccess))
                .toArray(ResolvedJavaField[]::new);
    }

    @Override
    public ResolvedJavaField[] getStaticFields() {
        TornadoResolvedField[] fields = tornadoType.getStaticFields();
        return Arrays.stream(fields)
                .map(f -> new JVMCIResolvedJavaFieldAdapter(f, metaAccess))
                .toArray(ResolvedJavaField[]::new);
    }

    @Override
    public ResolvedJavaMethod[] getDeclaredConstructors() {
        return getDeclaredConstructors(false);
    }

    @Override
    public ResolvedJavaMethod[] getDeclaredConstructors(boolean forceLink) {
        TornadoResolvedMethod[] constructors = tornadoType.getDeclaredConstructors();
        return Arrays.stream(constructors)
                .map(m -> new JVMCIResolvedJavaMethodAdapter(m, metaAccess))
                .toArray(ResolvedJavaMethod[]::new);
    }

    @Override
    public ResolvedJavaMethod[] getDeclaredMethods() {
        return getDeclaredMethods(false);
    }

    @Override
    public ResolvedJavaMethod[] getDeclaredMethods(boolean forceLink) {
        TornadoResolvedMethod[] methods = tornadoType.getDeclaredMethods();
        return Arrays.stream(methods)
                .map(m -> new JVMCIResolvedJavaMethodAdapter(m, metaAccess))
                .toArray(ResolvedJavaMethod[]::new);
    }

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return tornadoType.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return tornadoType.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return tornadoType.getDeclaredAnnotations();
    }

    @Override
    public boolean hasFinalizer() {
        return false;  // Stub
    }

    @Override
    public Assumptions.AssumptionResult<Boolean> hasFinalizableSubclass() {
        return new Assumptions.AssumptionResult<>(false);
    }

    @Override
    public Assumptions.AssumptionResult<ResolvedJavaType> findLeafConcreteSubtype() {
        return new Assumptions.AssumptionResult<>(this);
    }

    @Override
    public ResolvedJavaType getSingleImplementor() {
        return null;  // Stub
    }

    @Override
    public ResolvedJavaType findLeastCommonAncestor(ResolvedJavaType otherType) {
        if (isAssignableFrom(otherType)) {
            return this;
        }
        if (otherType.isAssignableFrom(this)) {
            return otherType;
        }
        return null;
    }

    @Override
    public ResolvedJavaMethod resolveMethod(ResolvedJavaMethod method, ResolvedJavaType callerType) {
        if (method instanceof JVMCIResolvedJavaMethodAdapter adapter) {
            TornadoResolvedMethod resolved = tornadoType.resolveMethod(adapter.getTornadoMethod());
            return new JVMCIResolvedJavaMethodAdapter(resolved, metaAccess);
        }
        return method;
    }

    @Override
    public Assumptions.AssumptionResult<ResolvedJavaMethod> findUniqueConcreteMethod(ResolvedJavaMethod method) {
        return new Assumptions.AssumptionResult<>(method);
    }

    @Override
    public ResolvedJavaField findInstanceFieldWithOffset(long offset, JavaKind expectedKind) {
        return null;  // Cannot determine without VM support
    }

    @Override
    public String getSourceFileName() {
        return null;  // Cannot determine without class file parsing
    }

    @Override
    public boolean isLocal() {
        return tornadoType.isLocal();
    }

    @Override
    public boolean isMember() {
        return tornadoType.isMember();
    }

    @Override
    public ResolvedJavaType getEnclosingType() {
        TornadoResolvedType enclosingType = tornadoType.getEnclosingType();
        if (enclosingType == null) {
            return null;
        }
        return new JVMCIResolvedJavaTypeAdapter(enclosingType, metaAccess);
    }

    @Override
    public ResolvedJavaMethod getClassInitializer() {
        return null;  // Stub
    }

    @Override
    public boolean isCloneableWithAllocation() {
        return Cloneable.class.isAssignableFrom(tornadoType.getJavaClass());
    }

    @Override
    public String toString() {
        return "JVMCIResolvedJavaTypeAdapter<" + tornadoType.getName() + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof JVMCIResolvedJavaTypeAdapter adapter) {
            return tornadoType.equals(adapter.tornadoType);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return tornadoType.hashCode();
    }
}
