package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.meta.JavaConstant;
import jdk.vm.ci.meta.ResolvedJavaField;
import jdk.vm.ci.meta.ResolvedJavaType;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedField;

import java.lang.annotation.Annotation;

/**
 * Adapter that implements JVMCI ResolvedJavaField interface by delegating to TornadoResolvedField.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 */
public class JVMCIResolvedJavaFieldAdapter implements ResolvedJavaField {
    private final TornadoResolvedField tornadoField;
    private final JVMCIMetaAccessAdapter metaAccess;

    public JVMCIResolvedJavaFieldAdapter(TornadoResolvedField tornadoField, JVMCIMetaAccessAdapter metaAccess) {
        this.tornadoField = tornadoField;
        this.metaAccess = metaAccess;
    }

    public TornadoResolvedField getTornadoField() {
        return tornadoField;
    }

    @Override
    public String getName() {
        return tornadoField.getName();
    }

    @Override
    public ResolvedJavaType getType() {
        return new JVMCIResolvedJavaTypeAdapter(tornadoField.getType(), metaAccess);
    }

    @Override
    public ResolvedJavaType getDeclaringClass() {
        return new JVMCIResolvedJavaTypeAdapter(tornadoField.getDeclaringClass(), metaAccess);
    }

    @Override
    public int getModifiers() {
        return tornadoField.getModifiers();
    }

    @Override
    public boolean isStatic() {
        return tornadoField.isStatic();
    }

    @Override
    public boolean isFinal() {
        return tornadoField.isFinal();
    }

    @Override
    public boolean isSynthetic() {
        return tornadoField.isSynthetic();
    }

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return tornadoField.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return tornadoField.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return tornadoField.getDeclaredAnnotations();
    }

    // ========== Stubs for features not needed by TornadoVM ==========

    @Override
    public int getOffset() {
        return 0;  // Cannot determine without VM support
    }

    @Override
    public boolean isInternal() {
        return false;
    }

    @Override
    public JavaConstant getConstantValue() {
        return null;  // Cannot determine without VM support
    }

    @Override
    public String toString() {
        return "JVMCIResolvedJavaFieldAdapter<" + tornadoField.getName() + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof JVMCIResolvedJavaFieldAdapter adapter) {
            return tornadoField.equals(adapter.tornadoField);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return tornadoField.hashCode();
    }
}
