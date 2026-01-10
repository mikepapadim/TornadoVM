package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.hotspot.HotSpotJVMCIRuntime;
import jdk.vm.ci.meta.*;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedMethod;

import java.lang.annotation.Annotation;
import java.lang.reflect.Type;

/**
 * Adapter that implements JVMCI ResolvedJavaMethod interface by delegating to TornadoResolvedMethod.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 * For bytecode access, it delegates to the real JVMCI ResolvedJavaMethod.
 */
public class JVMCIResolvedJavaMethodAdapter implements ResolvedJavaMethod {
    private final TornadoResolvedMethod tornadoMethod;
    private final JVMCIMetaAccessAdapter metaAccess;
    private ResolvedJavaMethod jvmciMethod;  // Lazily loaded for bytecode access

    public JVMCIResolvedJavaMethodAdapter(TornadoResolvedMethod tornadoMethod, JVMCIMetaAccessAdapter metaAccess) {
        this.tornadoMethod = tornadoMethod;
        this.metaAccess = metaAccess;
    }

    public TornadoResolvedMethod getTornadoMethod() {
        return tornadoMethod;
    }

    /**
     * Lazily get the real JVMCI ResolvedJavaMethod for bytecode access.
     */
    private ResolvedJavaMethod getJVMCIMethod() {
        if (jvmciMethod == null) {
            MetaAccessProvider jvmciMetaAccess = HotSpotJVMCIRuntime.runtime().getHostJVMCIBackend().getMetaAccess();
            jvmciMethod = jvmciMetaAccess.lookupJavaMethod(tornadoMethod.getExecutable());
        }
        return jvmciMethod;
    }

    @Override
    public String getName() {
        return tornadoMethod.getName();
    }

    @Override
    public ResolvedJavaType getDeclaringClass() {
        return new JVMCIResolvedJavaTypeAdapter(tornadoMethod.getDeclaringClass(), metaAccess);
    }

    @Override
    public Signature getSignature() {
        return new JVMCISignatureAdapter(tornadoMethod.getSignature(), metaAccess);
    }

    @Override
    public int getModifiers() {
        return tornadoMethod.getModifiers();
    }

    @Override
    public byte[] getCode() {
        // Delegate to real JVMCI method for bytecode access
        return getJVMCIMethod().getCode();
    }

    @Override
    public int getCodeSize() {
        // Delegate to real JVMCI method for bytecode access
        return getJVMCIMethod().getCodeSize();
    }

    @Override
    public int getMaxLocals() {
        // Delegate to real JVMCI method for bytecode access
        return getJVMCIMethod().getMaxLocals();
    }

    @Override
    public int getMaxStackSize() {
        // Delegate to real JVMCI method for bytecode access
        return getJVMCIMethod().getMaxStackSize();
    }

    @Override
    public boolean isConstructor() {
        return tornadoMethod.isConstructor();
    }

    @Override
    public boolean isClassInitializer() {
        return tornadoMethod.getName().equals("<clinit>");
    }

    @Override
    public boolean isDefault() {
        return false;  // Default methods are interface methods with implementation
    }

    @Override
    public boolean isStatic() {
        return tornadoMethod.isStatic();
    }

    @Override
    public boolean isFinal() {
        return tornadoMethod.isFinal();
    }

    @Override
    public boolean isSynchronized() {
        return tornadoMethod.isSynchronized();
    }

    @Override
    public boolean isBridge() {
        return tornadoMethod.isBridge();
    }

    @Override
    public boolean isVarArgs() {
        return tornadoMethod.isVarArgs();
    }

    @Override
    public boolean isSynthetic() {
        return tornadoMethod.isSynthetic();
    }

    @Override
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
        return tornadoMethod.getAnnotation(annotationClass);
    }

    @Override
    public Annotation[] getAnnotations() {
        return tornadoMethod.getAnnotations();
    }

    @Override
    public Annotation[] getDeclaredAnnotations() {
        return tornadoMethod.getDeclaredAnnotations();
    }

    @Override
    public Annotation[][] getParameterAnnotations() {
        return tornadoMethod.getParameterAnnotations();
    }

    // ========== Stubs for features not needed by TornadoVM ==========

    @Override
    public boolean canBeInlined() {
        return true;
    }

    @Override
    public boolean hasNeverInlineDirective() {
        return false;
    }

    @Override
    public boolean shouldBeInlined() {
        return true;
    }

    @Override
    public LineNumberTable getLineNumberTable() {
        return null;
    }

    @Override
    public LocalVariableTable getLocalVariableTable() {
        return null;
    }

    @Override
    public ConstantPool getConstantPool() {
        // Delegate to real JVMCI method for constant pool access
        return getJVMCIMethod().getConstantPool();
    }

    @Override
    public Parameter[] getParameters() {
        // Delegate to real JVMCI method for parameter information
        return getJVMCIMethod().getParameters();
    }

    @Override
    public void reprofile() {
        // Stub
    }

    @Override
    public boolean canBeStaticallyBound() {
        return tornadoMethod.isFinal() || tornadoMethod.isStatic();
    }

    @Override
    public ExceptionHandler[] getExceptionHandlers() {
        return new ExceptionHandler[0];
    }

    @Override
    public StackTraceElement asStackTraceElement(int bci) {
        return new StackTraceElement(
                tornadoMethod.getDeclaringClass().getName(),
                tornadoMethod.getName(),
                null,
                -1
        );
    }

    @Override
    public ProfilingInfo getProfilingInfo() {
        return null;
    }

    @Override
    public ProfilingInfo getProfilingInfo(boolean includeNormal, boolean includeOSR) {
        return null;
    }

    @Override
    public SpeculationLog getSpeculationLog() {
        return null;
    }

    @Override
    public boolean isInVirtualMethodTable(ResolvedJavaType resolved) {
        return false;  // Stub
    }

    @Override
    public Constant getEncoding() {
        return null;  // Stub
    }

    @Override
    public Type[] getGenericParameterTypes() {
        return null;  // Stub
    }

    @Override
    public String toString() {
        return "JVMCIResolvedJavaMethodAdapter<" + tornadoMethod.getName() + ">";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof JVMCIResolvedJavaMethodAdapter adapter) {
            return tornadoMethod.equals(adapter.tornadoMethod);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return tornadoMethod.hashCode();
    }
}
