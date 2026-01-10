package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.meta.*;
import uk.ac.manchester.tornado.runtime.types.TornadoConstantReflectionProvider;
import uk.ac.manchester.tornado.runtime.types.TornadoResolvedType;

/**
 * Adapter that implements JVMCI ConstantReflectionProvider interface by delegating to TornadoConstantReflectionProvider.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 */
public class JVMCIConstantReflectionAdapter implements ConstantReflectionProvider {
    private final TornadoConstantReflectionProvider tornadoConstantReflection;
    private final JVMCIMetaAccessAdapter metaAccess;

    public JVMCIConstantReflectionAdapter(TornadoConstantReflectionProvider tornadoConstantReflection, JVMCIMetaAccessAdapter metaAccess) {
        this.tornadoConstantReflection = tornadoConstantReflection;
        this.metaAccess = metaAccess;
    }

    @Override
    public Boolean constantEquals(Constant x, Constant y) {
        return tornadoConstantReflection.constantEquals(x, y);
    }

    @Override
    public Integer readArrayLength(JavaConstant array) {
        return null;  // Cannot read array length without HotSpot internals
    }

    @Override
    public JavaConstant readArrayElement(JavaConstant array, int index) {
        return null;  // Cannot read array elements without HotSpot internals
    }

    @Override
    public JavaConstant readFieldValue(ResolvedJavaField field, JavaConstant receiver) {
        return null;  // Cannot read field values without HotSpot internals
    }

    @Override
    public JavaConstant boxPrimitive(JavaConstant source) {
        return null;  // Boxing not needed for TornadoVM
    }

    @Override
    public JavaConstant unboxPrimitive(JavaConstant source) {
        return null;  // Unboxing not needed for TornadoVM
    }

    @Override
    public JavaConstant forString(String value) {
        return null;  // String constant creation not needed for TornadoVM
    }

    @Override
    public ResolvedJavaType asJavaType(Constant constant) {
        TornadoResolvedType tornadoType = tornadoConstantReflection.asJavaType(constant);
        if (tornadoType == null) {
            return null;
        }
        return new JVMCIResolvedJavaTypeAdapter(tornadoType, metaAccess);
    }

    @Override
    public MethodHandleAccessProvider getMethodHandleAccess() {
        return null;  // Method handle access not needed for TornadoVM
    }

    @Override
    public MemoryAccessProvider getMemoryAccessProvider() {
        return null;  // Memory access provider not needed for TornadoVM
    }

    @Override
    public JavaConstant asJavaClass(ResolvedJavaType type) {
        return null;  // Cannot create JavaConstant for Class<?> without HotSpot internals
    }

    @Override
    public Constant asObjectHub(ResolvedJavaType type) {
        return null;  // Object hub (klass pointer) not needed for TornadoVM
    }
}
