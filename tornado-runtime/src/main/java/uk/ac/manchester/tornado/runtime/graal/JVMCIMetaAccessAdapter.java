package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.meta.*;
import uk.ac.manchester.tornado.runtime.types.*;

import java.lang.reflect.Executable;
import java.lang.reflect.Field;

/**
 * Adapter that implements JVMCI MetaAccessProvider interface by delegating to TornadoMetaAccessProvider.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 */
public class JVMCIMetaAcc   essAdapter implements MetaAccessProvider {
    private final TornadoMetaAccessProvider tornadoMetaAccess;

    public JVMCIMetaAccessAdapter(TornadoMetaAccessProvider tornadoMetaAccess) {
        this.tornadoMetaAccess = tornadoMetaAccess;
    }

    @Override
    public ResolvedJavaType lookupJavaType(Class<?> clazz) {
        TornadoResolvedType tornadoType = tornadoMetaAccess.lookupJavaType(clazz);
        return new JVMCIResolvedJavaTypeAdapter(tornadoType, this);
    }

    @Override
    public ResolvedJavaMethod lookupJavaMethod(Executable executable) {
        TornadoResolvedMethod tornadoMethod = tornadoMetaAccess.lookupJavaMethod(executable);
        return new JVMCIResolvedJavaMethodAdapter(tornadoMethod, this);
    }

    @Override
    public ResolvedJavaField lookupJavaField(Field field) {
        TornadoResolvedField tornadoField = tornadoMetaAccess.lookupJavaField(field);
        return new JVMCIResolvedJavaFieldAdapter(tornadoField, this);
    }

    @Override
    public ResolvedJavaType lookupJavaType(JavaConstant constant) {
        // Delegate to real JVMCI MetaAccessProvider for JavaConstant lookups
        // This requires access to HotSpot internals which Tornado types don't have
        jdk.vm.ci.meta.MetaAccessProvider jvmciMetaAccess = jdk.vm.ci.hotspot.HotSpotJVMCIRuntime.runtime().getHostJVMCIBackend().getMetaAccess();
        return jvmciMetaAccess.lookupJavaType(constant);
    }

    @Override
    public int getArrayBaseOffset(JavaKind kind) {
        TornadoJavaKind tornadoKind = convertJavaKind(kind);
        return tornadoMetaAccess.getArrayBaseOffset(tornadoKind);
    }

    @Override
    public int getArrayIndexScale(JavaKind kind) {
        TornadoJavaKind tornadoKind = convertJavaKind(kind);
        return tornadoMetaAccess.getArrayIndexScale(tornadoKind);
    }

    @Override
    public JavaConstant encodeDeoptActionAndReason(DeoptimizationAction action, DeoptimizationReason reason, int debugId) {
        return null;  // Not needed
    }

    @Override
    public JavaConstant encodeSpeculation(SpeculationLog.Speculation speculation) {
        return null;  // Not needed
    }

    @Override
    public SpeculationLog.Speculation decodeSpeculation(JavaConstant constant, SpeculationLog speculationLog) {
        return null;  // Not needed
    }

    @Override
    public DeoptimizationReason decodeDeoptReason(JavaConstant constant) {
        return null;  // Not needed
    }

    @Override
    public DeoptimizationAction decodeDeoptAction(JavaConstant constant) {
        return null;  // Not needed
    }

    @Override
    public int decodeDebugId(JavaConstant constant) {
        return 0;  // Not needed
    }

    @Override
    public long getMemorySize(JavaConstant constant) {
        return 0;  // Not needed
    }

    @Override
    public Signature parseMethodDescriptor(String methodDescriptor) {
        throw new UnsupportedOperationException("parseMethodDescriptor not implemented");
    }

    // Helper method to convert JVMCI JavaKind to Tornado JavaKind
    static TornadoJavaKind convertJavaKind(JavaKind kind) {
        return switch (kind) {
            case Boolean -> TornadoJavaKind.Boolean;
            case Byte -> TornadoJavaKind.Byte;
            case Short -> TornadoJavaKind.Short;
            case Char -> TornadoJavaKind.Char;
            case Int -> TornadoJavaKind.Int;
            case Long -> TornadoJavaKind.Long;
            case Float -> TornadoJavaKind.Float;
            case Double -> TornadoJavaKind.Double;
            case Object -> TornadoJavaKind.Object;
            case Void -> TornadoJavaKind.Void;
            default -> TornadoJavaKind.Illegal;
        };
    }

    // Helper method to convert Tornado JavaKind to JVMCI JavaKind
    static JavaKind convertTornadoJavaKind(TornadoJavaKind kind) {
        return switch (kind) {
            case Boolean -> JavaKind.Boolean;
            case Byte -> JavaKind.Byte;
            case Short -> JavaKind.Short;
            case Char -> JavaKind.Char;
            case Int -> JavaKind.Int;
            case Long -> JavaKind.Long;
            case Float -> JavaKind.Float;
            case Double -> JavaKind.Double;
            case Object -> JavaKind.Object;
            case Void -> JavaKind.Void;
            default -> JavaKind.Illegal;
        };
    }
}
