package uk.ac.manchester.tornado.runtime.graal;

import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.JavaType;
import jdk.vm.ci.meta.ResolvedJavaType;
import jdk.vm.ci.meta.Signature;
import uk.ac.manchester.tornado.runtime.types.TornadoSignature;

/**
 * Adapter that implements JVMCI Signature interface by delegating to TornadoSignature.
 * This allows Graal compiler to work with Tornado's JVMCI-free type system.
 */
public class JVMCISignatureAdapter implements Signature {
    private final TornadoSignature tornadoSignature;
    private final JVMCIMetaAccessAdapter metaAccess;

    public JVMCISignatureAdapter(TornadoSignature tornadoSignature, JVMCIMetaAccessAdapter metaAccess) {
        this.tornadoSignature = tornadoSignature;
        this.metaAccess = metaAccess;
    }

    @Override
    public int getParameterCount(boolean receiver) {
        return tornadoSignature.getParameterCount(receiver);
    }

    @Override
    public JavaType getParameterType(int index, ResolvedJavaType accessingClass) {
        return new JVMCIResolvedJavaTypeAdapter(tornadoSignature.getParameterType(index), metaAccess);
    }

    @Override
    public JavaType getReturnType(ResolvedJavaType accessingClass) {
        return new JVMCIResolvedJavaTypeAdapter(tornadoSignature.getReturnType(), metaAccess);
    }

    @Override
    public JavaKind getParameterKind(int index) {
        return JVMCIMetaAccessAdapter.convertTornadoJavaKind(tornadoSignature.getParameterKind(index));
    }

    @Override
    public JavaKind getReturnKind() {
        return JVMCIMetaAccessAdapter.convertTornadoJavaKind(tornadoSignature.getReturnKind());
    }

    @Override
    public String toMethodDescriptor() {
        return tornadoSignature.toMethodDescriptor();
    }

    @Override
    public String toString() {
        return "JVMCISignatureAdapter<" + tornadoSignature.toMethodDescriptor() + ">";
    }
}
