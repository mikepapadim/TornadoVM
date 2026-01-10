package uk.ac.manchester.tornado.runtime.types;

/**
 * TornadoVM's representation of a method signature.
 * This is a replacement for jdk.vm.ci.meta.Signature with NO JVMCI dependency.
 */
public interface TornadoSignature {

    int getParameterCount(boolean receiver);

    TornadoResolvedType getParameterType(int index);

    TornadoResolvedType getReturnType();

    TornadoJavaKind getParameterKind(int index);

    TornadoJavaKind getReturnKind();

    String toMethodDescriptor();
}
