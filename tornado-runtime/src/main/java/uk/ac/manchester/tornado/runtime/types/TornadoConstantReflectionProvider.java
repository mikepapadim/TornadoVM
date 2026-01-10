package uk.ac.manchester.tornado.runtime.types;

/**
 * TornadoVM's constant reflection provider.
 * This is a replacement for jdk.vm.ci.meta.ConstantReflectionProvider with NO JVMCI dependency.
 *
 * Most methods are intentionally minimal as TornadoVM doesn't rely heavily on constant folding.
 */
public interface TornadoConstantReflectionProvider {

    // ========== Basic constant operations ==========

    Boolean constantEquals(Object x, Object y);

    TornadoResolvedType asJavaType(Object constant);
}
