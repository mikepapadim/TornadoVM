package uk.ac.manchester.tornado.runtime.types;

/**
 * Reflection-based implementation of TornadoConstantReflectionProvider.
 * Uses pure Java reflection with NO JVMCI dependency.
 *
 * Most methods are minimal as TornadoVM doesn't rely heavily on constant folding.
 */
public class TornadoReflectionConstantReflectionProvider implements TornadoConstantReflectionProvider {
    private final TornadoReflectionMetaAccessProvider metaAccess;

    public TornadoReflectionConstantReflectionProvider(TornadoReflectionMetaAccessProvider metaAccess) {
        this.metaAccess = metaAccess;
    }

    @Override
    public Boolean constantEquals(Object x, Object y) {
        if (x == null && y == null) return true;
        if (x == null || y == null) return false;
        return x.equals(y);
    }

    @Override
    public TornadoResolvedType asJavaType(Object constant) {
        if (constant instanceof Class<?> clazz) {
            return metaAccess.lookupJavaType(clazz);
        }
        return null;
    }
}
