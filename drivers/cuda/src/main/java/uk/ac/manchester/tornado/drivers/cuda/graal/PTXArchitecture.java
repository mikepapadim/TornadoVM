package uk.ac.manchester.tornado.drivers.cuda.graal;

import jdk.vm.ci.code.Architecture;
import jdk.vm.ci.code.Register.RegisterCategory;
import jdk.vm.ci.meta.JavaKind;
import jdk.vm.ci.meta.PlatformKind;
import org.graalvm.compiler.core.common.LIRKind;
import org.graalvm.compiler.lir.Variable;
import org.graalvm.compiler.lir.gen.LIRGeneratorTool;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.PTXAssemblerConstants;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.PTXKind;
import uk.ac.manchester.tornado.drivers.cuda.graal.lir.PTXLIRStmt;
import uk.ac.manchester.tornado.drivers.cuda.graal.meta.PTXMemorySpace;

import java.nio.ByteOrder;

import static jdk.vm.ci.code.MemoryBarriers.LOAD_STORE;
import static jdk.vm.ci.code.MemoryBarriers.STORE_STORE;
import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.shouldNotReachHere;

public class PTXArchitecture extends Architecture {

    public static final RegisterCategory PTX_ABI = new RegisterCategory("abi");

    public static final PTXMemoryBase globalSpace = new PTXMemoryBase(0, PTXMemorySpace.GLOBAL);
    public static final PTXMemoryBase paramSpace = new PTXMemoryBase(1, PTXMemorySpace.PARAM);

    public static PTXParam HEAP_POINTER;
    public static PTXParam STACK_POINTER;
    public static PTXParam[] abiRegisters;

    public static PTXBuiltInRegister ThreadIDX = new PTXBuiltInRegister("%tid.x");
    public static PTXBuiltInRegister ThreadIDY = new PTXBuiltInRegister("%tid.y");
    public static PTXBuiltInRegister ThreadIDZ = new PTXBuiltInRegister("%tid.z");

    public static PTXBuiltInRegister BlockDimX = new PTXBuiltInRegister("%ntid.x");
    public static PTXBuiltInRegister BlockDimY = new PTXBuiltInRegister("%ntid.y");
    public static PTXBuiltInRegister BlockDimZ = new PTXBuiltInRegister("%ntid.z");

    public static PTXBuiltInRegister BlockIDX = new PTXBuiltInRegister("%ctaid.x");
    public static PTXBuiltInRegister BlockIDY = new PTXBuiltInRegister("%ctaid.y");
    public static PTXBuiltInRegister BlockIDZ = new PTXBuiltInRegister("%ctaid.z");

    public static PTXBuiltInRegister GridDimX = new PTXBuiltInRegister("%nctaid.x");
    public static PTXBuiltInRegister GridDimY = new PTXBuiltInRegister("%nctaid.y");
    public static PTXBuiltInRegister GridDimZ = new PTXBuiltInRegister("%nctaid.z");

    public PTXArchitecture(PTXKind wordKind, ByteOrder byteOrder) {
        super("Tornado PTX",
                wordKind,
                byteOrder,
                false,
                null,
                LOAD_STORE | STORE_STORE,
                0,
                0
        );

        HEAP_POINTER = new PTXParam(PTXAssemblerConstants.HEAP_PTR_NAME, wordKind);
        STACK_POINTER = new PTXParam(PTXAssemblerConstants.STACK_PTR_NAME, wordKind);

        abiRegisters = new PTXParam[]{HEAP_POINTER, STACK_POINTER};
    }

    @Override
    public boolean canStoreValue(RegisterCategory category, PlatformKind kind) {
        return false;
    }

    @Override
    public PlatformKind getLargestStorableKind(RegisterCategory category) {
        return null;
    }

    @Override
    public PlatformKind getPlatformKind(JavaKind javaKind) {
        return null;
    }

    public String getABI() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < abiRegisters.length; i++) {
            sb.append(abiRegisters[i].getDeclaration());
            if (i < abiRegisters.length - 1) {
                sb.append(", ");
            }
        }
        return sb.toString();
    }

    public static class PTXRegister {
        public final int number;
        protected String name;
        public final PTXKind lirKind;

        public PTXRegister(int number, PTXKind lirKind) {
            this.number = number;
            this.lirKind = lirKind;
            this.name = "r" + lirKind.getTypeChar() + number;
        }

        public String getDeclaration() {
            return String.format(".reg .%s %s", lirKind.toString(), name);
        }

        public String getName() {
            return name;
        }
    }

    public static class PTXParam extends PTXRegister {

        private Variable allocatedTo;

        public PTXParam(String name, PTXKind lirKind) {
            super(0, lirKind);
            this.name = name;
        }

        @Override
        public String getDeclaration() {
            return String.format(".param .%s %s", lirKind.toString(), name);
        }

        public Variable getAllocatedVar() {
            return allocatedTo; }

        public void allocateToVar(Variable var) {
            allocatedTo = var;
        }
    }

    public static class PTXMemoryBase extends PTXRegister {

        public final PTXMemorySpace memorySpace;

        public PTXMemoryBase(int number, PTXMemorySpace memorySpace) {
            super(number, PTXKind.B64);
            this.memorySpace = memorySpace;
        }
    }

    public static class PTXBuiltInRegister extends Variable {
        private Variable allocatedTo;

        protected PTXBuiltInRegister(String name) {
            super(LIRKind.value(PTXKind.U32), 0);
            setName(name);
        }

        public Variable getAllocatedTo(LIRGeneratorTool tool) {
            if (allocatedTo == null) {
                allocatedTo = tool.newVariable(this.getValueKind());
                tool.append(new PTXLIRStmt.AssignStmt(allocatedTo, this));
            }

            return allocatedTo;
        }
    }

    public static class PTXBuiltInRegisterArray {
        public final PTXBuiltInRegister threadID;
        public final PTXBuiltInRegister blockID;
        public final PTXBuiltInRegister blockDim;
        public final PTXBuiltInRegister gridDim;

        public PTXBuiltInRegisterArray(int dim) {
            switch (dim) {
                case 0:
                    threadID = PTXArchitecture.ThreadIDX;
                    blockDim = PTXArchitecture.BlockDimX;
                    blockID = PTXArchitecture.BlockIDX;
                    gridDim = PTXArchitecture.GridDimX;
                    break;
                case 1:
                    threadID = PTXArchitecture.ThreadIDY;
                    blockDim = PTXArchitecture.BlockDimY;
                    blockID = PTXArchitecture.BlockIDY;
                    gridDim = PTXArchitecture.GridDimY;
                    break;
                case 2:
                    threadID = PTXArchitecture.ThreadIDZ;
                    blockDim = PTXArchitecture.BlockDimZ;
                    blockID = PTXArchitecture.BlockIDZ;
                    gridDim = PTXArchitecture.GridDimZ;
                    break;
                default:
                    shouldNotReachHere("Too many dimensions: %d", dim);
                    threadID = null;
                    blockDim = null;
                    blockID = null;
                    gridDim = null;
                    break;
            }
        }
    }
}
