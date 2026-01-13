/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

package uk.ac.manchester.tornado.drivers.ptx.graal.lir;

import static uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssemblerConstants.BRANCH;
import static uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssemblerConstants.DOT;
import static uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssemblerConstants.TAB;
import static uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssemblerConstants.UNI;

import org.graalvm.compiler.lir.LIRInstructionClass;
import org.graalvm.compiler.lir.LabelRef;

import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.api.exceptions.TornadoInternalError;
import uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssembler;
import uk.ac.manchester.tornado.drivers.ptx.graal.compiler.PTXCompilationResultBuilder;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXLIRStmt.AbstractInstruction;

public class PTXControlFlow {

    protected static void emitBlockRef(LabelRef labelRef, PTXAssembler asm) {
        asm.emitBlock(labelRef.label().getBlockId());
    }

    public static class LoopLabel extends AbstractInstruction {
        public static final LIRInstructionClass<LoopLabel> TYPE = LIRInstructionClass.create(LoopLabel.class);

        private final int blockId;

        public LoopLabel(int blockId) {
            super(TYPE);
            this.blockId = blockId;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            asm.emitLoopLabel(blockId);
        }
    }

    public static class LoopBreakOp extends Branch {

        public LoopBreakOp(LabelRef destination, boolean isConditional, boolean isLoopEdgeBack) {
            super(destination, isConditional, isLoopEdgeBack);
        }
    }

    public static class Branch extends AbstractInstruction {
        public static final LIRInstructionClass<Branch> TYPE = LIRInstructionClass.create(Branch.class);
        private final LabelRef destination;
        private final boolean isConditional;
        private final boolean isLoopEdgeBack;

        public Branch(LabelRef destination, boolean isConditional, boolean isLoopEdgeBack) {
            super(TYPE);
            this.destination = destination;
            this.isConditional = isConditional;
            this.isLoopEdgeBack = isLoopEdgeBack;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == uk.ac.manchester.tornado.drivers.ptx.graal.backend.CodeGenMode.CUDA) {
                emitCUDA(asm);
            } else {
                emitPTX(asm);
            }
        }

        private void emitPTX(PTXAssembler asm) {
            asm.emitSymbol(TAB);
            asm.emit(BRANCH);
            if (!isConditional) {
                asm.emit(DOT + UNI);
            }
            asm.emitSymbol(TAB);

            if (isLoopEdgeBack) {
                asm.emitLoop(destination.label().getBlockId());
            } else {
                emitBlockRef(destination, asm);
            }
            asm.delimiter();
            asm.eol();
        }

        private void emitCUDA(PTXAssembler asm) {
            // CUDA: Convert branch to goto
            asm.emitCudaIndent();
            asm.emit("goto ");
            if (isLoopEdgeBack) {
                asm.emit("LOOP_COND_" + destination.label().getBlockId());
            } else {
                asm.emit("BLOCK_" + destination.label().getBlockId());
            }
            asm.emit(";");
            asm.eol();
        }
    }

    public static class DeoptOp extends AbstractInstruction {

        public static final LIRInstructionClass<DeoptOp> TYPE = LIRInstructionClass.create(DeoptOp.class);
        @Use
        private final Value actionAndReason;

        public DeoptOp(Value actionAndReason) {
            super(TYPE);
            this.actionAndReason = actionAndReason;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            TornadoInternalError.unimplemented();
        }

    }

    // ============================================================
    // Structured Control Flow Operations (CUDA Mode)
    // Based on OpenCL backend patterns for generating C-style code
    // ============================================================

    /**
     * Emits the start of a for loop: "for ("
     * After this, loop initialization statements are emitted,
     * followed by LoopConditionOp and loop increment, then LoopPostOp.
     */
    public static class LoopInitOp extends AbstractInstruction {

        public static final LIRInstructionClass<LoopInitOp> TYPE = LIRInstructionClass.create(LoopInitOp.class);

        public LoopInitOp() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.emitCudaIndent();
                asm.emit("for (");
                // Disable EOL and indent temporarily so init, condition, increment are on same line
                asm.indentOff();
                asm.eolOff();
            } else {
                // PTX mode doesn't use structured loops
            }
        }
    }

    /**
     * Emits the end of the for loop header: ") {"
     * This is called after loop initialization, condition, and increment have been emitted.
     */
    public static class LoopPostOp extends AbstractInstruction {

        public static final LIRInstructionClass<LoopPostOp> TYPE = LIRInstructionClass.create(LoopPostOp.class);

        public LoopPostOp() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.emit(") {");
                asm.indentOn();
                asm.eolOn();
                asm.eol();
                asm.increaseIndent();
            } else {
                // PTX mode doesn't use structured loops
            }
        }
    }

    /**
     * Emits a loop condition.
     * Can generate either:
     * 1. Just the condition (for for-loop header)
     * 2. An if-break statement inside the loop body
     */
    public static class LoopConditionOp extends AbstractInstruction {

        public static final LIRInstructionClass<LoopConditionOp> TYPE = LIRInstructionClass.create(LoopConditionOp.class);
        @Use
        private final Value condition;
        private boolean generateIfBreakStatement = true;

        public LoopConditionOp(Value condition) {
            super(TYPE);
            this.condition = condition;
        }

        public void setGenerateIfBreakStatement(boolean value) {
            this.generateIfBreakStatement = value;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                if (generateIfBreakStatement) {
                    // Generate: if (!condition) break;
                    asm.emitCudaIndent();
                    asm.emit("if (!(");
                    asm.emit(asm.toStringWithMode(condition));
                    asm.emit(")) break;");
                    asm.eol();
                } else {
                    // Just emit the condition (for for-loop header)
                    asm.emit(asm.toStringWithMode(condition));
                }
            } else {
                // PTX mode doesn't use structured loops
            }
        }
    }

    /**
     * Emits an if statement: "if (condition) {"
     */
    public static class ConditionalBranchOp extends AbstractInstruction {

        public static final LIRInstructionClass<ConditionalBranchOp> TYPE = LIRInstructionClass.create(ConditionalBranchOp.class);
        @Use
        private final Value condition;

        public ConditionalBranchOp(Value condition) {
            super(TYPE);
            this.condition = condition;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.emitCudaIndent();
                asm.emit("if (");
                asm.emit(asm.toStringWithMode(condition));
                asm.emit(") {");
                asm.eol();
                asm.increaseIndent();
            } else {
                // PTX mode uses conditional branches
            }
        }
    }

    /**
     * Emits an else-if statement: "} else if (condition) {"
     */
    public static class LinkedConditionalBranchOp extends AbstractInstruction {

        public static final LIRInstructionClass<LinkedConditionalBranchOp> TYPE = LIRInstructionClass.create(LinkedConditionalBranchOp.class);
        @Use
        private final Value condition;

        public LinkedConditionalBranchOp(Value condition) {
            super(TYPE);
            this.condition = condition;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.decreaseIndent();
                asm.emitCudaIndent();
                asm.emit("} else if (");
                asm.emit(asm.toStringWithMode(condition));
                asm.emit(") {");
                asm.eol();
                asm.increaseIndent();
            } else {
                // PTX mode uses conditional branches
            }
        }
    }

    /**
     * Emits an else statement: "} else {"
     */
    public static class ElseBranchOp extends AbstractInstruction {

        public static final LIRInstructionClass<ElseBranchOp> TYPE = LIRInstructionClass.create(ElseBranchOp.class);

        public ElseBranchOp() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.decreaseIndent();
                asm.emitCudaIndent();
                asm.emit("} else {");
                asm.eol();
                asm.increaseIndent();
            } else {
                // PTX mode uses conditional branches
            }
        }
    }

    /**
     * Begins a scope: "{"
     */
    public static class BeginScopeOp extends AbstractInstruction {

        public static final LIRInstructionClass<BeginScopeOp> TYPE = LIRInstructionClass.create(BeginScopeOp.class);

        public BeginScopeOp() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.emitCudaIndent();
                asm.emit("{");
                asm.eol();
                asm.increaseIndent();
            } else {
                // PTX mode doesn't use scopes
            }
        }
    }

    /**
     * Ends a scope: "}"
     */
    public static class EndScopeOp extends AbstractInstruction {

        public static final LIRInstructionClass<EndScopeOp> TYPE = LIRInstructionClass.create(EndScopeOp.class);

        public EndScopeOp() {
            super(TYPE);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
                asm.decreaseIndent();
                asm.emitCudaIndent();
                asm.emit("}");
                asm.eol();
            } else {
                // PTX mode doesn't use scopes
            }
        }
    }
}
