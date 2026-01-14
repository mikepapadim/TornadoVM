/*
 * Copyright (c) 2020-2022 APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

package uk.ac.manchester.tornado.drivers.ptx.graal.compiler;

import static uk.ac.manchester.tornado.runtime.graal.TornadoLIRGenerator.trace;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.stream.IntStream;

import org.graalvm.collections.EconomicMap;
import org.graalvm.collections.Equivalence;
import org.graalvm.compiler.asm.Assembler;
import org.graalvm.compiler.code.CompilationResult;
import org.graalvm.compiler.core.common.spi.CodeGenProviders;
import org.graalvm.compiler.debug.DebugContext;
import org.graalvm.compiler.lir.InstructionValueProcedure;
import org.graalvm.compiler.lir.LIR;
import org.graalvm.compiler.lir.LIRInstruction;
import org.graalvm.compiler.lir.LIRInstruction.OperandFlag;
import org.graalvm.compiler.lir.LIRInstruction.OperandMode;
import org.graalvm.compiler.lir.Variable;
import org.graalvm.compiler.lir.asm.CompilationResultBuilder;
import org.graalvm.compiler.lir.asm.DataBuilder;
import org.graalvm.compiler.lir.asm.FrameContext;
import org.graalvm.compiler.lir.framemap.FrameMap;
import org.graalvm.compiler.nodes.AbstractBeginNode;
import org.graalvm.compiler.nodes.AbstractEndNode;
import org.graalvm.compiler.nodes.ControlSplitNode;
import org.graalvm.compiler.nodes.EndNode;
import org.graalvm.compiler.nodes.FixedNode;
import org.graalvm.compiler.nodes.IfNode;
import org.graalvm.compiler.nodes.LoopBeginNode;
import org.graalvm.compiler.nodes.LoopEndNode;
import org.graalvm.compiler.nodes.LoopExitNode;
import org.graalvm.compiler.nodes.MergeNode;
import org.graalvm.compiler.nodes.cfg.ControlFlowGraph;
import org.graalvm.compiler.nodes.cfg.HIRBlock;
import org.graalvm.compiler.options.OptionValues;

import jdk.vm.ci.code.Register;
import jdk.vm.ci.meta.ResolvedJavaMethod;
import jdk.vm.ci.meta.Value;
import uk.ac.manchester.tornado.api.exceptions.TornadoInternalError;
import uk.ac.manchester.tornado.drivers.ptx.PTXDeviceContext;
import uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssembler;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXControlFlow;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXControlFlow.LoopConditionOp;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXControlFlow.LoopInitOp;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXControlFlow.LoopPostOp;
import uk.ac.manchester.tornado.drivers.ptx.graal.lir.PTXLIRStmt.AssignStmt;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class PTXCompilationResultBuilder extends CompilationResultBuilder {
    protected LIR lir;
    HashSet<HIRBlock> rescheduledBasicBlocks;
    private boolean isKernel;
    private boolean isParallel;
    private Set<ResolvedJavaMethod> nonInlinedMethods;
    private PTXAssembler asm;
    private PTXDeviceContext deviceContext;
    private boolean includePrintf;
    private PTXLIRGenerationResult lirGenRes;
    private TaskDataContext meta;

    public PTXCompilationResultBuilder(CodeGenProviders providers, FrameMap frameMap, Assembler asm, DataBuilder dataBuilder, FrameContext frameContext, OptionValues options, DebugContext debug,
            CompilationResult compilationResult, LIR lir) {
        super(providers, frameMap, asm, dataBuilder, frameContext, options, debug, compilationResult, Register.None, EconomicMap.create(Equivalence.DEFAULT), NO_VERIFIERS, lir);

        nonInlinedMethods = new HashSet<>();
        this.asm = (PTXAssembler) asm;
    }

    private static void emitOp(CompilationResultBuilder crb, LIRInstruction op) {
        try {
            trace("op: " + op);
            op.emitCode(crb);
        } catch (AssertionError | RuntimeException t) {
            throw new TornadoInternalError(t);
        }
    }

    private static boolean isLoopBlock(HIRBlock HIRBlock, HIRBlock loopHeader) {

        Set<HIRBlock> visited = new HashSet<>();
        Stack<HIRBlock> stack = new Stack<>();
        stack.push(HIRBlock);

        while (!stack.isEmpty()) {

            HIRBlock b = stack.pop();
            visited.add(b);

            if (b.getId() < loopHeader.getId()) {
                return false;
            } else if (b == loopHeader) {
                return true;
            } else {
                HIRBlock[] successors = new HIRBlock[b.getSuccessorCount()];
                for (int i = 0; i < b.getSuccessorCount(); i++) {
                    successors[i] = b.getSuccessorAt(i);
                }
                for (HIRBlock bl : successors) {
                    if (!visited.contains(bl)) {
                        stack.push(bl);
                    }
                }
            }
        }
        return false;
    }

    public PTXLIRGenerationResult getPTXLIRGenerationResult() {
        return this.lirGenRes;
    }

    public void setPTXLIRGenerationResult(PTXLIRGenerationResult result) {
        this.lirGenRes = result;
    }

    public PTXAssembler getAssembler() {
        return asm;
    }

    public boolean getIncludePrintf() {
        return includePrintf;
    }

    public void setIncludePrintf(boolean value) {
        this.includePrintf = value;
    }

    public boolean getParallel() {
        return isParallel;
    }

    public void setParallel(boolean value) {
        isParallel = value;
    }

    public void addNonInlinedMethod(ResolvedJavaMethod method) {
        nonInlinedMethods.add(method);
    }

    public Set<ResolvedJavaMethod> getNonInlinedMethods() {
        return nonInlinedMethods;
    }

    public boolean isKernel() {
        return isKernel;
    }

    public void setKernel(boolean value) {
        isKernel = value;
    }

    /**
     * Emits code for {@code lir} in its {@linkplain LIR#codeEmittingOrder() code
     * emitting order}.
     */
    public void emit(LIR lir) {
        assert this.lir == null;
        assert currentBlockIndex == 0;
        this.lir = lir;
        this.currentBlockIndex = 0;
        frameContext.enter(this);

        final ControlFlowGraph cfg = (ControlFlowGraph) lir.getControlFlowGraph();
        trace("Traversing CFG: ", cfg.graph.name);

        cfg.computePostdominators();
        traverseControlFlowGraph(cfg, new PTXBlockVisitor(this, asm));

        trace("Finished traversing CFG");
        this.lir = null;
        this.currentBlockIndex = 0;
    }

    @Override
    public void finish() {
        int position = asm.position();
        compilationResult.setTargetCode(asm.close(true), position);
    }

    void emitBlock(HIRBlock HIRBlock) {
        if (HIRBlock == null) {
            return;
        }

        trace("HIRBlock: %d", HIRBlock.getId());

        if (Options.PrintLIRWithAssembly.getValue(getOptions())) {
            blockComment(String.format("HIRBlock B%d %s", HIRBlock.getId(), HIRBlock.getLoop()));
        }

        LIRInstruction breakInst = null;

        for (LIRInstruction op : lir.getLIRforBlock(HIRBlock)) {
            if (op != null) {
                if (Options.PrintLIRWithAssembly.getValue(getOptions())) {
                    blockComment(String.format("%d %s", op.id(), op));
                }

                if (op instanceof PTXControlFlow.LoopBreakOp) {
                    breakInst = op;
                    continue;
                }

                try {
                    emitOp(this, op);
                } catch (TornadoInternalError e) {
                    throw e.addContext("lir instruction", HIRBlock + "@" + op.id() + " " + op + "\n");
                }
            }
        }

        /*
         * Because of the way Graal handles Phi nodes, we generate the break instruction
         * before any phi nodes are updated, therefore we need to ensure that the break
         * is emitted as the end of the HIRBlock.
         */
        if (breakInst != null) {
            try {
                emitOp(this, breakInst);
            } catch (TornadoInternalError e) {
                throw e.addContext("lir instruction", HIRBlock + "@" + breakInst.id() + " " + breakInst + "\n");
            }
        }
    }

    /**
     * Checks if loop header can be formatted into structured for-loop.
     * Returns true if LoopConditionOp is close enough to LoopInitOp/LoopPostOp
     * allowing only LoopLabel and a few AssignStmts in between.
     *
     * @return true if the loop can be formatted.
     */
    private static boolean shouldFormatLoopHeader(List<LIRInstruction> instructions) {
        int loopInitOpIndex = -1, loopPostOpIndex = -1, loopConditionOpIndex = -1;

        for (int index = 0, instructionsSize = instructions.size(); index < instructionsSize; index++) {
            LIRInstruction instruction = instructions.get(index);
            if (instruction instanceof LoopInitOp) {
                loopInitOpIndex = index;
            }
            if (instruction instanceof LoopPostOp) {
                loopPostOpIndex = index;
            }
            if (instruction instanceof LoopConditionOp) {
                loopConditionOpIndex = index;
            }
        }

        // Check if we found all required operations
        if (loopInitOpIndex == -1 || loopPostOpIndex == -1 || loopConditionOpIndex == -1) {
            return false;
        }

        // Check order: LoopInitOp, LoopPostOp, ..., LoopConditionOp
        if (loopInitOpIndex >= loopPostOpIndex || loopPostOpIndex >= loopConditionOpIndex) {
            return false;
        }

        // Count instructions between LoopPostOp and LoopConditionOp
        // Allow: LoopLabel (ignored in CUDA mode) and a few AssignStmts
        int betweenCount = 0;
        for (int i = loopPostOpIndex + 1; i < loopConditionOpIndex; i++) {
            LIRInstruction instr = instructions.get(i);
            if (instr instanceof PTXControlFlow.LoopLabel) {
                // LoopLabel is a no-op in CUDA mode, skip it
                continue;
            }
            if (instr instanceof AssignStmt) {
                // AssignStmts can be moved by formatLoopHeader
                betweenCount++;
                continue;
            }
            // Other instructions prevent formatting
            return false;
        }

        // Allow up to 3 AssignStmts between (for condition calculation)
        return betweenCount <= 3;
    }

    /**
     * Checks if the loop condition is right after the loop header instructions.
     *
     * @return true if the {@param loopCondIndex} is right after the LIR
     *     instructions of a loop header ({@param loopPostOpIndex} and
     *     {@param loopInitOpIndex}).
     */
    private static boolean isLoopConditionRightAfterHeader(int loopCondIndex, int loopPostOpIndex, int loopInitOpIndex) {
        return (loopCondIndex - 1 == loopPostOpIndex) && (loopCondIndex - 2 == loopInitOpIndex);
    }

    /**
     * Formats a loop header by reorganizing instructions to create structured for-loop syntax.
     * This method:
     * 1. Finds the LoopConditionOp and tells it not to generate if/break
     * 2. Moves condition-dependent instructions after LoopPostOp
     * 3. Reorders instructions to form: for (init; condition; increment)
     */
    private static void formatLoopHeader(List<LIRInstruction> instructions) {
        int index = instructions.size() - 1;

        // Find the LoopConditionOp
        LIRInstruction condition = instructions.get(index);
        while (!(condition instanceof LoopConditionOp)) {
            index--;
            condition = instructions.get(index);
        }
        // Tell it to emit just the condition, not if/break
        ((LoopConditionOp) condition).setGenerateIfBreakStatement(false);

        // Remove condition from current position
        instructions.remove(index);

        // Find all values that the condition depends on
        final Set<Value> dependencies = new HashSet<>();
        DepFinder df = new DepFinder(dependencies);
        condition.forEachInput(df);

        // Move dependent assignments to be right before LoopPostOp
        index--;
        final List<LIRInstruction> moved = new ArrayList<>();
        LIRInstruction insn = instructions.get(index);
        while (!(insn instanceof LoopPostOp)) {
            if (insn instanceof AssignStmt) {
                AssignStmt assign = (AssignStmt) insn;
                if (assign.getResult() instanceof Variable) {
                    Variable var = (Variable) assign.getResult();
                    if (dependencies.contains(var)) {
                        moved.add(instructions.remove(index));
                    }
                }
            }
            index--;
            insn = instructions.get(index);
        }

        // Find LoopInitOp
        LIRInstruction loopInit = instructions.get(instructions.size() - 1);
        while (!(loopInit instanceof LoopInitOp)) {
            index--;
            loopInit = instructions.get(index);
        }

        // Insert condition after LoopInitOp
        instructions.add(index + 1, condition);
        // Insert moved dependencies before LoopPostOp
        instructions.addAll(index - 1, moved);
    }

    /**
     * Helper class to find instruction dependencies.
     */
    private static class DepFinder implements InstructionValueProcedure {
        private final Set<Value> dependencies;

        DepFinder(final Set<Value> dependencies) {
            this.dependencies = dependencies;
        }

        @Override
        public Value doValue(LIRInstruction instruction, Value value, OperandMode mode, EnumSet<OperandFlag> flags) {
            if (value instanceof Variable) {
                dependencies.add(value);
            }
            return value;
        }

        public Set<Value> getDependencies() {
            return dependencies;
        }
    }

    /**
     * Emits a loop header block with potential loop header formatting.
     * Used by structured control flow in CUDA mode.
     */
    void emitLoopBlock(HIRBlock block) {
        final List<LIRInstruction> headerInstructions = lir.getLIRforBlock(block);

        // TODO: Loop header formatting (for (init; cond; incr)) causes variable ordering issues
        // For now, use simple for() { if-break } structure which compiles successfully
        // Future work: Port OCL's loop header formatting correctly

        emitBlock(block);
    }

    private void traverseControlFlowGraph(ControlFlowGraph cfg, PTXBlockVisitor visitor) {
        traverseControlFlowGraph(cfg.getStartBlock(), visitor, new HashSet<>(), new HashMap<>());
        if (rescheduledBasicBlocks != null) {
            rescheduledBasicBlocks.clear();
        }
    }

    private void rescheduleBasicBlock(HIRBlock block, PTXBlockVisitor visitor, HashSet<HIRBlock> visited, HashMap<HIRBlock, HIRBlock> pending) {
        HIRBlock HIRBlock = pending.get(block);
        visitor.enter(HIRBlock);
        visitor.exit(HIRBlock, null);
        visited.add(HIRBlock);
        pending.remove(HIRBlock);
        if (rescheduledBasicBlocks == null) {
            rescheduledBasicBlocks = new HashSet<>();
        }
        rescheduledBasicBlocks.add(HIRBlock);
    }

    private HIRBlock getBlockTrueBranch(HIRBlock basicHIRBlock) {
        IfNode ifNode = (IfNode) basicHIRBlock.getDominator().getEndNode();
        for (int i = 0; i < basicHIRBlock.getDominator().getSuccessorCount(); i++) {
            if (ifNode.trueSuccessor() == basicHIRBlock.getDominator().getSuccessorAt(i).getBeginNode()) {
                return basicHIRBlock.getDominator().getSuccessorAt(i);
            }
        }
        return null;
    }

    private boolean isFalseSuccessorWithLoopEnd(IfNode ifNode, HIRBlock HIRBlock) {
        return isCurrentBlockAFalseBranch(ifNode, HIRBlock) && HIRBlock.getEndNode() instanceof LoopEndNode;
    }

    private boolean isCurrentBlockAFalseBranch(IfNode ifNode, HIRBlock HIRBlock) {
        return ifNode.falseSuccessor() == HIRBlock.getBeginNode();
    }

    private boolean isTrueBranchALoopExitNode(IfNode ifNode) {
        return ifNode.trueSuccessor() instanceof AbstractBeginNode;
    }

    private boolean isTrueBranchWithEndNodeOrNotControlSplit(HIRBlock blockTrueBranch) {
        return ((blockTrueBranch.getEndNode() instanceof AbstractEndNode) || !(blockTrueBranch.getEndNode() instanceof ControlSplitNode));
    }

    /**
     * From Graal 22.1.0 the graph traversal was changed. This method reschedules
     * the current basic HIRBlock to generate always the true condition before the
     * false condition only if we have a LoopEndNode node in the false branch, or we
     * have a LoopExit in the true branch contains a LoopExitNode or it is not a
     * control Split (due to nested control-flow).
     *
     * @param HIRBlock
     *     {@link HIRBlock}
     * @param visitor
     *     {@link PTXBlockVisitor}
     * @param visited
     *     {@link HashSet}
     * @param pending
     *     {@link HashMap}
     */
    private void rescheduleTrueBranchConditionsIfNeeded(HIRBlock basicBlock, PTXBlockVisitor visitor, HashSet<HIRBlock> visited, HashMap<HIRBlock, HIRBlock> pending) {
        if (!basicBlock.isLoopHeader() && basicBlock.getDominator() != null && basicBlock.getDominator().getEndNode() instanceof IfNode) {
            IfNode ifNode = (IfNode) basicBlock.getDominator().getEndNode();
            HIRBlock blockTrueBranch = getBlockTrueBranch(basicBlock);
            // implement rescheduling for condition if nodes
            if (isNotLoopBeginIf(ifNode)) {
                boolean shouldReschedule = isFalseSuccessorWithLoopEnd(ifNode, basicBlock) //
                        || (isCurrentBlockAFalseBranch(ifNode, basicBlock) //
                        && isTrueBranchALoopExitNode(ifNode) //
                        && isTrueBranchWithEndNodeOrNotControlSplit(blockTrueBranch));

                if (shouldReschedule) {
                    for (int i = 0; i < basicBlock.getDominator().getSuccessorCount(); i++) {
                        HIRBlock successor = basicBlock.getDominator().getSuccessorAt(i);
                        if (successor.getBeginNode() == ifNode.trueSuccessor() && !visited.contains(successor)) {
                            pending.put(basicBlock, successor);
                            rescheduleBasicBlock(basicBlock, visitor, visited, pending);
                        }
                    }
                }
            }
        }
    }

    /**
     * This function examines if a given {@code IfNode} is generated for a condition or if it is part of the loop logic (LoopBegin -> If).
     *
     * @param ifNode The {@code IfNode} to be examined.
     * @return true if the {@code IfNode} is not associated with the loop iteration, false otherwise.
     */
    private boolean isNotLoopBeginIf(IfNode ifNode) {
        return !(ifNode.predecessor() instanceof LoopBeginNode);
    }

    private void traverseControlFlowGraph(HIRBlock basicBlock, PTXBlockVisitor visitor, HashSet<HIRBlock> visited, HashMap<HIRBlock, HIRBlock> pending) {

        if (pending.containsKey(basicBlock) && !visited.contains(pending.get(basicBlock))) {
            rescheduleBasicBlock(basicBlock, visitor, visited, pending);
        }

        // New call due to the integration with Graal-IR 22.1.0
        rescheduleTrueBranchConditionsIfNeeded(basicBlock, visitor, visited, pending);

        visitor.enter(basicBlock);
        visited.add(basicBlock);

        HIRBlock firstDominated = basicBlock.getFirstDominated();
        LinkedList<HIRBlock> queue = new LinkedList<>();
        queue.add(firstDominated);

        if (basicBlock.isLoopHeader()) {
            HIRBlock[] successors = IntStream.range(0, basicBlock.getSuccessorCount()).mapToObj(basicBlock::getSuccessorAt).toArray(HIRBlock[]::new);
            LinkedList<HIRBlock> last = new LinkedList<>();
            LinkedList<HIRBlock> pendingList = new LinkedList<>();

            FixedNode endNode = basicBlock.getEndNode();
            IfNode ifNode = null;
            if (endNode instanceof IfNode) {
                ifNode = (IfNode) endNode;
            }
            for (HIRBlock block : successors) {
                boolean isInnerLoop = isLoopBlock(block, basicBlock);
                if (!isInnerLoop) {
                    assert ifNode != null;
                    if (ifNode.trueSuccessor() == block.getBeginNode() && block.getBeginNode() instanceof LoopExitNode && block.getEndNode() instanceof EndNode) {
                        pendingList.addFirst(block);
                        if (block.getPostdominator().getBeginNode() instanceof MergeNode) {
                            // We may need to reschedule this block if it is not closed before visiting the
                            // postDominator.
                            pending.put(block.getPostdominator(), block);
                        }
                    } else {
                        last.addLast(block);
                    }
                } else {
                    queue.addLast(block);
                }
            }

            for (HIRBlock l : pendingList) {
                last.addLast(l);
            }

            for (HIRBlock l : last) {
                queue.addLast(l);
            }
            queue.removeFirst();
        }

        for (HIRBlock block : queue) {
            firstDominated = block;
            while (firstDominated != null) {
                if (!visited.contains(firstDominated)) {
                    traverseControlFlowGraph(firstDominated, visitor, visited, pending);
                }
                firstDominated = firstDominated.getDominatedSibling();
            }
        }

        if (rescheduledBasicBlocks == null || (!rescheduledBasicBlocks.contains(basicBlock))) {
            visitor.exit(basicBlock, null);
        }
    }

    public PTXDeviceContext getDeviceContext() {
        return this.deviceContext;
    }

    public void setDeviceContext(PTXDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
    }

    public TaskDataContext getTaskMetaData() {
        return meta;
    }

    public void setTaskMetaData(TaskDataContext metaData) {
        this.meta = metaData;
    }
}
