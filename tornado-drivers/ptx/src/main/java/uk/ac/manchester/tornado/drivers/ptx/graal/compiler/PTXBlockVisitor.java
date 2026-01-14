/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

package uk.ac.manchester.tornado.drivers.ptx.graal.compiler;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.graalvm.compiler.nodes.BeginNode;
import org.graalvm.compiler.nodes.EndNode;
import org.graalvm.compiler.nodes.IfNode;
import org.graalvm.compiler.nodes.LoopBeginNode;
import org.graalvm.compiler.nodes.LoopEndNode;
import org.graalvm.compiler.nodes.LoopExitNode;
import org.graalvm.compiler.nodes.MergeNode;
import org.graalvm.compiler.nodes.cfg.ControlFlowGraph;
import org.graalvm.compiler.nodes.cfg.HIRBlock;

import uk.ac.manchester.tornado.drivers.ptx.graal.asm.PTXAssembler;
import uk.ac.manchester.tornado.drivers.ptx.graal.backend.CodeGenMode;

/**
 * Block visitor for PTX backend that supports both:
 * - PTX assembly mode (simple block labels)
 * - CUDA C++ mode (structured control flow with for/while/if/else)
 *
 * CUDA mode is based on OpenCL's OCLBlockVisitor approach.
 */
public class PTXBlockVisitor implements ControlFlowGraph.RecursiveVisitor<HIRBlock> {
    private final PTXCompilationResultBuilder crb;
    private PTXAssembler asm;

    // CUDA mode: track structured control flow
    private Set<HIRBlock> merges;
    private Map<HIRBlock, Integer> closedLoops;
    private Map<HIRBlock, Boolean> openBlocks;
    private Map<HIRBlock, Boolean> closedBlocks;
    private Set<HIRBlock> rmvEndBracket;
    private int loopCount;
    private int loopEnds;

    public PTXBlockVisitor(PTXCompilationResultBuilder resultBuilder, PTXAssembler asm) {
        this.crb = resultBuilder;
        this.asm = asm;

        // Initialize CUDA mode tracking structures
        this.merges = new HashSet<>();
        this.closedLoops = new HashMap<>();
        this.openBlocks = new HashMap<>();
        this.closedBlocks = new HashMap<>();
        this.rmvEndBracket = new HashSet<>();
        this.loopCount = 0;
        this.loopEnds = 0;
    }

    private static boolean isMergeBlock(HIRBlock block) {
        return block.getBeginNode() instanceof MergeNode;
    }

    private static boolean isIfBlock(HIRBlock block) {
        return block.getEndNode() instanceof IfNode;
    }

    @Override
    public HIRBlock enter(HIRBlock block) {
        if (asm.getCodeGenMode() == CodeGenMode.PTX) {
            enterPTX(block);
        } else {
            enterCUDA(block);
        }
        return null;
    }

    @Override
    public void exit(HIRBlock block, HIRBlock value) {
        if (asm.getCodeGenMode() == CodeGenMode.CUDA) {
            exitCUDA(block);
        }
    }

    // ==================== PTX Mode ====================

    private void enterPTX(HIRBlock block) {
        asm.eol();
        asm.emitBlockLabel(block);
        crb.emitBlock(block);
    }

    // ==================== CUDA Mode ====================

    private void enterCUDA(HIRBlock block) {
        markBlockOpen(block);
        boolean isMerge = block.getBeginNode() instanceof MergeNode;

        if (isMerge) {
            asm.eolOn();
            merges.add(block);
        }

        asm.eol();

        // Emit block label for goto targets (if there are any gotos to this block)
        // In structured code, most blocks won't have gotos, but some edge cases still need labels
        asm.emitBlockLabel(block);

        if (block.isLoopHeader()) {
            loopCount++;
            crb.emitLoopBlock(block);
        } else {
            final HIRBlock dom = block.getDominator();
            if (dom != null && !isMerge && !dom.isLoopHeader() && isIfBlock(dom)) {
                emitBeginBlockForElseStatement(dom, block);
            }
            crb.emitBlock(block);
        }
    }

    private void exitCUDA(HIRBlock block) {
        if (block.isLoopEnd()) {
            LoopEndNode loopEndNode = (LoopEndNode) block.getEndNode();
            LoopBeginNode loopBeginNode = loopEndNode.loopBegin();
            HIRBlock loopBeginBlock = loopBeginNode.graph().getLastSchedule().getNodeToBlockMap().get(loopBeginNode);

            loopEnds++;
            closeScope(block, loopBeginBlock);
        }

        if (block.getPostdominator() != null) {
            HIRBlock pdom = block.getPostdominator();

            if (!merges.contains(pdom) && isMergeBlock(pdom)) {
                if (!(pdom.getBeginNode() instanceof MergeNode && merges.contains(block) && block.getPredecessorCount() > 2)) {
                    if (!wasLoopBlockAlreadyClosed(block)) {
                        if (!rmvEndBracket.contains(block)) {
                            closeBlock(block);
                        }
                    }
                }
            } else {
                checkClosingBlockInsideIf(block, pdom);
            }
        } else if (block.getBeginNode() instanceof LoopExitNode && !wasLoopBlockAlreadyClosed(block)) {
            closeBlock(block);
        } else {
            closeBranchBlock(block);
        }
    }

    private void emitBeginBlockForElseStatement(HIRBlock dom, HIRBlock block) {
        final IfNode ifNode = (IfNode) dom.getEndNode();
        if (ifNode.falseSuccessor() == block.getBeginNode()) {
            asm.emitCudaIndent();
            asm.elseStmt();
            asm.eol();
        }
        asm.beginScope();
        asm.eolOn();
    }

    private void closeBlock(HIRBlock block) {
        if (openBlocks.getOrDefault(block, false) && !wasBlockAlreadyClosed(block)) {
            asm.endScope(String.format("BLOCK_%d", block.getId()));
            markBlockClosed(block);
        }
    }

    private void checkClosingBlockInsideIf(HIRBlock block, HIRBlock pdom) {
        if (pdom.isLoopHeader() && block.getDominator() != null && isIfBlock(block.getDominator())) {
            if ((block.getDominator().getDominator() != null && isIfBlock(block.getDominator().getDominator()))
                || !(block.getDominator().getBeginNode() instanceof LoopBeginNode)) {

                HIRBlock[] successors = new HIRBlock[block.getDominator().getSuccessorCount()];
                for (int i = 0; i < block.getDominator().getSuccessorCount(); i++) {
                    successors[i] = block.getDominator().getSuccessorAt(i);
                }

                int index = 0;
                if (successors[index] == block) {
                    index = 1;
                }

                if (successors[index] != block && block.getBeginNode() instanceof MergeNode) {
                    return;
                }

                if (!(successors[index].getBeginNode() instanceof LoopExitNode)) {
                    closeBlock(block);
                }
            }
        } else if (pdom.getBeginNode() instanceof MergeNode && block.getDominator() != null && isIfBlock(block.getDominator())) {
            HIRBlock dom2 = block.getDominator(2);
            if (dom2 != null && isIfBlock(dom2)) {
                HIRBlock[] successors = new HIRBlock[block.getDominator().getSuccessorCount()];
                for (int i = 0; i < block.getDominator().getSuccessorCount(); i++) {
                    successors[i] = block.getDominator().getSuccessorAt(i);
                }

                for (HIRBlock successor : successors) {
                    closeBlock(successor);
                }
            }
        }
    }

    private void closeScope(HIRBlock block, HIRBlock loopBeginBlock) {
        if (block.getBeginNode() instanceof LoopExitNode) {
            if (!(block.getDominator().getDominator() != null && block.getDominator().getDominator().getBeginNode() instanceof MergeNode)) {
                closeBlock(block);
                incrementClosedLoops(loopBeginBlock);
            }
        } else {
            closeBlock(block);
            incrementClosedLoops(loopBeginBlock);
        }
    }

    private void closeBranchBlock(HIRBlock block) {
        final HIRBlock dom = block.getDominator();
        if ((dom != null && wasLoopBlockAlreadyClosed(block)) || (block.isLoopEnd() && !(block.getBeginNode() instanceof LoopExitNode))) {
            return;
        }

        if (dom != null && !isMergeBlock(block) && !dom.isLoopHeader() && isIfBlock(dom)) {
            closeIfBlock(block, dom);
        }
    }

    private void closeIfBlock(HIRBlock block, HIRBlock dom) {
        final IfNode ifNode = (IfNode) dom.getEndNode();
        if ((ifNode.falseSuccessor() == block.getBeginNode()) || (ifNode.trueSuccessor() == block.getBeginNode())) {
            boolean isLoopEnd = block.getEndNode() instanceof LoopEndNode;
            boolean isTrueBranch = ifNode.trueSuccessor() == block.getBeginNode();
            if (!(isTrueBranch && isLoopEnd)) {
                closeBlock(block);
                if (block.getLoop() != null) {
                    incrementClosedLoops(block.getLoop().getHeader());
                }
            }
        }
    }

    private boolean wasLoopBlockAlreadyClosed(HIRBlock block) {
        HIRBlock dominator = block.getDominator();
        if (dominator.getLoop() != null) {
            int closeCount = closedLoops.getOrDefault(dominator.getLoop().getHeader(), 0);
            return closeCount == dominator.getLoop().getLoopExits().size();
        }
        return false;
    }

    private boolean wasBlockAlreadyClosed(HIRBlock block) {
        if (block != null) {
            return closedBlocks.getOrDefault(block, false);
        }
        return false;
    }

    private void markBlockOpen(HIRBlock block) {
        openBlocks.put(block, true);
    }

    private void markBlockClosed(HIRBlock block) {
        closedBlocks.put(block, true);
    }

    private void incrementClosedLoops(HIRBlock loopBeginBlock) {
        int closedLoopCount = closedLoops.getOrDefault(loopBeginBlock, 0);
        closedLoops.put(loopBeginBlock, closedLoopCount + 1);
    }
}
