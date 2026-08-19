# OQ-17 diagnosis: CUDA emitter miscompiles `A || (B && C)`

Item 1 of the upstream work session. Diagnosis only — no fix in this branch.

Built and tested against upstream `develop` at `99549c986` (fetched and rebased fresh before
starting), CUDA backend only, RTX 4090 / sm_89.

## Reproducer

Added `testLogicOrOfAnd` to `tornado-unittests/src/main/java/uk/ac/manchester/tornado/unittests/logic/TestLogic.java`
(this file already exists and already covers other short-circuit/nested-boolean shapes, so it's
the natural home rather than a new file). Exhaustive over the same 256-input set as the original
probe: 16 neighbour counts x {DEAD, ALIVE}, each repeated 8x.

```
tornado-test -V uk.ac.manchester.tornado.unittests.logic.TestLogic#testLogicOrOfAnd
...
Running test: testLogicOrOfAnd ................ [FAILED]
    [REASON] index 0: count=0 cell=0 expected:<0> but was:<-1>
```

Full `TestLogic` class run for context — the new test is the only failure, and the existing
`testLogicNestedBoolean` (a different nested-boolean shape, see "Existing coverage" below) passes:

```
Running test: testLogicShortCircuitSideEffect ................ [PASS]
Running test: testLogic01                ................ [PASS]
Running test: testLogic02                ................ [PASS]
Running test: testLogic03                ................ [PASS]
Running test: testLogicXorPattern        ................ [PASS]
Running test: testLogicNestedBoolean     ................ [PASS]
Running test: testLogicOrOfAnd           ................ [FAILED]
Test ran: 7, Failed: 1, Unsupported: 0
```

Generated CUDA (`--printKernel`), byte-identical in shape to the report that opened OQ-17:

```c
b_14  =  i_10 != 2 || i_12 != -1;
b_15  =  i_10 == 3 || i_10 != 2 && i_12 != -1;
i_16  =  (b_15 == true) ? -1 : 0;
```

For `count=0, cell=0`: correct value is `(0==3) || ((0==2)&&(0==-1))` = `false || (false&&false)` =
`false` → `0`. Generated: `b_15 = (0==3) || (0!=2 && 0!=-1) = false || (true&&true)` = `true` →
`-1`. The right operand of the outer `||` is `NOT(count==2) && NOT(cell==-1)`, not
`(count==2) && (cell==-1)` — the AND is combining the *negated* forms of both legs instead of
their plain forms.

## Root cause

`tornado-drivers/cuda/src/main/java/uk/ac/manchester/tornado/drivers/cuda/graal/compiler/CUDANodeLIRBuilder.java`

Graal represents `B && C` as a `ShortCircuitOrNode` with both legs negated
(`ShortCircuitOrNode(B, xNegated=true, C, yNegated=true)`, i.e. `!B || !C`, so that
`NOT(that node) == B && C`). `A || (B && C)` is then the outer `ShortCircuitOrNode(A, false,
<inner node>, yNegated=true)` — the outer disjunction needs the inner node's *negated* value.

TornadoVM's Graal operand cache (`tornado.graal.compiler.core.gen.NodeLIRBuilder.nodeOperands`,
confirmed by `javap` on `graalJars/tornado-graal-23.1.0.jar`) is a single `NodeMap<Value>`: exactly
one cached `Value` per graph node, with no way to distinguish "this node's plain value" from
"this node's negated value" — whichever is written last under a given node's identity is what
every later reader gets back.

`emitNegatedLogicNode(LogicNode node)` (defined at line 311, called whenever a node's *negated*
value is needed) ends with `setResult(node, result)` at **line 361**, writing the negated `Value`
into that single shared slot, keyed by the original (non-negated) node. This is fine in isolation
— but if the *same* node is later asked for its plain value (or is asked for its negated value a
second time from a different call site expecting to recompute), the stale cached value is returned
instead, via `operandOrConjunction` (lines 431-440), which trusts `operand(value) != null` and
never distinguishes what the caller actually needs.

For `A || (B && C)`, this fires concretely as:

1. The inner `ShortCircuitOrNode` (`B && C`) is visited first by the block scheduler and
   materialized on its own via `emitShortCircuitOrNode` (line 604) — its own value is
   `!B || !C`, so it calls `getProcessedOperand(B, xNegated=true)` → **`emitNegatedLogicNode(B)`**
   → computes `NOT(B)` and, at line 361, **caches `NOT(B)` under node `B`'s own identity.** Same
   for `C`. This produces `b_14 = i_10 != 2 || i_12 != -1` in the trace below — dead code in the
   final kernel, but the caching side effect is what matters.
2. The outer `ShortCircuitOrNode` (`A || (B && C)`) is visited next. Its `y` operand is the inner
   node with `yNegated=true`, so it calls `getProcessedOperand(innerNode, true)` →
   **`emitNegatedLogicNode(innerNode)`**. Because `innerNode` is itself a `ShortCircuitOrNode`,
   this hits the branch at **lines 345-348**:
   ```java
   } else if (node instanceof ShortCircuitOrNode shortCircuitOrNode) {
       final Value x = operandOrConjunction(shortCircuitOrNode.getX());   // = operandOrConjunction(B)
       final Value y = operandOrConjunction(shortCircuitOrNode.getY());   // = operandOrConjunction(C)
       result = getGen().getArithmetic().genBinaryExpr(CUDABinaryOp.LOGICAL_AND, boolLirKind, x, y);
   ```
   This is the correct De Morgan identity *if* `x`/`y` are `B`/`C`'s plain values — but
   `operandOrConjunction(B)` (lines 431-440) finds `operand(B) != null` (set in step 1) and
   returns the **already-cached `NOT(B)`** instead of recomputing `B`. Same for `C`. The AND then
   combines `NOT(B) && NOT(C)` instead of `B && C`.

Confirmed directly (not just inferred from the generated kernel) by re-running the same test with
`-Dtornado.logger.buildlir=True` (`Logger.traceBuildLIR`, gated by `TornadoOptions.TRACE_BUILD_LIR`
/ `tornado.logger.buildlir`, already present in the file — no instrumentation added):

```
emitNode: 27|ShortCircuitOr
emitShortCircuitOrNode: 27|ShortCircuitOr, (X: 19|== - isNegated: true) || (Y: 21|== - isNegated: true)
emitNegatedLogicNode: 19|==          <- caches NOT(B) under node 19
emitNegatedLogicNode: 21|==          <- caches NOT(C) under node 21
emitNode: 28|ShortCircuitOr
emitShortCircuitOrNode: 28|ShortCircuitOr, (X: 17|== - isNegated: false) || (Y: 27|ShortCircuitOr - isNegated: true)
emitLogicNode: 17|==
emitNegatedLogicNode: 27|ShortCircuitOr   <- operandOrConjunction(19) and (21) return the CACHED NOT(B)/NOT(C) from above
```

Node IDs 19/21/27/28 in the trace are exactly the nodes referenced above (`B`=`c==2`, `C`=`v==-1`,
27=inner `B&&C`, 28=outer `A||(B&&C)`). This is a direct, reproducible confirmation of the cache
aliasing, not a hypothesis.

### This refines, not just confirms, the OQ-17 hypothesis

The original hypothesis ("root of a select" vs "inlined as an operand of an enclosing `||`")
doesn't hold as the actual mechanism — the outer node in the failing kernel is not itself negated
and is not inlined into a select as a negated root; `emitConditional` just tests
`b_15 == true` directly, no arm-swap. The `B && C`-only kernel is correct not because it's a
select root, but because `B` and `C` are each visited **exactly once** in that kernel (their sole
use is computing the AND node's own value), so there's no second, conflicting request for their
plain form to be served a stale negated value. The bug is specifically: **a node whose negated
value is cached once via `emitNegatedLogicNode` corrupts every later reader of that node's plain
value, and this fires whenever a `ShortCircuitOrNode`'s leg is shared between (a) that node's own
materialization and (b) a second, later negation of an enclosing node that references it.**

## OpenCL / Metal — read, not tested

Per this session's CUDA-only scope, OpenCL and Metal were not built or run. Read only:
`tornado-drivers/opencl/src/main/java/uk/ac/manchester/tornado/drivers/opencl/graal/compiler/OCLNodeLIRBuilder.java`
and
`tornado-drivers/metal/src/main/java/uk/ac/manchester/tornado/drivers/metal/graal/compiler/MetalNodeLIRBuilder.java`
both contain `emitNegatedLogicNode`, `operandOrConjunction`, and `emitShortCircuitOrNode` with the
same structure and the same `setResult(node, result)` pattern (OpenCL: lines 311-420ish; Metal:
309-419ish — line numbers close but not identical to CUDA's). This is consistent with the same
defect existing there, but that is a reading-based hypothesis, not a measured result. The PTX
module and the removed SPIR-V backend were not examined.

## Existing test coverage

`tornado-unittests/src/main/java/uk/ac/manchester/tornado/unittests/logic/TestLogic.java` already
has `testLogicNestedBoolean`: `!(a>0 && b>0) || (a==b)`. It **passes** on this SHA. It doesn't hit
the same conflict because the AND node there is negated exactly once, at the top of the OR, and
is the *only* place that AND node's value is ever needed — no second, independent request for its
plain form exists to be served the stale cache entry. This is a different logical shape from
`A || (B && C)` (the AND leg isn't itself asked for both a positive and negated form), so it isn't
redundant coverage for OQ-17, it's coverage of a shape that happens not to trigger this particular
aliasing.

No other nested short-circuit coverage was found in `tornado-unittests/` (searched for `Logic`,
`ShortCircuit`, `KernelContext` boolean patterns).

## Not established

- Whether OpenCL/Metal actually reproduce this (read only, not run).
- Whether `&&` inside `&&`, or three-deep nestings, hit the same or a different aliasing pattern —
  not tested in this session (item 1 is diagnosis of the reported case only).
- Whether any *other* already-shipped kernel in the test suite or examples silently miscompiles
  the same way — this session only searched `tornado-unittests/logic/`, not the whole tree.

## What a fix needs to account for

Not attempted in this session (ground rules: diagnose only, don't fix on a guess). The shared
single-slot `nodeOperands` cache is a framework-level constraint (`tornado.graal.compiler.core.gen.NodeLIRBuilder`,
vendored Graal, shared by every backend) — a fix confined to `CUDANodeLIRBuilder` likely needs
`emitNegatedLogicNode`'s `ShortCircuitOrNode` branch to recompute its legs' plain values without
going through the shared cache (e.g. bypassing `operandOrConjunction` for legs it knows it just
negated elsewhere), rather than trying to give one node two cache slots.
