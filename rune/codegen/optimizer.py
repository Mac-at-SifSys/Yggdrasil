"""
optimizer.py — Graph optimization passes for Rune

Optimization passes:
1. Grade pruning: eliminate dead grades that are immediately projected away
2. Product fusion: adjacent geometric products -> single kernel call
3. Sandwich detection: A * B * ~A pattern -> fused sandwich kernel
4. Constant folding: fold multivector literal operations at compile time
"""

from typing import List, Optional
from rune.codegen.tracer import TraceGraph, TraceOp


def optimize_graph(graph: TraceGraph) -> TraceGraph:
    """Apply all optimization passes to a trace graph."""
    graph = grade_pruning(graph)
    graph = sandwich_detection(graph)
    graph = constant_folding(graph)
    return graph


def grade_pruning(graph: TraceGraph) -> TraceGraph:
    """
    Dead grade elimination.

    If a grade_project(k) immediately follows a geometric_product,
    the product kernel only needs to compute grade-k output components.
    """
    optimized = TraceGraph()
    optimized.inputs = graph.inputs
    optimized.outputs = graph.outputs

    ops = graph.ops
    i = 0
    while i < len(ops):
        if (i + 1 < len(ops)
            and ops[i].op_name == 'geometric_product'
            and ops[i+1].op_name == 'grade_project'
            and ops[i+1].inputs[0] == ops[i].output):

            # Fuse into grade_projected_product
            fused = TraceOp(
                op_name='grade_projected_product',
                inputs=ops[i].inputs,
                output=ops[i+1].output,
                kwargs={'grade': ops[i+1].kwargs.get('grade', 0)}
            )
            optimized.add_op(fused)
            i += 2
        else:
            optimized.add_op(ops[i])
            i += 1

    return optimized


def sandwich_detection(graph: TraceGraph) -> TraceGraph:
    """
    Detect A * B * ~A patterns and replace with fused sandwich kernel.
    """
    optimized = TraceGraph()
    optimized.inputs = graph.inputs
    optimized.outputs = graph.outputs

    ops = graph.ops
    i = 0
    while i < len(ops):
        if (i + 2 < len(ops)
            and ops[i].op_name == 'geometric_product'
            and ops[i+1].op_name == 'reverse'
            and ops[i+2].op_name == 'geometric_product'
            and ops[i+1].inputs[0] == ops[i].inputs[0]  # reverse of first operand
            and ops[i+2].inputs[0] == ops[i].output      # chain
            and ops[i+2].inputs[1] == ops[i+1].output):  # with reverse

            fused = TraceOp(
                op_name='sandwich',
                inputs=[ops[i].inputs[0], ops[i].inputs[1]],
                output=ops[i+2].output
            )
            optimized.add_op(fused)
            i += 3
        else:
            optimized.add_op(ops[i])
            i += 1

    return optimized


def constant_folding(graph: TraceGraph) -> TraceGraph:
    """
    Fold operations on constant multivector literals.
    (Placeholder for future implementation)
    """
    return graph  # No-op for now
