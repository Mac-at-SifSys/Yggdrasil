"""
Lower Fused Ops Pass -- Decompose fused IR operations back to primitives.

Fused ops (FUSED_GP_GRADE, FUSED_FFN, FUSED_ATTENTION) are created by the
fusion pass for potential kernel-level optimization. However, if the runtime
engine does not implement these fused kernels, this pass lowers them back
to primitive operations that the engine can execute.

This pass should run after fusion and before codegen.
"""

from typing import Dict, List
from rune.compiler.ir import (
    IRGraph, IRNode, IRType, OpCode,
    GRADE_SCALAR, GRADE_FULL,
)


class LowerFusedOpsPass:
    """
    Decompose fused operations back to their primitive components.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            'gp_grade_lowered': 0,
            'ffn_lowered': 0,
            'attention_lowered': 0,
            'layer_norm_lowered': 0,
            'sandwich_lowered': 0,
            'total_lowered': 0,
        }

    def run(self, graph: IRGraph) -> IRGraph:
        """Lower all fused ops to primitives. Modifies graph in place."""
        self.stats = {k: 0 for k in self.stats}

        self._lower_fused_gp_grade(graph)
        self._lower_fused_ffn(graph)
        self._lower_fused_attention(graph)
        self._lower_fused_layer_norm(graph)
        self._lower_fused_sandwich(graph)

        if self.verbose and self.stats['total_lowered'] > 0:
            print(f"LowerFusedOps: {self.stats['total_lowered']} fused ops lowered")

        return graph

    def _lower_fused_gp_grade(self, graph: IRGraph):
        """Lower FUSED_GP_GRADE back to GP + GRADE_PROJECT."""
        for node in graph.find_nodes_by_op(OpCode.FUSED_GP_GRADE):
            target_grade = node.attrs.get('target_grade', 0)

            # Create GP node
            gp_id = graph.add_node(
                op=OpCode.GEOMETRIC_PRODUCT,
                inputs=node.inputs[:],
                output_type=IRType(
                    shape=node.output_type.shape,
                    grade_mask=GRADE_FULL,
                ),
                name=f'lowered_gp_{node.id}',
            )

            # Create grade_project node
            proj_id = graph.add_node(
                op=OpCode.GRADE_PROJECT,
                inputs=[gp_id],
                output_type=node.output_type,
                attrs={'target_grade': target_grade},
                name=f'lowered_proj_{node.id}',
            )

            # Rewire consumers
            self._rewire_consumers(graph, node.id, proj_id)
            node.is_dead = True

            self.stats['gp_grade_lowered'] += 1
            self.stats['total_lowered'] += 1

    def _lower_fused_ffn(self, graph: IRGraph):
        """Lower FUSED_FFN back to LINEAR + GELU + LINEAR."""
        for node in graph.find_nodes_by_op(OpCode.FUSED_FFN):
            d_in = node.attrs.get('d_in', 0)
            d_ff = node.attrs.get('d_ff', 0)
            d_out = node.attrs.get('d_out', 0)

            # FUSED_FFN inputs: [up_weight, x, maybe_up_bias, down_weight, maybe_down_bias]
            # Reconstruct up linear inputs
            up_inputs = []
            remaining = []
            # First input is up weight, second is x
            if len(node.inputs) >= 2:
                up_inputs = [node.inputs[0], node.inputs[1]]
                remaining = node.inputs[2:]

            # Check for up bias
            up_has_bias = False
            if remaining and d_ff > 0:
                # Heuristic: if more than 1 remaining input, first might be up bias
                # Use the original attrs for guidance
                up_has_bias = node.attrs.get('original_up', -1) >= 0

            up_attrs = {'d_in': d_in, 'd_out': d_ff, 'has_bias': len(remaining) > 1}
            if len(remaining) > 1:
                up_inputs.append(remaining[0])
                remaining = remaining[1:]

            shape = node.output_type.shape
            B_S = shape[0] if len(shape) == 3 else 1
            S_dim = shape[1] if len(shape) == 3 else shape[0] if len(shape) == 2 else 1

            up_id = graph.add_node(
                op=OpCode.LINEAR,
                inputs=up_inputs,
                output_type=IRType(shape=node.output_type.shape[:-1] + (d_ff,), grade_mask=GRADE_FULL),
                attrs={'d_in': d_in, 'd_out': d_ff, 'has_bias': len(up_inputs) > 2},
                name=f'lowered_ffn_up_{node.id}',
            )

            gelu_id = graph.add_node(
                op=OpCode.GELU,
                inputs=[up_id],
                output_type=IRType(shape=node.output_type.shape[:-1] + (d_ff,), grade_mask=GRADE_FULL),
                name=f'lowered_ffn_gelu_{node.id}',
            )

            down_inputs = []
            if remaining:
                down_inputs = [remaining[0], gelu_id]
                if len(remaining) > 1:
                    down_inputs.append(remaining[1])
            else:
                down_inputs = [gelu_id]

            down_id = graph.add_node(
                op=OpCode.LINEAR,
                inputs=down_inputs,
                output_type=node.output_type,
                attrs={'d_in': d_ff, 'd_out': d_out, 'has_bias': len(down_inputs) > 2},
                name=f'lowered_ffn_down_{node.id}',
            )

            self._rewire_consumers(graph, node.id, down_id)
            node.is_dead = True

            self.stats['ffn_lowered'] += 1
            self.stats['total_lowered'] += 1

    def _lower_fused_attention(self, graph: IRGraph):
        """Lower FUSED_ATTENTION back to ATTN_SCORE + SOFTMAX."""
        for node in graph.find_nodes_by_op(OpCode.FUSED_ATTENTION):
            d_head = node.attrs.get('d_head', 0)
            scale = node.attrs.get('scale', 1.0)

            score_id = graph.add_node(
                op=OpCode.ATTN_SCORE,
                inputs=node.inputs[:],
                output_type=node.output_type,
                attrs={'d_head': d_head, 'scale': scale},
                name=f'lowered_attn_score_{node.id}',
            )

            softmax_id = graph.add_node(
                op=OpCode.SOFTMAX,
                inputs=[score_id],
                output_type=node.output_type,
                name=f'lowered_softmax_{node.id}',
            )

            self._rewire_consumers(graph, node.id, softmax_id)
            node.is_dead = True

            self.stats['attention_lowered'] += 1
            self.stats['total_lowered'] += 1

    def _lower_fused_layer_norm(self, graph: IRGraph):
        """Lower FUSED_LAYER_NORM back to LAYER_NORM."""
        for node in graph.find_nodes_by_op(OpCode.FUSED_LAYER_NORM):
            ln_id = graph.add_node(
                op=OpCode.LAYER_NORM,
                inputs=node.inputs[:],
                output_type=node.output_type,
                attrs=node.attrs,
                name=f'lowered_ln_{node.id}',
            )
            self._rewire_consumers(graph, node.id, ln_id)
            node.is_dead = True

            self.stats['layer_norm_lowered'] += 1
            self.stats['total_lowered'] += 1

    def _lower_fused_sandwich(self, graph: IRGraph):
        """Lower FUSED_SANDWICH back to GP + REVERSE + GP."""
        for node in graph.find_nodes_by_op(OpCode.FUSED_SANDWICH):
            sw_id = graph.add_node(
                op=OpCode.SANDWICH,
                inputs=node.inputs[:],
                output_type=node.output_type,
                attrs=node.attrs,
                name=f'lowered_sandwich_{node.id}',
            )
            self._rewire_consumers(graph, node.id, sw_id)
            node.is_dead = True

            self.stats['sandwich_lowered'] += 1
            self.stats['total_lowered'] += 1

    def _rewire_consumers(self, graph: IRGraph, old_id: int, new_id: int):
        """Replace all references to old_id with new_id in consumer inputs."""
        for node in graph.nodes:
            if node.is_dead and node.fused_into < 0:
                continue
            node.inputs = [new_id if x == old_id else x for x in node.inputs]
