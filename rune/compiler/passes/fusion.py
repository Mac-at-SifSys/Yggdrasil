"""
Fusion Pass -- Fuse adjacent operations into single compound operations.

Fusion patterns:
1. GP + grade_project -> fused_gp_grade (compute only target grade's components)
2. linear + gelu + linear -> fused_ffn (one kernel launch for entire FFN)
3. Q_proj + K_proj + score + softmax + V_proj + weighted_sum -> fused_attention
4. norm + scale(GP) + add -> fused_layer_norm

Each fusion:
- Creates a new compound node
- Marks the fused-in nodes as dead (fused_into = new_node_id)
- Rewires downstream references
"""

from typing import Dict, List, Optional, Set, Tuple
from rune.compiler.ir import (
    IRGraph, IRNode, IRType, OpCode,
    GRADE_SCALAR, GRADE_FULL,
    grade_mask_to_str,
)


class FusionPass:
    """
    Fuse adjacent operations into compound kernels.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            'gp_grade_fusions': 0,
            'ffn_fusions': 0,
            'attention_fusions': 0,
            'layer_norm_fusions': 0,
            'total_nodes_fused': 0,
        }

    def run(self, graph: IRGraph) -> IRGraph:
        """Run all fusion patterns. Modifies graph in place."""
        self.stats = {k: 0 for k in self.stats}

        # Order matters: fuse small patterns first, then larger compound ones
        self._fuse_gp_grade_project(graph)
        self._fuse_ffn(graph)
        self._fuse_attention_blocks(graph)

        if self.verbose:
            total = self.stats['total_nodes_fused']
            print(f"Fusion: {total} nodes fused "
                  f"(gp_grade={self.stats['gp_grade_fusions']}, "
                  f"ffn={self.stats['ffn_fusions']}, "
                  f"attn={self.stats['attention_fusions']})")

        return graph

    def _fuse_gp_grade_project(self, graph: IRGraph):
        """
        Pattern: GP node -> grade_project node (single consumer)
        Fuse into: fused_gp_grade node that only computes the target grade.
        """
        grade_proj_nodes = graph.find_nodes_by_op(OpCode.GRADE_PROJECT)

        for gp_node in grade_proj_nodes:
            if gp_node.is_dead or gp_node.fused_into >= 0:
                continue
            if not gp_node.inputs:
                continue

            producer_id = gp_node.inputs[0]
            producer = graph.get_node(producer_id)

            if producer.is_dead or producer.fused_into >= 0:
                continue
            if producer.op != OpCode.GEOMETRIC_PRODUCT:
                continue

            # Only fuse if the GP has a single consumer (the grade_project)
            users = graph.get_users(producer_id)
            if len(users) != 1 or users[0] != gp_node.id:
                continue

            target_grade = gp_node.attrs.get('target_grade', 0)

            # Create fused node
            fused_id = graph.add_node(
                op=OpCode.FUSED_GP_GRADE,
                inputs=producer.inputs[:],
                output_type=gp_node.output_type,
                attrs={
                    'target_grade': target_grade,
                    'original_gp': producer_id,
                    'original_proj': gp_node.id,
                },
                name=f'fused_gp_grade({producer.name}->{gp_node.name})',
            )

            # Rewire: anything that used gp_node's output now uses fused node
            self._rewire_consumers(graph, gp_node.id, fused_id)

            # Mark originals as fused
            producer.is_dead = True
            producer.fused_into = fused_id
            gp_node.is_dead = True
            gp_node.fused_into = fused_id

            self.stats['gp_grade_fusions'] += 1
            self.stats['total_nodes_fused'] += 2

            if self.verbose:
                print(f"  Fuse GP+grade_project: [{producer_id}]+[{gp_node.id}] "
                      f"-> [{fused_id}] (grade {target_grade})")

    def _fuse_ffn(self, graph: IRGraph):
        """
        Pattern: LINEAR(up) -> GELU -> LINEAR(down) with single-consumer chains.
        Fuse into: FUSED_FFN node.
        """
        gelu_nodes = graph.find_nodes_by_op(OpCode.GELU)

        for gelu_node in gelu_nodes:
            if gelu_node.is_dead or gelu_node.fused_into >= 0:
                continue
            if not gelu_node.inputs:
                continue

            # Check upstream: should be LINEAR (up projection)
            up_id = gelu_node.inputs[0]
            up_node = graph.get_node(up_id)
            if up_node.is_dead or up_node.fused_into >= 0:
                continue
            if up_node.op != OpCode.LINEAR:
                continue

            # Check that up_node only feeds into this gelu
            up_users = graph.get_users(up_id)
            if len(up_users) != 1 or up_users[0] != gelu_node.id:
                continue

            # Check downstream: gelu should feed into exactly one LINEAR (down)
            gelu_users = graph.get_users(gelu_node.id)
            if len(gelu_users) != 1:
                continue
            down_id = gelu_users[0]
            down_node = graph.get_node(down_id)
            if down_node.is_dead or down_node.fused_into >= 0:
                continue
            if down_node.op != OpCode.LINEAR:
                continue

            # Verify it's an FFN pattern (up expands dim, down contracts)
            d_in = up_node.attrs.get('d_in', 0)
            d_ff = up_node.attrs.get('d_out', 0)
            d_out = down_node.attrs.get('d_out', 0)
            if d_ff <= d_in:
                continue  # Not an expand-contract pattern

            # Collect all inputs: up's inputs (weight, x, maybe bias) +
            # down's weight and maybe bias (but not gelu output)
            fused_inputs = []
            # up_node inputs: [weight, x, maybe_bias]
            fused_inputs.extend(up_node.inputs)
            # down_node inputs: [weight, gelu_out, maybe_bias]
            # Skip gelu_out (inputs[1]), keep weight and bias
            for i, inp_id in enumerate(down_node.inputs):
                if inp_id != gelu_node.id:
                    fused_inputs.append(inp_id)

            fused_id = graph.add_node(
                op=OpCode.FUSED_FFN,
                inputs=fused_inputs,
                output_type=down_node.output_type,
                attrs={
                    'd_in': d_in,
                    'd_ff': d_ff,
                    'd_out': d_out,
                    'activation': 'gelu',
                    'original_up': up_id,
                    'original_gelu': gelu_node.id,
                    'original_down': down_id,
                },
                name=f'fused_ffn({up_node.name})',
            )

            # Rewire consumers of down_node
            self._rewire_consumers(graph, down_id, fused_id)

            # Mark originals
            up_node.is_dead = True
            up_node.fused_into = fused_id
            gelu_node.is_dead = True
            gelu_node.fused_into = fused_id
            down_node.is_dead = True
            down_node.fused_into = fused_id

            self.stats['ffn_fusions'] += 1
            self.stats['total_nodes_fused'] += 3

            if self.verbose:
                print(f"  Fuse FFN: [{up_id}]+[{gelu_node.id}]+[{down_id}] "
                      f"-> [{fused_id}] (d_in={d_in}, d_ff={d_ff}, d_out={d_out})")

    def _fuse_attention_blocks(self, graph: IRGraph):
        """
        Pattern: Look for ATTN_SCORE nodes and walk their neighborhood to
        find Q_proj + K_proj + score + softmax + V_proj + weighted_sum patterns.
        Fuse the score + softmax into a single node when possible.
        """
        score_nodes = graph.find_nodes_by_op(OpCode.ATTN_SCORE)

        for score_node in score_nodes:
            if score_node.is_dead or score_node.fused_into >= 0:
                continue

            # Check downstream: score -> softmax (single consumer)
            score_users = graph.get_users(score_node.id)
            if len(score_users) != 1:
                continue

            softmax_id = score_users[0]
            softmax_node = graph.get_node(softmax_id)
            if softmax_node.op != OpCode.SOFTMAX:
                continue
            if softmax_node.is_dead or softmax_node.fused_into >= 0:
                continue

            # Fuse score + softmax into a single attention scoring node
            fused_id = graph.add_node(
                op=OpCode.FUSED_ATTENTION,
                inputs=score_node.inputs[:],
                output_type=softmax_node.output_type,
                attrs={
                    **score_node.attrs,
                    'includes_softmax': True,
                    'original_score': score_node.id,
                    'original_softmax': softmax_id,
                },
                name=f'fused_attn_score_softmax({score_node.name})',
            )

            self._rewire_consumers(graph, softmax_id, fused_id)

            score_node.is_dead = True
            score_node.fused_into = fused_id
            softmax_node.is_dead = True
            softmax_node.fused_into = fused_id

            self.stats['attention_fusions'] += 1
            self.stats['total_nodes_fused'] += 2

            if self.verbose:
                print(f"  Fuse ATTN: [{score_node.id}]+[{softmax_id}] "
                      f"-> [{fused_id}]")

    def _rewire_consumers(self, graph: IRGraph, old_id: int, new_id: int):
        """Replace all references to old_id with new_id in consumer inputs."""
        for node in graph.nodes:
            if node.is_dead and node.fused_into < 0:
                continue
            node.inputs = [new_id if x == old_id else x for x in node.inputs]
