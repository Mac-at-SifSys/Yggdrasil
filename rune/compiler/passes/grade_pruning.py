"""
Grade Pruning -- Eliminate dead grade computation.

For each node, compute which output grades are actually needed by downstream
consumers. If a consumer only reads grade-0 (e.g., attention scoring), the
producer doesn't need to compute grades 1-3.

This is a backward dataflow analysis:
1. Start from outputs: mark all grades as needed
2. For each consumer, determine which input grades it needs
3. Propagate requirements backward to producers
4. If a producer's output grade is not needed by ANY consumer, mark it dead

Grade requirements per operation:
- grade_project(x, k): needs only grade k from x
- scalar_product(a, b): needs all grades from a and b
- GP(a, b): needs all grades from both (grade mixing)
- reverse(x): needs same grades as output needs (just sign flips)
- add(a, b): needs same grades from both as output needs
- linear(W, x): needs all grades from both (GP inside)
- attention scoring: needs grade-0 output only (scalar scores)
- softmax: input is scalar, output is scalar
- weighted_sum: weights are scalar, values need all output grades
"""

from typing import Dict, List, Set, Tuple
from rune.compiler.ir import (
    IRGraph, IRNode, IRType, OpCode,
    GRADE_SCALAR, GRADE_VECTOR, GRADE_BIVECTOR, GRADE_TRIVECTOR,
    GRADE_EVEN, GRADE_ODD, GRADE_FULL,
    grade_mask_to_str, count_components,
)


# These ops are still executed by dense 8-lane Clifford kernels in the
# persistent runtime. Their logical grade mask may narrow during pruning,
# but their physical storage must stay dense until the engine grows
# scalar-/grade-aware variants for them.
_DENSE_CLIFFORD_STORAGE_OPS = {
    OpCode.GEOMETRIC_PRODUCT,
    OpCode.REVERSE,
    OpCode.SANDWICH,
    OpCode.BIVECTOR_EXP,
    OpCode.LINEAR,
    OpCode.ATTENTION,
    OpCode.ATTN_SCORE,
    OpCode.WEIGHTED_SUM,
    OpCode.LAYER_NORM,
    OpCode.GELU,
    OpCode.ADD,
    OpCode.SCALE,
    OpCode.EMBED_LOOKUP,
    OpCode.MEAN_POOL_SEQ,
    OpCode.MEMORY_READ,
    OpCode.MEMORY_WRITE,
    OpCode.MEMORY_GATE,
}


class GradePruningPass:
    """
    Backward dataflow analysis to determine minimal grade requirements.

    After running, each node's output_type.grade_mask is narrowed to only
    the grades actually needed by downstream consumers.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {
            'nodes_pruned': 0,
            'grades_eliminated': 0,
            'floats_saved': 0,
        }

    def run(self, graph: IRGraph) -> IRGraph:
        """
        Run grade pruning on the graph. Modifies in place and returns it.
        """
        self.stats = {'nodes_pruned': 0, 'grades_eliminated': 0, 'floats_saved': 0}

        # Step 1: Compute required grade masks via backward analysis
        required = self._backward_analysis(graph)

        # Step 2: Apply narrowed grade masks to output types
        for node in graph.live_nodes():
            nid = node.id
            old_mask = node.output_grade_mask
            new_mask = required.get(nid, old_mask)

            if node.output_type is not None and new_mask != old_mask:
                old_components = count_components(old_mask)
                new_components = count_components(new_mask)
                eliminated = old_components - new_components

                if eliminated > 0:
                    self.stats['grades_eliminated'] += eliminated
                    total_elements = node.output_type.total_elements
                    self.stats['floats_saved'] += total_elements * eliminated

                    if self.verbose:
                        print(f"  Prune [{nid}] {node.op}: "
                              f"{grade_mask_to_str(old_mask)} -> "
                              f"{grade_mask_to_str(new_mask)} "
                              f"(-{eliminated} components)")

                new_type = node.output_type.with_grade_mask(new_mask)
                if (
                    node.op in _DENSE_CLIFFORD_STORAGE_OPS
                    and node.output_type.dtype in ('float16', 'float32', 'float64', 'bfloat16')
                    and node.output_type.physical_components == 8
                    and new_type.physical_components != 8
                ):
                    new_type = IRType(
                        shape=new_type.shape,
                        grade_mask=new_type.grade_mask,
                        dtype=new_type.dtype,
                        storage_components=8,
                    )

                node.output_type = new_type
                node.required_grade_mask = new_mask

        # Step 3: Mark fully dead nodes (grade_mask == 0 and not structural)
        structural_ops = {
            OpCode.INPUT, OpCode.OUTPUT, OpCode.CONSTANT,
            OpCode.COPY, OpCode.SOFTMAX,
            OpCode.BACKWARD_GP, OpCode.BACKWARD_LINEAR, OpCode.BACKWARD_ATTENTION,
            OpCode.BACKWARD_FFN, OpCode.BACKWARD_NORM, OpCode.BACKWARD_GELU,
            OpCode.BACKWARD_EMBED, OpCode.BACKWARD_CE, OpCode.BACKWARD_ADD,
            OpCode.BACKWARD_MATMUL_SCALAR, OpCode.BACKWARD_GRADE_PROJECT,
            OpCode.BACKWARD_BVEXP, OpCode.BACKWARD_SOFTMAX,
            OpCode.BACKWARD_WEIGHTED_SUM, OpCode.BACKWARD_COPY,
            OpCode.BACKWARD_MEMORY_GATE,
            OpCode.ADAM_M_UPDATE, OpCode.ADAM_V_UPDATE,
            OpCode.ADAM_PARAM_UPDATE, OpCode.ADAM_FULL_UPDATE,
            OpCode.MEMORY_WRITE,
        }
        for node in graph.live_nodes():
            if (node.output_type is not None
                    and node.output_type.grade_mask == 0
                    and node.op not in structural_ops):
                node.is_dead = True
                self.stats['nodes_pruned'] += 1

        if self.verbose:
            print(f"Grade pruning: eliminated {self.stats['grades_eliminated']} "
                  f"grade components, saved {self.stats['floats_saved']} floats, "
                  f"killed {self.stats['nodes_pruned']} nodes")

        return graph

    def _backward_analysis(self, graph: IRGraph) -> Dict[int, int]:
        """
        Backward dataflow: for each node, compute the union of grades
        required by all its consumers.

        Returns: mapping from node_id to required grade mask.
        """
        # Start with each node requiring whatever its current type says
        required: Dict[int, int] = {}
        for node in graph.live_nodes():
            required[node.id] = 0  # Start with nothing required

        # Output nodes: all grades required
        for node in graph.find_nodes_by_op(OpCode.OUTPUT):
            for inp_id in node.inputs:
                inp_node = graph.get_node(inp_id)
                required[inp_id] = inp_node.output_grade_mask

        # Training/effectful nodes also have to survive even if the scalar loss
        # is the only formal graph output.
        effectful_ops = {
            OpCode.BACKWARD_GP, OpCode.BACKWARD_LINEAR, OpCode.BACKWARD_ATTENTION,
            OpCode.BACKWARD_FFN, OpCode.BACKWARD_NORM, OpCode.BACKWARD_GELU,
            OpCode.BACKWARD_EMBED, OpCode.BACKWARD_CE, OpCode.BACKWARD_ADD,
            OpCode.BACKWARD_MATMUL_SCALAR, OpCode.BACKWARD_GRADE_PROJECT,
            OpCode.BACKWARD_BVEXP, OpCode.BACKWARD_SOFTMAX,
            OpCode.BACKWARD_WEIGHTED_SUM, OpCode.BACKWARD_COPY,
            OpCode.BACKWARD_MEMORY_GATE,
            OpCode.ADAM_M_UPDATE, OpCode.ADAM_V_UPDATE,
            OpCode.ADAM_PARAM_UPDATE, OpCode.ADAM_FULL_UPDATE,
            OpCode.MEMORY_WRITE,
        }
        for node in graph.live_nodes():
            if node.op in effectful_ops and node.output_type is not None:
                required[node.id] = node.output_grade_mask

        # Process in reverse topological order
        topo = graph.topological_order()

        for nid in reversed(topo):
            node = graph.get_node(nid)
            if node.is_dead:
                continue

            my_required = required.get(nid, 0)
            if my_required == 0 and node.op not in {OpCode.OUTPUT, OpCode.INPUT}:
                # Nobody needs our output -- we might be dead
                continue

            # Propagate requirements to inputs
            input_reqs = self._get_input_requirements(node, my_required, graph)

            for inp_id, inp_req in input_reqs.items():
                required[inp_id] = required.get(inp_id, 0) | inp_req

        return required

    def _get_input_requirements(self, node: IRNode, output_required: int,
                                graph: IRGraph) -> Dict[int, int]:
        """
        Given a node and which grades of its output are required,
        determine which grades of each input are required.

        Returns: mapping from input_node_id to required grade mask.
        """
        reqs: Dict[int, int] = {}
        op = node.op

        if op == OpCode.GRADE_PROJECT:
            # grade_project only needs the target grade from input
            target_grade = node.attrs.get('target_grade', 0)
            target_mask = 1 << target_grade
            if node.inputs:
                reqs[node.inputs[0]] = target_mask

        elif op == OpCode.ATTN_SCORE:
            # Attention scoring produces scalar output only.
            # It computes scalar_part(Q * ~K) which in Cl(3,0) is a full
            # Euclidean dot over all 8 components. So it needs ALL grades
            # from Q and K to produce the correct scalar product.
            if len(node.inputs) >= 2:
                q_id, k_id = node.inputs[0], node.inputs[1]
                reqs[q_id] = GRADE_FULL
                reqs[k_id] = GRADE_FULL

        elif op == OpCode.SOFTMAX:
            # Softmax is scalar-only in/out
            if node.inputs:
                reqs[node.inputs[0]] = GRADE_SCALAR

        elif op == OpCode.WEIGHTED_SUM:
            # Weights are scalar, values need whatever grades output needs
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = GRADE_SCALAR  # attention weights
                reqs[node.inputs[1]] = output_required  # values

        elif op == OpCode.ADD:
            # Addition: each input needs the same grades as output
            for inp_id in node.inputs:
                reqs[inp_id] = output_required

        elif op == OpCode.REVERSE:
            # Reverse just flips signs; needs same grades as output
            if node.inputs:
                reqs[node.inputs[0]] = output_required

        elif op == OpCode.GEOMETRIC_PRODUCT:
            # GP mixes all grades. If output needs ANY grade, we generally
            # need all input grades because of grade mixing.
            # Exception: if output only needs even grades and one input is
            # known to be even, the other only needs even grades too.
            if output_required != 0:
                for inp_id in node.inputs:
                    reqs[inp_id] = GRADE_FULL

        elif op == OpCode.LINEAR:
            # CliffordLinear uses GP internally -- same reasoning
            if output_required != 0:
                for inp_id in node.inputs:
                    inp_node = graph.get_node(inp_id)
                    if inp_node.op == OpCode.CONSTANT:
                        # Weight/bias parameters: need full grades
                        reqs[inp_id] = GRADE_FULL
                    else:
                        reqs[inp_id] = GRADE_FULL

        elif op == OpCode.LAYER_NORM:
            # LayerNorm uses GP(gamma, x_hat) + beta
            # The norm itself uses all grades (sum of squares)
            # If output needs only certain grades, we still need all input
            # grades for the normalization step
            for inp_id in node.inputs:
                reqs[inp_id] = GRADE_FULL

        elif op == OpCode.FFN:
            # FFN is compound: linear -> gelu -> linear
            # Needs all grades from input
            if node.inputs:
                reqs[node.inputs[0]] = GRADE_FULL

        elif op == OpCode.GELU:
            # GELU is element-wise: needs same grades as output requires
            if node.inputs:
                reqs[node.inputs[0]] = output_required

        elif op == OpCode.EMBED_LOOKUP:
            # Lookup: needs whatever grades output needs from the weight table
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = 0  # token_ids are not multivectors
                reqs[node.inputs[1]] = output_required  # embedding weight

        elif op == OpCode.MEAN_POOL_SEQ:
            if node.inputs:
                reqs[node.inputs[0]] = output_required

        elif op == OpCode.MEMORY_READ:
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = GRADE_FULL
                reqs[node.inputs[1]] = GRADE_FULL

        elif op == OpCode.MEMORY_GATE:
            if len(node.inputs) >= 3:
                reqs[node.inputs[0]] = output_required
                reqs[node.inputs[1]] = GRADE_FULL
                reqs[node.inputs[2]] = GRADE_SCALAR

        elif op == OpCode.BACKWARD_MEMORY_GATE:
            if len(node.inputs) >= 3:
                reqs[node.inputs[0]] = GRADE_FULL
                reqs[node.inputs[1]] = GRADE_FULL
                reqs[node.inputs[2]] = GRADE_SCALAR

        elif op == OpCode.MEMORY_WRITE:
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = GRADE_FULL
                reqs[node.inputs[1]] = GRADE_FULL

        elif op == OpCode.BIVECTOR_EXP:
            # Produces even-graded output from bivector input
            if node.inputs:
                reqs[node.inputs[0]] = GRADE_BIVECTOR

        elif op == OpCode.SANDWICH:
            # Sandwich product: needs all grades
            for inp_id in node.inputs:
                reqs[inp_id] = GRADE_FULL

        elif op == OpCode.COPY:
            # Copy/concat: pass through requirements
            for inp_id in node.inputs:
                reqs[inp_id] = output_required

        elif op == OpCode.SCALE:
            # Scale: scalar * multivector
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = GRADE_SCALAR  # scale factor
                reqs[node.inputs[1]] = output_required  # operand

        elif op == OpCode.CROSS_ENTROPY:
            # Loss: scalar output, needs scalar logits + integer targets
            if len(node.inputs) >= 2:
                reqs[node.inputs[0]] = GRADE_SCALAR  # logits
                reqs[node.inputs[1]] = 0              # target_ids

        elif op == OpCode.BACKWARD_CE:
            # Cross-entropy backward consumes scalar logits and integer targets
            # to produce scalar grad_logits. The loss input is scalar too.
            if len(node.inputs) >= 1:
                reqs[node.inputs[0]] = GRADE_SCALAR
            if len(node.inputs) >= 2:
                reqs[node.inputs[1]] = 0
            if len(node.inputs) >= 3:
                reqs[node.inputs[2]] = GRADE_SCALAR

        elif op == OpCode.BACKWARD_GRADE_PROJECT:
            # Backward through grade projection only needs the projected input
            # gradient lanes. The original forward input is not consumed.
            target_grade = node.attrs.get('target_grade', 0)
            target_mask = 1 << target_grade
            if node.inputs:
                reqs[node.inputs[0]] = target_mask
            for inp_id in node.inputs[1:]:
                reqs[inp_id] = 0

        elif op == OpCode.CONSTANT:
            pass  # No inputs

        elif op == OpCode.INPUT:
            pass  # No inputs

        elif op == OpCode.OUTPUT:
            # Output: pass through
            for inp_id in node.inputs:
                inp_node = graph.get_node(inp_id)
                reqs[inp_id] = inp_node.output_grade_mask

        elif op in (OpCode.SCALAR_PRODUCT, OpCode.MATMUL_SCALAR):
            if op == OpCode.SCALAR_PRODUCT:
                # Scalar product in Cl(3,0) consumes all grades from both inputs.
                for inp_id in node.inputs:
                    reqs[inp_id] = GRADE_FULL
            else:
                # MATMUL_SCALAR is a dense scalar matmul over the stored lanes.
                # It should preserve compact scalar/bivector layouts instead of
                # widening them back to full 8-lane multivectors.
                for inp_id in node.inputs:
                    inp_node = graph.get_node(inp_id)
                    reqs[inp_id] = inp_node.output_grade_mask

        elif op == OpCode.BACKWARD_MATMUL_SCALAR:
            # Scalar matmul backward should preserve the compact storage of the
            # scalar/bivector lanes it actually consumes. Widening these inputs
            # to GRADE_FULL corrupts the tied-head path in training graphs.
            if len(node.inputs) >= 1:
                reqs[node.inputs[0]] = GRADE_SCALAR  # grad_output / grad_logits
            for inp_id in node.inputs[1:]:
                inp_node = graph.get_node(inp_id)
                reqs[inp_id] = inp_node.output_grade_mask

        else:
            # Unknown op: conservatively require all grades from all inputs
            for inp_id in node.inputs:
                reqs[inp_id] = GRADE_FULL

        return reqs
