"""
Rune Codegen -- Emit execution plans from optimized IR.

Takes an optimized IRGraph and produces:
1. A memory layout (buffer sizes and offsets)
2. A command sequence for the persistent engine
3. A mapping from IR nodes to GPU memory addresses

The execution plan is a list of EngineCommands that the persistent
CUDA engine can execute sequentially. Each IRNode maps to one or more
commands.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from rune.compiler.ir import (
    IRGraph, IRNode, IRType, OpCode,
    GRADE_SCALAR, GRADE_FULL,
    grade_mask_to_str, count_components,
)
from rune.compiler.passes.memory_plan import MemoryPlan


class EngineOp:
    """Engine-level operation codes (lower level than IR opcodes)."""
    # Data movement
    LOAD_EMBEDDING = 'load_embedding'
    STORE_OUTPUT = 'store_output'
    COPY_BUFFER = 'copy_buffer'
    ZERO_BUFFER = 'zero_buffer'

    # Clifford kernels
    GEOMETRIC_PRODUCT = 'kern_gp'
    GEOMETRIC_PRODUCT_GRADED = 'kern_gp_graded'  # Only compute target grades
    REVERSE = 'kern_reverse'
    SANDWICH = 'kern_sandwich'
    BIVECTOR_EXP = 'kern_bvexp'
    GRADE_PROJECT = 'kern_grade_project'
    ADD = 'kern_add'
    SCALE = 'kern_scale'

    # Compound kernels
    CLIFFORD_LINEAR = 'kern_clifford_linear'
    ATTENTION_SCORE = 'kern_attn_score'
    FUSED_ATTENTION = 'kern_fused_attn'
    SOFTMAX = 'kern_softmax'
    WEIGHTED_SUM = 'kern_weighted_sum'
    MEAN_POOL_SEQ = 'kern_mean_pool_seq'
    LAYER_NORM = 'kern_layer_norm'
    GELU = 'kern_gelu'
    FUSED_FFN = 'kern_fused_ffn'
    FUSED_GP_GRADE = 'kern_fused_gp_grade'

    # Scalar ops
    MATMUL = 'kern_matmul'
    CROSS_ENTROPY = 'kern_cross_entropy'

    # Backward kernels
    BACKWARD_CE = 'kern_backward_ce'
    BACKWARD_LINEAR = 'kern_backward_linear'
    BACKWARD_GP = 'kern_backward_gp'
    BACKWARD_NORM = 'kern_backward_norm'
    BACKWARD_GELU = 'kern_backward_gelu'
    BACKWARD_EMBED = 'kern_backward_embed'
    BACKWARD_ADD = 'kern_backward_add'
    BACKWARD_MATMUL = 'kern_backward_matmul'
    BACKWARD_GRADE_PROJECT = 'kern_backward_grade_project'
    BACKWARD_BVEXP = 'kern_backward_bvexp'
    BACKWARD_SOFTMAX = 'kern_backward_softmax'
    BACKWARD_WEIGHTED_SUM = 'kern_backward_weighted_sum'
    BACKWARD_COPY = 'kern_backward_copy'
    BACKWARD_MEMORY_GATE = 'kern_backward_memory_gate'
    BACKWARD_ATTENTION = 'kern_backward_attention'
    BACKWARD_FFN = 'kern_backward_ffn'

    # Optimizer kernels
    ADAM_FULL = 'kern_adam_full'
    ADAM_M_UPDATE = 'kern_adam_m_update'
    ADAM_V_UPDATE = 'kern_adam_v_update'
    ADAM_PARAM_UPDATE = 'kern_adam_param_update'

    # Memory bank
    MEMORY_READ = 'kern_memory_read'
    MEMORY_WRITE = 'kern_memory_write'
    MEMORY_GATE = 'kern_memory_gate'

    # Sync
    BARRIER = 'barrier'


@dataclass
class EngineCommand:
    """A single command for the persistent engine."""
    op: str                                     # EngineOp
    output_buffer: int = -1                     # Buffer ID for output
    input_buffers: List[int] = field(default_factory=list)  # Buffer IDs for inputs
    params: Dict = field(default_factory=dict)  # Operation parameters
    ir_node_id: int = -1                        # Source IR node
    name: str = ''

    def __repr__(self):
        ins = ','.join(str(b) for b in self.input_buffers)
        return (f"Cmd({self.op} buf[{ins}] -> buf[{self.output_buffer}]"
                f"{' ' + self.name if self.name else ''})")


@dataclass
class ExecutionPlan:
    """Complete execution plan for the persistent engine."""
    commands: List[EngineCommand] = field(default_factory=list)
    memory_plan: MemoryPlan = None
    n_parameters: int = 0
    n_buffers: int = 0
    total_memory_bytes: int = 0

    # Maps for the engine
    param_buffer_map: Dict[int, int] = field(default_factory=dict)  # param_id -> buffer_id
    param_grad_buffer_map: Dict[int, int] = field(default_factory=dict)  # param_id -> grad buffer_id
    param_name_map: Dict[int, str] = field(default_factory=dict)    # param_id -> parameter name
    constant_buffer_data: Dict[int, np.ndarray] = field(default_factory=dict)  # buffer_id -> static payload
    input_buffers: List[int] = field(default_factory=list)   # buffer IDs for model inputs
    output_buffers: List[int] = field(default_factory=list)  # buffer IDs for model outputs
    input_buffer_names: Dict[str, int] = field(default_factory=dict)
    output_buffer_names: Dict[str, int] = field(default_factory=dict)
    buffer_specs: Dict[int, Dict] = field(default_factory=dict)     # buffer_id -> metadata

    @property
    def command_count(self) -> int:
        return len(self.commands)

    def dump(self) -> str:
        """Pretty-print the execution plan."""
        lines = [
            f"ExecutionPlan: {self.command_count} commands, "
            f"{self.n_buffers} buffers, "
            f"{self.total_memory_bytes / 1024 / 1024:.1f} MB",
            "=" * 60,
        ]
        for i, cmd in enumerate(self.commands):
            lines.append(f"  [{i:3d}] {cmd}")
        return '\n'.join(lines)

    def __repr__(self):
        return (f"ExecutionPlan({self.command_count} cmds, "
                f"{self.n_buffers} bufs, "
                f"{self.total_memory_bytes / 1024 / 1024:.1f} MB)")


class Codegen:
    """
    Generate execution plans from optimized IR graphs.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def generate(self, graph: IRGraph, memory_plan: MemoryPlan) -> ExecutionPlan:
        """
        Generate an execution plan from an optimized IR and memory plan.

        Args:
            graph: Optimized IRGraph
            memory_plan: Buffer allocation from MemoryPlanPass

        Returns:
            ExecutionPlan ready for the persistent engine
        """
        plan = ExecutionPlan()
        plan.memory_plan = memory_plan
        plan.n_buffers = memory_plan.n_buffers
        plan.total_memory_bytes = memory_plan.total_bytes

        # Track parameter buffers
        for node in graph.find_nodes_by_op(OpCode.CONSTANT):
            buf_id = memory_plan.node_to_buffer.get(node.id, -1)
            if buf_id < 0:
                continue
            if 'const_data' in node.attrs:
                plan.constant_buffer_data[buf_id] = np.ascontiguousarray(node.attrs['const_data'])
                continue
            param_id = node.attrs.get('param_id', node.id)
            if buf_id >= 0:
                plan.param_buffer_map[param_id] = buf_id
                plan.param_name_map[param_id] = node.name
                plan.n_parameters += 1

        for node in graph.live_nodes():
            if node.output_type is None:
                continue
            buf_id = memory_plan.node_to_buffer.get(node.id, -1)
            if buf_id < 0:
                continue
            spec = plan.buffer_specs.setdefault(buf_id, {})
            spec.setdefault('name', node.name)
            spec.setdefault('shape', node.output_type.shape)
            spec.setdefault('dtype', node.output_type.dtype)
            spec.setdefault('grade_mask', node.output_type.grade_mask)
            spec.setdefault('storage_components', node.output_type.physical_components)
            spec.setdefault('size_bytes', memory_plan.buffers[buf_id].size_bytes)

        # Process nodes in topological order
        topo = graph.topological_order()
        for nid in topo:
            node = graph.get_node(nid)
            if node.is_dead or node.fused_into >= 0:
                continue

            if node.op == OpCode.ADAM_FULL_UPDATE and len(node.inputs) >= 2:
                param_node = graph.get_node(node.inputs[0])
                grad_node = graph.get_node(node.inputs[1])
                param_id = param_node.attrs.get('param_id', param_node.id)
                grad_buf = memory_plan.node_to_buffer.get(grad_node.id, -1)
                if grad_buf >= 0:
                    plan.param_grad_buffer_map[param_id] = grad_buf

            cmds = self._emit_node(node, graph, memory_plan)
            for cmd in cmds:
                cmd.ir_node_id = nid
                plan.commands.append(cmd)

            # Track input/output buffers
            if node.op == OpCode.INPUT:
                buf_id = memory_plan.node_to_buffer.get(nid, -1)
                if buf_id >= 0:
                    plan.input_buffers.append(buf_id)
                    plan.input_buffer_names[node.name] = buf_id
            elif node.op == OpCode.OUTPUT:
                if node.inputs:
                    inp_buf = memory_plan.node_to_buffer.get(node.inputs[0], -1)
                    if inp_buf >= 0:
                        plan.output_buffers.append(inp_buf)
                        plan.output_buffer_names[node.name] = inp_buf

        if self.verbose:
            print(f"Codegen: {plan.command_count} commands, "
                  f"{plan.n_parameters} parameters, "
                  f"{plan.n_buffers} buffers")

        return plan

    def _get_buffer(self, node_id: int, memory_plan: MemoryPlan) -> int:
        """Get buffer ID for a node."""
        return memory_plan.node_to_buffer.get(node_id, -1)

    def _input_buffers(self, node: IRNode, memory_plan: MemoryPlan) -> List[int]:
        """Get buffer IDs for all inputs of a node."""
        return [self._get_buffer(inp, memory_plan) for inp in node.inputs]

    def _emit_node(self, node: IRNode, graph: IRGraph,
                   memory_plan: MemoryPlan) -> List[EngineCommand]:
        """Emit engine commands for a single IR node."""
        out_buf = self._get_buffer(node.id, memory_plan)
        in_bufs = self._input_buffers(node, memory_plan)
        op = node.op

        if op == OpCode.INPUT:
            return [EngineCommand(
                op=EngineOp.ZERO_BUFFER,
                output_buffer=out_buf,
                name=node.name,
            )]

        elif op == OpCode.CONSTANT:
            # Constants are loaded at init time, no runtime command needed
            return []

        elif op == OpCode.OUTPUT:
            if node.inputs:
                src_buf = self._get_buffer(node.inputs[0], memory_plan)
                return [EngineCommand(
                    op=EngineOp.STORE_OUTPUT,
                    output_buffer=src_buf,
                    input_buffers=[src_buf],
                    name=node.name,
                )]
            return []

        elif op == OpCode.EMBED_LOOKUP:
            out_shape = node.output_type.shape
            n_tokens = out_shape[0] * out_shape[1]
            return [EngineCommand(
                op=EngineOp.LOAD_EMBEDDING,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n_tokens': n_tokens,
                    'd_model': out_shape[2],
                },
                name=node.name,
            )]

        elif op == OpCode.BIVECTOR_EXP:
            return [EngineCommand(
                op=EngineOp.BIVECTOR_EXP,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'n': node.output_type.total_elements,
                    'input_components': node.attrs.get(
                        'input_components',
                        graph.get_node(node.inputs[0]).output_type.physical_components,
                    ),
                },
                name=node.name,
            )]

        elif op == OpCode.GEOMETRIC_PRODUCT:
            return [EngineCommand(
                op=EngineOp.GEOMETRIC_PRODUCT,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n': node.output_type.total_elements,
                },
                name=node.name,
            )]

        elif op == OpCode.REVERSE:
            return [EngineCommand(
                op=EngineOp.REVERSE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n': node.output_type.total_elements,
                },
                name=node.name,
            )]

        elif op == OpCode.SANDWICH:
            return [EngineCommand(
                op=EngineOp.SANDWICH,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n': node.output_type.total_elements,
                },
                name=node.name,
            )]

        elif op == OpCode.GRADE_PROJECT:
            return [EngineCommand(
                op=EngineOp.GRADE_PROJECT,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'target_grade': node.attrs.get('target_grade', 0),
                    'n': node.output_type.total_elements,
                    'output_components': node.output_type.physical_components,
                },
                name=node.name,
            )]

        elif op == OpCode.ADD:
            return [EngineCommand(
                op=EngineOp.ADD,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n': node.output_type.total_elements,
                    'components': node.output_type.physical_components,
                },
                name=node.name,
            )]

        elif op == OpCode.SCALE:
            return [EngineCommand(
                op=EngineOp.SCALE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={'n': node.output_type.total_elements},
                name=node.name,
            )]

        elif op == OpCode.LINEAR:
            out_shape = node.output_type.shape
            batch_tokens = out_shape[0] * out_shape[1]
            return [EngineCommand(
                op=EngineOp.CLIFFORD_LINEAR,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'batch_tokens': batch_tokens,
                    'd_in': node.attrs.get('d_in'),
                    'd_out': node.attrs.get('d_out'),
                    'has_bias': node.attrs.get('has_bias', False),
                    'grade_mask': node.output_grade_mask,
                },
                name=node.name,
            )]

        elif op == OpCode.ATTN_SCORE:
            out_shape = node.output_type.shape
            return [EngineCommand(
                op=EngineOp.ATTENTION_SCORE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'batch': out_shape[0],
                    'n_heads': 1,
                    'seq': out_shape[1],
                    'd_head': node.attrs.get('d_head'),
                    'scale': node.attrs.get('scale'),
                },
                name=node.name,
            )]

        elif op == OpCode.SOFTMAX:
            out_shape = node.output_type.shape
            return [EngineCommand(
                op=EngineOp.SOFTMAX,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'n_rows': out_shape[0] * out_shape[1],
                    'row_len': out_shape[2],
                },
                name=node.name,
            )]

        elif op == OpCode.WEIGHTED_SUM:
            out_shape = node.output_type.shape
            return [EngineCommand(
                op=EngineOp.WEIGHTED_SUM,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'batch': out_shape[0],
                    'n_heads': 1,
                    'seq': out_shape[1],
                    'd_head': node.attrs.get('d_head'),
                    'grade_mask': node.output_grade_mask,
                },
                name=node.name,
            )]

        elif op == OpCode.MEAN_POOL_SEQ:
            return [EngineCommand(
                op=EngineOp.MEAN_POOL_SEQ,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'batch': node.attrs.get('batch', node.output_type.shape[0]),
                    'seq': node.attrs.get('seq', 1),
                    'd_model': node.attrs.get('d_model', node.output_type.shape[-1]),
                },
                name=node.name,
            )]

        elif op == OpCode.LAYER_NORM:
            out_shape = node.output_type.shape
            return [EngineCommand(
                op=EngineOp.LAYER_NORM,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'batch_tokens': out_shape[0] * out_shape[1],
                    'd_model': node.attrs.get('d_model'),
                    'eps': node.attrs.get('eps', 1e-6),
                    'grade_mask': node.output_grade_mask,
                },
                name=node.name,
            )]

        elif op == OpCode.GELU:
            return [EngineCommand(
                op=EngineOp.GELU,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'grade_mask': node.output_grade_mask,
                    'n': node.output_type.total_elements,
                },
                name=node.name,
            )]

        elif op == OpCode.FFN:
            return [EngineCommand(
                op=EngineOp.FUSED_FFN,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.COPY:
            # Concat or copy: multiple inputs -> one output
            return [EngineCommand(
                op=EngineOp.COPY_BUFFER,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.CROSS_ENTROPY:
            return [EngineCommand(
                op=EngineOp.CROSS_ENTROPY,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'n': node.inputs and graph.get_node(node.inputs[0]).output_type.shape[0] *
                         graph.get_node(node.inputs[0]).output_type.shape[1] or 0,
                    'vocab': graph.get_node(node.inputs[0]).output_type.shape[2] if node.inputs else 0,
                },
                name=node.name,
            )]

        elif op == OpCode.MATMUL_SCALAR:
            return [EngineCommand(
                op=EngineOp.MATMUL,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'm': node.attrs.get('m'),
                    'k': node.attrs.get('k'),
                    'n': node.attrs.get('n'),
                },
                name=node.name,
            )]

        elif op == OpCode.ZERO:
            return [EngineCommand(
                op=EngineOp.ZERO_BUFFER,
                output_buffer=out_buf,
                params={'n_floats': node.output_type.total_floats},
                name=node.name,
            )]

        # Fused operations
        elif op == OpCode.FUSED_GP_GRADE:
            return [EngineCommand(
                op=EngineOp.FUSED_GP_GRADE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'target_grade': node.attrs.get('target_grade', 0),
                    'grade_mask': node.output_grade_mask,
                },
                name=node.name,
            )]

        elif op == OpCode.FUSED_FFN:
            return [EngineCommand(
                op=EngineOp.FUSED_FFN,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'd_in': node.attrs.get('d_in'),
                    'd_ff': node.attrs.get('d_ff'),
                    'd_out': node.attrs.get('d_out'),
                    'activation': node.attrs.get('activation', 'gelu'),
                    'grade_mask': node.output_grade_mask,
                },
                name=node.name,
            )]

        elif op == OpCode.FUSED_ATTENTION:
            return [EngineCommand(
                op=EngineOp.FUSED_ATTENTION,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'd_head': node.attrs.get('d_head'),
                    'scale': node.attrs.get('scale'),
                    'includes_softmax': node.attrs.get('includes_softmax', True),
                },
                name=node.name,
            )]

        elif op == OpCode.FUSED_LAYER_NORM:
            return [EngineCommand(
                op=EngineOp.LAYER_NORM,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.FUSED_SANDWICH:
            return [EngineCommand(
                op=EngineOp.SANDWICH,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.SCALAR_PRODUCT:
            return [EngineCommand(
                op=EngineOp.GEOMETRIC_PRODUCT,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={'grade_mask': GRADE_SCALAR},
                name=node.name,
            )]

        # --- Backward ops ---
        elif op == OpCode.BACKWARD_CE:
            return [EngineCommand(
                op=EngineOp.BACKWARD_CE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'n': node.output_type.shape[0] * node.output_type.shape[1],
                    'vocab': node.output_type.shape[2],
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_LINEAR:
            grad_shape = graph.get_node(node.inputs[0]).output_type.shape if node.inputs else ()
            if len(grad_shape) >= 3:
                batch_tokens = grad_shape[0] * grad_shape[1]
            else:
                batch_tokens = 0
            return [EngineCommand(
                op=EngineOp.BACKWARD_LINEAR,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'batch_tokens': batch_tokens,
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_GP:
            params = dict(node.attrs)
            params.setdefault('n', node.output_type.total_elements)
            if 'd_head' in params and len(node.output_type.shape) >= 3:
                params.setdefault('batch', node.output_type.shape[0])
                params.setdefault('seq', node.output_type.shape[1])
            return [EngineCommand(
                op=EngineOp.BACKWARD_GP,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=params,
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_NORM:
            grad_shape = graph.get_node(node.inputs[0]).output_type.shape if node.inputs else ()
            if len(grad_shape) >= 3:
                batch_tokens = grad_shape[0] * grad_shape[1]
            else:
                batch_tokens = 0
            return [EngineCommand(
                op=EngineOp.BACKWARD_NORM,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={**node.attrs, 'batch_tokens': batch_tokens},
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_GELU:
            return [EngineCommand(
                op=EngineOp.BACKWARD_GELU,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={'n': node.output_type.total_elements},
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_EMBED:
            return [EngineCommand(
                op=EngineOp.BACKWARD_EMBED,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'n_tokens': graph.get_node(node.inputs[0]).output_type.shape[0] *
                                graph.get_node(node.inputs[0]).output_type.shape[1],
                    'd_model': graph.get_node(node.inputs[0]).output_type.shape[2],
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_ADD:
            return [EngineCommand(
                op=EngineOp.BACKWARD_ADD,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={**node.attrs, 'n': node.output_type.total_elements},
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_MATMUL_SCALAR:
            mode = node.attrs.get('mode', 'input')
            if mode == 'input':
                out_shape = node.output_type.shape
                m = out_shape[0] * out_shape[1]
                k = graph.get_node(node.inputs[0]).output_type.shape[2]
                n = out_shape[2]
            else:
                out_shape = node.output_type.shape
                m = out_shape[0]
                k = graph.get_node(node.inputs[1]).output_type.shape[0] * graph.get_node(node.inputs[1]).output_type.shape[1]
                n = out_shape[1]
            return [EngineCommand(
                op=EngineOp.BACKWARD_MATMUL,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'm': m,
                    'k': k,
                    'n': n,
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_GRADE_PROJECT:
            return [EngineCommand(
                op=EngineOp.BACKWARD_GRADE_PROJECT,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={**node.attrs, 'n': node.output_type.total_elements},
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_BVEXP:
            return [EngineCommand(
                op=EngineOp.BACKWARD_BVEXP,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'n': node.output_type.total_elements,
                    'input_components': node.attrs.get(
                        'input_components',
                        node.output_type.physical_components,
                    ),
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_SOFTMAX:
            out_shape = node.output_type.shape
            return [EngineCommand(
                op=EngineOp.BACKWARD_SOFTMAX,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    'n_rows': out_shape[0] * out_shape[1],
                    'row_len': out_shape[2],
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_WEIGHTED_SUM:
            mode = node.attrs.get('mode', 'values')
            value_shape = graph.get_node(node.inputs[2]).output_type.shape
            batch = value_shape[0]
            seq = value_shape[1]
            d_head = value_shape[2]
            return [EngineCommand(
                op=EngineOp.BACKWARD_WEIGHTED_SUM,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'batch': batch,
                    'n_heads': 1,
                    'seq': seq,
                    'd_head': d_head,
                    'mode': mode,
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_COPY:
            return [EngineCommand(
                op=EngineOp.BACKWARD_COPY,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'n': node.output_type.total_elements,
                    'd_out': node.output_type.shape[-1] if node.output_type.shape else 1,
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_MEMORY_GATE:
            return [EngineCommand(
                op=EngineOp.BACKWARD_MEMORY_GATE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'batch': node.attrs.get('batch', node.output_type.shape[0] if node.output_type.shape else 1),
                    'seq': node.attrs.get('seq', 1),
                    'd_model': node.attrs.get('d_model', node.output_type.shape[-1] if node.output_type.shape else 1),
                },
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_ATTENTION:
            return [EngineCommand(
                op=EngineOp.BACKWARD_ATTENTION,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.BACKWARD_FFN:
            return [EngineCommand(
                op=EngineOp.BACKWARD_FFN,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        # --- Optimizer ops ---
        elif op == OpCode.ADAM_FULL_UPDATE:
            return [EngineCommand(
                op=EngineOp.ADAM_FULL,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params={
                    **node.attrs,
                    'n_floats': node.output_type.total_floats,
                    'step': 1,
                },
                name=node.name,
            )]

        elif op == OpCode.ADAM_M_UPDATE:
            return [EngineCommand(
                op=EngineOp.ADAM_M_UPDATE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.ADAM_V_UPDATE:
            return [EngineCommand(
                op=EngineOp.ADAM_V_UPDATE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.ADAM_PARAM_UPDATE:
            return [EngineCommand(
                op=EngineOp.ADAM_PARAM_UPDATE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        # --- Memory bank ops ---
        elif op == OpCode.MEMORY_READ:
            return [EngineCommand(
                op=EngineOp.MEMORY_READ,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.MEMORY_WRITE:
            return [EngineCommand(
                op=EngineOp.MEMORY_WRITE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        elif op == OpCode.MEMORY_GATE:
            return [EngineCommand(
                op=EngineOp.MEMORY_GATE,
                output_buffer=out_buf,
                input_buffers=in_bufs,
                params=node.attrs,
                name=node.name,
            )]

        else:
            raise ValueError(
                f"Codegen: unsupported IR opcode '{op}' in node {node.id} "
                f"({node.name}). All IR ops must be lowered to engine commands "
                f"before codegen. If this is a fused op, run LowerFusedOpsPass first."
            )
