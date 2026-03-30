"""
Rune IR -- Typed operation graph with Cl(3,0) grade annotations.

Each IRNode knows:
- What operation it performs
- What grades its inputs have
- What grades its output will have
- The shapes of all tensors

Grade masks: 4-bit int where bit k means grade k is present.
  0x01 = scalar only
  0x02 = vector only
  0x04 = bivector only
  0x08 = trivector only
  0x05 = even (scalar + bivector)
  0x0F = full multivector
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# Grade mask constants
GRADE_SCALAR = 0x01
GRADE_VECTOR = 0x02
GRADE_BIVECTOR = 0x04
GRADE_TRIVECTOR = 0x08
GRADE_EVEN = 0x05       # scalar + bivector
GRADE_ODD = 0x0A        # vector + trivector
GRADE_FULL = 0x0F       # all grades

# Component counts per grade in Cl(3,0)
GRADE_SIZES = {0: 1, 1: 3, 2: 3, 3: 1}

# Component index ranges per grade in the 8-component layout
GRADE_RANGES = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}


def grade_mask_to_str(mask: int) -> str:
    """Human-readable grade mask string."""
    names = {0x01: 'S', 0x02: 'V', 0x04: 'B', 0x08: 'T'}
    special = {0x05: 'even', 0x0A: 'odd', 0x0F: 'full'}
    if mask in special:
        return special[mask]
    parts = []
    for bit, name in names.items():
        if mask & bit:
            parts.append(name)
    return '+'.join(parts) if parts else 'empty'


def count_components(grade_mask: int) -> int:
    """Number of live scalar components for a given grade mask."""
    count = 0
    for g in range(4):
        if grade_mask & (1 << g):
            count += GRADE_SIZES[g]
    return count


@dataclass
class IRType:
    """Type of a tensor in the IR."""
    shape: Tuple[int, ...]          # Batch shape (excludes trailing 8)
    grade_mask: int = GRADE_FULL    # Which grades are live
    dtype: str = 'float32'
    storage_components: Optional[int] = None  # Override physical lane count

    @property
    def n_components(self) -> int:
        """Number of live scalar components per multivector."""
        return count_components(self.grade_mask)

    @property
    def physical_components(self) -> int:
        """
        Physical storage lanes per logical element.

        Multivector tensors use the dense 8-lane Cl(3,0) layout.
        Scalar and integer tensors are stored densely as 1 lane/value.
        """
        if self.storage_components is not None:
            return self.storage_components
        if self.grade_mask == GRADE_SCALAR:
            return 1
        if self.dtype not in ('float16', 'float32', 'float64', 'bfloat16'):
            return 1
        return 8

    @property
    def itemsize(self) -> int:
        """Byte width of one stored lane."""
        import numpy as _np
        return _np.dtype(self.dtype).itemsize

    @property
    def total_elements(self) -> int:
        """Total number of multivectors."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def total_floats(self) -> int:
        """Total stored scalar lanes."""
        return self.total_elements * self.physical_components

    @property
    def live_floats(self) -> int:
        """Total live scalar lanes implied by the grade mask."""
        if self.grade_mask == 0:
            return self.total_elements * self.physical_components
        return self.total_elements * min(self.n_components, self.physical_components)

    @property
    def memory_bytes(self) -> int:
        return self.total_floats * self.itemsize

    def with_grade_mask(self, new_mask: int) -> 'IRType':
        """Return a copy with a different grade mask."""
        storage = self.storage_components
        if storage is None and new_mask == GRADE_SCALAR:
            storage = 1
        return IRType(
            shape=self.shape,
            grade_mask=new_mask,
            dtype=self.dtype,
            storage_components=storage,
        )

    def __repr__(self):
        return (
            f"IRType(shape={self.shape}, grades={grade_mask_to_str(self.grade_mask)}, "
            f"storage={self.physical_components})"
        )


class OpCode:
    """IR operation codes."""
    # Core Clifford ops
    GEOMETRIC_PRODUCT = 'gp'
    REVERSE = 'reverse'
    SANDWICH = 'sandwich'
    BIVECTOR_EXP = 'bvexp'
    GRADE_PROJECT = 'grade_project'
    SCALAR_PRODUCT = 'scalar_product'

    # Arithmetic
    ADD = 'add'
    SCALE = 'scale'

    # Layer ops (compound)
    LINEAR = 'linear'           # CliffordLinear: sum_i GP(W[j,i], x[i]) + b[j]
    ATTENTION = 'attention'     # Full attention block
    ATTN_SCORE = 'attn_score'   # Just the scoring: grade_0(Q * ~K)
    SOFTMAX = 'softmax'
    WEIGHTED_SUM = 'weighted_sum'
    FFN = 'ffn'                 # Full FFN: linear -> activation -> linear
    LAYER_NORM = 'layer_norm'

    # Activations
    GELU = 'gelu'

    # Memory
    EMBED_LOOKUP = 'embed_lookup'
    MATMUL_SCALAR = 'matmul_scalar'  # Standard float matmul
    MEAN_POOL_SEQ = 'mean_pool_seq'  # Mean-pool sequence dim: (B,S,D,8) -> (B,1,D,8)

    # Loss
    CROSS_ENTROPY = 'cross_entropy'

    # Control
    CONSTANT = 'constant'    # Learnable parameter (weight, bias)
    INPUT = 'input'          # Model input (token_ids)
    OUTPUT = 'output'        # Model output (logits)
    COPY = 'copy'
    ZERO = 'zero'

    # Fused ops (produced by fusion pass)
    FUSED_GP_GRADE = 'fused_gp_grade'       # GP + grade_project
    FUSED_FFN = 'fused_ffn'                 # linear -> gelu -> linear
    FUSED_ATTENTION = 'fused_attention'     # full attention kernel
    FUSED_LAYER_NORM = 'fused_layer_norm'   # norm + scale(GP) + add
    FUSED_SANDWICH = 'fused_sandwich'       # GP + reverse + GP

    # Backward ops
    BACKWARD_GP = 'backward_gp'
    BACKWARD_LINEAR = 'backward_linear'
    BACKWARD_ATTENTION = 'backward_attention'
    BACKWARD_FFN = 'backward_ffn'
    BACKWARD_NORM = 'backward_norm'
    BACKWARD_GELU = 'backward_gelu'
    BACKWARD_EMBED = 'backward_embed'
    BACKWARD_CE = 'backward_ce'
    BACKWARD_ADD = 'backward_add'
    BACKWARD_MATMUL_SCALAR = 'backward_matmul_scalar'
    BACKWARD_GRADE_PROJECT = 'backward_grade_project'
    BACKWARD_BVEXP = 'backward_bvexp'
    BACKWARD_SOFTMAX = 'backward_softmax'
    BACKWARD_WEIGHTED_SUM = 'backward_weighted_sum'
    BACKWARD_COPY = 'backward_copy'
    BACKWARD_MEMORY_GATE = 'backward_memory_gate'

    # Optimizer ops
    ADAM_M_UPDATE = 'adam_m_update'            # m = beta1*m + (1-beta1)*g
    ADAM_V_UPDATE = 'adam_v_update'            # v = beta2*v + (1-beta2)*g^2
    ADAM_PARAM_UPDATE = 'adam_param_update'    # p -= lr * m_hat / (sqrt(v_hat) + eps)
    ADAM_FULL_UPDATE = 'adam_full_update'      # Complete Adam step for one parameter

    # Memory bank ops
    MEMORY_READ = 'memory_read'     # query against persistent bank -> retrieved MV
    MEMORY_WRITE = 'memory_write'   # post-step write of a chunk summary into bank state
    MEMORY_GATE = 'memory_gate'     # gated residual injection of retrieved context


@dataclass
class IRNode:
    """A node in the computation graph."""
    id: int
    op: str                                         # OpCode
    inputs: List[int] = field(default_factory=list)  # Input node IDs
    output_type: IRType = None                       # Output tensor type
    attrs: Dict = field(default_factory=dict)        # Operation-specific attributes
    name: str = ''                                   # Human-readable name

    # Optimization annotations
    is_dead: bool = False               # Marked for removal
    fused_into: int = -1                # If fused, points to the fusion target
    required_grade_mask: int = GRADE_FULL  # What grades downstream actually needs

    @property
    def output_grade_mask(self) -> int:
        return self.output_type.grade_mask if self.output_type else GRADE_FULL

    def __repr__(self):
        status = ''
        if self.is_dead:
            status = ' DEAD'
        elif self.fused_into >= 0:
            status = f' ->fused({self.fused_into})'
        grade_str = grade_mask_to_str(self.output_grade_mask)
        return (f"IRNode({self.id}: {self.op} "
                f"inputs={self.inputs} "
                f"grades={grade_str}{status}"
                f"{' ' + self.name if self.name else ''})")


@dataclass
class IRGraph:
    """The full computation graph."""
    nodes: List[IRNode] = field(default_factory=list)
    _next_id: int = 0

    # Metadata
    batch_size: int = 0
    seq_len: int = 0
    config_attrs: Dict = field(default_factory=dict)

    def add_node(self, op: str, inputs: List[int] = None, output_type: IRType = None,
                 attrs: Dict = None, name: str = '') -> int:
        """Add a node and return its ID."""
        node = IRNode(
            id=self._next_id,
            op=op,
            inputs=inputs or [],
            output_type=output_type,
            attrs=attrs or {},
            name=name,
        )
        self.nodes.append(node)
        self._next_id += 1
        return node.id

    def get_node(self, node_id: int) -> IRNode:
        return self.nodes[node_id]

    def get_users(self, node_id: int) -> List[int]:
        """Get IDs of all nodes that use this node as input."""
        return [n.id for n in self.nodes if node_id in n.inputs and not n.is_dead]

    def topological_order(self) -> List[int]:
        """Return node IDs in topological (dependency) order."""
        visited = set()
        order = []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = self.get_node(nid)
            for inp in node.inputs:
                visit(inp)
            order.append(nid)

        for n in self.nodes:
            if not n.is_dead:
                visit(n.id)
        return order

    def live_nodes(self) -> List[IRNode]:
        """Return all non-dead, non-fused nodes."""
        return [n for n in self.nodes if not n.is_dead and n.fused_into < 0]

    def live_node_count(self) -> int:
        """Count of live nodes."""
        return len(self.live_nodes())

    def find_nodes_by_op(self, op: str) -> List[IRNode]:
        """Find all live nodes with a given opcode."""
        return [n for n in self.live_nodes() if n.op == op]

    def total_memory_bytes(self) -> int:
        """Sum of output memory for all live nodes."""
        total = 0
        for n in self.live_nodes():
            if n.output_type is not None:
                total += n.output_type.memory_bytes
        return total

    def total_live_floats(self) -> int:
        """Sum of live (non-dead-grade) floats across all live nodes."""
        total = 0
        for n in self.live_nodes():
            if n.output_type is not None:
                total += n.output_type.live_floats
        return total

    def validate(self) -> List[str]:
        """Validate graph integrity. Returns list of error messages (empty = ok)."""
        errors = []
        live_ids = {n.id for n in self.live_nodes()}
        for node in self.live_nodes():
            for inp_id in node.inputs:
                if inp_id not in live_ids:
                    # Check if it was fused into something that IS live
                    inp_node = self.get_node(inp_id)
                    if inp_node.fused_into >= 0:
                        if inp_node.fused_into not in live_ids:
                            errors.append(
                                f"Node {node.id} ({node.op}) references fused "
                                f"input {inp_id} whose target {inp_node.fused_into} is dead"
                            )
                    elif inp_node.is_dead:
                        errors.append(
                            f"Node {node.id} ({node.op}) references dead input {inp_id}"
                        )
            if node.output_type is None and node.op != OpCode.OUTPUT:
                errors.append(f"Node {node.id} ({node.op}) has no output_type")
        return errors

    def dump(self, show_dead: bool = False) -> str:
        """Pretty-print the graph."""
        lines = [f"IRGraph: {self.live_node_count()} live nodes "
                 f"(batch={self.batch_size}, seq={self.seq_len})"]
        lines.append("=" * 60)
        for node in self.nodes:
            if not show_dead and (node.is_dead or node.fused_into >= 0):
                continue
            grade_str = grade_mask_to_str(node.output_grade_mask)
            shape_str = str(node.output_type.shape) if node.output_type else '?'
            line = f"  [{node.id:3d}] {node.op:20s} inputs={str(node.inputs):20s} "
            line += f"shape={shape_str:25s} grades={grade_str:6s}"
            if node.name:
                line += f"  # {node.name}"
            lines.append(line)
        return '\n'.join(lines)

    def __repr__(self):
        live = self.live_node_count()
        total = len(self.nodes)
        return f"IRGraph({live} live / {total} total nodes)"
