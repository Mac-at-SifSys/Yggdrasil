"""
YGGDRASIL L2 Runtime — runtime.py
Pure-Python fallback implementation of the C++ runtime.

Fully functional without the C++ library. Uses numpy for efficient
multivector arithmetic in Cl(3,0).

Components:
  - MVValue: 8-component multivector for Cl(3,0)
  - EagerRuntime: immediate execution mode
  - GraphRuntime: trace-and-optimize execution
  - GradeMemoryPool: Python-accessible grade-stratified allocator
  - AutodiffTape: records ops, replays backward
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# ============================================================================
# Constants for Cl(3,0)
# ============================================================================

NUM_GRADES = 4
MV_DIM = 8
GRADE_DIM = (1, 3, 3, 1)  # components per grade

# Grade bitmasks
GRADE_0: int = 0x01
GRADE_1: int = 0x02
GRADE_2: int = 0x04
GRADE_3: int = 0x08
ALL_GRADES: int = 0x0F

# Offsets into the 8-component layout
_GRADE_OFFSET = [0, 1, 4, 7]  # cumulative sum of GRADE_DIM


def grade_offset(g: int) -> int:
    return _GRADE_OFFSET[g]


def grade_slice(g: int) -> slice:
    start = _GRADE_OFFSET[g]
    return slice(start, start + GRADE_DIM[g])


# ============================================================================
# OpType enumeration (mirrors the C++ enum)
# ============================================================================


class OpType(enum.IntEnum):
    INPUT = 0
    CONSTANT = 1
    NEGATE = 10
    REVERSE = 11
    GRADE_INVOLUTION = 12
    CLIFFORD_CONJ = 13
    DUAL = 14
    GRADE_PROJECT = 15
    NORM = 16
    GEOMETRIC_PROD = 20
    OUTER_PROD = 21
    INNER_PROD = 22
    SCALAR_PROD = 23
    REGRESSIVE_PROD = 24
    SANDWICH = 25
    SCALAR_MUL = 30
    SCALAR_ADD = 31
    ADD = 40
    SUB = 41
    EXP = 50
    LOG = 51
    FUSED_GP_GRADE = 100
    FUSED_SANDWICH = 101
    FUSED_GP_CHAIN = 102


# ============================================================================
# MVValue — multivector in Cl(3,0)
# ============================================================================


class MVValue:
    """
    8-component multivector for Cl(3,0).
    Layout: [s, e1, e2, e3, e12, e13, e23, e123]
    """

    __slots__ = ("data",)

    def __init__(self, data: Optional[np.ndarray] = None):
        if data is None:
            self.data = np.zeros(MV_DIM, dtype=np.float32)
        else:
            self.data = np.asarray(data, dtype=np.float32).ravel()[:MV_DIM].copy()
            if len(self.data) < MV_DIM:
                self.data = np.pad(self.data, (0, MV_DIM - len(self.data)))

    # ---- Factories ----------------------------------------------------------

    @staticmethod
    def zero() -> "MVValue":
        return MVValue()

    @staticmethod
    def scalar(s: float) -> "MVValue":
        v = MVValue()
        v.data[0] = s
        return v

    @staticmethod
    def vector(x: float, y: float, z: float) -> "MVValue":
        v = MVValue()
        v.data[1], v.data[2], v.data[3] = x, y, z
        return v

    @staticmethod
    def bivector(xy: float, xz: float, yz: float) -> "MVValue":
        v = MVValue()
        v.data[4], v.data[5], v.data[6] = xy, xz, yz
        return v

    @staticmethod
    def pseudoscalar(p: float) -> "MVValue":
        v = MVValue()
        v.data[7] = p
        return v

    # ---- Grade access -------------------------------------------------------

    def grade(self, g: int) -> np.ndarray:
        return self.data[grade_slice(g)].copy()

    def grade_project(self, mask: int) -> "MVValue":
        r = MVValue()
        for g in range(NUM_GRADES):
            if mask & (1 << g):
                r.data[grade_slice(g)] = self.data[grade_slice(g)]
        return r

    @property
    def scalar_part(self) -> float:
        return float(self.data[0])

    # ---- Arithmetic ---------------------------------------------------------

    def __add__(self, other: "MVValue") -> "MVValue":
        return MVValue(self.data + other.data)

    def __sub__(self, other: "MVValue") -> "MVValue":
        return MVValue(self.data - other.data)

    def __neg__(self) -> "MVValue":
        return MVValue(-self.data)

    def __mul__(self, other: Union["MVValue", float]) -> "MVValue":
        if isinstance(other, (int, float)):
            return MVValue(self.data * float(other))
        return geometric_product(self, other)

    def __rmul__(self, other: float) -> "MVValue":
        return MVValue(self.data * float(other))

    def __repr__(self) -> str:
        parts = []
        labels = ["1", "e1", "e2", "e3", "e12", "e13", "e23", "e123"]
        for i, (v, l) in enumerate(zip(self.data, labels)):
            if abs(v) > 1e-7:
                parts.append(f"{v:.4f}*{l}")
        return "MV(" + " + ".join(parts) + ")" if parts else "MV(0)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MVValue):
            return NotImplemented
        return np.allclose(self.data, other.data, atol=1e-6)

    def norm_sq(self) -> float:
        rev = reverse(self)
        prod = geometric_product(self, rev)
        return float(prod.data[0])

    def norm(self) -> float:
        return float(np.sqrt(abs(self.norm_sq())))


# ============================================================================
# Clifford algebra operations for Cl(3,0)
# ============================================================================

# Sign tables for reverse and grade involution
_REVERSE_SIGN = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32)
_INVOLUTE_SIGN = np.array([1, -1, -1, -1, 1, 1, 1, -1], dtype=np.float32)


def reverse(a: MVValue) -> MVValue:
    return MVValue(a.data * _REVERSE_SIGN)


def grade_involution(a: MVValue) -> MVValue:
    return MVValue(a.data * _INVOLUTE_SIGN)


def clifford_conjugate(a: MVValue) -> MVValue:
    return reverse(grade_involution(a))


def grade_project(a: MVValue, mask: int) -> MVValue:
    return a.grade_project(mask)


def dual(a: MVValue) -> MVValue:
    """Poincare dual: a* = a * e123^{-1}.  In Cl(3,0), e123^{-1} = -e123."""
    e123_inv = MVValue.pseudoscalar(-1.0)
    return geometric_product(a, e123_inv)


# ---- Geometric product (full Cl(3,0) Cayley table) -------------------------

def geometric_product(a: MVValue, b: MVValue) -> MVValue:
    """Full geometric product in Cl(3,0) with e1^2 = e2^2 = e3^2 = +1."""
    A = a.data
    B = b.data
    R = np.zeros(MV_DIM, dtype=np.float32)

    # Indices: 0=1, 1=e1, 2=e2, 3=e3, 4=e12, 5=e13, 6=e23, 7=e123

    R[0] = (A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3]
            - A[4]*B[4] - A[5]*B[5] - A[6]*B[6] - A[7]*B[7])

    R[1] = (A[0]*B[1] + A[1]*B[0] - A[2]*B[4] - A[3]*B[5]
            + A[4]*B[2] + A[5]*B[3] - A[6]*B[7] - A[7]*B[6])

    R[2] = (A[0]*B[2] + A[2]*B[0] + A[1]*B[4] - A[3]*B[6]
            - A[4]*B[1] + A[6]*B[3] + A[5]*B[7] + A[7]*B[5])

    R[3] = (A[0]*B[3] + A[3]*B[0] + A[1]*B[5] + A[2]*B[6]
            - A[5]*B[1] - A[6]*B[2] - A[4]*B[7] - A[7]*B[4])

    R[4] = (A[0]*B[4] + A[4]*B[0] + A[1]*B[2] - A[2]*B[1]
            + A[3]*B[7] + A[7]*B[3] - A[5]*B[6] + A[6]*B[5])

    R[5] = (A[0]*B[5] + A[5]*B[0] + A[1]*B[3] - A[3]*B[1]
            - A[2]*B[7] - A[7]*B[2] + A[4]*B[6] - A[6]*B[4])

    R[6] = (A[0]*B[6] + A[6]*B[0] + A[2]*B[3] - A[3]*B[2]
            + A[1]*B[7] + A[7]*B[1] - A[4]*B[5] + A[5]*B[4])

    R[7] = (A[0]*B[7] + A[7]*B[0] + A[1]*B[6] - A[2]*B[5] + A[3]*B[4]
            + A[4]*B[3] - A[5]*B[2] + A[6]*B[1])

    return MVValue(R)


def outer_product(a: MVValue, b: MVValue) -> MVValue:
    """Outer (wedge) product: keep only grade-increasing terms."""
    result = MVValue()
    for ga in range(NUM_GRADES):
        ag = a.grade_project(1 << ga)
        for gb in range(NUM_GRADES):
            target = ga + gb
            if target > 3:
                continue
            bg = b.grade_project(1 << gb)
            prod = geometric_product(ag, bg)
            proj = prod.grade_project(1 << target)
            result = result + proj
    return result


def inner_product(a: MVValue, b: MVValue) -> MVValue:
    """Left contraction: <a_r * b_s>_{s-r} for s >= r."""
    result = MVValue()
    for ga in range(NUM_GRADES):
        ag = a.grade_project(1 << ga)
        for gb in range(ga, NUM_GRADES):
            target = gb - ga
            bg = b.grade_project(1 << gb)
            prod = geometric_product(ag, bg)
            proj = prod.grade_project(1 << target)
            result = result + proj
    return result


def sandwich(r: MVValue, x: MVValue) -> MVValue:
    """Sandwich product: r * x * reverse(r)."""
    return geometric_product(geometric_product(r, x), reverse(r))


def mv_norm(a: MVValue) -> MVValue:
    """Return scalar MV with the norm."""
    return MVValue.scalar(a.norm())


# ============================================================================
# GradeMemoryPool — grade-stratified allocator
# ============================================================================


class GradeMemoryPool:
    """
    Pre-allocates pools for each grade (0..3) to avoid fragmentation.
    Each pool stores blocks of a fixed number of multivectors.
    """

    def __init__(self, block_mv: int = 256, num_blocks: int = 64):
        self.block_mv = block_mv
        self.num_blocks = num_blocks
        self._pools: List[List[np.ndarray]] = []  # free lists per grade
        self._arenas: List[List[np.ndarray]] = []  # all allocated blocks

        for g in range(NUM_GRADES):
            dim = GRADE_DIM[g]
            blocks = []
            arena = []
            for _ in range(num_blocks):
                buf = np.zeros((block_mv, dim), dtype=np.float32)
                blocks.append(buf)
                arena.append(buf)
            self._pools.append(blocks)
            self._arenas.append(arena)

    def allocate(self, grade: int) -> np.ndarray:
        """Allocate a block for a specific grade. Returns (block_mv, grade_dim) array."""
        if grade < 0 or grade >= NUM_GRADES:
            raise ValueError(f"grade must be 0..{NUM_GRADES - 1}")
        pool = self._pools[grade]
        if not pool:
            raise RuntimeError(f"Grade-{grade} pool exhausted")
        return pool.pop()

    def free(self, grade: int, buf: np.ndarray) -> None:
        """Return a block to the pool."""
        if grade < 0 or grade >= NUM_GRADES:
            raise ValueError(f"grade must be 0..{NUM_GRADES - 1}")
        buf[:] = 0  # zero before returning
        self._pools[grade].append(buf)

    def stats(self) -> Dict[int, Dict[str, int]]:
        return {
            g: {
                "total": self.num_blocks,
                "free": len(self._pools[g]),
                "used": self.num_blocks - len(self._pools[g]),
            }
            for g in range(NUM_GRADES)
        }


# ============================================================================
# AutodiffTape — records multivector ops, replays backward
# ============================================================================


class _TapeEntry:
    __slots__ = (
        "id", "op", "input_ids", "forward_value",
        "saved_inputs", "grade_mask",
    )

    def __init__(
        self,
        id: int,
        op: OpType,
        input_ids: List[int],
        forward_value: MVValue,
        saved_inputs: List[MVValue],
        grade_mask: int = ALL_GRADES,
    ):
        self.id = id
        self.op = op
        self.input_ids = input_ids
        self.forward_value = forward_value
        self.saved_inputs = saved_inputs
        self.grade_mask = grade_mask


class AutodiffTape:
    """
    Records multivector-level operations and replays backward to compute
    gradients using Clifford derivative rules.
    """

    def __init__(self):
        self._entries: List[_TapeEntry] = []
        self._adjoints: Dict[int, MVValue] = {}

    # ---- Recording API (forward pass) ----

    def record_input(self, value: MVValue, label: str = "") -> int:
        eid = len(self._entries)
        self._entries.append(_TapeEntry(
            id=eid, op=OpType.INPUT, input_ids=[],
            forward_value=value, saved_inputs=[],
        ))
        return eid

    def record_unary(
        self, op: OpType, input_id: int, result: MVValue,
        mask: int = ALL_GRADES,
    ) -> int:
        eid = len(self._entries)
        self._entries.append(_TapeEntry(
            id=eid, op=op, input_ids=[input_id],
            forward_value=result,
            saved_inputs=[self._entries[input_id].forward_value],
            grade_mask=mask,
        ))
        return eid

    def record_binary(
        self, op: OpType, lhs_id: int, rhs_id: int, result: MVValue,
        mask: int = ALL_GRADES,
    ) -> int:
        eid = len(self._entries)
        self._entries.append(_TapeEntry(
            id=eid, op=op, input_ids=[lhs_id, rhs_id],
            forward_value=result,
            saved_inputs=[
                self._entries[lhs_id].forward_value,
                self._entries[rhs_id].forward_value,
            ],
            grade_mask=mask,
        ))
        return eid

    # ---- Backward pass ----

    def backward(self, output_id: int, seed: Optional[MVValue] = None) -> None:
        if seed is None:
            seed = MVValue.scalar(1.0)
        self._adjoints.clear()
        self._adjoints[output_id] = seed

        for i in range(output_id, -1, -1):
            if i not in self._adjoints:
                continue
            entry = self._entries[i]
            if entry.op == OpType.INPUT:
                continue

            out_adj = self._adjoints[i]
            input_adjs = self._backward_rule(entry, out_adj)

            for k, inp_id in enumerate(entry.input_ids):
                if k >= len(input_adjs):
                    break
                if inp_id in self._adjoints:
                    self._adjoints[inp_id] = self._adjoints[inp_id] + input_adjs[k]
                else:
                    self._adjoints[inp_id] = input_adjs[k]

    def grad(self, id: int) -> MVValue:
        return self._adjoints.get(id, MVValue.zero())

    # ---- Utilities ----

    def value(self, id: int) -> MVValue:
        return self._entries[id].forward_value

    def is_leaf(self, id: int) -> bool:
        return self._entries[id].op == OpType.INPUT

    def clear(self) -> None:
        self._entries.clear()
        self._adjoints.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # ---- Backward rules ----

    @staticmethod
    def _backward_rule(entry: _TapeEntry, out_adj: MVValue) -> List[MVValue]:
        op = entry.op

        if op == OpType.NEGATE:
            return [-out_adj]

        if op == OpType.REVERSE:
            return [reverse(out_adj)]

        if op == OpType.GRADE_INVOLUTION:
            return [grade_involution(out_adj)]

        if op == OpType.CLIFFORD_CONJ:
            return [reverse(grade_involution(out_adj))]

        if op == OpType.GRADE_PROJECT:
            return [out_adj.grade_project(entry.grade_mask)]

        if op == OpType.NORM:
            a = entry.saved_inputs[0]
            n = entry.forward_value.scalar_part
            if abs(n) < 1e-12:
                return [MVValue.zero()]
            a_rev = reverse(a)
            return [MVValue(a_rev.data * (out_adj.scalar_part / n))]

        if op == OpType.ADD:
            return [out_adj, out_adj]

        if op == OpType.SUB:
            return [out_adj, -out_adj]

        if op == OpType.SCALAR_MUL:
            scalar_mv = entry.saved_inputs[0]
            operand = entry.saved_inputs[1]
            s = scalar_mv.scalar_part
            grad_operand = MVValue(out_adj.data * s)
            ds = float(np.dot(operand.data, out_adj.data))
            grad_scalar = MVValue.scalar(ds)
            return [grad_scalar, grad_operand]

        if op == OpType.GEOMETRIC_PROD:
            a, b = entry.saved_inputs
            b_rev = reverse(b)
            a_rev = reverse(a)
            grad_a = geometric_product(out_adj, b_rev)
            grad_b = geometric_product(a_rev, out_adj)
            return [grad_a, grad_b]

        if op in (OpType.OUTER_PROD, OpType.INNER_PROD):
            a, b = entry.saved_inputs
            b_rev = reverse(b)
            a_rev = reverse(a)
            grad_a = geometric_product(out_adj, b_rev)
            grad_b = geometric_product(a_rev, out_adj)
            return [grad_a, grad_b]

        if op == OpType.SANDWICH:
            rot, x = entry.saved_inputs
            r_rev = reverse(rot)
            grad_x = sandwich(r_rev, out_adj)
            x_rev = reverse(x)
            term1 = geometric_product(geometric_product(out_adj, r_rev), x_rev)
            term2 = geometric_product(geometric_product(x, r_rev), out_adj)
            grad_r = term1 + term2
            return [grad_r, grad_x]

        if op == OpType.SCALAR_ADD:
            grad_scalar = MVValue.scalar(out_adj.scalar_part)
            return [grad_scalar, out_adj]

        # Default: zero gradients
        return [MVValue.zero() for _ in entry.input_ids]


# ============================================================================
# Graph node (Python equivalent of C++ GraphNode)
# ============================================================================


class _GraphNode:
    __slots__ = (
        "id", "op", "inputs", "num_mv", "grade_mask",
        "label", "const_data", "topo_order", "batch_id",
    )

    def __init__(
        self,
        id: int = -1,
        op: OpType = OpType.INPUT,
        inputs: Optional[List[int]] = None,
        num_mv: int = 1,
        grade_mask: int = ALL_GRADES,
        label: str = "",
        const_data: Optional[np.ndarray] = None,
    ):
        self.id = id
        self.op = op
        self.inputs = inputs or []
        self.num_mv = num_mv
        self.grade_mask = grade_mask
        self.label = label
        self.const_data = const_data
        self.topo_order = -1
        self.batch_id = -1


# ============================================================================
# EagerRuntime — immediate execution mode
# ============================================================================


class EagerRuntime:
    """
    Execute Clifford algebra operations immediately.
    Optionally records to an AutodiffTape for backward pass.
    """

    def __init__(self, tape: Optional[AutodiffTape] = None):
        self.tape = tape
        self.pool = GradeMemoryPool()

    def input(self, value: MVValue, label: str = "") -> Tuple[int, MVValue]:
        """Register an input multivector. Returns (tape_id, value)."""
        tid = -1
        if self.tape is not None:
            tid = self.tape.record_input(value, label)
        return tid, value

    def negate(self, tid: int, val: MVValue) -> Tuple[int, MVValue]:
        result = -val
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_unary(OpType.NEGATE, tid, result)
        return new_tid, result

    def reverse_op(self, tid: int, val: MVValue) -> Tuple[int, MVValue]:
        result = reverse(val)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_unary(OpType.REVERSE, tid, result)
        return new_tid, result

    def grade_project_op(
        self, tid: int, val: MVValue, mask: int,
    ) -> Tuple[int, MVValue]:
        result = val.grade_project(mask)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_unary(
                OpType.GRADE_PROJECT, tid, result, mask
            )
        return new_tid, result

    def add(
        self, tid_a: int, a: MVValue, tid_b: int, b: MVValue,
    ) -> Tuple[int, MVValue]:
        result = a + b
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(OpType.ADD, tid_a, tid_b, result)
        return new_tid, result

    def sub(
        self, tid_a: int, a: MVValue, tid_b: int, b: MVValue,
    ) -> Tuple[int, MVValue]:
        result = a - b
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(OpType.SUB, tid_a, tid_b, result)
        return new_tid, result

    def geometric_product_op(
        self, tid_a: int, a: MVValue, tid_b: int, b: MVValue,
    ) -> Tuple[int, MVValue]:
        result = geometric_product(a, b)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(
                OpType.GEOMETRIC_PROD, tid_a, tid_b, result
            )
        return new_tid, result

    def outer_product_op(
        self, tid_a: int, a: MVValue, tid_b: int, b: MVValue,
    ) -> Tuple[int, MVValue]:
        result = outer_product(a, b)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(
                OpType.OUTER_PROD, tid_a, tid_b, result
            )
        return new_tid, result

    def inner_product_op(
        self, tid_a: int, a: MVValue, tid_b: int, b: MVValue,
    ) -> Tuple[int, MVValue]:
        result = inner_product(a, b)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(
                OpType.INNER_PROD, tid_a, tid_b, result
            )
        return new_tid, result

    def sandwich_op(
        self, tid_r: int, r: MVValue, tid_x: int, x: MVValue,
    ) -> Tuple[int, MVValue]:
        result = sandwich(r, x)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_binary(
                OpType.SANDWICH, tid_r, tid_x, result
            )
        return new_tid, result

    def norm_op(self, tid: int, val: MVValue) -> Tuple[int, MVValue]:
        result = mv_norm(val)
        new_tid = -1
        if self.tape is not None:
            new_tid = self.tape.record_unary(OpType.NORM, tid, result)
        return new_tid, result

    def backward(self, output_tid: int, seed: Optional[MVValue] = None) -> None:
        if self.tape is None:
            raise RuntimeError("No tape attached to EagerRuntime")
        self.tape.backward(output_tid, seed)

    def grad(self, tid: int) -> MVValue:
        if self.tape is None:
            raise RuntimeError("No tape attached to EagerRuntime")
        return self.tape.grad(tid)


# ============================================================================
# GraphRuntime — trace-and-optimize execution
# ============================================================================


class GraphRuntime:
    """
    Build a computation graph, optimize it with fusion passes, then execute.
    """

    def __init__(self):
        self._nodes: List[_GraphNode] = []
        self._alive: List[bool] = []
        self._values: Dict[int, MVValue] = {}  # node_id → computed value
        self._exec_order: List[int] = []

    # ---- Graph construction -------------------------------------------------

    def add_input(
        self, value: MVValue, label: str = "", mask: int = ALL_GRADES,
    ) -> int:
        nid = len(self._nodes)
        self._nodes.append(_GraphNode(
            id=nid, op=OpType.INPUT, label=label, grade_mask=mask,
        ))
        self._alive.append(True)
        self._values[nid] = value
        return nid

    def add_constant(self, value: MVValue, label: str = "") -> int:
        nid = len(self._nodes)
        self._nodes.append(_GraphNode(
            id=nid, op=OpType.CONSTANT, label=label,
            const_data=value.data.copy(),
        ))
        self._alive.append(True)
        self._values[nid] = value
        return nid

    def add_unary(self, op: OpType, input_id: int, mask: int = ALL_GRADES) -> int:
        nid = len(self._nodes)
        self._nodes.append(_GraphNode(
            id=nid, op=op, inputs=[input_id], grade_mask=mask,
        ))
        self._alive.append(True)
        return nid

    def add_binary(
        self, op: OpType, lhs: int, rhs: int, mask: int = ALL_GRADES,
    ) -> int:
        nid = len(self._nodes)
        self._nodes.append(_GraphNode(
            id=nid, op=op, inputs=[lhs, rhs], grade_mask=mask,
        ))
        self._alive.append(True)
        return nid

    # ---- Topological sort ---------------------------------------------------

    def topological_sort(self) -> List[int]:
        n = len(self._nodes)
        in_deg = [0] * n
        for i in range(n):
            if not self._alive[i]:
                continue
            for inp in self._nodes[i].inputs:
                if self._alive[inp]:
                    in_deg[i] += 1

        from collections import deque
        q = deque()
        for i in range(n):
            if self._alive[i] and in_deg[i] == 0:
                q.append(i)

        order = []
        topo = 0
        while q:
            u = q.popleft()
            self._nodes[u].topo_order = topo
            topo += 1
            order.append(u)

            for v in range(n):
                if not self._alive[v]:
                    continue
                if u in self._nodes[v].inputs:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        q.append(v)

        self._exec_order = order
        return order

    # ---- Fusion passes ------------------------------------------------------

    def fuse_gp_grade(self) -> int:
        """Fuse grade_project(geometric_product(a,b)) -> FUSED_GP_GRADE."""
        consumers = self._build_consumers()
        count = 0
        for nid in list(self._live_ids()):
            node = self._nodes[nid]
            if node.op != OpType.GRADE_PROJECT:
                continue
            if len(node.inputs) != 1:
                continue
            gp_id = node.inputs[0]
            gp = self._nodes[gp_id]
            if gp.op != OpType.GEOMETRIC_PROD:
                continue
            if len(consumers.get(gp_id, [])) > 1:
                continue

            # Fuse
            self._nodes[nid] = _GraphNode(
                id=nid, op=OpType.FUSED_GP_GRADE,
                inputs=gp.inputs, grade_mask=node.grade_mask,
                label="fused_gp_grade",
            )
            self._alive[gp_id] = False
            count += 1
        return count

    def fuse_sandwich(self) -> int:
        """Detect R * x * reverse(R) -> FUSED_SANDWICH."""
        consumers = self._build_consumers()
        count = 0
        for nid in list(self._live_ids()):
            node = self._nodes[nid]
            if node.op != OpType.GEOMETRIC_PROD or len(node.inputs) != 2:
                continue
            inner_id, rev_id = node.inputs
            inner = self._nodes[inner_id]
            rev_node = self._nodes[rev_id]
            if inner.op != OpType.GEOMETRIC_PROD or rev_node.op != OpType.REVERSE:
                continue
            if len(inner.inputs) != 2 or len(rev_node.inputs) != 1:
                continue
            r_in_inner = inner.inputs[0]
            r_in_rev = rev_node.inputs[0]
            if r_in_inner != r_in_rev:
                continue
            if len(consumers.get(inner_id, [])) > 1:
                continue
            if len(consumers.get(rev_id, [])) > 1:
                continue

            x_id = inner.inputs[1]
            self._nodes[nid] = _GraphNode(
                id=nid, op=OpType.FUSED_SANDWICH,
                inputs=[r_in_inner, x_id], grade_mask=node.grade_mask,
                label="fused_sandwich",
            )
            self._alive[inner_id] = False
            self._alive[rev_id] = False
            count += 1
        return count

    def run_fusion_passes(self) -> Dict[str, int]:
        sw = self.fuse_sandwich()
        gp = self.fuse_gp_grade()
        return {"sandwich": sw, "gp_grade": gp}

    # ---- Execution ----------------------------------------------------------

    def execute(self) -> Dict[int, MVValue]:
        """Execute the graph in topological order, return all computed values."""
        order = self.topological_sort()

        for nid in order:
            node = self._nodes[nid]
            if nid in self._values:
                continue  # input or constant already set

            inputs = [self._values[i] for i in node.inputs]
            result = self._eval_op(node.op, inputs, node.grade_mask)
            self._values[nid] = result

        return dict(self._values)

    def get_value(self, nid: int) -> MVValue:
        return self._values.get(nid, MVValue.zero())

    # ---- Internal helpers ---------------------------------------------------

    def _build_consumers(self) -> Dict[int, List[int]]:
        consumers: Dict[int, List[int]] = {}
        for nid in self._live_ids():
            for inp in self._nodes[nid].inputs:
                consumers.setdefault(inp, []).append(nid)
        return consumers

    def _live_ids(self):
        return [i for i in range(len(self._nodes)) if self._alive[i]]

    @staticmethod
    def _eval_op(op: OpType, inputs: List[MVValue], mask: int) -> MVValue:
        if op == OpType.NEGATE:
            return -inputs[0]
        if op == OpType.REVERSE:
            return reverse(inputs[0])
        if op == OpType.GRADE_INVOLUTION:
            return grade_involution(inputs[0])
        if op == OpType.CLIFFORD_CONJ:
            return clifford_conjugate(inputs[0])
        if op == OpType.DUAL:
            return dual(inputs[0])
        if op == OpType.GRADE_PROJECT:
            return inputs[0].grade_project(mask)
        if op == OpType.NORM:
            return mv_norm(inputs[0])
        if op == OpType.GEOMETRIC_PROD:
            return geometric_product(inputs[0], inputs[1])
        if op == OpType.OUTER_PROD:
            return outer_product(inputs[0], inputs[1])
        if op == OpType.INNER_PROD:
            return inner_product(inputs[0], inputs[1])
        if op == OpType.SANDWICH:
            return sandwich(inputs[0], inputs[1])
        if op == OpType.ADD:
            return inputs[0] + inputs[1]
        if op == OpType.SUB:
            return inputs[0] - inputs[1]
        if op == OpType.SCALAR_MUL:
            return MVValue(inputs[0].data[0] * inputs[1].data)
        if op == OpType.SCALAR_ADD:
            r = MVValue(inputs[1].data.copy())
            r.data[0] += inputs[0].data[0]
            return r
        # Fused ops
        if op == OpType.FUSED_GP_GRADE:
            prod = geometric_product(inputs[0], inputs[1])
            return prod.grade_project(mask)
        if op == OpType.FUSED_SANDWICH:
            return sandwich(inputs[0], inputs[1])
        if op == OpType.FUSED_GP_CHAIN:
            tmp = geometric_product(inputs[0], inputs[1])
            return geometric_product(tmp, inputs[2])

        raise ValueError(f"Unknown op: {op}")
