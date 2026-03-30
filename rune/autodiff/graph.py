"""
graph.py — Computation graph for Clifford autodiff

Nodes represent multivector-valued operations.
Edges carry multivector tensors (not scalar tensors).
The tape records operations during forward pass for backward replay.
"""

from typing import List, Optional, Tuple, Any
import numpy as np


class GradNode:
    """A node in the computation graph."""

    __slots__ = ('op', 'inputs', 'output_data', 'grad_data', 'backward_fn')

    def __init__(self, op: str, inputs: List[Any], output_data: np.ndarray,
                 backward_fn=None):
        self.op = op
        self.inputs = inputs  # List of (Multivector or CliffordTensor) inputs
        self.output_data = output_data
        self.grad_data = None  # Populated during backward
        self.backward_fn = backward_fn

    def __repr__(self):
        return f"GradNode(op={self.op}, shape={self.output_data.shape})"


class ComputationGraph:
    """
    Computation graph / tape for reverse-mode autodiff.

    Records operations during forward pass, replays backward using
    Clifford derivative rules.
    """

    def __init__(self):
        self._tape: List[GradNode] = []
        self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def record(self, node: GradNode):
        """Record an operation on the tape."""
        if self._enabled:
            self._tape.append(node)

    def clear(self):
        """Clear the tape."""
        self._tape.clear()

    def backward(self, output_grad: np.ndarray = None):
        """
        Replay tape backward, computing gradients via Clifford rules.

        Args:
            output_grad: Gradient of the loss w.r.t. the output.
                         If None, assumes scalar loss with grad = 1.
        """
        if not self._tape:
            return

        # Seed gradient at the output
        last_node = self._tape[-1]
        if output_grad is None:
            last_node.grad_data = np.ones_like(last_node.output_data)
        else:
            last_node.grad_data = output_grad

        # Replay backward
        for node in reversed(self._tape):
            if node.backward_fn is not None and node.grad_data is not None:
                node.backward_fn(node.grad_data)

    @property
    def tape(self) -> List[GradNode]:
        return self._tape


# Global computation graph
_global_graph = ComputationGraph()


def get_global_graph() -> ComputationGraph:
    return _global_graph


def reset_global_graph():
    global _global_graph
    _global_graph = ComputationGraph()
