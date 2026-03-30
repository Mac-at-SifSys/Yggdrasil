"""
engine.py — Autodiff engine for Clifford algebra operations

Reverse-mode automatic differentiation using Clifford-specific rules.
NOT generic autograd through scalar decomposition.
"""

import numpy as np
from contextlib import contextmanager
from typing import List, Optional
from rune.autodiff.graph import get_global_graph, GradNode
from rune.autodiff.clifford_rules import CliffordDerivativeRules


_grad_enabled = True


def enable_grad():
    global _grad_enabled
    _grad_enabled = True


@contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def backward(output, grad_output=None):
    """
    Compute gradients via reverse-mode autodiff with Clifford rules.

    Args:
        output: The output Multivector/CliffordTensor whose graph to differentiate
        grad_output: Upstream gradient. Defaults to all-ones if None.
    """
    from rune.types.multivector import Multivector
    from rune.types.tensor import CliffordTensor

    if not hasattr(output, '_grad_fn') or output._grad_fn is None:
        return

    # Build the backward chain
    _backward_recursive(output, grad_output)


def _backward_recursive(node, grad_output=None):
    """Recursively compute gradients through the computation graph."""
    from rune.types.multivector import Multivector

    if not hasattr(node, '_grad_fn') or node._grad_fn is None:
        return

    op_name = node._grad_fn[0]
    args = node._grad_fn[1:]

    if grad_output is None:
        grad_output = np.ones_like(node._data)

    if op_name == 'geometric_product':
        a, b = args
        grad_a, grad_b = CliffordDerivativeRules.geometric_product_backward(
            grad_output, a._data, b._data
        )
        _accumulate_grad(a, grad_a)
        _accumulate_grad(b, grad_b)
        _backward_recursive(a, grad_a)
        _backward_recursive(b, grad_b)

    elif op_name == 'add':
        a, b = args
        grad_a, grad_b = CliffordDerivativeRules.add_backward(grad_output)
        _accumulate_grad(a, grad_a)
        _accumulate_grad(b, grad_b)
        _backward_recursive(a, grad_a)
        _backward_recursive(b, grad_b)

    elif op_name == 'reverse':
        (x,) = args
        grad_x = CliffordDerivativeRules.reverse_backward(grad_output)
        _accumulate_grad(x, grad_x)
        _backward_recursive(x, grad_x)

    elif op_name == 'grade_project':
        x, grade = args
        grad_x = CliffordDerivativeRules.grade_project_backward(grad_output, grade)
        _accumulate_grad(x, grad_x)
        _backward_recursive(x, grad_x)

    elif op_name == 'scale':
        x, s = args
        grad_x = CliffordDerivativeRules.scale_backward(grad_output, s)
        _accumulate_grad(x, grad_x)
        _backward_recursive(x, grad_x)


def _accumulate_grad(node, grad):
    """Accumulate gradient into a node."""
    if hasattr(node, '_requires_grad') and node._requires_grad:
        if node._grad is None:
            from rune.types.multivector import Multivector
            node._grad = Multivector(grad.copy())
        else:
            node._grad._data += grad
