"""
forge.optimizers -- Clifford-algebra-aware optimizers.

Optimizers that respect the graded structure of Cl(3,0) multivectors:
per-grade learning rate scaling, geometric-product-based moment estimation,
and multivector-norm gradient clipping.
"""

from forge.optimizers.clifford_adam import CliffordAdam
from forge.optimizers.clifford_sgd import CliffordSGD
from forge.optimizers.clifford_grad_clip import clifford_grad_clip

__all__ = ["CliffordAdam", "CliffordSGD", "clifford_grad_clip"]
