"""
forge.losses -- Clifford-algebra-aware loss functions.

Loss functions that operate on multivector representations:
  - Cross-entropy via scalar (grade-0) projection
  - Algebraic consistency penalties (rotor unitarity, grade stability)
  - Grade entropy regularization (encourage use of all grades)
"""

from forge.losses.cross_entropy import clifford_cross_entropy
from forge.losses.algebraic_consistency import algebraic_consistency_loss
from forge.losses.grade_entropy import grade_entropy_penalty

__all__ = [
    "clifford_cross_entropy",
    "algebraic_consistency_loss",
    "grade_entropy_penalty",
]
