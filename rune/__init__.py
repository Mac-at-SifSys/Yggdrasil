"""
Rune — Domain-Specific Language for Clifford Algebra Cl(3,0)

A Python-embedded DSL where multivectors and geometric products are
syntactic primitives, not library calls on top of flat-vector operations.

The grade is a compile-time type parameter, not a runtime tag.
"""

from rune.types.multivector import Multivector
from rune.types.graded import Scalar, Vector, Bivector, Trivector, Even, Odd
from rune.types.tensor import CliffordTensor
from rune.autodiff.engine import enable_grad, no_grad, backward

__version__ = "0.1.0"
__all__ = [
    "Multivector", "Scalar", "Vector", "Bivector", "Trivector",
    "Even", "Odd", "CliffordTensor",
    "enable_grad", "no_grad", "backward",
]
