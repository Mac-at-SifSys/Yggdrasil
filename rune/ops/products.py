"""
products.py — Outer, inner, scalar, sandwich products
"""

from rune.types.multivector import Multivector


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """Outer (wedge) product."""
    return a.outer_product(b)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """Left contraction inner product."""
    return a.inner_product(b)


def scalar_product(a: Multivector, b: Multivector):
    """Scalar part of geometric product."""
    return a.scalar_product(b)


def sandwich(r: Multivector, x: Multivector) -> Multivector:
    """Sandwich product: r * x * ~r."""
    return r.sandwich(x)


def commutator(a: Multivector, b: Multivector) -> Multivector:
    """Commutator product: (ab - ba) / 2."""
    ab = a.geometric_product(b)
    ba = b.geometric_product(a)
    return (ab - ba) * 0.5
