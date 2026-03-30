"""
geometric_product.py — Optimized geometric product operations

These are functional-style wrappers that can be traced by the JIT compiler.
"""

import numpy as np
from rune.types.multivector import Multivector, PRODUCT_IDX, PRODUCT_SIGN


def geom_prod(a: Multivector, b: Multivector) -> Multivector:
    """Geometric product of two multivectors."""
    return a.geometric_product(b)


def batched_geom_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Raw batched geometric product on numpy arrays.
    a, b: (..., 8) float32
    Returns: (..., 8) float32

    Optimized loop order for cache efficiency.
    """
    result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

    for i in range(8):
        for j in range(8):
            idx = PRODUCT_IDX[i, j]
            sign = PRODUCT_SIGN[i, j]
            result[..., idx] += sign * a[..., i] * b[..., j]

    return result


def geom_prod_sparse(a: np.ndarray, b: np.ndarray,
                     a_mask: int = 0x0F, b_mask: int = 0x0F) -> np.ndarray:
    """
    Grade-aware sparse geometric product.
    Only computes contributions from active grades.

    a_mask/b_mask: bitmask of active grades in a/b.
    """
    from rune.types.multivector import BASIS_GRADE

    result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

    for i in range(8):
        if not (a_mask & (1 << BASIS_GRADE[i])):
            continue
        for j in range(8):
            if not (b_mask & (1 << BASIS_GRADE[j])):
                continue
            idx = PRODUCT_IDX[i, j]
            sign = PRODUCT_SIGN[i, j]
            result[..., idx] += sign * a[..., i] * b[..., j]

    return result
