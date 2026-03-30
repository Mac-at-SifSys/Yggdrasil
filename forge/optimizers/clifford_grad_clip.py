"""
clifford_grad_clip.py -- Gradient clipping using multivector norm.

Standard gradient clipping operates element-wise.  Here we compute the
*Clifford norm* of the gradient (via the self-reverse product) so that the
clipping threshold respects the algebraic magnitude rather than treating
each float independently.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import xp
from typing import List

from rune.types.multivector import Multivector
from forge.param_utils import get_param_grad


def _mv_norm_sq(data: np.ndarray) -> float:
    """Positive-definite squared norm of a multivector gradient array.

    We use the *grade-wise* Clifford norm: for each grade k we compute
    the squared norm with the appropriate sign, then take absolute value
    per grade before summing.  This gives a positive-definite quantity
    that still respects the algebraic grade structure.

    Concretely:
        ||x||^2 = |<x_0>|^2 + |<x_1>|^2 + |<x_2>|^2 + |<x_3>|^2

    where <x_k> denotes the grade-k component and |.|^2 is the Euclidean
    norm of that grade's coefficients.
    """
    if data.shape[-1] != 8:
        return float(xp.sum(data ** 2))

    # Grade-wise Euclidean norms (always positive)
    s0 = float(xp.sum(data[..., 0:1] ** 2))  # scalar
    s1 = float(xp.sum(data[..., 1:4] ** 2))  # vector
    s2 = float(xp.sum(data[..., 4:7] ** 2))  # bivector
    s3 = float(xp.sum(data[..., 7:8] ** 2))  # trivector
    return s0 + s1 + s2 + s3


def clifford_grad_clip(
    params: List[dict],
    max_norm: float,
) -> float:
    """Clip gradients of all parameters based on total multivector norm.

    Computes the total Clifford norm across all parameter gradients,
    then scales every gradient down proportionally if the total exceeds
    ``max_norm``.

    Parameters
    ----------
    params : list of dict
        Each dict has ``"mv"`` whose ``._grad`` may hold a gradient Multivector.
    max_norm : float
        Maximum allowed total norm.

    Returns
    -------
    float
        The total norm *before* clipping (useful for logging).
    """
    # Accumulate total squared norm
    total_sq = 0.0
    grad_arrays = []

    for p in params:
        g = get_param_grad(p)
        if g is None:
            continue
        grad_arrays.append(g)
        total_sq += _mv_norm_sq(g)

    total_norm = float(xp.sqrt(total_sq)) if total_sq > 0.0 else 0.0

    if total_norm > max_norm and total_norm > 0.0:
        scale = max_norm / total_norm
        for g in grad_arrays:
            g *= scale

    return total_norm
