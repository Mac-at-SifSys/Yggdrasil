"""
grade_entropy.py -- Grade entropy penalty.

In Clifford-algebra language models we want representations to actually *use*
all four grades (scalar, vector, bivector, trivector).  If the network
collapses to using only grade-0 it is just doing standard scalar arithmetic
and all the geometric structure is wasted.

The grade entropy penalty computes:
  p_k = ||x_k||^2 / sum_j ||x_j||^2      (energy fraction for grade k)
  H   = -sum_k  p_k * log(p_k)            (Shannon entropy, max = log(4))

and returns a penalty that is *large* when entropy is *low* (collapsed):
  penalty = max_entropy - H
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import xp
from rune.types.multivector import Multivector


def grade_energy(data: np.ndarray) -> np.ndarray:
    """Compute squared norm (energy) per grade.

    Parameters
    ----------
    data : np.ndarray, shape (..., 8)

    Returns
    -------
    np.ndarray, shape (4,)
        Mean energy in each grade across all batch elements.
    """
    e = xp.zeros(4, dtype=xp.float64)
    e[0] = float(xp.mean(data[..., 0:1] ** 2))
    e[1] = float(xp.mean(data[..., 1:4] ** 2))
    e[2] = float(xp.mean(data[..., 4:7] ** 2))
    e[3] = float(xp.mean(data[..., 7:8] ** 2))
    return e


def grade_entropy(data: np.ndarray) -> float:
    """Shannon entropy of the grade energy distribution.

    Parameters
    ----------
    data : np.ndarray, shape (..., 8)

    Returns
    -------
    float
        Entropy in nats.  Maximum is log(4) ~ 1.386 when all grades are equal.
    """
    e = grade_energy(data)
    total = e.sum()
    if total < 1e-30:
        return 0.0
    p = e / total
    # Avoid log(0)
    p_safe = xp.clip(p, 1e-30, None)
    return float(-xp.sum(p * xp.log(p_safe)))


def grade_entropy_penalty(
    x,
    weight: float = 1.0,
) -> float:
    """Penalize low grade entropy (encourages full use of all grades).

    Parameters
    ----------
    x : Multivector or np.ndarray, shape (..., 8)
        The representation to evaluate.
    weight : float
        Multiplier on the penalty.

    Returns
    -------
    float
        ``weight * (max_entropy - H)``.  Zero when perfectly balanced.
    """
    if isinstance(x, Multivector):
        data = x._data
    elif hasattr(x, "shape"):
        data = x
    else:
        raise TypeError(f"Expected Multivector or ndarray, got {type(x)}")

    max_entropy = float(xp.log(4.0))  # ~ 1.3863
    H = grade_entropy(data)

    return weight * (max_entropy - H)
