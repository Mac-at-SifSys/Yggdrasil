"""
grade_projection.py — Grade projection operations
"""

import numpy as np
from rune.types.multivector import Multivector, GRADE_SLICES


def grade_project(mv: Multivector, k: int) -> Multivector:
    """Extract grade-k component."""
    return mv.grade(k)


def even_project(mv: Multivector) -> Multivector:
    """Extract even sub-algebra (grades 0 + 2)."""
    return mv.even()


def odd_project(mv: Multivector) -> Multivector:
    """Extract odd sub-algebra (grades 1 + 3)."""
    return mv.odd()


def grade_energy(mv: Multivector) -> np.ndarray:
    """
    Compute the energy (squared norm) per grade.
    Returns array of shape (..., 4) with energy for grades 0-3.
    """
    data = mv.data
    energies = np.zeros((*data.shape[:-1], 4), dtype=np.float32)
    energies[..., 0] = data[..., 0] ** 2
    energies[..., 1] = np.sum(data[..., 1:4] ** 2, axis=-1)
    energies[..., 2] = np.sum(data[..., 4:7] ** 2, axis=-1)
    energies[..., 3] = data[..., 7] ** 2
    return energies


def grade_distribution(mv: Multivector) -> np.ndarray:
    """
    Normalized grade energy distribution.
    Returns array of shape (..., 4) summing to 1.
    """
    e = grade_energy(mv)
    total = np.sum(e, axis=-1, keepdims=True)
    return e / np.maximum(total, 1e-12)
