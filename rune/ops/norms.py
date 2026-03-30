"""
norms.py — Norm operations for multivectors
"""

import numpy as np
from rune.types.multivector import Multivector


def norm_squared(mv: Multivector) -> np.ndarray:
    return mv.norm_squared()


def norm(mv: Multivector) -> np.ndarray:
    return mv.norm()


def normalize(mv: Multivector) -> Multivector:
    return mv.normalize()
