"""
grade_decay.py — Entropy-based memory fade.

Higher grades decay faster than lower grades:
  Grade 0 (scalar):    0.9999  — facts persist longest
  Grade 1 (vector):    0.9995  — directions fade
  Grade 2 (bivector):  0.9990  — relationships fade faster
  Grade 3 (trivector): 0.9980  — orientation fades fastest

When total grade energy drops below threshold, the slot is evictable.
"""

import numpy as np
from rune.backend import xp

# Grade component index ranges in the 8-component multivector
GRADE_INDICES = {
    0: slice(0, 1),    # scalar
    1: slice(1, 4),    # vector (e1, e2, e3)
    2: slice(4, 7),    # bivector (e12, e13, e23)
    3: slice(7, 8),    # trivector (e123)
}

DEFAULT_DECAY_RATES = [0.9999, 0.9995, 0.9990, 0.9980]
EVICTION_THRESHOLD = 1e-6


def compute_grade_energy(bank: np.ndarray) -> np.ndarray:
    """
    Compute per-grade energy for each slot.

    Args:
        bank: (n_slots, 8) — memory bank
    Returns:
        (n_slots, 4) — energy per grade per slot
    """
    energy = xp.zeros((bank.shape[0], 4), dtype=xp.float32)
    for g in range(4):
        slc = GRADE_INDICES[g]
        energy[:, g] = xp.sum(bank[:, slc] ** 2, axis=-1)
    return energy


def apply_grade_decay(bank: np.ndarray, grade_energy: np.ndarray,
                      decay_rates=None, n_valid: int = None):
    """
    Apply per-grade decay to memory bank in-place.

    Args:
        bank: (n_slots, 8) — memory bank, modified in-place
        grade_energy: (n_slots, 4) — updated in-place
        decay_rates: list of 4 floats, one per grade
        n_valid: number of valid slots (only decay those)
    Returns:
        evictable: (n_valid,) bool array — True for slots below threshold
    """
    if decay_rates is None:
        decay_rates = DEFAULT_DECAY_RATES

    n = n_valid if n_valid is not None else bank.shape[0]

    for g in range(4):
        slc = GRADE_INDICES[g]
        bank[:n, slc] *= decay_rates[g]

    # Update grade energy
    grade_energy[:n] = compute_grade_energy(bank[:n])

    # Mark evictable slots
    total_energy = grade_energy[:n].sum(axis=1)
    evictable = total_energy < EVICTION_THRESHOLD

    return evictable
