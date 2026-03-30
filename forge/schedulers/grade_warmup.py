"""
grade_warmup.py -- Gradually activate higher Clifford grades during training.

Motivation: if all grades start active, the optimiser can get trapped in a
local minimum that ignores the higher-grade structure.  By *warming up* grades
one at a time we force the scalar channel to learn a reasonable baseline,
then layer in vector, bivector, and finally trivector representations.

The schedule is specified by a list of step thresholds at which each
successive grade becomes active.
"""

import numpy as np
from typing import List, Optional


def grade_mask_at_step(
    step: int,
    schedule: Optional[List[int]] = None,
) -> int:
    """Return a 4-bit grade bitmask indicating which grades are active.

    Bit 0 = grade 0 (scalar), bit 1 = grade 1 (vector), etc.

    Parameters
    ----------
    step : int
        Current training step.
    schedule : list of int, length 4, optional
        Step at which each grade becomes active.
        Default: [0, 500, 2000, 5000] --
        scalar from step 0, vector from 500, bivector from 2000,
        trivector from 5000.

    Returns
    -------
    int
        Bitmask in range [0x01, 0x0F].
    """
    if schedule is None:
        schedule = [0, 500, 2000, 5000]

    mask = 0
    for grade, threshold in enumerate(schedule):
        if step >= threshold:
            mask |= (1 << grade)
    return mask


def grade_mask_to_component_mask(grade_mask: int) -> np.ndarray:
    """Convert a 4-bit grade mask to an 8-element float mask for Cl(3,0).

    Parameters
    ----------
    grade_mask : int
        Bitmask: bit k set means grade k is active.

    Returns
    -------
    np.ndarray, shape (8,), dtype float32
        1.0 for active components, 0.0 for masked.
    """
    from rune.types.multivector import GRADE_SLICES
    mask = np.zeros(8, dtype=np.float32)
    for grade, slc in GRADE_SLICES.items():
        if grade_mask & (1 << grade):
            mask[slc] = 1.0
    return mask


class GradeWarmupScheduler:
    """Stateful scheduler that tracks grade activation.

    Parameters
    ----------
    schedule : list of int, length 4
        Step thresholds for each grade.
    """

    def __init__(self, schedule: Optional[List[int]] = None):
        self.schedule = schedule or [0, 500, 2000, 5000]
        self._prev_mask = 0

    def get_mask(self, step: int) -> int:
        """Return the grade bitmask for the current step."""
        mask = grade_mask_at_step(step, self.schedule)
        if mask != self._prev_mask:
            self._prev_mask = mask
        return mask

    def get_component_mask(self, step: int) -> np.ndarray:
        """Return the 8-element component mask for the current step."""
        return grade_mask_to_component_mask(self.get_mask(step))

    def is_grade_active(self, step: int, grade: int) -> bool:
        """Check if a specific grade is active at the given step."""
        return bool(self.get_mask(step) & (1 << grade))

    def all_active_step(self) -> int:
        """Return the step at which all grades are active."""
        return max(self.schedule)
