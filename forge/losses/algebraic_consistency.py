"""
algebraic_consistency.py -- Penalize representations that break algebraic invariants.

Two main terms:

1. **Rotor unitarity**: For any rotor R used inside RotorLinear layers,
   we want  |R * ~R - 1|  to stay small.  Deviation means the layer
   is no longer an isometry, which can cause gradient explosion / vanishing.

2. **Grade ratio stability**: The energy distribution across grades should
   not shift too abruptly between consecutive layers.  We penalize the
   L2 distance between successive grade distributions.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import xp
from typing import List, Optional
from rune.types.multivector import Multivector, GRADE_SLICES
from rune.ops.batched import batched_geom_prod, batched_reverse


def _rotor_unitarity_penalty(rotor_data: np.ndarray) -> float:
    """Penalty for deviation of a rotor from unit norm.

    For a unit rotor R we have  R * ~R = 1  (scalar 1, all other grades 0).
    The penalty is  mean(| R*~R - 1 |^2).

    Parameters
    ----------
    rotor_data : np.ndarray, shape (..., 8)
        Raw data of rotor multivectors (even subalgebra).

    Returns
    -------
    float
        Mean squared deviation from unitarity.
    """
    rev = batched_reverse(rotor_data)
    prod = batched_geom_prod(rotor_data, rev)

    # Ideal: grade-0 = 1, everything else = 0
    target = xp.zeros_like(prod)
    target[..., 0] = 1.0

    diff = prod - target
    return float(xp.mean(diff ** 2))


def _grade_ratio_penalty(
    representations: List[np.ndarray],
) -> float:
    """Penalize abrupt shifts in grade energy distribution across layers.

    Parameters
    ----------
    representations : list of np.ndarray, each shape (..., 8)
        Activations at successive layers.

    Returns
    -------
    float
        Mean L2 distance between consecutive normalised grade distributions.
    """
    if len(representations) < 2:
        return 0.0

    def _grade_dist(data: np.ndarray) -> np.ndarray:
        """Normalised grade energy distribution, shape (4,)."""
        e = xp.zeros(4, dtype=xp.float32)
        e[0] = float(xp.mean(data[..., 0:1] ** 2))
        e[1] = float(xp.mean(data[..., 1:4] ** 2))
        e[2] = float(xp.mean(data[..., 4:7] ** 2))
        e[3] = float(xp.mean(data[..., 7:8] ** 2))
        total = e.sum()
        if total < 1e-12:
            return xp.ones(4, dtype=xp.float32) * 0.25
        return e / total

    total_penalty = 0.0
    prev_dist = _grade_dist(representations[0])
    for rep in representations[1:]:
        cur_dist = _grade_dist(rep)
        total_penalty += float(xp.sum((cur_dist - prev_dist) ** 2))
        prev_dist = cur_dist

    return total_penalty / (len(representations) - 1)


def algebraic_consistency_loss(
    rotor_params: Optional[List[np.ndarray]] = None,
    layer_representations: Optional[List[np.ndarray]] = None,
    rotor_weight: float = 1.0,
    grade_ratio_weight: float = 0.1,
) -> float:
    """Combined algebraic consistency loss.

    Parameters
    ----------
    rotor_params : list of np.ndarray, optional
        Raw data arrays of rotor parameters (from RotorLinear layers).
    layer_representations : list of np.ndarray, optional
        Activations at each layer, for grade ratio monitoring.
    rotor_weight : float
        Weight of the rotor unitarity term.
    grade_ratio_weight : float
        Weight of the grade ratio stability term.

    Returns
    -------
    float
        Weighted sum of penalties.
    """
    loss = 0.0

    if rotor_params:
        rotor_penalties = [_rotor_unitarity_penalty(r) for r in rotor_params]
        loss += rotor_weight * float(
            xp.mean(xp.asarray(rotor_penalties, dtype=xp.float32))
        )

    if layer_representations and len(layer_representations) >= 2:
        loss += grade_ratio_weight * _grade_ratio_penalty(layer_representations)

    return loss
