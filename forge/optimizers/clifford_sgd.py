"""
clifford_sgd.py -- Grade-aware SGD with momentum.

Each Cl(3,0) grade can have its own momentum coefficient, allowing finer
control over how quickly scalar vs. vector vs. bivector vs. trivector
components adapt.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import xp
from typing import List, Optional

from rune.types.multivector import Multivector, GRADE_SLICES
from forge.param_utils import (
    clear_param_grad,
    copy_array_,
    grade_scale_for_data,
    get_param_data,
    get_param_grad,
    zero_state_like,
)


class CliffordSGD:
    """Stochastic Gradient Descent with per-grade momentum.

    Parameters
    ----------
    params : list of dict
        Each dict has ``"mv"`` (a Multivector parameter).
    lr : float
        Base learning rate.
    momentum : float
        Default momentum coefficient (used for any grade not overridden).
    grade_momentum : list of float, length 4, optional
        Per-grade momentum coefficients [grade0, grade1, grade2, grade3].
        If None, ``momentum`` is used for all grades.
    weight_decay : float
        L2 weight decay.
    grade_lr_scale : list of float, length 4, optional
        Per-grade learning rate multiplier.  Default all 1.0.
    """

    def __init__(
        self,
        params: List[dict],
        lr: float = 1e-2,
        momentum: float = 0.9,
        grade_momentum: Optional[List[float]] = None,
        weight_decay: float = 0.0,
        grade_lr_scale: Optional[List[float]] = None,
    ):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

        self.grade_momentum = grade_momentum or [momentum] * 4
        self.grade_lr_scale = grade_lr_scale or [1.0, 1.0, 1.0, 1.0]

        # Velocity buffers
        self.velocity: List[np.ndarray] = []
        for p in self.params:
            self.velocity.append(zero_state_like(get_param_data(p)))

    def step(self):
        """Perform one SGD step with per-grade momentum."""
        for p, v in zip(self.params, self.velocity):
            data = get_param_data(p)
            g = get_param_grad(p)
            if g is None:
                continue

            # Weight decay
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * data

            # Per-grade momentum update
            comp_momentum = grade_scale_for_data(data, self.grade_momentum)
            v[:] = comp_momentum * v + g

            # Apply per-grade LR scaling
            comp_lr = grade_scale_for_data(data, self.grade_lr_scale)
            data -= self.lr * comp_lr * v

            clear_param_grad(p)

    def zero_grad(self):
        for p in self.params:
            clear_param_grad(p)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "grade_momentum": self.grade_momentum,
            "grade_lr_scale": self.grade_lr_scale,
            "weight_decay": self.weight_decay,
            "velocity": [v.copy() for v in self.velocity],
        }

    def load_state_dict(self, sd: dict):
        self.lr = sd["lr"]
        self.grade_momentum = sd["grade_momentum"]
        self.grade_lr_scale = sd["grade_lr_scale"]
        self.weight_decay = sd["weight_decay"]
        for v, saved in zip(self.velocity, sd["velocity"]):
            copy_array_(v, saved)
