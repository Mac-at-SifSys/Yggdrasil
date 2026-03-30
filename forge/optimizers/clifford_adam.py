"""
clifford_adam.py -- Adam optimizer with Clifford algebra awareness.

Key differences from standard Adam:
  1. Per-grade learning rate scaling via grade_lr_scale
  2. Second moment (v) uses the geometric self-product (grad * ~grad) for the
     Clifford norm, rather than element-wise squaring.  This makes the adaptive
     step size respect the algebraic structure of Cl(3,0).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import to_numpy, xp
from typing import List, Tuple, Optional

from rune.types.multivector import Multivector, GRADE_SLICES
from rune.ops.batched import batched_geom_prod, batched_reverse
from forge.param_utils import (
    clear_param_grad,
    copy_array_,
    grade_scale_for_data,
    get_param_data,
    get_param_grad,
    zero_state_like,
)


def _geometric_self_product(data: np.ndarray) -> np.ndarray:
    """Compute the geometric product  data * ~data  (scalar of self-reverse product).

    For a multivector g, the quantity  g * ~g  is a *multivector* whose grade-0
    component equals the squared Clifford norm.  We keep the full result so that
    the adaptive denominator is grade-aware.

    Parameters
    ----------
    data : np.ndarray, shape (..., 8)
        Raw multivector components.

    Returns
    -------
    np.ndarray, shape (..., 8)
        Components of  data * reverse(data).
    """
    if data.shape[-1] != 8:
        return xp.sum(data ** 2, axis=-1, keepdims=True)
    rev = batched_reverse(data)
    return batched_geom_prod(data, rev)


class CliffordAdam:
    """Adam optimiser with per-grade learning-rate scaling and Clifford-norm v.

    Parameters
    ----------
    params : list of dict
        Each dict has ``"mv"`` (a Multivector with ._data as the parameter
        array) and optionally ``"name"`` for logging.
    lr : float
        Base learning rate.
    betas : tuple of float
        (beta1, beta2) for exponential moving averages.
    eps : float
        Numerical stability term.
    weight_decay : float
        L2 weight decay (decoupled, applied before the update).
    grade_lr_scale : list of float, length 4
        Per-grade multiplier on the learning rate.  Default [1, 1, 1, 0.5]
        dampens the trivector grade, which is often noisy early in training.
    """

    def __init__(
        self,
        params: List[dict],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grade_lr_scale: Optional[List[float]] = None,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grade_lr_scale = grade_lr_scale or [1.0, 1.0, 1.0, 0.5]

        self.t = 0  # step counter

        # State: first moment (m), second moment (v) per parameter
        self.state: List[dict] = []
        for p in self.params:
            data = get_param_data(p)
            self.state.append({
                "m": zero_state_like(data),
                "v": zero_state_like(data),
            })

    # ------------------------------------------------------------------
    def step(self):
        """Perform a single optimisation step over all parameters."""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for p, s in zip(self.params, self.state):
            data = get_param_data(p)
            g = get_param_grad(p)
            if g is None:
                continue

            # Decoupled weight decay
            if self.weight_decay != 0.0:
                data -= self.weight_decay * self.lr * data

            # First moment: standard EMA
            s["m"] = self.beta1 * s["m"] + (1.0 - self.beta1) * g

            # Second moment: use the Clifford norm of the gradient.
            # v_scalar = scalar_part( g * ~g ) = the squared Clifford norm,
            # broadcast as a single positive scalar across all 8 components.
            # This ensures the adaptive denominator reflects algebraic magnitude
            # rather than element-wise squares.
            g_clamped = xp.clip(g, -1e4, 1e4)
            gp = _geometric_self_product(g_clamped)
            # Extract scalar part (index 0) = squared Clifford norm
            norm_sq = xp.abs(gp[..., 0:1])  # shape (..., 1)
            # Broadcast to all 8 components
            v_update = xp.broadcast_to(norm_sq, g.shape).copy()
            s["v"] = self.beta2 * s["v"] + (1.0 - self.beta2) * v_update

            # Bias-corrected estimates
            m_hat = s["m"] / bc1
            v_hat = s["v"] / bc2

            # Update with per-grade LR scaling
            lr_scale = grade_scale_for_data(data, self.grade_lr_scale)
            update = m_hat / (xp.sqrt(xp.maximum(v_hat, 1e-30)) + self.eps)
            update *= lr_scale

            data -= self.lr * update

            # Zero out the gradient for next iteration
            clear_param_grad(p)

    # ------------------------------------------------------------------
    def zero_grad(self):
        """Clear all parameter gradients."""
        for p in self.params:
            clear_param_grad(p)

    def state_dict(self) -> dict:
        """Return serialisable optimiser state."""
        return {
            "t": self.t,
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grade_lr_scale": self.grade_lr_scale,
            "states": [
                {"m": to_numpy(s["m"]).copy(), "v": to_numpy(s["v"]).copy()}
                for s in self.state
            ],
        }

    def load_state_dict(self, sd: dict):
        """Restore optimiser state from a dict."""
        self.t = sd["t"]
        self.lr = sd["lr"]
        self.beta1, self.beta2 = sd["betas"]
        self.eps = sd["eps"]
        self.weight_decay = sd["weight_decay"]
        self.grade_lr_scale = sd["grade_lr_scale"]
        for s, saved in zip(self.state, sd["states"]):
            copy_array_(s["m"], saved["m"])
            copy_array_(s["v"], saved["v"])
