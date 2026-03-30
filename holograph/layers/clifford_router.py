"""
clifford_router.py — Grade-energy based Mixture-of-Experts router.

Routes tokens to experts based on their per-grade energy distribution.
A token heavy in bivector content (rotation information) might route to
a different expert than one heavy in scalar content (magnitude information).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from rune.types.multivector import GRADE_SLICES


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / (xp.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


class CliffordRouter:
    """
    Mixture-of-Experts router using grade energy distribution.

    For each token x of shape (d_model, 8):
        grade_energy[k] = sum_d ||grade_k(x[d])||^2   for k in 0..3
    This gives a 4-dimensional energy signature.

    A learnable linear map projects this to n_experts logits,
    then softmax gives routing probabilities.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        """
        Args:
            d_model: number of multivector channels
            n_experts: number of experts to route to
            top_k: number of experts each token is routed to
        """
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router projection: grade_energy (4,) -> expert_logits (n_experts,)
        self.router_weight = xp.random.randn(n_experts, 4).astype(xp.float32) * 0.1
        self.router_bias = xp.zeros(n_experts, dtype=xp.float32)

        # Load balancing: track expert usage
        self._expert_counts = xp.zeros(n_experts, dtype=xp.float32)

    def compute_grade_energy(self, x: np.ndarray) -> np.ndarray:
        """
        Compute per-grade energy for each token.

        Args:
            x: (..., d_model, 8)
        Returns:
            (..., 4) — energy per grade
        """
        energy = xp.zeros((*x.shape[:-2], 4), dtype=xp.float32)
        for g in range(4):
            slc = GRADE_SLICES[g]
            grade_data = x[..., slc]  # (..., d_model, grade_size)
            # Sum of squared norms across d_model
            energy[..., g] = xp.sum(grade_data ** 2, axis=(-2, -1))
        return energy

    def forward(self, x: np.ndarray) -> tuple:
        """
        Route tokens to experts.

        Args:
            x: (..., d_model, 8)
        Returns:
            (expert_indices, expert_weights, grade_energy)
            expert_indices: (..., top_k) — indices of selected experts
            expert_weights: (..., top_k) — softmax weights for selected experts
            grade_energy: (..., 4) — grade energy distribution
        """
        # Compute grade energy: (..., 4)
        energy = self.compute_grade_energy(x)

        # Normalize energy for stable routing
        energy_norm = energy / (xp.sum(energy, axis=-1, keepdims=True) + 1e-12)

        # Project to expert logits: (..., n_experts)
        logits = xp.einsum('...g,eg->...e', energy_norm, self.router_weight) + self.router_bias

        # Full softmax for weight computation
        probs = _softmax(logits, axis=-1)

        # Top-k selection
        # Get indices of top_k highest probabilities
        top_k_indices = xp.argsort(probs, axis=-1)[..., -self.top_k:]  # (..., top_k)

        # Gather top-k weights
        batch_shape = probs.shape[:-1]
        top_k_weights = xp.take_along_axis(probs, top_k_indices, axis=-1)

        # Re-normalize top-k weights
        top_k_weights = top_k_weights / (xp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-12)

        return top_k_indices, top_k_weights, energy

    def load_balance_loss(self) -> float:
        """
        Compute load balancing loss to encourage even expert usage.
        Returns scalar loss value.
        """
        if xp.sum(self._expert_counts) < 1:
            return 0.0
        freq = self._expert_counts / (xp.sum(self._expert_counts) + 1e-12)
        uniform = xp.ones_like(freq) / self.n_experts
        return float(xp.sum((freq - uniform) ** 2))

    def parameters(self):
        return [self.router_weight, self.router_bias]

    def __repr__(self):
        return f"CliffordRouter(n_experts={self.n_experts}, top_k={self.top_k})"
