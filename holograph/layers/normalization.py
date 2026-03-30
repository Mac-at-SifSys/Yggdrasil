"""
normalization.py -- CliffordLayerNorm for Cl(3,0) multivectors.

Normalizes total multivector norm while preserving grade ratios.
Scaling uses the geometric product, not element-wise multiplication.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from rune.ops.batched import batched_geom_prod
from rune.autodiff.clifford_rules import CliffordDerivativeRules
from holograph.layers.clifford_linear import _accumulate_grad, _zero_grad_params


class CliffordLayerNorm:
    """
    Layer normalization for multivector sequences.

    Given input x of shape (batch, seq, d_model, 8):
    1. Compute per-element multivector norm: ||x_i|| = sqrt(<x_i ~x_i>_0)
    2. Compute RMS norm across d_model
    3. Normalize: x_hat = x / rms
    4. Scale via geometric product: y = gamma * x_hat
    5. Shift via addition: y = y + beta

    gamma and beta are learnable multivectors of shape (d_model, 8).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Learnable scale: initialized to scalar 1 (identity under GP)
        self.gamma = xp.zeros((d_model, 8), dtype=xp.float32)
        self.gamma[:, 0] = 1.0

        # Learnable shift
        self.beta = xp.zeros((d_model, 8), dtype=xp.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (..., d_model, 8) -- batched multivector sequence
        Returns:
            (..., d_model, 8) -- normalized output
        """
        # In Cl(3,0), scalar_part(x * ~x) is exactly the Euclidean sum of
        # squared Clifford components. Keep the original numerics exactly:
        # rms^2 = mean(norm_sq + eps) + eps = mean(norm_sq) + 2*eps.
        norm_sq = xp.sum(x * x, axis=-1)  # (..., d_model)
        rms_sq = xp.mean(norm_sq, axis=-1, keepdims=True) + (2.0 * self.eps)
        rms = xp.sqrt(rms_sq)  # (..., 1)

        x_hat = x / rms[..., xp.newaxis]
        y = batched_geom_prod(self.gamma, x_hat)
        y = y + self.beta

        self._cache = {
            "x": x.copy(),
            "x_hat": x_hat,
            "rms": rms,
        }
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through CliffordLayerNorm.

        Forward: y = GP(gamma, x / rms) + beta
        where rms = sqrt(mean(sum_c x^2) + const)
        """
        x = self._cache["x"]
        x_hat = self._cache["x_hat"]
        rms = self._cache["rms"]

        batch_shape = x.shape[:-2]
        n_batch = len(batch_shape)
        reduce_axes = tuple(range(n_batch)) if n_batch else None

        grad_beta = grad_output.sum(axis=reduce_axes) if reduce_axes is not None else grad_output
        _accumulate_grad(self.beta, grad_beta)

        gamma_bc = xp.broadcast_to(self.gamma, x_hat.shape).copy()
        grad_gamma_full, grad_x_hat = CliffordDerivativeRules.geometric_product_backward(
            grad_output, gamma_bc, x_hat
        )
        grad_gamma = grad_gamma_full.sum(axis=reduce_axes) if reduce_axes is not None else grad_gamma_full
        _accumulate_grad(self.gamma, grad_gamma)

        # Closed-form RMS backward:
        # dL/dx = dL/dx_hat / rms - x * <dL/dx_hat, x> / (d_model * rms^3)
        inner = xp.sum(grad_x_hat * x, axis=(-2, -1), keepdims=True)
        inv_rms = 1.0 / (rms + 1e-12)
        correction = x * (inner / (self.d_model * (rms[..., xp.newaxis] ** 3 + 1e-12)))
        return grad_x_hat * inv_rms[..., xp.newaxis] - correction

    def zero_grad(self):
        """Reset gradient accumulators for all parameters."""
        _zero_grad_params(self.parameters())

    def parameters(self):
        """Return list of parameter arrays."""
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"CliffordLayerNorm(d_model={self.d_model})"
