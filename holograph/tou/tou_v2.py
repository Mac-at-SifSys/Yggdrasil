"""
tou_v2.py — Tensor of Understanding v2.0 integration class.

Combines primitives and blades into a unified processing framework
that routes multivector inputs through blade-specific pathways,
with primitive-based gating and sandwich product transformations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from rune.types.multivector import PRODUCT_IDX, PRODUCT_SIGN, GRADE_SLICES
from holograph.tou.primitives import ToUPrimitives
from holograph.tou.blades import ToUBlades, N_BLADES
from rune.ops.batched import batched_norm, batched_scalar_product, batched_sandwich


def _softmax(x, axis=-1):
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / (xp.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


class ToUV2:
    """
    Tensor of Understanding v2.0.

    Processing pipeline:
    1. Compute affinity of input with primitives (scalar product)
    2. Route through 9 blades based on grade content
    3. Transform each blade output via sandwich product with learned rotors
    4. Recombine with primitive-weighted gating

    Parameters:
        d_model: multivector channel dimension
        n_primitives: number of active primitives (can sub-sample from 1486)
        n_blades: number of blades (9)
    """

    def __init__(self, d_model: int, n_active_primitives: int = 256):
        self.d_model = d_model
        self.n_active = min(n_active_primitives, 1486)

        # Initialize all 1486 primitives and 9 blades
        self.primitives = ToUPrimitives(learnable=True)
        self.blades = ToUBlades()
        self.blade_masks = xp.stack([blade.mask for blade in self.blades.blades], axis=0)

        # Active primitive indices (subset for efficiency)
        # Select evenly across categories
        self.active_indices = xp.linspace(0, 1485, self.n_active, dtype=xp.int32)

        # Learnable rotor per blade: (n_blades, d_model, 8)
        # Each rotor transforms the blade output via sandwich product
        self.blade_rotors = xp.zeros((N_BLADES, d_model, 8), dtype=xp.float32)
        self.blade_rotors[:, :, 0] = 1.0  # identity rotors
        # Small bivector perturbation
        self.blade_rotors[:, :, 4:7] = xp.random.randn(N_BLADES, d_model, 3).astype(xp.float32) * 0.01

        # Primitive-to-blade routing weights: (n_active, n_blades)
        self.prim_blade_weights = xp.random.randn(self.n_active, N_BLADES).astype(xp.float32) * 0.1

        # Output mixing: learnable scalar per blade
        self.blade_mix = xp.ones(N_BLADES, dtype=xp.float32) / N_BLADES

    def _get_unit_rotors(self) -> np.ndarray:
        """Normalize blade rotors to unit rotors."""
        r = self.blade_rotors.copy()
        r[..., 1:4] = 0.0
        r[..., 7:8] = 0.0
        norm = batched_norm(r)
        norm = xp.maximum(norm, 1e-12)
        return r / norm[..., xp.newaxis]

    def compute_primitive_affinity(self, x: np.ndarray) -> np.ndarray:
        """
        Compute affinity between input and active primitives.

        x: (..., d_model, 8) -> (..., n_active)
        Affinity = sum over d_model of scalar_part(x * primitive)
        """
        active_prims = self.primitives.data[self.active_indices]  # (n_active, 8)
        # x: (..., d_model, 8), prims: (n_active, 8)
        # For each primitive, compute scalar product with mean of x across d_model
        x_mean = xp.mean(x, axis=-2)  # (..., 8)

        # Scalar part of geometric product using batched_scalar_product
        # x_mean: (..., 8) -> (..., 1, 8), active_prims: (n_active, 8)
        x_exp = x_mean[..., xp.newaxis, :]  # (..., 1, 8)
        affinity = batched_scalar_product(x_exp, active_prims)  # (..., n_active)

        return affinity

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        ToU v2.0 forward pass.

        Args:
            x: (..., d_model, 8)
        Returns:
            (..., d_model, 8) — transformed output
        """
        # 1. Compute primitive affinity: (..., n_active)
        affinity = self.compute_primitive_affinity(x)
        prim_weights = _softmax(affinity, axis=-1)  # (..., n_active)

        # 2. Compute blade routing from primitives: (..., n_blades)
        # prim_weights: (..., n_active), prim_blade_weights: (n_active, n_blades)
        blade_weights = xp.einsum('...p,pb->...b', prim_weights, _softmax(self.prim_blade_weights, axis=-1))
        # Normalize
        blade_weights = blade_weights / (xp.sum(blade_weights, axis=-1, keepdims=True) + 1e-12)

        # 3. Project through all blades at once.
        mask_view = self.blade_masks.reshape((N_BLADES,) + (1,) * (x.ndim - 1) + (8,))
        blade_outputs = x[xp.newaxis, ...] * mask_view  # (n_blades, ..., d_model, 8)

        # 4. Transform all blade outputs via sandwich products in one batched call.
        rotors = self._get_unit_rotors()  # (n_blades, d_model, 8)
        rotor_view = rotors.reshape((N_BLADES,) + (1,) * (x.ndim - 2) + rotors.shape[1:])
        transformed_outputs = batched_sandwich(rotor_view, blade_outputs)

        # 5. Recombine with blade weights
        # blade_weights: (..., n_blades) -> (..., n_blades, 1, 1)
        bw_expanded = blade_weights[..., :, xp.newaxis, xp.newaxis]  # (..., n_blades, 1, 1)
        stacked = xp.moveaxis(transformed_outputs, 0, -3)  # (..., n_blades, d_model, 8)
        output = xp.sum(bw_expanded * stacked, axis=-3)   # (..., d_model, 8)

        return output

    def parameters(self):
        params = self.primitives.parameters()
        params.extend([self.blade_rotors, self.prim_blade_weights, self.blade_mix])
        return params

    def __repr__(self):
        return f"ToUV2(d_model={self.d_model}, n_active={self.n_active}, n_blades={N_BLADES})"
