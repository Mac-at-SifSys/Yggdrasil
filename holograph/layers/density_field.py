"""
density_field.py — DensityField layer for Cl(3,0).

Each token is a density in multivector space, not a point.
A DensityField has:
  - mean: a multivector (the central element)
  - concentration: an even-subalgebra element (how tightly concentrated)
  - grade_weights: [4] weights for how much each grade contributes

Density interactions use the geometric product, allowing
two density fields to combine in a way that respects the algebraic structure.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from rune.types.multivector import GRADE_SLICES
from holograph.layers.clifford_linear import CliffordLinear
from rune.ops.batched import batched_geom_prod


class DensityField:
    """
    A density field in multivector space.

    Represents a distribution centered at 'mean' with 'concentration'
    controlling the spread (higher concentration = more peaked).

    Attributes:
        mean: (..., 8) — center multivector
        concentration: (..., 8) — even-subalgebra element (grades 0, 2)
        grade_weights: (..., 4) — per-grade contribution weights
    """

    def __init__(self, mean: np.ndarray, concentration: np.ndarray,
                 grade_weights: np.ndarray):
        """
        Args:
            mean: (..., 8) multivector data
            concentration: (..., 8) even-subalgebra element
            grade_weights: (..., 4) weights for grades 0-3
        """
        self.mean = mean.astype(xp.float32)
        self.concentration = concentration.astype(xp.float32)
        self.grade_weights = grade_weights.astype(xp.float32)

    @staticmethod
    def from_multivector(mv_data: np.ndarray, init_concentration: float = 1.0) -> 'DensityField':
        """
        Create a density field from a multivector, with default concentration.
        """
        shape = mv_data.shape[:-1]
        # Concentration: scalar = init_concentration, bivector = 0
        conc = xp.zeros((*shape, 8), dtype=xp.float32)
        conc[..., 0] = init_concentration

        # Grade weights: uniform initially
        gw = xp.ones((*shape, 4), dtype=xp.float32) * 0.25
        return DensityField(mv_data, conc, gw)

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """
        Evaluate density at a point in multivector space.

        density(x) = exp(-concentration_scalar * ||x - mean||^2)
        weighted by grade_weights per grade.

        Args:
            point: (..., 8) — query multivector
        Returns:
            (...,) — density value
        """
        diff = point - self.mean  # (..., 8)

        # Per-grade squared norms
        grade_norms_sq = xp.zeros((*diff.shape[:-1], 4), dtype=xp.float32)
        for g in range(4):
            slc = GRADE_SLICES[g]
            grade_norms_sq[..., g] = xp.sum(diff[..., slc] ** 2, axis=-1)

        # Weighted norm
        weighted_norm_sq = xp.sum(self.grade_weights * grade_norms_sq, axis=-1)

        # Concentration scalar
        conc_scalar = self.concentration[..., 0]

        return xp.exp(-conc_scalar * weighted_norm_sq)


def density_interaction(field_a: DensityField, field_b: DensityField) -> DensityField:
    """
    Two density fields interact via the geometric product.

    The result mean is the geometric product of the two means.
    The result concentration combines both concentrations (geometric product).
    Grade weights are element-wise averaged.
    """
    # Mean: geometric product
    new_mean = batched_geom_prod(field_a.mean, field_b.mean)

    # Concentration: geometric product of even elements
    new_conc = batched_geom_prod(field_a.concentration, field_b.concentration)
    # Project back to even sub-algebra
    new_conc[..., 1:4] = 0.0
    new_conc[..., 7:8] = 0.0

    # Grade weights: geometric mean (sqrt of product)
    new_gw = xp.sqrt(xp.abs(field_a.grade_weights * field_b.grade_weights) + 1e-12)
    # Re-normalize
    gw_sum = xp.sum(new_gw, axis=-1, keepdims=True) + 1e-12
    new_gw = new_gw / gw_sum

    return DensityField(new_mean, new_conc, new_gw)


class DensityFieldLayer:
    """
    Learnable transformation of density fields.

    Transforms mean via CliffordLinear, updates concentration via a
    learnable even-subalgebra transformation, and adjusts grade weights
    via a small MLP.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model

        # Transform mean via CliffordLinear
        self.mean_transform = CliffordLinear(d_model, d_model, bias=True)

        # Transform concentration: learnable even-subalgebra scaling
        # (d_model, 8) but only even components active
        self.conc_scale = xp.zeros((d_model, 8), dtype=xp.float32)
        self.conc_scale[:, 0] = 1.0  # identity initially

        # Grade weight transform: (4, 4) linear + bias
        self.gw_weight = xp.eye(4, dtype=xp.float32) * 0.1 + xp.random.randn(4, 4).astype(xp.float32) * 0.01
        self.gw_bias = xp.zeros(4, dtype=xp.float32)

    def forward(self, field: DensityField) -> DensityField:
        """
        Transform a density field.

        Args:
            field: DensityField with mean (..., d_model, 8),
                   concentration (..., d_model, 8), grade_weights (..., d_model, 4)
        Returns:
            Transformed DensityField
        """
        # Transform mean
        new_mean = self.mean_transform.forward(field.mean)

        # Transform concentration via geometric product with learnable scale
        # conc_scale: (d_model, 8), concentration: (..., d_model, 8)
        new_conc = batched_geom_prod(self.conc_scale, field.concentration)
        # Project to even
        new_conc[..., 1:4] = 0.0
        new_conc[..., 7:8] = 0.0

        # Transform grade weights
        # field.grade_weights: (..., d_model, 4)
        # Linear: W @ gw + b, then softmax
        gw = field.grade_weights
        new_gw = xp.einsum('...i,ji->...j', gw, self.gw_weight) + self.gw_bias
        # Softmax
        exp_gw = xp.exp(new_gw - xp.max(new_gw, axis=-1, keepdims=True))
        new_gw = exp_gw / (xp.sum(exp_gw, axis=-1, keepdims=True) + 1e-12)

        return DensityField(new_mean, new_conc, new_gw)

    def parameters(self):
        params = self.mean_transform.parameters()
        params.extend([self.conc_scale, self.gw_weight, self.gw_bias])
        return params

    def __repr__(self):
        return f"DensityFieldLayer(d_model={self.d_model})"
