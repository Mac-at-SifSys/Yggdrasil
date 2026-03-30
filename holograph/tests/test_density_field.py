"""
test_density_field.py — Tests for DensityField and interactions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from holograph.layers.density_field import DensityField, DensityFieldLayer, density_interaction


def test_density_field_creation():
    """Test creating density fields from multivectors."""
    d_model = 8
    mv_data = np.random.randn(d_model, 8).astype(np.float32) * 0.1
    field = DensityField.from_multivector(mv_data, init_concentration=2.0)

    assert field.mean.shape == (d_model, 8)
    assert field.concentration.shape == (d_model, 8)
    assert field.grade_weights.shape == (d_model, 4)
    assert np.allclose(field.concentration[:, 0], 2.0), "Concentration scalar not set correctly"
    print("  [PASS] DensityField creation correct")


def test_density_evaluation():
    """Test density evaluation at points."""
    d_model = 4
    mean = np.zeros((d_model, 8), dtype=np.float32)
    mean[:, 0] = 1.0  # scalar mean = 1

    conc = np.zeros((d_model, 8), dtype=np.float32)
    conc[:, 0] = 1.0  # concentration = 1

    gw = np.ones((d_model, 4), dtype=np.float32) * 0.25

    field = DensityField(mean, conc, gw)

    # Evaluate at the mean: should give max density
    density_at_mean = field.evaluate(mean)
    # At mean, diff = 0, so density = exp(0) = 1
    assert np.allclose(density_at_mean, 1.0, atol=1e-5), \
        f"Density at mean should be 1.0, got {density_at_mean}"

    # Evaluate far from mean: should give low density
    far_point = np.ones((d_model, 8), dtype=np.float32) * 10.0
    density_far = field.evaluate(far_point)
    assert np.all(density_far < 0.01), f"Density far from mean should be low, got {density_far}"

    print("  [PASS] Density evaluation correct")


def test_density_interaction():
    """Test density field interaction."""
    d_model = 4
    mv_a = np.random.randn(d_model, 8).astype(np.float32) * 0.1
    mv_b = np.random.randn(d_model, 8).astype(np.float32) * 0.1

    field_a = DensityField.from_multivector(mv_a)
    field_b = DensityField.from_multivector(mv_b)

    result = density_interaction(field_a, field_b)

    assert result.mean.shape == (d_model, 8)
    assert result.concentration.shape == (d_model, 8)
    assert result.grade_weights.shape == (d_model, 4)

    # Concentration should be even (no odd grades)
    assert np.allclose(result.concentration[:, 1:4], 0), "Concentration has odd components"
    assert np.allclose(result.concentration[:, 7:8], 0), "Concentration has trivector component"

    # Grade weights should sum to ~1
    gw_sums = np.sum(result.grade_weights, axis=-1)
    assert np.allclose(gw_sums, 1.0, atol=0.1), f"Grade weights don't sum to 1: {gw_sums}"

    print("  [PASS] Density interaction correct")


def test_density_field_layer():
    """Test DensityFieldLayer forward pass."""
    d_model = 4
    layer = DensityFieldLayer(d_model)

    mv_data = np.random.randn(2, 3, d_model, 8).astype(np.float32) * 0.1
    field = DensityField.from_multivector(mv_data)

    result = layer.forward(field)
    assert result.mean.shape == (2, 3, d_model, 8)
    assert result.concentration.shape == (2, 3, d_model, 8)
    assert result.grade_weights.shape == (2, 3, d_model, 4)

    # Concentration should remain even
    assert np.allclose(result.concentration[..., 1:4], 0)
    assert np.allclose(result.concentration[..., 7:8], 0)

    print("  [PASS] DensityFieldLayer forward correct")


def test_density_no_nan():
    """Ensure no NaN in density operations."""
    d_model = 8
    mv = np.random.randn(4, d_model, 8).astype(np.float32)
    field = DensityField.from_multivector(mv)

    point = np.random.randn(4, d_model, 8).astype(np.float32)
    density = field.evaluate(point)
    assert not np.any(np.isnan(density)), "NaN in density evaluation"

    layer = DensityFieldLayer(d_model)
    result = layer.forward(field)
    assert not np.any(np.isnan(result.mean)), "NaN in layer output mean"
    assert not np.any(np.isnan(result.concentration)), "NaN in layer output concentration"

    print("  [PASS] No NaN in density operations")


if __name__ == '__main__':
    np.random.seed(42)
    print("Testing DensityField...")
    test_density_field_creation()
    test_density_evaluation()
    test_density_interaction()
    test_density_field_layer()
    test_density_no_nan()
    print("\nAll DensityField tests passed!")
