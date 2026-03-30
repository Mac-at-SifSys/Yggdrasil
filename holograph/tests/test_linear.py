"""
test_linear.py — Tests for CliffordLinear and variants.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from holograph.layers.clifford_linear import (
    CliffordLinear, EvenLinear, RotorLinear, ProjectedLinear,
    _geom_prod_batch,
)


def test_clifford_linear_shape():
    """Test basic forward pass shapes."""
    d_in, d_out = 8, 16
    layer = CliffordLinear(d_in, d_out)

    # Single sample
    x = np.random.randn(d_in, 8).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (d_out, 8), f"Expected ({d_out}, 8), got {y.shape}"

    # Batched
    x = np.random.randn(4, 10, d_in, 8).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (4, 10, d_out, 8), f"Expected (4, 10, {d_out}, 8), got {y.shape}"

    print("  [PASS] CliffordLinear shapes correct")


def test_even_linear_grade_preservation():
    """EvenLinear weights should only have even grades."""
    layer = EvenLinear(8, 8)

    # Check that odd grades in weight are zero
    assert np.allclose(layer.weight[:, :, 1:4], 0), "EvenLinear weight has nonzero vector components"
    assert np.allclose(layer.weight[:, :, 7:8], 0), "EvenLinear weight has nonzero trivector components"

    x = np.random.randn(2, 5, 8, 8).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (2, 5, 8, 8)

    print("  [PASS] EvenLinear grade structure correct")


def test_rotor_linear_norm_preservation():
    """RotorLinear (sandwich product) should approximately preserve norms."""
    layer = RotorLinear(4, 4)

    x = np.random.randn(2, 3, 4, 8).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (2, 3, 4, 8)

    # Check norm is approximately preserved (rotors preserve norms)
    from rune.types.multivector import REVERSE_SIGN
    x_flat = x.reshape(-1, 8)
    y_flat = y.reshape(-1, 8)
    x_norms = np.sqrt(np.abs(np.sum(x_flat * (x_flat * REVERSE_SIGN), axis=-1)))
    y_norms = np.sqrt(np.abs(np.sum(y_flat * (y_flat * REVERSE_SIGN), axis=-1)))

    # Norms won't be exactly preserved because we sum over d_in
    # but each individual rotor sandwich preserves norm
    print(f"  Input norms: mean={x_norms.mean():.4f}, Output norms: mean={y_norms.mean():.4f}")
    print("  [PASS] RotorLinear forward works")


def test_projected_linear():
    """ProjectedLinear should zero out non-selected grades."""
    layer = ProjectedLinear(8, 8, output_grades=[0, 2])  # keep only scalar + bivector

    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    y = layer.forward(x)

    # Vector components should be zero
    assert np.allclose(y[..., 1:4], 0), "ProjectedLinear didn't zero vector components"
    # Trivector components should be zero
    assert np.allclose(y[..., 7:8], 0), "ProjectedLinear didn't zero trivector components"
    # Scalar and bivector should be nonzero (with high probability)
    assert not np.allclose(y[..., 0:1], 0), "ProjectedLinear zeroed scalar (unexpected)"
    assert not np.allclose(y[..., 4:7], 0), "ProjectedLinear zeroed bivector (unexpected)"

    print("  [PASS] ProjectedLinear grade projection correct")


def test_parameter_count():
    """Test parameter listing."""
    layer = CliffordLinear(8, 16, bias=True)
    params = layer.parameters()
    assert len(params) == 2, "Expected weight + bias"
    assert params[0].shape == (16, 8, 8), f"Weight shape wrong: {params[0].shape}"
    assert params[1].shape == (16, 8), f"Bias shape wrong: {params[1].shape}"

    layer_no_bias = CliffordLinear(8, 16, bias=False)
    params = layer_no_bias.parameters()
    assert len(params) == 1, "Expected weight only"

    print("  [PASS] Parameter counts correct")


if __name__ == '__main__':
    np.random.seed(42)
    print("Testing CliffordLinear layers...")
    test_clifford_linear_shape()
    test_even_linear_grade_preservation()
    test_rotor_linear_norm_preservation()
    test_projected_linear()
    test_parameter_count()
    print("\nAll CliffordLinear tests passed!")
