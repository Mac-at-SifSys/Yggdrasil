"""
test_attention.py — Tests for CliffordAttention.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from holograph.layers.clifford_attention import CliffordAttention


def test_attention_shape():
    """Test basic forward pass shape."""
    d_model, n_heads = 8, 2
    attn = CliffordAttention(d_model, n_heads, dropout=0.0)

    batch, seq = 2, 4
    x = np.random.randn(batch, seq, d_model, 8).astype(np.float32) * 0.1
    y = attn.forward(x)

    assert y.shape == (batch, seq, d_model, 8), \
        f"Expected ({batch}, {seq}, {d_model}, 8), got {y.shape}"
    print("  [PASS] CliffordAttention shape correct")


def test_attention_causal_mask():
    """Test that causal masking prevents attending to future tokens."""
    d_model, n_heads = 4, 2
    attn = CliffordAttention(d_model, n_heads, dropout=0.0)

    batch, seq = 1, 6
    x = np.random.randn(batch, seq, d_model, 8).astype(np.float32) * 0.1

    # Causal mask: lower triangular
    mask = np.tril(np.ones((seq, seq), dtype=bool))

    y_masked = attn.forward(x, mask=mask)
    assert y_masked.shape == (batch, seq, d_model, 8)

    # First token output should only depend on first token input
    # Change a later token and verify first token output is unchanged
    x2 = x.copy()
    x2[:, -1, :, :] = np.random.randn(d_model, 8).astype(np.float32) * 10.0
    y2_masked = attn.forward(x2, mask=mask)

    # First token output should be identical
    assert np.allclose(y_masked[:, 0], y2_masked[:, 0], atol=1e-5), \
        "Causal mask failed: first token affected by future token change"
    print("  [PASS] Causal masking works correctly")


def test_attention_no_nan():
    """Ensure no NaNs in output."""
    d_model, n_heads = 8, 4
    attn = CliffordAttention(d_model, n_heads, dropout=0.0)

    x = np.random.randn(2, 8, d_model, 8).astype(np.float32) * 0.1
    y = attn.forward(x)

    assert not np.any(np.isnan(y)), "Attention output contains NaN"
    assert not np.any(np.isinf(y)), "Attention output contains Inf"
    print("  [PASS] No NaN or Inf in attention output")


def test_attention_parameters():
    """Test parameter listing."""
    d_model, n_heads = 8, 2
    attn = CliffordAttention(d_model, n_heads)
    params = attn.parameters()
    assert len(params) > 0, "No parameters returned"
    # Each head has Q, K, V projections (weight only), plus output projection (weight + bias)
    # Per head: 3 weights. Total: n_heads*3 weights + 1 output weight + 1 output bias
    expected = n_heads * 3 + 2  # 3 proj per head (no bias) + out_weight + out_bias
    assert len(params) == expected, f"Expected {expected} params, got {len(params)}"
    print("  [PASS] Parameter count correct")


if __name__ == '__main__':
    np.random.seed(42)
    print("Testing CliffordAttention...")
    test_attention_shape()
    test_attention_causal_mask()
    test_attention_no_nan()
    test_attention_parameters()
    print("\nAll CliffordAttention tests passed!")
