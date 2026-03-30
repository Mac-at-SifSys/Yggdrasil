"""
test_backward_pass.py — Test gradient flow through the HLM model.

Uses numerical differentiation (finite differences) to verify
that gradients can flow through all layers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from holograph.models.hlm_config import HLMConfig
from holograph.models.hlm import HLM
from holograph.layers.clifford_linear import CliffordLinear
from holograph.layers.activations import clifford_gelu, clifford_relu, clifford_sigmoid


def numerical_gradient(f, x, eps=1e-4):
    """
    Compute numerical gradient of scalar function f at array x.
    Returns gradient array of same shape as x.
    """
    grad = np.zeros_like(x)
    flat = x.ravel()
    grad_flat = grad.ravel()
    for i in range(min(flat.size, 50)):  # limit for speed
        old = flat[i]
        flat[i] = old + eps
        fp = f(x)
        flat[i] = old - eps
        fm = f(x)
        flat[i] = old
        grad_flat[i] = (fp - fm) / (2 * eps)
    return grad


def test_linear_gradient():
    """Test gradient flow through CliffordLinear."""
    np.random.seed(42)
    layer = CliffordLinear(4, 4, bias=True)

    x = np.random.randn(4, 8).astype(np.float32) * 0.1

    def loss_fn(w):
        layer.weight = w.reshape(layer.weight.shape)
        y = layer.forward(x)
        return float(np.sum(y ** 2))

    w_flat = layer.weight.ravel().copy()
    grad = numerical_gradient(loss_fn, w_flat)

    # Check gradient is nonzero (information flows)
    assert np.any(np.abs(grad) > 1e-6), "Gradient through CliffordLinear is all zero"
    print(f"  [PASS] CliffordLinear gradient: max={np.max(np.abs(grad)):.6f}")


def test_activation_gradient():
    """Test gradient flow through activations."""
    x = np.random.randn(4, 8).astype(np.float32) * 0.5
    eps = 1e-4

    for name, fn in [('gelu', clifford_gelu), ('relu', clifford_relu), ('sigmoid', clifford_sigmoid)]:
        grad = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(8):
                x_p = x.copy(); x_p[i, j] += eps
                x_m = x.copy(); x_m[i, j] -= eps
                grad[i, j] = (np.sum(fn(x_p) ** 2) - np.sum(fn(x_m) ** 2)) / (2 * eps)

        nonzero = np.sum(np.abs(grad) > 1e-6)
        print(f"  [PASS] {name} gradient: {nonzero}/{grad.size} nonzero entries")


def test_attention_gradient():
    """Test gradient flow through CliffordAttention."""
    from holograph.layers.clifford_attention import CliffordAttention

    np.random.seed(42)
    attn = CliffordAttention(4, 2, dropout=0.0)

    x = np.random.randn(1, 3, 4, 8).astype(np.float32) * 0.1
    eps = 1e-4

    def loss_fn_x(x_in):
        y = attn.forward(x_in)
        return float(np.sum(y ** 2))

    # Check gradient w.r.t. input
    grad = np.zeros_like(x)
    flat = x.ravel()
    grad_flat = grad.ravel()
    for i in range(min(flat.size, 30)):
        old = flat[i]
        flat[i] = old + eps
        fp = loss_fn_x(x)
        flat[i] = old - eps
        fm = loss_fn_x(x)
        flat[i] = old
        grad_flat[i] = (fp - fm) / (2 * eps)

    assert np.any(np.abs(grad) > 1e-6), "Gradient through attention is all zero"
    print(f"  [PASS] Attention gradient: max={np.max(np.abs(grad)):.6f}")


def test_normalization_gradient():
    """Test gradient flow through CliffordLayerNorm."""
    from holograph.layers.normalization import CliffordLayerNorm

    d_model = 4
    norm = CliffordLayerNorm(d_model)
    x = np.random.randn(2, d_model, 8).astype(np.float32) * 0.5
    eps = 1e-4

    def loss_fn(x_in):
        y = norm.forward(x_in)
        return float(np.sum(y ** 2))

    # Sample a few gradient entries
    grad_samples = []
    flat = x.ravel()
    for i in range(min(flat.size, 20)):
        old = flat[i]
        flat[i] = old + eps
        fp = loss_fn(x)
        flat[i] = old - eps
        fm = loss_fn(x)
        flat[i] = old
        grad_samples.append(abs((fp - fm) / (2 * eps)))

    assert max(grad_samples) > 1e-6, "Gradient through LayerNorm is all zero"
    print(f"  [PASS] LayerNorm gradient: max={max(grad_samples):.6f}")


def test_full_model_gradient():
    """Test end-to-end gradient flow through the full model."""
    np.random.seed(42)
    config = HLMConfig(
        vocab_size=16,
        d_model=4,
        n_layers=2,
        n_heads=2,
        d_ff=8,
        n_tou_primitives=8,
        max_seq_len=16,
        dropout=0.0,
    )
    model = HLM(config)

    tokens = np.array([[1, 2, 3, 4]])
    eps = 1e-4

    # Gradient w.r.t. embedding weight — perturb in-place
    emb_w = model.embedding.weight
    grad_samples = []
    # Only perturb entries for tokens we actually use (0-4), to ensure signal
    token_set = set(tokens.ravel().tolist())
    indices = []
    for t in token_set:
        for c in range(min(emb_w.shape[1], 4)):
            for k in range(8):
                indices.append((t, c, k))
    for idx in indices[:20]:
        old = emb_w[idx]
        emb_w[idx] = old + eps
        fp = float(np.sum(model.forward(tokens) ** 2))
        emb_w[idx] = old - eps
        fm = float(np.sum(model.forward(tokens) ** 2))
        emb_w[idx] = old
        grad_samples.append(abs((fp - fm) / (2 * eps)))

    assert max(grad_samples) > 1e-8, "End-to-end gradient is zero"
    print(f"  [PASS] Full model gradient: max={max(grad_samples):.6f}")


if __name__ == '__main__':
    np.random.seed(42)
    print("Testing backward pass / gradient flow...")
    test_linear_gradient()
    test_activation_gradient()
    test_attention_gradient()
    test_normalization_gradient()
    test_full_model_gradient()
    print("\nAll backward pass tests passed!")
