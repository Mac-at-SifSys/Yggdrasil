"""
test_full_model.py -- Test full HLM forward pass with small config.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from holograph.models.hlm_config import HLMConfig
from holograph.models.hlm import HLM


def get_test_config():
    """Small config for fast testing."""
    return HLMConfig(
        vocab_size=64,
        d_model=4,
        n_layers=2,
        n_heads=2,
        d_ff=8,
        n_tou_primitives=16,
        n_blades=9,
        tou_layer_interval=2,
        max_seq_len=32,
        dropout=0.0,
    )


def get_spec_config():
    """Config matching the spec: vocab=256, d_model=32, 2 layers, 4 heads."""
    return HLMConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        n_tou_primitives=32,
        n_blades=9,
        tou_layer_interval=2,
        max_seq_len=64,
        dropout=0.0,
    )


def test_full_forward():
    """Test full forward pass produces correct output shape."""
    config = get_test_config()
    model = HLM(config)

    batch, seq = 2, 8
    tokens = np.random.randint(0, config.vocab_size, size=(batch, seq))

    logits = model.forward(tokens)

    assert logits.shape == (batch, seq, config.vocab_size), \
        f"Expected ({batch}, {seq}, {config.vocab_size}), got {logits.shape}"
    print(f"  [PASS] Full forward: logits shape {logits.shape}")


def test_spec_forward():
    """Test forward pass with spec config (vocab=256, d_model=32, 2 layers, 4 heads).

    Verifies output shape is (batch, seq_len, vocab_size).
    """
    config = get_spec_config()
    model = HLM(config)

    batch, seq = 2, 16
    tokens = np.random.randint(0, config.vocab_size, size=(batch, seq))

    logits = model.forward(tokens)

    expected_shape = (batch, seq, config.vocab_size)
    assert logits.shape == expected_shape, \
        f"Expected {expected_shape}, got {logits.shape}"
    print(f"  [PASS] Spec forward: logits shape {logits.shape} "
          f"(vocab=256, d_model=32, 2 layers, 4 heads)")


def test_spec_gradient_flow():
    """Verify gradient flow through the full model (basic backward check).

    Uses numerical gradient approximation to verify that small perturbations
    in the embedding propagate through to the output, confirming that no
    layer completely blocks gradient flow.
    """
    config = get_spec_config()
    model = HLM(config)

    batch, seq = 1, 8
    tokens = np.random.randint(0, config.vocab_size, size=(batch, seq))

    # Forward pass: baseline
    logits_base = model.forward(tokens).copy()

    # Perturb one embedding parameter and check that output changes.
    # This is a basic "gradient flow" sanity check: if the output
    # changes in response to an input perturbation, gradients can flow.
    epsilon = 1e-3

    params = model.parameters()
    assert len(params) > 0, "Model has no parameters"

    # Perturb the embedding weight for a token that actually appears
    # in our input sequence. The embedding table is (vocab, d_model, 8),
    # so we index into the row for token tokens[0, 0].
    emb_weight = model.embedding.weight  # (vocab, d_model, 8)
    tok_id = int(tokens[0, 0])
    original_val = emb_weight[tok_id, 0, 0]
    emb_weight[tok_id, 0, 0] = original_val + epsilon

    logits_perturbed = model.forward(tokens)

    # Restore
    emb_weight[tok_id, 0, 0] = original_val

    # The output should have changed
    diff = np.abs(logits_perturbed - logits_base).max()
    assert diff > 1e-10, (
        f"Output did not change after parameter perturbation "
        f"(max diff = {diff:.2e}). Gradient flow may be blocked."
    )
    print(f"  [PASS] Gradient flow verified: max output diff = {diff:.6f} "
          f"for epsilon = {epsilon}")


def test_single_token():
    """Test with single token sequence."""
    config = get_test_config()
    model = HLM(config)

    tokens = np.array([[5]])  # batch=1, seq=1
    logits = model.forward(tokens)
    assert logits.shape == (1, 1, config.vocab_size)
    print("  [PASS] Single token forward works")


def test_no_nan():
    """Ensure no NaN or Inf in model output."""
    config = get_test_config()
    model = HLM(config)

    tokens = np.random.randint(0, config.vocab_size, size=(2, 6))
    logits = model.forward(tokens)

    assert not np.any(np.isnan(logits)), "Model output contains NaN"
    assert not np.any(np.isinf(logits)), "Model output contains Inf"
    print("  [PASS] No NaN or Inf in model output")


def test_parameter_count():
    """Test parameter counting."""
    config = get_test_config()
    model = HLM(config)

    n_params = model.count_parameters()
    assert n_params > 0, "Parameter count is zero"
    print(f"  [PASS] Total parameters: {n_params:,}")


def test_embedding():
    """Test CliffordEmbedding directly."""
    from holograph.models.hlm import CliffordEmbedding

    emb = CliffordEmbedding(vocab_size=32, d_model=4)
    tokens = np.array([[0, 1, 2], [3, 4, 5]])
    out = emb.forward(tokens)
    assert out.shape == (2, 3, 4, 8), f"Embedding shape: {out.shape}"

    # Different tokens should give different embeddings
    assert not np.allclose(out[0, 0], out[0, 1]), "Different tokens gave same embedding"
    print("  [PASS] CliffordEmbedding correct")


def test_model_repr():
    """Test model string representation."""
    config = get_test_config()
    model = HLM(config)
    repr_str = repr(model)
    assert 'HLM' in repr_str
    assert str(config.vocab_size) in repr_str
    print(f"  [PASS] Model repr: {repr_str}")


def test_moe_config():
    """Test model with MoE enabled."""
    config = HLMConfig(
        vocab_size=32,
        d_model=4,
        n_layers=2,
        n_heads=2,
        d_ff=8,
        n_tou_primitives=16,
        max_seq_len=16,
        dropout=0.0,
        use_moe=True,
        n_experts=2,
        moe_top_k=1,
    )
    model = HLM(config)
    tokens = np.random.randint(0, config.vocab_size, size=(1, 4))
    logits = model.forward(tokens)
    assert logits.shape == (1, 4, config.vocab_size)
    print("  [PASS] MoE model forward works")


if __name__ == '__main__':
    np.random.seed(42)
    print("Testing full HLM model...")
    test_full_forward()
    test_spec_forward()
    test_spec_gradient_flow()
    test_single_token()
    test_no_nan()
    test_parameter_count()
    test_embedding()
    test_model_repr()
    test_moe_config()
    print("\nAll full model tests passed!")
