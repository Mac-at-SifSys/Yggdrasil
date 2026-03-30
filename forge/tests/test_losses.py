"""
test_losses.py -- Test all loss functions in forge.losses.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.types.multivector import Multivector
from forge.losses.cross_entropy import clifford_cross_entropy, clifford_cross_entropy_with_grad
from forge.losses.algebraic_consistency import algebraic_consistency_loss
from forge.losses.grade_entropy import grade_entropy_penalty, grade_entropy, grade_energy


# ---- Cross-entropy tests ----

def test_ce_perfect_prediction():
    """When the correct class has a very high logit, loss should be near zero."""
    batch, vocab = 4, 10
    logits = np.zeros((batch, vocab, 8), dtype=np.float32)
    targets = np.array([3, 7, 0, 5], dtype=np.int64)

    # Set high scalar logit at the target position
    for i in range(batch):
        logits[i, targets[i], 0] = 100.0

    loss = clifford_cross_entropy(logits, targets)
    assert loss < 0.01, f"Expected near-zero loss for perfect prediction, got {loss}"
    print(f"[PASS] CE perfect prediction: loss={loss:.6f}")


def test_ce_uniform_prediction():
    """With uniform logits, loss should be log(vocab_size)."""
    batch, vocab = 8, 20
    logits = np.zeros((batch, vocab, 8), dtype=np.float32)
    targets = np.arange(batch, dtype=np.int64) % vocab

    loss = clifford_cross_entropy(logits, targets)
    expected = np.log(vocab)
    assert abs(loss - expected) < 0.01, (
        f"Uniform logits should give loss ~{expected:.4f}, got {loss:.4f}"
    )
    print(f"[PASS] CE uniform: loss={loss:.4f} (expected ~{expected:.4f})")


def test_ce_gradient_shape():
    """Gradient should have the same shape as input logits."""
    batch, vocab = 4, 10
    logits = np.random.randn(batch, vocab, 8).astype(np.float32)
    targets = np.random.randint(0, vocab, size=batch).astype(np.int64)

    loss, grad = clifford_cross_entropy_with_grad(logits, targets)
    assert grad.shape == logits.shape, f"Grad shape {grad.shape} != logits shape {logits.shape}"

    # Only grade-0 should have nonzero gradient
    assert np.allclose(grad[..., 1:], 0.0), "Non-scalar grades should have zero CE gradient"
    print(f"[PASS] CE gradient: shape={grad.shape}, loss={loss:.4f}")


# ---- Algebraic consistency tests ----

def test_unit_rotor_no_penalty():
    """A unit rotor (scalar=1) should have near-zero unitarity penalty."""
    rotor = np.zeros((4, 8), dtype=np.float32)
    rotor[:, 0] = 1.0  # identity rotor

    loss = algebraic_consistency_loss(rotor_params=[rotor])
    assert loss < 0.01, f"Unit rotor penalty should be ~0, got {loss}"
    print(f"[PASS] Unit rotor consistency: loss={loss:.6f}")


def test_non_unit_rotor_has_penalty():
    """A rotor with norm != 1 should have nonzero penalty."""
    rotor = np.zeros((4, 8), dtype=np.float32)
    rotor[:, 0] = 2.0  # scalar = 2, not unit

    loss = algebraic_consistency_loss(rotor_params=[rotor])
    assert loss > 0.1, f"Non-unit rotor should have significant penalty, got {loss}"
    print(f"[PASS] Non-unit rotor penalty: loss={loss:.4f}")


def test_grade_ratio_stability():
    """Identical layer representations should have zero ratio penalty."""
    rep = np.random.randn(8, 16, 8).astype(np.float32)
    loss = algebraic_consistency_loss(layer_representations=[rep, rep, rep])
    assert loss < 1e-6, f"Identical layers should have ~0 ratio penalty, got {loss}"
    print(f"[PASS] Grade ratio stability (identical): loss={loss:.8f}")


def test_grade_ratio_shift_detected():
    """Different layer representations should produce nonzero ratio penalty."""
    rep1 = np.zeros((4, 8), dtype=np.float32)
    rep1[:, 0] = 1.0  # only scalar
    rep2 = np.zeros((4, 8), dtype=np.float32)
    rep2[:, 1:4] = 1.0  # only vector

    loss = algebraic_consistency_loss(
        layer_representations=[rep1, rep2],
        grade_ratio_weight=1.0,
    )
    assert loss > 0.1, f"Grade shift should be detected, got {loss}"
    print(f"[PASS] Grade ratio shift: loss={loss:.4f}")


# ---- Grade entropy tests ----

def test_max_entropy():
    """Equal energy in all grades -> entropy = log(4)."""
    # grade_energy uses np.mean over each grade's components, so:
    # grade 0: mean of 1 component  -> val^2
    # grade 1: mean of 3 components -> val^2
    # grade 2: mean of 3 components -> val^2
    # grade 3: mean of 1 component  -> val^2
    # To get equal mean energies, set all components to the same magnitude.
    data = np.ones(8, dtype=np.float32)

    # Check energies are equal
    e = grade_energy(data)
    assert np.allclose(e, e[0], atol=1e-5), f"Energies not equal: {e}"

    H = grade_entropy(data)
    max_H = np.log(4.0)
    assert abs(H - max_H) < 0.01, f"Expected entropy ~{max_H:.4f}, got {H:.4f}"

    penalty = grade_entropy_penalty(data)
    assert penalty < 0.01, f"Max-entropy should give ~0 penalty, got {penalty}"
    print(f"[PASS] Max entropy: H={H:.4f}, penalty={penalty:.6f}")


def test_collapsed_entropy():
    """All energy in one grade -> entropy ~ 0 -> large penalty."""
    data = np.zeros(8, dtype=np.float32)
    data[0] = 5.0  # only scalar

    H = grade_entropy(data)
    assert H < 0.01, f"Collapsed should have ~0 entropy, got {H}"

    penalty = grade_entropy_penalty(data)
    max_H = np.log(4.0)
    assert penalty > max_H - 0.1, f"Collapsed should give ~{max_H:.2f} penalty, got {penalty}"
    print(f"[PASS] Collapsed entropy: H={H:.6f}, penalty={penalty:.4f}")


def test_entropy_with_multivector():
    """grade_entropy_penalty should accept Multivector objects."""
    mv = Multivector(np.random.randn(8).astype(np.float32))
    penalty = grade_entropy_penalty(mv)
    assert isinstance(penalty, float), f"Expected float, got {type(penalty)}"
    assert penalty >= 0.0, f"Penalty should be non-negative, got {penalty}"
    print(f"[PASS] Entropy with Multivector: penalty={penalty:.4f}")


if __name__ == "__main__":
    test_ce_perfect_prediction()
    test_ce_uniform_prediction()
    test_ce_gradient_shape()
    test_unit_rotor_no_penalty()
    test_non_unit_rotor_has_penalty()
    test_grade_ratio_stability()
    test_grade_ratio_shift_detected()
    test_max_entropy()
    test_collapsed_entropy()
    test_entropy_with_multivector()
    print("\nAll loss tests passed.")
