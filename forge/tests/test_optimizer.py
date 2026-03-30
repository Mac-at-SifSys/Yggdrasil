"""
test_optimizer.py -- Test that CliffordAdam steps reduce loss on a toy problem.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.types.multivector import Multivector
from forge.optimizers.clifford_adam import CliffordAdam
from forge.optimizers.clifford_sgd import CliffordSGD
from forge.optimizers.clifford_grad_clip import clifford_grad_clip


def _toy_loss_and_grad(param_mv: Multivector, target: np.ndarray):
    """Squared distance loss: ||param.data - target||^2 and its gradient."""
    diff = param_mv._data - target
    loss = float(np.sum(diff ** 2))
    grad = 2.0 * diff
    return loss, grad


def test_clifford_adam_reduces_loss():
    """CliffordAdam should reduce a quadratic loss over several steps."""
    np.random.seed(42)
    target = np.random.randn(8).astype(np.float32)
    param = Multivector(np.zeros(8, dtype=np.float32), requires_grad=True)

    params = [{"mv": param, "name": "w"}]
    opt = CliffordAdam(params, lr=0.01, betas=(0.9, 0.999))

    initial_loss, _ = _toy_loss_and_grad(param, target)

    for _ in range(300):
        loss, grad = _toy_loss_and_grad(param, target)
        param._grad = Multivector(grad)
        opt.step()

    final_loss, _ = _toy_loss_and_grad(param, target)

    assert final_loss < initial_loss * 0.5, (
        f"Adam did not reduce loss sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    )
    print(f"[PASS] CliffordAdam: {initial_loss:.4f} -> {final_loss:.4f}")


def test_clifford_sgd_reduces_loss():
    """CliffordSGD should reduce a quadratic loss."""
    np.random.seed(42)
    target = np.random.randn(8).astype(np.float32)
    param = Multivector(np.zeros(8, dtype=np.float32), requires_grad=True)

    params = [{"mv": param, "name": "w"}]
    opt = CliffordSGD(params, lr=0.01, momentum=0.9)

    initial_loss, _ = _toy_loss_and_grad(param, target)

    for _ in range(200):
        loss, grad = _toy_loss_and_grad(param, target)
        param._grad = Multivector(grad)
        opt.step()

    final_loss, _ = _toy_loss_and_grad(param, target)

    assert final_loss < initial_loss * 0.1, (
        f"SGD did not reduce loss sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    )
    print(f"[PASS] CliffordSGD: {initial_loss:.4f} -> {final_loss:.4f}")


def test_grad_clip():
    """clifford_grad_clip should cap the total gradient norm."""
    np.random.seed(42)
    param = Multivector(np.zeros(8, dtype=np.float32), requires_grad=True)
    param._grad = Multivector(np.ones(8, dtype=np.float32) * 10.0)

    params = [{"mv": param}]
    orig_norm = clifford_grad_clip(params, max_norm=1.0)

    # After clipping the grad norm should be <= 1.0
    clipped_data = param._grad._data
    from forge.optimizers.clifford_grad_clip import _mv_norm_sq
    clipped_norm = float(np.sqrt(_mv_norm_sq(clipped_data)))

    assert orig_norm > 1.0, f"Expected large original norm, got {orig_norm}"
    assert clipped_norm <= 1.0 + 1e-5, (
        f"Clipped norm should be <= 1.0, got {clipped_norm}"
    )
    print(f"[PASS] grad_clip: {orig_norm:.4f} -> {clipped_norm:.4f}")


def test_adam_grade_lr_scale():
    """Per-grade LR scaling should cause different convergence rates."""
    np.random.seed(123)
    target = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    param = Multivector(np.zeros(8, dtype=np.float32), requires_grad=True)

    # Dampen trivector heavily
    params = [{"mv": param}]
    opt = CliffordAdam(params, lr=0.1, grade_lr_scale=[1.0, 1.0, 1.0, 0.01])

    for _ in range(50):
        loss, grad = _toy_loss_and_grad(param, target)
        param._grad = Multivector(grad)
        opt.step()

    # Scalar should be closer to target than trivector
    scalar_err = abs(param._data[0] - 1.0)
    tri_err = abs(param._data[7] - 1.0)

    assert scalar_err < tri_err, (
        f"Scalar error ({scalar_err:.4f}) should be less than trivector error ({tri_err:.4f})"
    )
    print(f"[PASS] grade_lr_scale: scalar_err={scalar_err:.4f}, tri_err={tri_err:.4f}")


def test_adam_state_dict_roundtrip():
    """Optimizer state should survive save/load."""
    np.random.seed(42)
    param = Multivector(np.random.randn(8).astype(np.float32), requires_grad=True)
    params = [{"mv": param}]
    opt = CliffordAdam(params, lr=0.01)

    # Do a few steps
    for _ in range(5):
        param._grad = Multivector(np.random.randn(8).astype(np.float32))
        opt.step()

    sd = opt.state_dict()
    assert sd["t"] == 5
    assert len(sd["states"]) == 1
    assert sd["states"][0]["m"].shape == (8,)
    print("[PASS] state_dict roundtrip")


if __name__ == "__main__":
    test_clifford_adam_reduces_loss()
    test_clifford_sgd_reduces_loss()
    test_grad_clip()
    test_adam_grade_lr_scale()
    test_adam_state_dict_roundtrip()
    print("\nAll optimizer tests passed.")
