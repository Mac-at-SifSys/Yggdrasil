"""Tests for Clifford-native autodiff."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rune.types.multivector import Multivector
from rune.autodiff.clifford_rules import CliffordDerivativeRules


def numerical_grad(fn, x_data, eps=1e-4):
    """Numerical gradient via finite differences on all 8 components."""
    grad = np.zeros_like(x_data)
    for i in range(8):
        x_plus = x_data.copy()
        x_plus[i] += eps
        x_minus = x_data.copy()
        x_minus[i] -= eps
        grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad


def test_geom_prod_grad_a():
    """d(scalar_part(A*B)) / dA via analytical vs numerical."""
    np.random.seed(42)
    a_data = np.random.randn(8).astype(np.float32)
    b_data = np.random.randn(8).astype(np.float32)

    # Forward: loss = scalar_part(A * B) = (A*B)[0]
    def loss_fn(a):
        mv_a = Multivector(a)
        mv_b = Multivector(b_data)
        prod = mv_a * mv_b
        return prod.data[0]  # scalar part

    # Numerical gradient
    num_grad = numerical_grad(loss_fn, a_data)

    # Analytical gradient: d(scalar(A*B))/dA
    # grad_output is [1, 0, 0, 0, 0, 0, 0, 0] (only scalar part matters)
    grad_output = np.zeros(8, dtype=np.float32)
    grad_output[0] = 1.0

    grad_a, grad_b = CliffordDerivativeRules.geometric_product_backward(
        grad_output, a_data, b_data
    )

    assert np.allclose(grad_a, num_grad, atol=2e-3), \
        f"Analytical grad: {grad_a}\nNumerical grad: {num_grad}"


def test_geom_prod_grad_b():
    """d(scalar_part(A*B)) / dB via analytical vs numerical."""
    np.random.seed(43)
    a_data = np.random.randn(8).astype(np.float32)
    b_data = np.random.randn(8).astype(np.float32)

    def loss_fn(b):
        mv_a = Multivector(a_data)
        mv_b = Multivector(b)
        prod = mv_a * mv_b
        return prod.data[0]

    num_grad = numerical_grad(loss_fn, b_data)

    grad_output = np.zeros(8, dtype=np.float32)
    grad_output[0] = 1.0
    _, grad_b = CliffordDerivativeRules.geometric_product_backward(
        grad_output, a_data, b_data
    )

    assert np.allclose(grad_b, num_grad, atol=2e-3), \
        f"Analytical grad: {grad_b}\nNumerical grad: {num_grad}"


def test_reverse_grad():
    """Gradient through reverse operation."""
    np.random.seed(44)
    x_data = np.random.randn(8).astype(np.float32)

    def loss_fn(x):
        mv = Multivector(x)
        rev = ~mv
        return rev.data[0]  # scalar part of reverse

    num_grad = numerical_grad(loss_fn, x_data)

    grad_output = np.zeros(8, dtype=np.float32)
    grad_output[0] = 1.0
    anal_grad = CliffordDerivativeRules.reverse_backward(grad_output)

    assert np.allclose(anal_grad, num_grad, atol=2e-3)


def test_grade_project_grad():
    """Gradient through grade projection."""
    np.random.seed(45)
    x_data = np.random.randn(8).astype(np.float32)

    for grade in range(4):
        def loss_fn(x, g=grade):
            mv = Multivector(x)
            proj = mv.grade(g)
            return np.sum(proj.data)

        num_grad = np.zeros(8, dtype=np.float32)
        for i in range(8):
            xp = x_data.copy(); xp[i] += 1e-4
            xm = x_data.copy(); xm[i] -= 1e-4
            num_grad[i] = (loss_fn(xp) - loss_fn(xm)) / 2e-4

        grad_output = np.ones(8, dtype=np.float32)
        anal_grad = CliffordDerivativeRules.grade_project_backward(grad_output, grade)

        assert np.allclose(anal_grad, num_grad, atol=2e-3), \
            f"Grade {grade}: analytical={anal_grad}, numerical={num_grad}"


def test_bivector_exp_grad():
    """Gradient through bivector exponential."""
    np.random.seed(46)
    bv_data = np.zeros(8, dtype=np.float32)
    bv_data[4:7] = np.random.randn(3).astype(np.float32) * 0.5

    # Loss = sum of all components of exp(B)
    def loss_fn(bv):
        mv = Multivector(bv)
        r = Multivector.bivector_exp(mv)
        return np.sum(r.data)

    num_grad = np.zeros(8, dtype=np.float32)
    for i in range(8):
        xp = bv_data.copy(); xp[i] += 1e-4
        xm = bv_data.copy(); xm[i] -= 1e-4
        num_grad[i] = (loss_fn(xp) - loss_fn(xm)) / 2e-4

    grad_output = np.ones(8, dtype=np.float32)
    anal_grad = CliffordDerivativeRules.bivector_exp_backward(grad_output, bv_data)

    # Only bivector components should have gradient
    assert np.allclose(anal_grad[4:7], num_grad[4:7], atol=2e-3), \
        f"Analytical: {anal_grad[4:7]}\nNumerical: {num_grad[4:7]}"


def test_full_chain_gradient():
    """Test gradient through a chain: grade_0(A * B * ~A)."""
    np.random.seed(47)
    a_data = np.random.randn(8).astype(np.float32) * 0.5
    b_data = np.random.randn(8).astype(np.float32) * 0.5

    def loss_fn(a):
        mv_a = Multivector(a)
        mv_b = Multivector(b_data)
        sandwich = mv_a * mv_b * (~mv_a)
        return sandwich.data[0]  # scalar part

    num_grad = numerical_grad(loss_fn, a_data, eps=1e-4)

    # This tests that the chain rule works through multiple ops
    # Not testing analytical here, just verifying numerical is sensible
    assert not np.allclose(num_grad, 0, atol=1e-4), \
        "Gradient should be nonzero for generic multivectors"


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
    print("Done.")
