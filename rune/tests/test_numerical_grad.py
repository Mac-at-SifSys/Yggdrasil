"""
test_numerical_grad.py — Compare analytical Clifford gradients vs numerical

This is the critical verification: every analytical derivative rule must match
the numerical (finite-difference) gradient to within tolerance.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rune.types.multivector import Multivector
from rune.autodiff.clifford_rules import CliffordDerivativeRules


EPS = 1e-4
TOL = 2e-3


def numerical_jacobian(fn, x_data, output_size=8):
    """Compute full Jacobian df/dx via finite differences."""
    jac = np.zeros((output_size, 8), dtype=np.float32)
    for i in range(8):
        xp = x_data.copy(); xp[i] += EPS
        xm = x_data.copy(); xm[i] -= EPS
        jac[:, i] = (fn(xp) - fn(xm)) / (2 * EPS)
    return jac


def test_geom_prod_full_jacobian():
    """Full 8x8 Jacobian of geometric product w.r.t. A."""
    np.random.seed(100)
    a_data = np.random.randn(8).astype(np.float32) * 0.5
    b_data = np.random.randn(8).astype(np.float32) * 0.5

    def fn(a):
        mv_a = Multivector(a)
        mv_b = Multivector(b_data)
        return (mv_a * mv_b).data

    num_jac = numerical_jacobian(fn, a_data)

    # Analytical: for each output component c, compute dL/dA with dL/dC = e_c
    anal_jac = np.zeros((8, 8), dtype=np.float32)
    for c in range(8):
        grad_output = np.zeros(8, dtype=np.float32)
        grad_output[c] = 1.0
        grad_a, _ = CliffordDerivativeRules.geometric_product_backward(
            grad_output, a_data, b_data
        )
        anal_jac[c, :] = grad_a

    assert np.allclose(anal_jac, num_jac, atol=TOL), \
        f"Max diff: {np.max(np.abs(anal_jac - num_jac))}"


def test_geom_prod_full_jacobian_b():
    """Full 8x8 Jacobian of geometric product w.r.t. B."""
    np.random.seed(101)
    a_data = np.random.randn(8).astype(np.float32) * 0.5
    b_data = np.random.randn(8).astype(np.float32) * 0.5

    def fn(b):
        mv_a = Multivector(a_data)
        mv_b = Multivector(b)
        return (mv_a * mv_b).data

    num_jac = numerical_jacobian(fn, b_data)

    anal_jac = np.zeros((8, 8), dtype=np.float32)
    for c in range(8):
        grad_output = np.zeros(8, dtype=np.float32)
        grad_output[c] = 1.0
        _, grad_b = CliffordDerivativeRules.geometric_product_backward(
            grad_output, a_data, b_data
        )
        anal_jac[c, :] = grad_b

    assert np.allclose(anal_jac, num_jac, atol=TOL), \
        f"Max diff: {np.max(np.abs(anal_jac - num_jac))}"


def test_reverse_jacobian():
    """Jacobian of reverse is diagonal with REVERSE_SIGN."""
    np.random.seed(102)
    x_data = np.random.randn(8).astype(np.float32)

    def fn(x):
        return Multivector(x).reverse().data

    num_jac = numerical_jacobian(fn, x_data)

    # Analytical: reverse is linear, Jacobian is diagonal
    from rune.types.multivector import REVERSE_SIGN
    anal_jac = np.diag(REVERSE_SIGN)

    assert np.allclose(anal_jac, num_jac, atol=TOL)


def test_norm_squared_gradient():
    """Gradient of norm_squared = <x * ~x>_0."""
    np.random.seed(103)
    x_data = np.random.randn(8).astype(np.float32) * 0.5

    def fn(x):
        mv = Multivector(x)
        return np.array([mv.norm_squared()])

    num_grad = np.zeros(8, dtype=np.float32)
    for i in range(8):
        xp = x_data.copy(); xp[i] += EPS
        xm = x_data.copy(); xm[i] -= EPS
        num_grad[i] = (fn(xp)[0] - fn(xm)[0]) / (2 * EPS)

    anal_grad = CliffordDerivativeRules.norm_squared_backward(
        np.array([1.0], dtype=np.float32), x_data
    )

    assert np.allclose(anal_grad, num_grad, atol=TOL), \
        f"Analytical: {anal_grad}\nNumerical: {num_grad}"


def test_bivector_exp_jacobian():
    """Jacobian of bivector_exp w.r.t. bivector components."""
    np.random.seed(104)
    bv_data = np.zeros(8, dtype=np.float32)
    bv_data[4:7] = np.random.randn(3).astype(np.float32) * 0.5

    def fn(bv):
        return Multivector.bivector_exp(Multivector(bv)).data

    num_jac = numerical_jacobian(fn, bv_data)

    # Only check bivector-to-output Jacobian (other inputs should be zero)
    anal_jac = np.zeros((8, 8), dtype=np.float32)
    for c in range(8):
        grad_output = np.zeros(8, dtype=np.float32)
        grad_output[c] = 1.0
        grad_bv = CliffordDerivativeRules.bivector_exp_backward(grad_output, bv_data)
        anal_jac[c, :] = grad_bv

    # Only bivector inputs (4,5,6) should have nonzero Jacobian columns
    for col in [4, 5, 6]:
        assert np.allclose(anal_jac[:, col], num_jac[:, col], atol=TOL), \
            f"Col {col}: anal={anal_jac[:, col]}, num={num_jac[:, col]}"


def test_sandwich_gradient_x():
    """Gradient of sandwich product R*X*~R w.r.t. X."""
    np.random.seed(105)
    # Use a unit rotor
    bv = np.zeros(8, dtype=np.float32)
    bv[4:7] = np.random.randn(3).astype(np.float32) * 0.3
    r_data = Multivector.bivector_exp(Multivector(bv)).data
    x_data = np.random.randn(8).astype(np.float32) * 0.5

    def fn(x):
        r = Multivector(r_data)
        mx = Multivector(x)
        return r.sandwich(mx).data

    num_jac = numerical_jacobian(fn, x_data)

    # Check a few output component gradients
    for c in [0, 1, 4, 7]:
        grad_output = np.zeros(8, dtype=np.float32)
        grad_output[c] = 1.0
        _, grad_x = CliffordDerivativeRules.sandwich_backward(
            grad_output, r_data, x_data
        )
        assert np.allclose(grad_x, num_jac[c, :], atol=TOL), \
            f"Output {c}: anal={grad_x}, num={num_jac[c, :]}"


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
    print("Done.")
