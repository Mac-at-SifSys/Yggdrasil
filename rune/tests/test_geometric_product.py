"""Tests for geometric product in Rune DSL."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rune.types.multivector import Multivector
from rune.types.graded import Vector, Bivector, Even


def test_identity():
    one = Multivector.make_scalar(1.0)
    a = Multivector([2, 3, -1, 0.5, 1.5, -0.5, 2, -1])
    r = one * a
    assert np.allclose(r.data, a.data, atol=1e-5)

def test_scalar_mul():
    s = Multivector.make_scalar(3.0)
    a = Multivector([1, 2, 3, 4, 5, 6, 7, 8])
    r = s * a
    assert np.allclose(r.data, a.data * 3.0, atol=1e-5)

def test_vector_squares():
    """e_i^2 = +1 for Cl(3,0)."""
    for i in range(3):
        v = Multivector.make_vector(**{f'e{i+1}': 1.0})
        sq = v * v
        assert abs(sq.data[0] - 1.0) < 1e-5, f"e{i+1}^2 should be +1"
        assert np.allclose(sq.data[1:], 0, atol=1e-5)

def test_anticommutation():
    """e_i * e_j = -e_j * e_i for i != j."""
    e1 = Multivector.make_vector(e1=1)
    e2 = Multivector.make_vector(e2=1)
    ab = e1 * e2
    ba = e2 * e1
    assert np.allclose((ab + ba).data, 0, atol=1e-5)

def test_pseudoscalar_square():
    """e123^2 = -1."""
    ps = Multivector.make_pseudoscalar(1.0)
    sq = ps * ps
    assert abs(sq.data[0] - (-1.0)) < 1e-5

def test_associativity():
    np.random.seed(42)
    for _ in range(50):
        a = Multivector.random()
        b = Multivector.random()
        c = Multivector.random()
        ab_c = (a * b) * c
        a_bc = a * (b * c)
        assert np.allclose(ab_c.data, a_bc.data, atol=1e-4), "Associativity violated"

def test_distributivity():
    np.random.seed(123)
    for _ in range(50):
        a = Multivector.random()
        b = Multivector.random()
        c = Multivector.random()
        lhs = a * (b + c)
        rhs = (a * b) + (a * c)
        assert np.allclose(lhs.data, rhs.data, atol=1e-4), "Distributivity violated"

def test_reverse_antiautomorphism():
    """~(ab) = ~b * ~a"""
    np.random.seed(77)
    for _ in range(50):
        a = Multivector.random()
        b = Multivector.random()
        rev_ab = ~(a * b)
        rev_b_rev_a = (~b) * (~a)
        assert np.allclose(rev_ab.data, rev_b_rev_a.data, atol=1e-4)

def test_outer_product_grade():
    """Outer product of two vectors should be a bivector."""
    v1 = Multivector.make_vector(e1=1, e2=0, e3=0)
    v2 = Multivector.make_vector(e1=0, e2=1, e3=0)
    w = v1 ^ v2
    # Should be e12
    assert abs(w.data[4] - 1.0) < 1e-5
    assert np.allclose(w.data[:4], 0, atol=1e-5)
    assert np.allclose(w.data[5:], 0, atol=1e-5)

def test_batch_geometric_product():
    a = Multivector.random((5,))
    b = Multivector.random((5,))
    c = a * b
    assert c.data.shape == (5, 8)
    # Check first element matches single product
    a0 = Multivector(a.data[0])
    b0 = Multivector(b.data[0])
    c0 = a0 * b0
    assert np.allclose(c.data[0], c0.data, atol=1e-5)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
    print("Done.")
