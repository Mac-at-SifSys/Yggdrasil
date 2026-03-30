"""Tests for Rune type system."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rune.types.multivector import Multivector
from rune.types.graded import Scalar, Vector, Bivector, Trivector, Even, Odd


def test_scalar_creation():
    s = Scalar(3.0)
    assert abs(s.data[0] - 3.0) < 1e-6
    assert np.allclose(s.data[1:], 0)

def test_vector_creation():
    v = Vector(1.0, 2.0, 3.0)
    assert np.allclose(v.data[1:4], [1, 2, 3])
    assert abs(v.data[0]) < 1e-6

def test_bivector_creation():
    b = Bivector(1.0, -1.0, 0.5)
    assert np.allclose(b.data[4:7], [1, -1, 0.5])

def test_even_creation():
    e = Even(s=1.0, e12=0.5, e13=-0.3, e23=0.1)
    assert abs(e.data[0] - 1.0) < 1e-6
    assert abs(e.data[4] - 0.5) < 1e-6
    assert np.allclose(e.data[1:4], 0)  # no vector
    assert abs(e.data[7]) < 1e-6  # no trivector

def test_grade_projection():
    m = Multivector([1, 2, 3, 4, 5, 6, 7, 8])
    g0 = m.grade(0)
    assert abs(g0.data[0] - 1.0) < 1e-6
    assert np.allclose(g0.data[1:], 0)

    g1 = m.grade(1)
    assert np.allclose(g1.data[1:4], [2, 3, 4])
    assert abs(g1.data[0]) < 1e-6

    g2 = m.grade(2)
    assert np.allclose(g2.data[4:7], [5, 6, 7])

    g3 = m.grade(3)
    assert abs(g3.data[7] - 8.0) < 1e-6

def test_grade_completeness():
    m = Multivector([1, 2, 3, 4, 5, 6, 7, 8])
    total = m.grade(0) + m.grade(1) + m.grade(2) + m.grade(3)
    assert np.allclose(total.data, m.data, atol=1e-6)

def test_even_odd_decomposition():
    m = Multivector([1, 2, 3, 4, 5, 6, 7, 8])
    e = m.even()
    o = m.odd()
    recon = e + o
    assert np.allclose(recon.data, m.data, atol=1e-6)

def test_random_multivector():
    m = Multivector.random()
    assert m.data.shape == (8,)
    m_batch = Multivector.random((4, 3))
    assert m_batch.data.shape == (4, 3, 8)

def test_batch_operations():
    a = Multivector.random((10,))
    b = Multivector.random((10,))
    c = a + b
    assert c.data.shape == (10, 8)
    assert np.allclose(c.data, a.data + b.data)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
    print("Done.")
