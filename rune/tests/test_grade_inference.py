"""Tests for grade inference engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rune.types.type_checking import (
    GradeSignature, SCALAR, VECTOR, BIVECTOR, TRIVECTOR, EVEN, ODD, FULL,
    infer_geometric_product, infer_outer_product, infer_inner_product,
    infer_reverse, infer_grade_projection, infer_addition, infer_sandwich,
    infer_bivector_exp, GradeChecker,
)


def test_scalar_times_anything():
    """Scalar * X preserves X's grade."""
    assert infer_geometric_product(SCALAR, VECTOR) == GradeSignature({1})
    assert infer_geometric_product(SCALAR, BIVECTOR) == GradeSignature({2})
    assert infer_geometric_product(SCALAR, FULL) == FULL

def test_vector_times_vector():
    """Vector * Vector produces scalar + bivector."""
    result = infer_geometric_product(VECTOR, VECTOR)
    assert result.active_grades == {0, 2}

def test_bivector_times_bivector():
    """Bivector * Bivector produces scalar + bivector."""
    result = infer_geometric_product(BIVECTOR, BIVECTOR)
    assert 0 in result.active_grades
    assert 2 in result.active_grades

def test_outer_product_vectors():
    """Outer product of two vectors is a bivector."""
    result = infer_outer_product(VECTOR, VECTOR)
    assert result == BIVECTOR

def test_outer_product_overflow():
    """Outer product that would exceed grade 3 produces nothing for those."""
    result = infer_outer_product(BIVECTOR, BIVECTOR)
    # grade 2 + grade 2 = grade 4, which exceeds max
    assert len(result.active_grades) == 0 or max(result.active_grades) <= 3

def test_inner_product_vector_bivector():
    """Left contraction of vector into bivector gives vector."""
    result = infer_inner_product(VECTOR, BIVECTOR)
    assert result == VECTOR

def test_reverse_preserves_grade():
    assert infer_reverse(EVEN) == EVEN
    assert infer_reverse(FULL) == FULL

def test_grade_projection():
    assert infer_grade_projection(FULL, 2) == BIVECTOR
    assert infer_grade_projection(EVEN, 1).active_grades == set()

def test_addition_merges():
    assert infer_addition(SCALAR, BIVECTOR) == EVEN
    assert infer_addition(VECTOR, TRIVECTOR) == ODD

def test_sandwich_even():
    """Even rotor sandwich preserves grade."""
    result = infer_sandwich(EVEN, VECTOR)
    assert result == VECTOR

def test_bivector_exp():
    assert infer_bivector_exp(BIVECTOR) == EVEN

def test_component_count():
    assert SCALAR.component_count == 1
    assert VECTOR.component_count == 3
    assert BIVECTOR.component_count == 3
    assert TRIVECTOR.component_count == 1
    assert EVEN.component_count == 4
    assert ODD.component_count == 4
    assert FULL.component_count == 8

def test_grade_checker():
    checker = GradeChecker()
    checker.declare('x', FULL)
    checker.declare('w', EVEN)
    checker.declare('b', SCALAR)
    report = checker.optimization_report()
    assert report['variables']['w']['savings'] == 4
    assert report['variables']['b']['savings'] == 7


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
    print("Done.")
