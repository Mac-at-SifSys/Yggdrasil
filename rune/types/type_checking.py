"""
type_checking.py — Grade inference engine for the Rune type system

Performs compile-time (or trace-time) grade inference:
- Tracks which grades are active through operations
- Detects grade-violating operations at trace time
- Enables dead-grade elimination optimization
"""

from typing import Set, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GradeSignature:
    """Tracks which grades are active in a multivector expression."""
    active_grades: Set[int]  # subset of {0, 1, 2, 3}

    @property
    def mask(self) -> int:
        """Convert to bitmask."""
        m = 0
        for g in self.active_grades:
            m |= (1 << g)
        return m

    @property
    def component_count(self) -> int:
        """Number of scalar components needed."""
        sizes = {0: 1, 1: 3, 2: 3, 3: 1}
        return sum(sizes[g] for g in self.active_grades)

    @property
    def is_even(self) -> bool:
        return self.active_grades <= {0, 2}

    @property
    def is_odd(self) -> bool:
        return self.active_grades <= {1, 3}

    @property
    def is_scalar(self) -> bool:
        return self.active_grades == {0}

    @property
    def is_vector(self) -> bool:
        return self.active_grades == {1}

    @property
    def is_bivector(self) -> bool:
        return self.active_grades == {2}

    def __repr__(self):
        grade_names = {0: 'scalar', 1: 'vector', 2: 'bivector', 3: 'trivector'}
        names = [grade_names[g] for g in sorted(self.active_grades)]
        return f"GradeSignature({'+'.join(names)})"


# Pre-defined signatures
SCALAR = GradeSignature({0})
VECTOR = GradeSignature({1})
BIVECTOR = GradeSignature({2})
TRIVECTOR = GradeSignature({3})
EVEN = GradeSignature({0, 2})
ODD = GradeSignature({1, 3})
FULL = GradeSignature({0, 1, 2, 3})


def infer_geometric_product(a: GradeSignature, b: GradeSignature) -> GradeSignature:
    """
    Infer output grades of a geometric product.

    The geometric product of grade-r and grade-s can produce grades
    |r-s|, |r-s|+2, ..., r+s (stepping by 2) capped at 3.
    """
    result_grades = set()
    for ga in a.active_grades:
        for gb in b.active_grades:
            # Possible output grades from grade ga * grade gb
            min_grade = abs(ga - gb)
            max_grade = min(ga + gb, 3)
            for g in range(min_grade, max_grade + 1, 2):
                result_grades.add(g)
    return GradeSignature(result_grades)


def infer_outer_product(a: GradeSignature, b: GradeSignature) -> GradeSignature:
    """Infer output grades of outer product (grade goes up)."""
    result_grades = set()
    for ga in a.active_grades:
        for gb in b.active_grades:
            out_grade = ga + gb
            if out_grade <= 3:
                result_grades.add(out_grade)
    return GradeSignature(result_grades)


def infer_inner_product(a: GradeSignature, b: GradeSignature) -> GradeSignature:
    """Infer output grades of left contraction (grade goes down)."""
    result_grades = set()
    for ga in a.active_grades:
        for gb in b.active_grades:
            if ga <= gb:
                out_grade = gb - ga
                result_grades.add(out_grade)
    return GradeSignature(result_grades) if result_grades else GradeSignature({0})


def infer_reverse(a: GradeSignature) -> GradeSignature:
    """Reverse preserves grades (only changes signs)."""
    return GradeSignature(a.active_grades.copy())


def infer_grade_projection(a: GradeSignature, k: int) -> GradeSignature:
    """Grade projection always produces a single grade."""
    if k in a.active_grades:
        return GradeSignature({k})
    return GradeSignature(set())  # Empty if grade not present


def infer_addition(a: GradeSignature, b: GradeSignature) -> GradeSignature:
    """Addition merges active grades."""
    return GradeSignature(a.active_grades | b.active_grades)


def infer_sandwich(r: GradeSignature, x: GradeSignature) -> GradeSignature:
    """Sandwich product RxR†: preserves the grade of x when R is a versor."""
    if r.is_even:
        return GradeSignature(x.active_grades.copy())
    # General case: could mix grades
    temp = infer_geometric_product(r, x)
    return infer_geometric_product(temp, infer_reverse(r))


def infer_bivector_exp(bv: GradeSignature) -> GradeSignature:
    """exp(bivector) always produces even sub-algebra (scalar + bivector)."""
    return EVEN


class GradeChecker:
    """
    Static grade checker for Rune programs.

    Tracks grade signatures through operations and detects:
    - Dead grades (computed but never used)
    - Grade violations (type errors)
    - Optimization opportunities (sparse grade operations)
    """

    def __init__(self):
        self._variables: Dict[str, GradeSignature] = {}
        self._warnings = []
        self._errors = []

    def declare(self, name: str, sig: GradeSignature):
        self._variables[name] = sig

    def get(self, name: str) -> Optional[GradeSignature]:
        return self._variables.get(name)

    def check_assignment(self, target: GradeSignature,
                         source: GradeSignature) -> bool:
        """Check if source grades fit within target grades."""
        if not source.active_grades <= target.active_grades:
            extra = source.active_grades - target.active_grades
            self._errors.append(
                f"Grade violation: source has grades {extra} not in target {target}"
            )
            return False
        return True

    def optimization_report(self) -> Dict:
        """Report optimization opportunities."""
        report = {
            'variables': {},
            'savings': 0,
        }
        for name, sig in self._variables.items():
            full_size = 8  # full multivector
            actual_size = sig.component_count
            savings = full_size - actual_size
            if savings > 0:
                report['variables'][name] = {
                    'signature': str(sig),
                    'components': actual_size,
                    'savings': savings,
                }
                report['savings'] += savings
        return report

    @property
    def errors(self):
        return self._errors

    @property
    def warnings(self):
        return self._warnings
