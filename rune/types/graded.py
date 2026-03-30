"""
graded.py — Grade-specific subtypes for Cl(3,0)

These are compile-time grade annotations. Each subtype constrains which
grades are active, enabling grade-aware optimization (dead-grade elimination,
reduced memory allocation, etc.).
"""

import numpy as np
from rune.types.multivector import Multivector


class Scalar(Multivector):
    """Grade-0 only: 1 component."""

    def __init__(self, val: float = 0.0, requires_grad=False):
        data = np.zeros(8, dtype=np.float32)
        data[0] = val
        super().__init__(data, requires_grad=requires_grad, grade_mask=0x01)

    @staticmethod
    def from_multivector(mv: Multivector) -> 'Scalar':
        s = Scalar(0.0)
        s._data = mv.grade(0)._data
        return s


class Vector(Multivector):
    """Grade-1 only: 3 components (e1, e2, e3)."""

    def __init__(self, e1: float = 0, e2: float = 0, e3: float = 0,
                 data=None, requires_grad=False):
        if data is not None:
            super().__init__(data, requires_grad=requires_grad, grade_mask=0x02)
        else:
            d = np.zeros(8, dtype=np.float32)
            d[1] = e1; d[2] = e2; d[3] = e3
            super().__init__(d, requires_grad=requires_grad, grade_mask=0x02)

    @staticmethod
    def from_multivector(mv: Multivector) -> 'Vector':
        v = Vector()
        v._data = mv.grade(1)._data
        return v


class Bivector(Multivector):
    """Grade-2 only: 3 components (e12, e13, e23)."""

    def __init__(self, e12: float = 0, e13: float = 0, e23: float = 0,
                 data=None, requires_grad=False):
        if data is not None:
            super().__init__(data, requires_grad=requires_grad, grade_mask=0x04)
        else:
            d = np.zeros(8, dtype=np.float32)
            d[4] = e12; d[5] = e13; d[6] = e23
            super().__init__(d, requires_grad=requires_grad, grade_mask=0x04)

    @staticmethod
    def from_multivector(mv: Multivector) -> 'Bivector':
        b = Bivector()
        b._data = mv.grade(2)._data
        return b

    def exp(self) -> 'Even':
        """Exponentiate this bivector to get a rotor."""
        return Even.from_multivector(Multivector.bivector_exp(self))


class Trivector(Multivector):
    """Grade-3 only: 1 component (e123 / pseudoscalar)."""

    def __init__(self, val: float = 0.0, requires_grad=False):
        data = np.zeros(8, dtype=np.float32)
        data[7] = val
        super().__init__(data, requires_grad=requires_grad, grade_mask=0x08)


class Even(Multivector):
    """Even sub-algebra: Grade 0 + Grade 2 (rotor/spinor space). 4 components."""

    def __init__(self, s: float = 0, e12: float = 0, e13: float = 0, e23: float = 0,
                 data=None, requires_grad=False):
        if data is not None:
            super().__init__(data, requires_grad=requires_grad, grade_mask=0x05)
        else:
            d = np.zeros(8, dtype=np.float32)
            d[0] = s; d[4] = e12; d[5] = e13; d[6] = e23
            super().__init__(d, requires_grad=requires_grad, grade_mask=0x05)

    @staticmethod
    def from_multivector(mv: Multivector) -> 'Even':
        e = Even()
        e._data = mv.even()._data
        return e

    @staticmethod
    def identity() -> 'Even':
        """Identity rotor (scalar 1)."""
        return Even(s=1.0)

    @staticmethod
    def from_bivector_angle(bv: Bivector) -> 'Even':
        """Create rotor from bivector via exponential."""
        return Even.from_multivector(Multivector.bivector_exp(bv))


class Odd(Multivector):
    """Odd sub-algebra: Grade 1 + Grade 3. 4 components."""

    def __init__(self, e1: float = 0, e2: float = 0, e3: float = 0,
                 e123: float = 0, data=None, requires_grad=False):
        if data is not None:
            super().__init__(data, requires_grad=requires_grad, grade_mask=0x0A)
        else:
            d = np.zeros(8, dtype=np.float32)
            d[1] = e1; d[2] = e2; d[3] = e3; d[7] = e123
            super().__init__(d, requires_grad=requires_grad, grade_mask=0x0A)

    @staticmethod
    def from_multivector(mv: Multivector) -> 'Odd':
        o = Odd()
        o._data = mv.odd()._data
        return o
