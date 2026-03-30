"""
multivector.py — Core Multivector type for Cl(3,0)

Grade-stratified storage: the multivector is stored as 4 grade blocks,
not a flat 8-element array. Grade projection is a slice, not a gather.

Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
"""

import numpy as np
from typing import Optional, Union, Tuple

# Cl(3,0) multiplication table
# product_idx[i][j] = index of result basis element
# product_sign[i][j] = sign of the result
PRODUCT_IDX = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 4, 5, 2, 3, 7, 6],
    [2, 4, 0, 6, 1, 7, 3, 5],
    [3, 5, 6, 0, 7, 1, 2, 4],
    [4, 2, 1, 7, 0, 6, 5, 3],
    [5, 3, 7, 1, 6, 0, 4, 2],
    [6, 7, 3, 2, 5, 4, 0, 1],
    [7, 6, 5, 4, 3, 2, 1, 0],
], dtype=np.int32)

PRODUCT_SIGN = np.array([
    [+1, +1, +1, +1, +1, +1, +1, +1],
    [+1, +1, +1, +1, +1, +1, +1, +1],
    [+1, -1, +1, +1, -1, -1, +1, -1],
    [+1, -1, -1, +1, +1, -1, -1, +1],
    [+1, -1, +1, +1, -1, -1, +1, -1],
    [+1, -1, -1, +1, +1, -1, -1, +1],
    [+1, +1, -1, +1, -1, +1, -1, -1],
    [+1, +1, -1, +1, -1, +1, -1, -1],
], dtype=np.float32)

# Grade of each basis element
BASIS_GRADE = np.array([0, 1, 1, 1, 2, 2, 2, 3], dtype=np.int32)

# Grade slices in flat 8-component representation
GRADE_SLICES = {
    0: slice(0, 1),    # scalar
    1: slice(1, 4),    # vector (e1, e2, e3)
    2: slice(4, 7),    # bivector (e12, e13, e23)
    3: slice(7, 8),    # trivector (e123)
}

GRADE_SIZES = {0: 1, 1: 3, 2: 3, 3: 1}

# Reverse sign: (-1)^(k*(k-1)/2) for each basis element
REVERSE_SIGN = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32)

# Involution sign: (-1)^k for each basis element
INVOLUTION_SIGN = np.array([1, -1, -1, -1, 1, 1, 1, -1], dtype=np.float32)

# Conjugate sign = reverse * involution
CONJUGATE_SIGN = REVERSE_SIGN * INVOLUTION_SIGN


class Multivector:
    """
    Cl(3,0) multivector with grade-stratified storage.

    Supports batched operations: shape is (*batch_dims, 8) where the last
    dimension holds the 8 components in basis order.

    All arithmetic uses the geometric product as the default multiplication.
    """

    __slots__ = ('_data', '_requires_grad', '_grad', '_grad_fn', '_grade_mask')

    def __init__(self, data: Union[np.ndarray, list, float] = None,
                 requires_grad: bool = False,
                 grade_mask: int = 0x0F):
        """
        Args:
            data: Components in basis order {1, e1, e2, e3, e12, e13, e23, e123}.
                  Shape: (..., 8)
            requires_grad: Track gradients for autodiff
            grade_mask: Which grades are active (bitmask)
        """
        if data is None:
            self._data = np.zeros(8, dtype=np.float32)
        elif isinstance(data, (int, float)):
            self._data = np.zeros(8, dtype=np.float32)
            self._data[..., 0] = data
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)

        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None
        self._grade_mask = grade_mask

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> Tuple:
        return self._data.shape[:-1] if self._data.ndim > 1 else ()

    @property
    def batch_shape(self) -> Tuple:
        return self._data.shape[:-1]

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> Optional['Multivector']:
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    # --- Grade accessors (zero-cost: just slices) ---

    @property
    def scalar_part(self) -> np.ndarray:
        """Grade-0 (scalar) component."""
        return self._data[..., 0:1]

    @property
    def vector_part(self) -> np.ndarray:
        """Grade-1 (vector) components: e1, e2, e3."""
        return self._data[..., 1:4]

    @property
    def bivector_part(self) -> np.ndarray:
        """Grade-2 (bivector) components: e12, e13, e23."""
        return self._data[..., 4:7]

    @property
    def trivector_part(self) -> np.ndarray:
        """Grade-3 (trivector/pseudoscalar) component: e123."""
        return self._data[..., 7:8]

    def grade(self, k: int) -> 'Multivector':
        """Extract grade-k projection. Returns a new multivector with only grade-k nonzero."""
        result = np.zeros_like(self._data)
        slc = GRADE_SLICES[k]
        result[..., slc] = self._data[..., slc]
        mv = Multivector(result, requires_grad=self._requires_grad,
                         grade_mask=(1 << k))
        if self._requires_grad:
            mv._grad_fn = ('grade_project', self, k)
        return mv

    def even(self) -> 'Multivector':
        """Even sub-algebra: grade 0 + grade 2 (rotor space)."""
        result = np.zeros_like(self._data)
        result[..., 0:1] = self._data[..., 0:1]
        result[..., 4:7] = self._data[..., 4:7]
        return Multivector(result, grade_mask=0x05)

    def odd(self) -> 'Multivector':
        """Odd sub-algebra: grade 1 + grade 3."""
        result = np.zeros_like(self._data)
        result[..., 1:4] = self._data[..., 1:4]
        result[..., 7:8] = self._data[..., 7:8]
        return Multivector(result, grade_mask=0x0A)

    # --- Fundamental operations ---

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """Geometric product: self * other using the Cl(3,0) multiplication table.

        Uses native mjolnir kernels when available for single-element ops,
        falls back to NumPy for batched operations.
        """
        a = self._data
        b = other._data

        # Try native path for single multivectors (non-batched)
        if a.ndim == 1 and b.ndim == 1:
            try:
                from rune.bindings.mjolnir_ffi import is_native_available, native_geometric_product
                if is_native_available():
                    result = native_geometric_product(a, b)
                    mv = Multivector(result, requires_grad=self._requires_grad or other._requires_grad)
                    if mv._requires_grad:
                        mv._grad_fn = ('geometric_product', self, other)
                    return mv
            except (ImportError, RuntimeError):
                pass

        # NumPy fallback (handles batched dimensions)
        result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                idx = PRODUCT_IDX[i, j]
                sign = PRODUCT_SIGN[i, j]
                result[..., idx] += sign * a[..., i] * b[..., j]

        mv = Multivector(result, requires_grad=self._requires_grad or other._requires_grad)
        if mv._requires_grad:
            mv._grad_fn = ('geometric_product', self, other)
        return mv

    def outer_product(self, other: 'Multivector') -> 'Multivector':
        """Outer (wedge) product: grade-raising part of geometric product."""
        a = self._data
        b = other._data
        result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                idx = PRODUCT_IDX[i, j]
                if BASIS_GRADE[idx] == BASIS_GRADE[i] + BASIS_GRADE[j]:
                    result[..., idx] += PRODUCT_SIGN[i, j] * a[..., i] * b[..., j]

        return Multivector(result, requires_grad=self._requires_grad or other._requires_grad)

    def inner_product(self, other: 'Multivector') -> 'Multivector':
        """Left contraction inner product."""
        a = self._data
        b = other._data
        result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                idx = PRODUCT_IDX[i, j]
                ga, gb, gc = BASIS_GRADE[i], BASIS_GRADE[j], BASIS_GRADE[idx]
                if ga <= gb and gc == gb - ga:
                    result[..., idx] += PRODUCT_SIGN[i, j] * a[..., i] * b[..., j]

        return Multivector(result, requires_grad=self._requires_grad or other._requires_grad)

    def scalar_product(self, other: 'Multivector') -> np.ndarray:
        """Scalar part of geometric product."""
        return self.geometric_product(other).scalar_part

    # --- Involutions ---

    def reverse(self) -> 'Multivector':
        """Reversion: ~self. Grade k -> (-1)^(k*(k-1)/2) * grade k."""
        result = self._data * REVERSE_SIGN
        mv = Multivector(result, requires_grad=self._requires_grad)
        if self._requires_grad:
            mv._grad_fn = ('reverse', self)
        return mv

    def involution(self) -> 'Multivector':
        """Grade involution: grade k -> (-1)^k."""
        return Multivector(self._data * INVOLUTION_SIGN, requires_grad=self._requires_grad)

    def conjugate(self) -> 'Multivector':
        """Clifford conjugate: reverse(involution(self))."""
        return Multivector(self._data * CONJUGATE_SIGN, requires_grad=self._requires_grad)

    # --- Norms ---

    def norm_squared(self) -> np.ndarray:
        """Squared norm: scalar part of self * ~self."""
        rev = self.reverse()
        result = self.scalar_product(rev)
        if isinstance(result, np.ndarray) and result.ndim > 0:
            return result.squeeze(-1)
        return result

    def norm(self) -> np.ndarray:
        """Norm: sqrt(|norm_squared|)."""
        return np.sqrt(np.abs(self.norm_squared()))

    def normalize(self) -> 'Multivector':
        """Normalize to unit norm."""
        n = self.norm()
        if np.isscalar(n):
            if n < 1e-12:
                return Multivector(np.zeros_like(self._data))
            return Multivector(self._data / n)
        n = np.maximum(n, 1e-12)
        return Multivector(self._data / n[..., np.newaxis])

    # --- Sandwich product ---

    def sandwich(self, x: 'Multivector') -> 'Multivector':
        """Sandwich product: self * x * ~self."""
        rev = self.reverse()
        return self.geometric_product(x).geometric_product(rev)

    # --- Exponential ---

    @staticmethod
    def bivector_exp(bv: 'Multivector') -> 'Multivector':
        """exp(B) for a pure bivector B. Returns a rotor (even multivector).
        exp(B) = cos(|B|) + sin(|B|)/|B| * B
        """
        b = bv.bivector_part  # shape (..., 3)
        mag_sq = np.sum(b ** 2, axis=-1, keepdims=True)  # (..., 1)
        mag = np.sqrt(mag_sq)  # (..., 1)

        # Handle near-zero magnitude
        small = (mag < 1e-12).squeeze(-1)

        cos_mag = np.cos(mag)
        sinc = np.where(mag > 1e-12, np.sin(mag) / mag, 1.0)

        result = np.zeros(bv._data.shape, dtype=np.float32)
        result[..., 0:1] = cos_mag
        result[..., 4:7] = sinc * b

        return Multivector(result, grade_mask=0x05)

    @staticmethod
    def rotor_log(rotor: 'Multivector') -> 'Multivector':
        """Logarithm of a rotor. Returns a pure bivector."""
        s = rotor.scalar_part  # (..., 1)
        b = rotor.bivector_part  # (..., 3)
        bv_mag = np.sqrt(np.sum(b ** 2, axis=-1, keepdims=True))

        angle = np.arctan2(bv_mag, s)
        scale = np.where(bv_mag > 1e-12, angle / bv_mag, 0.0)

        result = np.zeros(rotor._data.shape, dtype=np.float32)
        result[..., 4:7] = scale * b
        return Multivector(result, grade_mask=0x04)

    # --- Operator overloading ---

    def __mul__(self, other):
        """Geometric product (default multiplication)."""
        if isinstance(other, Multivector):
            return self.geometric_product(other)
        elif isinstance(other, (int, float, np.floating)):
            return Multivector(self._data * float(other), requires_grad=self._requires_grad)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Multivector(self._data * float(other), requires_grad=self._requires_grad)
        return NotImplemented

    def __xor__(self, other):
        """Outer (wedge) product: a ^ b."""
        if isinstance(other, Multivector):
            return self.outer_product(other)
        return NotImplemented

    def __or__(self, other):
        """Inner product: a | b."""
        if isinstance(other, Multivector):
            return self.inner_product(other)
        return NotImplemented

    def __invert__(self):
        """Reverse: ~a."""
        return self.reverse()

    def __add__(self, other):
        if isinstance(other, Multivector):
            data = self._data + other._data
            mv = Multivector(data, requires_grad=self._requires_grad or other._requires_grad)
            if mv._requires_grad:
                mv._grad_fn = ('add', self, other)
            return mv
        elif isinstance(other, (int, float)):
            data = self._data.copy()
            data[..., 0] += other
            return Multivector(data, requires_grad=self._requires_grad)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Multivector):
            return Multivector(self._data - other._data,
                             requires_grad=self._requires_grad or other._requires_grad)
        elif isinstance(other, (int, float)):
            data = self._data.copy()
            data[..., 0] -= other
            return Multivector(data, requires_grad=self._requires_grad)
        return NotImplemented

    def __neg__(self):
        return Multivector(-self._data, requires_grad=self._requires_grad)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return Multivector(self._data / float(other), requires_grad=self._requires_grad)
        return NotImplemented

    # --- Representation ---

    def __repr__(self):
        names = ['1', 'e1', 'e2', 'e3', 'e12', 'e13', 'e23', 'e123']
        if self._data.ndim == 1:
            terms = []
            for i, (coeff, name) in enumerate(zip(self._data, names)):
                if abs(coeff) > 1e-10:
                    terms.append(f"{coeff:.4f}*{name}" if i > 0 else f"{coeff:.4f}")
            return f"Multivector({' + '.join(terms) if terms else '0'})"
        else:
            return f"Multivector(shape={self.shape}, dtype={self._data.dtype})"

    # --- Constructors ---

    @staticmethod
    def make_scalar(val: float) -> 'Multivector':
        """Create a scalar multivector."""
        data = np.zeros(8, dtype=np.float32)
        data[0] = val
        return Multivector(data, grade_mask=0x01)

    @staticmethod
    def make_vector(e1: float = 0, e2: float = 0, e3: float = 0) -> 'Multivector':
        """Create a pure vector."""
        data = np.zeros(8, dtype=np.float32)
        data[1] = e1; data[2] = e2; data[3] = e3
        return Multivector(data, grade_mask=0x02)

    @staticmethod
    def make_bivector(e12: float = 0, e13: float = 0, e23: float = 0) -> 'Multivector':
        """Create a pure bivector."""
        data = np.zeros(8, dtype=np.float32)
        data[4] = e12; data[5] = e13; data[6] = e23
        return Multivector(data, grade_mask=0x04)

    @staticmethod
    def make_pseudoscalar(val: float = 1.0) -> 'Multivector':
        """Create a pseudoscalar (e123)."""
        data = np.zeros(8, dtype=np.float32)
        data[7] = val
        return Multivector(data, grade_mask=0x08)

    # Convenience aliases (class-level, won't shadow property on instances)
    @classmethod
    def vector(cls, e1: float = 0, e2: float = 0, e3: float = 0) -> 'Multivector':
        return cls.make_vector(e1, e2, e3)

    @classmethod
    def bivector(cls, e12: float = 0, e13: float = 0, e23: float = 0) -> 'Multivector':
        return cls.make_bivector(e12, e13, e23)

    @classmethod
    def pseudoscalar(cls, val: float = 1.0) -> 'Multivector':
        return cls.make_pseudoscalar(val)

    @staticmethod
    def random(shape=(), requires_grad=False) -> 'Multivector':
        """Random multivector with all grades."""
        if isinstance(shape, int):
            shape = (shape,)
        data = np.random.randn(*shape, 8).astype(np.float32)
        return Multivector(data, requires_grad=requires_grad)

    @staticmethod
    def zeros(shape=()) -> 'Multivector':
        if isinstance(shape, int):
            shape = (shape,)
        return Multivector(np.zeros((*shape, 8), dtype=np.float32))

    @staticmethod
    def ones(shape=()) -> 'Multivector':
        """All-ones is just scalar 1 at each position."""
        if isinstance(shape, int):
            shape = (shape,)
        data = np.zeros((*shape, 8), dtype=np.float32)
        data[..., 0] = 1.0
        return Multivector(data)
