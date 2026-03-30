"""
tensor.py — Batched multivector tensor operations

A CliffordTensor wraps a batch of multivectors with shape (*batch_dims, d_model)
where each element is an 8-component Cl(3,0) multivector.

Storage: shape (*batch_dims, d_model, 8) as float32
"""

import numpy as np
from typing import Tuple, Optional, Union
from rune.types.multivector import Multivector, PRODUCT_IDX, PRODUCT_SIGN


class CliffordTensor:
    """
    Batched tensor of Cl(3,0) multivectors.

    Shape convention: (*batch, d_model) where each position is a multivector.
    Internal storage: (*batch, d_model, 8) float32.
    """

    __slots__ = ('_data', '_requires_grad', '_grad', '_grad_fn')

    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        """
        Args:
            data: Shape (..., 8) where last dim is the multivector components
        """
        self._data = data.astype(np.float32) if not isinstance(data, np.ndarray) else data
        self._requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> Tuple:
        """Logical shape (excludes the trailing 8)."""
        return self._data.shape[:-1]

    @property
    def full_shape(self) -> Tuple:
        return self._data.shape

    @property
    def d_model(self) -> int:
        return self._data.shape[-2] if self._data.ndim >= 2 else 1

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    # --- Grade accessors ---

    def grade(self, k: int) -> 'CliffordTensor':
        from rune.types.multivector import GRADE_SLICES
        result = np.zeros_like(self._data)
        slc = GRADE_SLICES[k]
        result[..., slc] = self._data[..., slc]
        return CliffordTensor(result, requires_grad=self._requires_grad)

    @property
    def scalar(self) -> np.ndarray:
        return self._data[..., 0]

    @property
    def vector(self) -> np.ndarray:
        return self._data[..., 1:4]

    @property
    def bivector(self) -> np.ndarray:
        return self._data[..., 4:7]

    @property
    def trivector(self) -> np.ndarray:
        return self._data[..., 7]

    # --- Geometric product matmul ---

    def geom_matmul(self, other: 'CliffordTensor') -> 'CliffordTensor':
        """
        Geometric-product matrix multiply.
        self: (..., M, K, 8), other: (..., K, N, 8) -> (..., M, N, 8)

        out[..., i, j, :] = sum_k self[..., i, k, :] * other[..., k, j, :]
        where * is the geometric product.
        """
        a = self._data   # (..., M, K, 8)
        b = other._data   # (..., K, N, 8)

        M = a.shape[-3] if a.ndim >= 3 else 1
        K = a.shape[-2]
        N = b.shape[-1 - 1]  # second-to-last before 8

        # Reshape for broadcasting
        # a: (..., M, K, 1, 8)
        # b: (..., 1, K, N, 8)
        a_exp = a[..., :, :, np.newaxis, :]  # (..., M, K, 1, 8)
        b_exp = b[..., np.newaxis, :, :, :]  # (..., 1, K, N, 8)

        # Compute element-wise geometric products and sum over K
        result = np.zeros((*a.shape[:-3], M, N, 8), dtype=np.float32)

        for k_idx in range(K):
            a_k = a[..., :, k_idx, :]  # (..., M, 8)
            b_k = b[..., k_idx, :, :]  # (..., N, 8)

            # Compute geometric product for all M x N pairs
            for i in range(8):
                for j in range(8):
                    idx = PRODUCT_IDX[i, j]
                    sign = PRODUCT_SIGN[i, j]
                    result[..., :, :, idx] += sign * (
                        a_k[..., :, np.newaxis, i] * b_k[..., np.newaxis, :, j]
                    )

        ct = CliffordTensor(result, requires_grad=self._requires_grad or other._requires_grad)
        if ct._requires_grad:
            ct._grad_fn = ('geom_matmul', self, other)
        return ct

    # --- Element-wise operations ---

    def geometric_product(self, other: 'CliffordTensor') -> 'CliffordTensor':
        """Element-wise geometric product."""
        a = self._data
        b = other._data
        result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                idx = PRODUCT_IDX[i, j]
                sign = PRODUCT_SIGN[i, j]
                result[..., idx] += sign * a[..., i] * b[..., j]

        return CliffordTensor(result, requires_grad=self._requires_grad or other._requires_grad)

    def reverse(self) -> 'CliffordTensor':
        from rune.types.multivector import REVERSE_SIGN
        return CliffordTensor(self._data * REVERSE_SIGN, requires_grad=self._requires_grad)

    def norm(self) -> np.ndarray:
        """Per-multivector norm. Returns (...) shaped array."""
        rev = self.reverse()
        prod = self.geometric_product(rev)
        return np.sqrt(np.abs(prod.data[..., 0]))

    def normalize(self) -> 'CliffordTensor':
        n = np.maximum(self.norm(), 1e-12)
        return CliffordTensor(self._data / n[..., np.newaxis], requires_grad=self._requires_grad)

    def sandwich(self, x: 'CliffordTensor') -> 'CliffordTensor':
        """Sandwich product: self * x * ~self."""
        rev = self.reverse()
        return self.geometric_product(x).geometric_product(rev)

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, CliffordTensor):
            return CliffordTensor(self._data + other._data,
                                requires_grad=self._requires_grad or other._requires_grad)
        elif isinstance(other, (int, float)):
            data = self._data.copy()
            data[..., 0] += other
            return CliffordTensor(data, requires_grad=self._requires_grad)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, CliffordTensor):
            return self.geometric_product(other)
        elif isinstance(other, (int, float, np.ndarray)):
            if isinstance(other, np.ndarray):
                return CliffordTensor(self._data * other[..., np.newaxis],
                                    requires_grad=self._requires_grad)
            return CliffordTensor(self._data * other, requires_grad=self._requires_grad)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return CliffordTensor(self._data * other, requires_grad=self._requires_grad)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, CliffordTensor):
            return CliffordTensor(self._data - other._data)
        return NotImplemented

    def __neg__(self):
        return CliffordTensor(-self._data, requires_grad=self._requires_grad)

    # --- Constructors ---

    @staticmethod
    def random(*shape, requires_grad=False) -> 'CliffordTensor':
        data = np.random.randn(*shape, 8).astype(np.float32)
        return CliffordTensor(data, requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape) -> 'CliffordTensor':
        return CliffordTensor(np.zeros((*shape, 8), dtype=np.float32))

    def __repr__(self):
        return f"CliffordTensor(shape={self.shape})"
