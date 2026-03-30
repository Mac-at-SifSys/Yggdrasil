"""
clifford_rules.py — Derivative rules for Clifford algebra operations

These are NOT generic autograd rules. They use the algebraic structure
of Cl(3,0) to compute gradients analytically.

Key derivative rules:
  d(AB)/dA = grad * ~B (right-multiply grad by reverse of B)
  d(AB)/dB = ~A * grad (left-multiply grad by reverse of A)
  d(exp(B))/dB = closed form
  d(grade_k(X))/dX = grade_k(grad) (projection is linear)
  d(~X)/dX = ~grad (reverse is linear, self-adjoint)
  d(norm(X))/dX = X / norm(X)
"""

import numpy as np
from rune.types.multivector import Multivector, PRODUCT_IDX, PRODUCT_SIGN, REVERSE_SIGN
from rune.ops.batched import batched_geom_prod, batched_reverse, batched_grade_project


def _xp(arr):
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


class CliffordDerivativeRules:
    """Collection of backward rules for Clifford operations."""

    @staticmethod
    def geometric_product_backward(grad_output: np.ndarray,
                                    a_data: np.ndarray,
                                    b_data: np.ndarray):
        """
        Backward for C = A * B (geometric product).

        Forward: C[product_idx[i][j]] += product_sign[i][j] * A[i] * B[j]

        By chain rule:
          dL/dA_i = sum_j sign[i][j] * grad_C[idx[i][j]] * B[j]
          dL/dB_j = sum_i sign[i][j] * grad_C[idx[i][j]] * A[i]
        """
        # Under the Euclidean component inner product used by the stack:
        #   dL/dA = grad_output * ~B
        #   dL/dB = ~A * grad_output
        # This lets the backward path reuse the native mjolnir GP kernels.
        grad_a = batched_geom_prod(grad_output, batched_reverse(b_data))
        grad_b = batched_geom_prod(batched_reverse(a_data), grad_output)
        return grad_a, grad_b

    @staticmethod
    def reverse_backward(grad_output: np.ndarray):
        """
        Backward for Y = ~X (reverse).
        Reverse is linear and self-adjoint: dL/dX = ~(dL/dY)
        """
        return batched_reverse(grad_output)

    @staticmethod
    def grade_project_backward(grad_output: np.ndarray, grade: int):
        """
        Backward for Y = grade_k(X).
        Grade projection is linear: dL/dX has grad only in grade k.
        """
        return batched_grade_project(grad_output, grade)

    @staticmethod
    def add_backward(grad_output: np.ndarray):
        """
        Backward for C = A + B.
        dL/dA = dL/dC, dL/dB = dL/dC
        """
        return grad_output, grad_output.copy()

    @staticmethod
    def scale_backward(grad_output: np.ndarray, scale: float):
        """
        Backward for Y = s * X (scalar multiplication).
        dL/dX = s * dL/dY
        """
        return grad_output * scale

    @staticmethod
    def norm_squared_backward(grad_output: np.ndarray, x_data: np.ndarray):
        """
        Backward for y = <x * ~x>_0 (norm squared).
        In Cl(3,0), <x * ~x>_0 is exactly the Euclidean sum of squared
        components, so d(norm_sq)/dX = 2X.
        """
        if grad_output.ndim < x_data.ndim:
            grad_output = grad_output[..., np.newaxis]
        return 2.0 * x_data * grad_output

    @staticmethod
    def sandwich_backward(grad_output: np.ndarray,
                          r_data: np.ndarray,
                          x_data: np.ndarray):
        """
        Backward for Y = R * X * ~R (sandwich product).

        Y = R * (X * ~R) = (R * X) * ~R

        Using chain rule through two geometric products:
        Let T = R * X, then Y = T * ~R

        dL/dT, dL/d(~R) from outer product backward
        dL/dR, dL/dX from inner product backward
        Plus dL/dR contribution from ~R dependency
        """
        r_rev = batched_reverse(r_data)

        # Forward intermediates
        rx = batched_geom_prod(r_data, x_data)  # R * X

        # Backward through Y = GP(RX, ~R)
        grad_rx, grad_rrev = CliffordDerivativeRules.geometric_product_backward(
            grad_output, rx, r_rev
        )

        # Backward through RX = GP(R, X)
        grad_r, grad_x = CliffordDerivativeRules.geometric_product_backward(
            grad_rx, r_data, x_data
        )

        # Add contribution from ~R: d(~R)/dR_k = REVERSE_SIGN[k]
        grad_r += batched_reverse(grad_rrev)

        return grad_r, grad_x

    @staticmethod
    def bivector_exp_backward(grad_output: np.ndarray, bv_data: np.ndarray):
        """
        Backward for R = exp(B) where B is a pure bivector.

        d(exp(B))/dB for Cl(3,0) bivectors.
        """
        xp = _xp(bv_data)
        b = bv_data[..., 4:7]  # bivector components
        mag_sq = xp.sum(b ** 2, axis=-1, keepdims=True)
        mag = xp.sqrt(mag_sq + 1e-24)

        cos_mag = xp.cos(mag)
        sin_mag = xp.sin(mag)
        sinc = xp.where(mag > 1e-12, sin_mag / mag, 1.0)
        # d(sinc)/d(mag) = (cos(mag) - sinc) / mag
        d_sinc = xp.where(mag > 1e-12, (cos_mag - sinc) / mag, 0.0)

        grad_bv = xp.zeros_like(bv_data)

        # Gradient of scalar part: d(cos(mag))/dB_k = -sin(mag) * B_k / mag
        grad_from_scalar = grad_output[..., 0:1]  # grad w.r.t. scalar output
        for k in range(3):
            bk = b[..., k:k+1]
            grad_bv[..., 4+k:5+k] += (
                grad_from_scalar * (-sin_mag) * bk / (mag + 1e-24)
            )

        # Gradient of bivector parts: d(sinc * B_k)/dB_m
        for k in range(3):
            grad_from_bk = grad_output[..., 4+k:5+k]
            for m in range(3):
                bm = b[..., m:m+1]
                bk = b[..., k:k+1]
                if k == m:
                    # d(sinc * B_k)/dB_k = sinc + B_k * d_sinc * B_k/mag
                    grad_bv[..., 4+m:5+m] += grad_from_bk * (
                        sinc + d_sinc * bk * bm / (mag + 1e-24)
                    )
                else:
                    # d(sinc * B_k)/dB_m = B_k * d_sinc * B_m/mag
                    grad_bv[..., 4+m:5+m] += grad_from_bk * (
                        d_sinc * bk * bm / (mag + 1e-24)
                    )

        return grad_bv
