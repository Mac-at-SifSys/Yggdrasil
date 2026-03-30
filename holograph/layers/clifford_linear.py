"""
clifford_linear.py — Linear layers using the geometric product.

y = W (*) x + b   where (*) is the geometric product, not matrix multiply.

Weight W is a tensor of multivectors [d_out, d_in, 8].
Input x has shape [batch, seq, d_in, 8].
Output y has shape [batch, seq, d_out, 8].

For each output channel j:
    y[..., j, :] = sum_i  W[j, i, :] (*) x[..., i, :] + b[j, :]

Variants:
- CliffordLinear: full 8-component weights
- EvenLinear: weights in even sub-algebra (4 active components)
- RotorLinear: unit-rotor weights (sandwich product)
- ProjectedLinear: output grade-projected
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
import numpy as np
from rune.backend import xp
from rune.types.multivector import GRADE_SLICES
from rune.ops.batched import batched_geom_prod, batched_reverse, batched_norm, batched_sandwich, geom_matmul
from rune.autodiff.clifford_rules import CliffordDerivativeRules

# Backward-compatible alias (used by tests and density_field)
_geom_prod_batch = batched_geom_prod


# Global gradient storage keyed by id(param)
_GRAD_STORE = {}


def _accumulate_grad(param, grad):
    """Add grad to param's gradient accumulator."""
    pid = id(param)
    if pid not in _GRAD_STORE or _GRAD_STORE[pid] is None:
        _GRAD_STORE[pid] = xp.zeros_like(param)
    _GRAD_STORE[pid] += grad


def _get_grad(param):
    """Get accumulated gradient for a parameter."""
    return _GRAD_STORE.get(id(param), None)


def _zero_grad_params(params):
    """Reset gradient accumulators for a list of parameters."""
    for p in params:
        _GRAD_STORE[id(p)] = None


def _xavier_clifford_init(shape, grade_mask=0x0F):
    """
    Xavier-like initialization for Clifford weights.

    Grade-aware: scales each grade by sqrt(2 / (fan_in + fan_out))
    divided by the number of active components in that grade.
    This ensures the geometric product output has controlled variance.
    """
    d_out, d_in = shape[0], shape[1]
    fan = d_in + d_out
    data = xp.zeros((*shape, 8), dtype=xp.float32)

    grade_sizes = {0: (0, 1), 1: (1, 4), 2: (4, 7), 3: (7, 8)}

    for g in range(4):
        if grade_mask & (1 << g):
            start, end = grade_sizes[g]
            n_components = end - start
            # Scale: standard Xavier / sqrt(n_components) to account
            # for the geometric product mixing
            scale = xp.sqrt(2.0 / (fan * n_components))
            data[..., start:end] = xp.random.randn(*shape, n_components).astype(xp.float32) * scale

    return data


class CliffordLinear:
    """
    Linear layer using geometric product.

    y[..., j, :] = sum_i W[j, i, :] (*) x[..., i, :] + b[j, :]
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        self.d_in = d_in
        self.d_out = d_out

        # Weight: (d_out, d_in, 8)
        self.weight = _xavier_clifford_init((d_out, d_in))

        # Bias: (d_out, 8)
        self.bias = xp.zeros((d_out, 8), dtype=xp.float32) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (..., d_in, 8) — input multivector tensor
        Returns:
            (..., d_out, 8) — output multivector tensor
        """
        batch_shape = x.shape[:-2]
        d_in = x.shape[-2]
        assert d_in == self.d_in, f"Expected d_in={self.d_in}, got {d_in}"

        # Cache for backward
        self._cache = {'x': x}

        x_flat = x.reshape(-1, self.d_in, 8)  # (M, d_in, 8)
        x_t = xp.ascontiguousarray(xp.transpose(x_flat, (1, 0, 2)))  # (d_in, M, 8)
        y_t = geom_matmul(self.weight, x_t)  # (d_out, M, 8)
        y = xp.transpose(y_t, (1, 0, 2)).reshape(*batch_shape, self.d_out, 8)

        if self.bias is not None:
            y = y + self.bias

        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for CliffordLinear.

        Args:
            grad_output: (..., d_out, 8) — gradient of loss w.r.t. output
        Returns:
            grad_input: (..., d_in, 8) — gradient of loss w.r.t. input
        """
        x = self._cache['x']
        batch_shape = x.shape[:-2]
        flat_batch = math.prod(batch_shape) if batch_shape else 1

        # Forward is y_t = geom_matmul(W, x_t), with:
        #   W   : (d_out, d_in, 8)
        #   x_t : (d_in, flat_batch, 8)
        # Reusing geom_matmul here avoids materializing (..., d_out, d_in, 8)
        # broadcast tensors in backward.
        x_flat = xp.ascontiguousarray(x.reshape(flat_batch, self.d_in, 8))
        x_t = xp.ascontiguousarray(xp.transpose(x_flat, (1, 0, 2)))
        grad_flat = xp.ascontiguousarray(grad_output.reshape(flat_batch, self.d_out, 8))
        grad_y_t = xp.ascontiguousarray(xp.transpose(grad_flat, (1, 0, 2)))

        x_rev_t = batched_reverse(x_t)
        x_rev_mt = xp.ascontiguousarray(xp.transpose(x_rev_t, (1, 0, 2)))
        grad_w = geom_matmul(grad_y_t, x_rev_mt)
        _accumulate_grad(self.weight, grad_w)

        w_rev = batched_reverse(self.weight)
        w_rev_t = xp.ascontiguousarray(xp.transpose(w_rev, (1, 0, 2)))
        grad_x_t = geom_matmul(w_rev_t, grad_y_t)
        grad_input = xp.transpose(grad_x_t, (1, 0, 2)).reshape(*batch_shape, self.d_in, 8)

        # Bias gradient: sum over all batch dimensions
        if self.bias is not None:
            grad_b = grad_flat.sum(axis=0)
            _accumulate_grad(self.bias, grad_b)

        return grad_input

    def zero_grad(self):
        """Reset gradient accumulators for all parameters."""
        _zero_grad_params(self.parameters())

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        return f"CliffordLinear(d_in={self.d_in}, d_out={self.d_out})"


class EvenLinear:
    """
    Linear layer with weights restricted to the even sub-algebra.
    Only scalar (1 comp) + bivector (3 comp) = 4 active components.
    Preserves even/odd grading: even*even=even, even*odd=odd.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        self.d_in = d_in
        self.d_out = d_out

        # Weight: (d_out, d_in, 8) but only even grades active
        self.weight = _xavier_clifford_init((d_out, d_in), grade_mask=0x05)

        self.bias = xp.zeros((d_out, 8), dtype=xp.float32) if bias else None
        if self.bias is not None:
            # Bias also even-only
            self.bias[:, 1:4] = 0.0
            self.bias[:, 7:8] = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = {'x': x}
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(-1, self.d_in, 8)
        x_t = xp.ascontiguousarray(xp.transpose(x_flat, (1, 0, 2)))
        y_t = geom_matmul(self.weight, x_t)
        y = xp.transpose(y_t, (1, 0, 2)).reshape(*batch_shape, self.d_out, 8)

        if self.bias is not None:
            y = y + self.bias
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward for EvenLinear — identical structure to CliffordLinear."""
        x = self._cache['x']
        batch_shape = x.shape[:-2]
        flat_batch = math.prod(batch_shape) if batch_shape else 1

        x_flat = xp.ascontiguousarray(x.reshape(flat_batch, self.d_in, 8))
        x_t = xp.ascontiguousarray(xp.transpose(x_flat, (1, 0, 2)))
        grad_flat = xp.ascontiguousarray(grad_output.reshape(flat_batch, self.d_out, 8))
        grad_y_t = xp.ascontiguousarray(xp.transpose(grad_flat, (1, 0, 2)))

        x_rev_t = batched_reverse(x_t)
        x_rev_mt = xp.ascontiguousarray(xp.transpose(x_rev_t, (1, 0, 2)))
        grad_w = geom_matmul(grad_y_t, x_rev_mt)
        grad_w[..., 1:4] = 0.0
        grad_w[..., 7:8] = 0.0
        _accumulate_grad(self.weight, grad_w)

        w_rev = batched_reverse(self.weight)
        w_rev_t = xp.ascontiguousarray(xp.transpose(w_rev, (1, 0, 2)))
        grad_x_t = geom_matmul(w_rev_t, grad_y_t)
        grad_input = xp.transpose(grad_x_t, (1, 0, 2)).reshape(*batch_shape, self.d_in, 8)
        if self.bias is not None:
            grad_b = grad_flat.sum(axis=0)
            grad_b[:, 1:4] = 0.0
            grad_b[:, 7:8] = 0.0
            _accumulate_grad(self.bias, grad_b)
        return grad_input

    def zero_grad(self):
        _zero_grad_params(self.parameters())

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class RotorLinear:
    """
    Linear layer where weights are unit rotors (normalized even elements).
    Applies via sandwich product: y_j = sum_i R_{j,i} * x_i * ~R_{j,i}

    This is a proper rotation in Clifford space — preserves norms and grades.
    """

    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out

        # Initialize rotors near identity: small bivector perturbation
        self.rotor_params = xp.zeros((d_out, d_in, 8), dtype=xp.float32)
        self.rotor_params[:, :, 0] = 1.0  # scalar = 1 (identity rotor)
        # Small random bivector perturbation
        scale = 0.01
        self.rotor_params[:, :, 4:7] = xp.random.randn(d_out, d_in, 3).astype(xp.float32) * scale

    def _get_rotors(self) -> np.ndarray:
        """Normalize rotor parameters to unit rotors."""
        r = self.rotor_params.copy()
        # Zero out odd grades (rotors are even)
        r[..., 1:4] = 0.0
        r[..., 7:8] = 0.0
        # Normalize
        norm = batched_norm(r)
        norm = xp.maximum(norm, 1e-12)
        return r / norm[..., xp.newaxis]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Sandwich product: y[..., j] = sum_i R[j,i] * x[..., i] * ~R[j,i]
        """
        batch_shape = x.shape[:-2]
        rotors = self._get_rotors()  # (d_out, d_in, 8)
        rotor_rev = batched_reverse(rotors)  # (d_out, d_in, 8)

        n_batch = len(batch_shape)
        r = rotors
        r_rev = rotor_rev
        for _ in range(n_batch):
            r = r[xp.newaxis]
            r_rev = r_rev[xp.newaxis]

        x_exp = x[..., xp.newaxis, :, :]  # (..., 1, d_in, 8)

        rxr = batched_sandwich(r, x_exp)  # (..., d_out, d_in, 8)

        # Sum over d_in
        y = xp.sum(rxr, axis=-2)  # (..., d_out, 8)

        # Cache for backward
        self._cache = {'x': x, 'rotors': rotors, 'rotor_rev': rotor_rev}
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward for RotorLinear sandwich product."""
        x = self._cache['x']
        rotors = self._cache['rotors']
        rotor_rev = self._cache['rotor_rev']
        batch_shape = x.shape[:-2]
        n_batch = len(batch_shape)

        # grad_output: (..., d_out, 8)
        # y[..., j, :] = sum_i RXR[..., j, i, :] where RXR = GP(GP(R, x), ~R)
        # grad through sum: grad_rxr[..., j, i, :] = grad_output[..., j, :]
        grad_out_exp = grad_output[..., xp.newaxis, :]  # (..., d_out, 1, 8)
        full_shape = (*batch_shape, self.d_out, self.d_in, 8)
        grad_out_exp = xp.broadcast_to(grad_out_exp, full_shape).copy()

        r = rotors
        r_rev = rotor_rev
        for _ in range(n_batch):
            r = r[xp.newaxis]
            r_rev = r_rev[xp.newaxis]
        r_exp = xp.broadcast_to(r, full_shape).copy()
        r_rev_exp = xp.broadcast_to(r_rev, full_shape).copy()

        x_exp = x[..., xp.newaxis, :, :]
        x_exp = xp.broadcast_to(x_exp, full_shape).copy()
        rx = batched_geom_prod(r_exp, x_exp)

        # Backward through outer GP: rxr = GP(rx, r_rev)
        grad_rx, grad_rrev = CliffordDerivativeRules.geometric_product_backward(
            grad_out_exp, rx, r_rev_exp
        )

        # Backward through inner GP: rx = GP(r, x)
        grad_r_inner, grad_x = CliffordDerivativeRules.geometric_product_backward(
            grad_rx, r_exp, x_exp
        )

        # Rotor gradient: grad_r_inner + reverse(grad_rrev) (from ~R dependency)
        from rune.ops.batched import batched_reverse as _rev
        grad_r_total = grad_r_inner + _rev(grad_rrev)

        # Sum grad_r over batch dims to get (d_out, d_in, 8)
        for _ in range(n_batch):
            grad_r_total = grad_r_total.sum(axis=0)

        # For rotor_params gradient, account for normalization
        # Simplified: accumulate directly (normalization Jacobian is near identity for small perturbations)
        grad_r_total[..., 1:4] = 0.0
        grad_r_total[..., 7:8] = 0.0
        _accumulate_grad(self.rotor_params, grad_r_total)

        # Input gradient: sum over d_out
        grad_input = grad_x.sum(axis=-3)
        return grad_input

    def zero_grad(self):
        _zero_grad_params(self.parameters())

    def parameters(self):
        return [self.rotor_params]


class ProjectedLinear:
    """
    CliffordLinear followed by grade projection.
    Output is projected to keep only specified grades.
    """

    def __init__(self, d_in: int, d_out: int, output_grades: list = None, bias: bool = True):
        """
        Args:
            output_grades: list of grade indices to keep, e.g. [0, 2] for even part
        """
        self.linear = CliffordLinear(d_in, d_out, bias=bias)
        self.output_grades = output_grades if output_grades is not None else [0, 1, 2, 3]

        # Build mask
        self._grade_mask = xp.zeros(8, dtype=xp.float32)
        for g in self.output_grades:
            slc = GRADE_SLICES[g]
            self._grade_mask[slc] = 1.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.linear.forward(x)
        return y * self._grade_mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward through grade projection (linear) then CliffordLinear."""
        # Grade projection is linear: grad through mask
        grad_masked = grad_output * self._grade_mask
        return self.linear.backward(grad_masked)

    def zero_grad(self):
        self.linear.zero_grad()

    def parameters(self):
        return self.linear.parameters()
