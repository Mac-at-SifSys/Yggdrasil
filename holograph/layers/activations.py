"""
activations.py — Grade-aware activation functions for Cl(3,0) multivectors.

Each grade gets a different activation:
- Scalar (grade 0): standard activations (GELU, ReLU, sigmoid)
- Vector (grade 1): magnitude-gated (preserve direction, gate magnitude)
- Bivector (grade 2): magnitude-gated (preserve rotation plane, gate magnitude)
- Trivector (grade 3): tanh squashing (pseudoscalar orientation)
"""

import numpy as np
from rune.backend import xp


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + xp.tanh(xp.sqrt(2.0 / xp.pi) * (x + 0.044715 * x ** 3)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return xp.where(x >= 0,
                    1.0 / (1.0 + xp.exp(-x)),
                    xp.exp(x) / (1.0 + xp.exp(x)))


def _magnitude_gate(components: np.ndarray, activation_fn) -> np.ndarray:
    """
    Gate a k-vector by its magnitude.
    Preserves direction, applies activation to the magnitude.

    components: (..., k) where k is the number of components in the grade
    Returns: (..., k) gated components
    """
    mag = xp.sqrt(xp.sum(components ** 2, axis=-1, keepdims=True) + 1e-12)
    direction = components / mag
    gated_mag = activation_fn(mag)
    return direction * gated_mag


def clifford_gelu(data: np.ndarray) -> np.ndarray:
    """
    Grade-aware GELU activation on raw multivector data.

    data: (..., 8) — raw multivector components
    Returns: (..., 8) — activated components

    Grade 0 (scalar):    GELU
    Grade 1 (vector):    magnitude-gated GELU
    Grade 2 (bivector):  magnitude-gated GELU
    Grade 3 (trivector): tanh
    """
    result = xp.zeros_like(data)
    # Grade 0: scalar GELU
    result[..., 0:1] = _gelu(data[..., 0:1])
    # Grade 1: magnitude-gated GELU
    result[..., 1:4] = _magnitude_gate(data[..., 1:4], _gelu)
    # Grade 2: magnitude-gated GELU
    result[..., 4:7] = _magnitude_gate(data[..., 4:7], _gelu)
    # Grade 3: tanh
    result[..., 7:8] = xp.tanh(data[..., 7:8])
    return result


def clifford_relu(data: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    """
    Grade-aware ReLU with per-grade thresholds.

    data: (..., 8) — raw multivector components
    thresholds: (4,) — one threshold per grade [t0, t1, t2, t3]
                 Default: [0.0, 0.0, 0.0, 0.0]
    Returns: (..., 8)
    """
    if thresholds is None:
        thresholds = xp.zeros(4, dtype=xp.float32)

    result = xp.zeros_like(data)

    # Grade 0: threshold on scalar value
    result[..., 0:1] = xp.maximum(data[..., 0:1], thresholds[0])

    # Grade 1: threshold on magnitude, preserve direction
    vec = data[..., 1:4]
    vec_mag = xp.sqrt(xp.sum(vec ** 2, axis=-1, keepdims=True) + 1e-12)
    vec_dir = vec / vec_mag
    vec_gated = xp.maximum(vec_mag - thresholds[1], 0.0)
    result[..., 1:4] = vec_dir * vec_gated

    # Grade 2: threshold on magnitude, preserve direction
    bv = data[..., 4:7]
    bv_mag = xp.sqrt(xp.sum(bv ** 2, axis=-1, keepdims=True) + 1e-12)
    bv_dir = bv / bv_mag
    bv_gated = xp.maximum(bv_mag - thresholds[2], 0.0)
    result[..., 4:7] = bv_dir * bv_gated

    # Grade 3: threshold on absolute value, preserve sign
    tv = data[..., 7:8]
    tv_sign = xp.sign(tv)
    tv_gated = xp.maximum(xp.abs(tv) - thresholds[3], 0.0)
    result[..., 7:8] = tv_sign * tv_gated

    return result


def clifford_sigmoid(data: np.ndarray) -> np.ndarray:
    """
    Grade-aware sigmoid gating.

    Grade 0: sigmoid(scalar) — squash to [0, 1]
    Grade 1: magnitude sigmoid gating — direction * sigmoid(magnitude)
    Grade 2: magnitude sigmoid gating — direction * sigmoid(magnitude)
    Grade 3: tanh (maps to [-1, 1] for pseudoscalar orientation)
    """
    result = xp.zeros_like(data)

    # Grade 0: standard sigmoid
    result[..., 0:1] = _sigmoid(data[..., 0:1])

    # Grade 1: magnitude-gated sigmoid
    result[..., 1:4] = _magnitude_gate(data[..., 1:4], _sigmoid)

    # Grade 2: magnitude-gated sigmoid
    result[..., 4:7] = _magnitude_gate(data[..., 4:7], _sigmoid)

    # Grade 3: tanh
    result[..., 7:8] = xp.tanh(data[..., 7:8])

    return result


# ============================================================================
# Backward functions for activations
# ============================================================================

def _gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU activation."""
    c = xp.sqrt(2.0 / xp.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = xp.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    d_inner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner


def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = _sigmoid(x)
    return s * (1.0 - s)


def _magnitude_gate_backward(components: np.ndarray, grad_out: np.ndarray,
                               activation_fn, activation_deriv_fn) -> np.ndarray:
    """
    Backward through magnitude gating: output = direction * activation(magnitude).

    components: (..., k) — input components
    grad_out: (..., k) — gradient of loss w.r.t. output
    Returns: (..., k) — gradient of loss w.r.t. input components
    """
    eps = 1e-12
    mag = xp.sqrt(xp.sum(components ** 2, axis=-1, keepdims=True) + eps)
    direction = components / mag
    gated_mag = activation_fn(mag)
    act_deriv = activation_deriv_fn(mag)

    # output = direction * gated_mag = (comp / mag) * act(mag)
    # d_output_k / d_comp_m:
    #   = (delta_{km} / mag - comp_k * comp_m / mag^3) * act(mag)
    #     + (comp_k / mag) * act'(mag) * (comp_m / mag)
    #   = act(mag) * (delta_{km}/mag - comp_k*comp_m/mag^3) + act'(mag)*comp_k*comp_m/mag^2

    # Using chain rule efficiently:
    # grad_comp = (act(mag)/mag) * grad_out
    #           + (act'(mag)/mag - act(mag)/mag^2) * direction * dot(direction, grad_out)
    dot_dg = xp.sum(direction * grad_out, axis=-1, keepdims=True)
    term1 = (gated_mag / mag) * grad_out
    term2 = (act_deriv / mag - gated_mag / (mag ** 2)) * direction * dot_dg * mag
    # Simplify term2: (act'(mag) - act(mag)/mag) * direction * dot(direction, grad_out)
    term2 = (act_deriv - gated_mag / mag) * direction * dot_dg

    return term1 + term2


def clifford_gelu_backward(data: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    Backward pass for grade-aware GELU activation.

    Args:
        data: (..., 8) — the INPUT to the forward pass (cached)
        grad_output: (..., 8) — gradient w.r.t. the output
    Returns:
        grad_input: (..., 8) — gradient w.r.t. the input
    """
    grad_input = xp.zeros_like(data)

    # Grade 0: scalar GELU derivative
    grad_input[..., 0:1] = grad_output[..., 0:1] * _gelu_derivative(data[..., 0:1])

    # Grade 1: magnitude-gated GELU backward
    grad_input[..., 1:4] = _magnitude_gate_backward(
        data[..., 1:4], grad_output[..., 1:4], _gelu, _gelu_derivative
    )

    # Grade 2: magnitude-gated GELU backward
    grad_input[..., 4:7] = _magnitude_gate_backward(
        data[..., 4:7], grad_output[..., 4:7], _gelu, _gelu_derivative
    )

    # Grade 3: tanh derivative = 1 - tanh^2
    tanh_val = xp.tanh(data[..., 7:8])
    grad_input[..., 7:8] = grad_output[..., 7:8] * (1.0 - tanh_val ** 2)

    return grad_input


def clifford_relu_backward(data: np.ndarray, grad_output: np.ndarray,
                            thresholds: np.ndarray = None) -> np.ndarray:
    """Backward pass for grade-aware ReLU."""
    if thresholds is None:
        thresholds = xp.zeros(4, dtype=xp.float32)

    grad_input = xp.zeros_like(data)
    eps = 1e-12

    # Grade 0: threshold on scalar value
    grad_input[..., 0:1] = grad_output[..., 0:1] * (data[..., 0:1] > thresholds[0]).astype(xp.float32)

    # Grade 1: threshold on magnitude
    vec = data[..., 1:4]
    vec_mag = xp.sqrt(xp.sum(vec ** 2, axis=-1, keepdims=True) + eps)
    active = (vec_mag > thresholds[1]).astype(xp.float32)
    direction = vec / vec_mag
    dot_dg = xp.sum(direction * grad_output[..., 1:4], axis=-1, keepdims=True)
    # When active: output = direction * (mag - threshold)
    # grad = (1/mag) * ((mag - threshold) * (grad - direction*dot) + direction*dot*mag)
    # Simplified: active * grad_output (for threshold=0 this is just direction projection)
    gated_mag = xp.maximum(vec_mag - thresholds[1], 0.0)
    grad_input[..., 1:4] = active * (
        (gated_mag / vec_mag) * grad_output[..., 1:4]
        + (1.0 - gated_mag / vec_mag) * direction * dot_dg
    )

    # Grade 2: same structure as grade 1
    bv = data[..., 4:7]
    bv_mag = xp.sqrt(xp.sum(bv ** 2, axis=-1, keepdims=True) + eps)
    active = (bv_mag > thresholds[2]).astype(xp.float32)
    direction = bv / bv_mag
    dot_dg = xp.sum(direction * grad_output[..., 4:7], axis=-1, keepdims=True)
    gated_mag = xp.maximum(bv_mag - thresholds[2], 0.0)
    grad_input[..., 4:7] = active * (
        (gated_mag / bv_mag) * grad_output[..., 4:7]
        + (1.0 - gated_mag / bv_mag) * direction * dot_dg
    )

    # Grade 3: threshold on absolute value with sign preservation
    tv = data[..., 7:8]
    active = (xp.abs(tv) > thresholds[3]).astype(xp.float32)
    grad_input[..., 7:8] = grad_output[..., 7:8] * active

    return grad_input


def clifford_sigmoid_backward(data: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """Backward pass for grade-aware sigmoid."""
    grad_input = xp.zeros_like(data)

    # Grade 0: sigmoid derivative
    grad_input[..., 0:1] = grad_output[..., 0:1] * _sigmoid_derivative(data[..., 0:1])

    # Grade 1: magnitude-gated sigmoid backward
    grad_input[..., 1:4] = _magnitude_gate_backward(
        data[..., 1:4], grad_output[..., 1:4], _sigmoid, _sigmoid_derivative
    )

    # Grade 2: magnitude-gated sigmoid backward
    grad_input[..., 4:7] = _magnitude_gate_backward(
        data[..., 4:7], grad_output[..., 4:7], _sigmoid, _sigmoid_derivative
    )

    # Grade 3: tanh derivative
    tanh_val = xp.tanh(data[..., 7:8])
    grad_input[..., 7:8] = grad_output[..., 7:8] * (1.0 - tanh_val ** 2)

    return grad_input
