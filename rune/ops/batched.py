"""
batched.py — Batched Clifford algebra operations.

All functions accept arrays from the active backend (numpy or cupy)
with shape (..., 8) and return arrays in the same backend.

Dispatch order:
  1. Active backend (numpy or cupy) array operations
  2. mjolnir CUDA native (when on GPU and mjolnir CUDA FFI available)
  3. mjolnir CPU native (when on CPU and mjolnir FFI available)

When CuPy is the backend and mjolnir CUDA is available, the heavy
Clifford ops dispatch into native CUDA kernels directly.
"""

import numpy as np
from rune.types.multivector import PRODUCT_IDX, PRODUCT_SIGN, REVERSE_SIGN, GRADE_SLICES

# ========================================================================
# Backend-aware array module
# ========================================================================

def _xp():
    """Get the active array module (numpy or cupy)."""
    from rune.backend import xp
    return xp

def set_device(device: str):
    """Set compute device: 'cpu' or 'cuda'. Also switches array backend."""
    from rune.backend import set_backend
    set_backend(device)

def get_device() -> str:
    """Return current compute device."""
    from rune.backend import get_backend
    return get_backend()

# ========================================================================
# CPU native detection (mjolnir FFI)
# ========================================================================

_NATIVE = None
_CUDA_NATIVE = None

def _native_available():
    global _NATIVE
    if _NATIVE is None:
        try:
            from rune.bindings.mjolnir_ffi import is_native_available
            _NATIVE = is_native_available()
        except ImportError:
            _NATIVE = False
    return _NATIVE


def _cuda_native_available():
    global _CUDA_NATIVE
    if _CUDA_NATIVE is None:
        try:
            from rune.bindings.mjolnir_cuda_ffi import is_cuda_available

            _CUDA_NATIVE = is_cuda_available()
        except ImportError:
            _CUDA_NATIVE = False
    return _CUDA_NATIVE

# ========================================================================
# Core implementations using the active array backend (xp)
# These work on both numpy and cupy arrays transparently.
# ========================================================================

def _xp_geom_prod(a, b):
    """Geometric product using active backend. Works on GPU via CuPy."""
    xp = _xp()
    out_shape = xp.broadcast_shapes(a.shape, b.shape)
    result = xp.zeros(out_shape, dtype=xp.float32)
    for i in range(8):
        for j in range(8):
            idx = int(PRODUCT_IDX[i, j])
            sign = float(PRODUCT_SIGN[i, j])
            result[..., idx] += sign * a[..., i] * b[..., j]
    return result

def _xp_reverse(a):
    xp = _xp()
    rev = xp.asarray(REVERSE_SIGN, dtype=xp.float32)
    return a * rev

def _xp_sandwich(r, x):
    rx = _xp_geom_prod(r, x)
    r_rev = _xp_reverse(r)
    return _xp_geom_prod(rx, r_rev)

def _xp_bivector_exp(bv):
    xp = _xp()
    b_components = bv[..., 4:7]
    mag_sq = xp.sum(b_components ** 2, axis=-1, keepdims=True)
    mag = xp.sqrt(mag_sq + 1e-12)
    cos_mag = xp.cos(mag)
    sinc = xp.where(mag > 1e-12, xp.sin(mag) / mag, xp.ones_like(mag))
    result = xp.zeros(bv.shape, dtype=xp.float32)
    result[..., 0:1] = cos_mag
    result[..., 4:7] = sinc * b_components
    return result

def _xp_norm(a):
    xp = _xp()
    a_rev = _xp_reverse(a)
    scalar = xp.zeros(a.shape[:-1], dtype=xp.float32)
    for i in range(8):
        for j in range(8):
            if int(PRODUCT_IDX[i, j]) == 0:
                scalar += float(PRODUCT_SIGN[i, j]) * a[..., i] * a_rev[..., j]
    return xp.sqrt(xp.abs(scalar))

def _xp_normalize(a):
    xp = _xp()
    n = _xp_norm(a)
    n = xp.maximum(n, xp.float32(1e-12))
    return a / n[..., None]

def _xp_scalar_product(a, b):
    xp = _xp()
    result = xp.zeros(xp.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=xp.float32)
    for i in range(8):
        for j in range(8):
            if int(PRODUCT_IDX[i, j]) == 0:
                result += float(PRODUCT_SIGN[i, j]) * a[..., i] * b[..., j]
    return result

def _xp_grade_project(a, grade):
    xp = _xp()
    result = xp.zeros_like(a)
    slc = GRADE_SLICES[grade]
    result[..., slc] = a[..., slc]
    return result

# ========================================================================
# Public API
# ========================================================================

def _ensure_f32(a):
    """Ensure array is float32 in the active backend."""
    xp = _xp()
    if not isinstance(a, xp.ndarray):
        # Convert from numpy if backend is cupy, or vice versa
        a = xp.asarray(a, dtype=xp.float32)
    elif a.dtype != xp.float32:
        a = a.astype(xp.float32)
    return a

def _broadcast_pair(a, b):
    """Broadcast two arrays to matching shapes."""
    xp = _xp()
    out_shape = xp.broadcast_shapes(a.shape, b.shape)
    if a.shape != out_shape:
        a = xp.broadcast_to(a, out_shape).copy()
    if b.shape != out_shape:
        b = xp.broadcast_to(b, out_shape).copy()
    return a, b


def batched_geom_prod(a, b):
    """Batched geometric product. a, b: (..., 8) -> (..., 8)."""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    a, b = _broadcast_pair(a, b)

    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_gp
            return gpu_batch_gp(a, b)
        except Exception:
            pass

    # On CPU, try mjolnir native first
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_gp
            return native_batch_gp(a, b)
        except Exception:
            pass

    return _xp_geom_prod(a, b)

def batched_reverse(a):
    """Batched reverse (reversion). a: (..., 8) -> (..., 8)."""
    a = _ensure_f32(a)
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_reverse
            return native_batch_reverse(a)
        except Exception:
            pass
    return _xp_reverse(a)

def batched_sandwich(r, x):
    """Batched sandwich product: r * x * ~r."""
    r = _ensure_f32(r)
    x = _ensure_f32(x)
    r, x = _broadcast_pair(r, x)
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_sandwich
            return gpu_batch_sandwich(r, x)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_sandwich
            return native_batch_sandwich(r, x)
        except Exception:
            pass
    return _xp_sandwich(r, x)

def batched_bivector_exp(bv):
    """Batched bivector exp. bv: (..., 8) -> (..., 8) rotor."""
    bv = _ensure_f32(bv)
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_bvexp
            return gpu_batch_bvexp(bv)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_bvexp
            return native_batch_bvexp(bv)
        except Exception:
            pass
    return _xp_bivector_exp(bv)

def batched_norm(a):
    """Batched Clifford norm. a: (..., 8) -> (...)."""
    a = _ensure_f32(a)
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_norm
            return gpu_batch_norm(a)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_norm
            return native_batch_norm(a)
        except Exception:
            pass
    return _xp_norm(a)

def batched_normalize(a):
    """Batched normalize. a: (..., 8) -> (..., 8)."""
    a = _ensure_f32(a)
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_normalize
            return gpu_batch_normalize(a)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_normalize
            return native_batch_normalize(a)
        except Exception:
            pass
    return _xp_normalize(a)

def batched_add(a, b):
    """Batched multivector addition."""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    return a + b

def batched_scale(a, s: float):
    """Batched scalar multiply."""
    a = _ensure_f32(a)
    xp = _xp()
    return a * xp.float32(s)

def batched_grade_project(a, grade: int):
    """Batched grade projection. a: (..., 8), grade: 0-3 -> (..., 8)."""
    a = _ensure_f32(a)
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_grade_proj
            return gpu_batch_grade_proj(a, grade)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_grade_project
            return native_batch_grade_project(a, grade)
        except Exception:
            pass
    return _xp_grade_project(a, grade)

def batched_scalar_product(a, b):
    """Scalar part of batched GP. a, b: (..., 8) -> (...)."""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    xp = _xp()
    out_shape = xp.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    if a.shape != (*out_shape, 8):
        a = xp.broadcast_to(a, (*out_shape, 8)).copy()
    if b.shape != (*out_shape, 8):
        b = xp.broadcast_to(b, (*out_shape, 8)).copy()
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_batch_scalar_product
            return gpu_batch_scalar_product(a, b)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_batch_scalar_product
            return native_batch_scalar_product(a, b)
        except Exception:
            pass
    return _xp_scalar_product(a, b)

def geom_matmul(a, b):
    """Geometric matrix multiply. a: (M,K,8), b: (K,N,8) -> (M,N,8)."""
    xp = _xp()
    a = xp.ascontiguousarray(_ensure_f32(a))
    b = xp.ascontiguousarray(_ensure_f32(b))
    assert a.ndim == 3 and b.ndim == 3
    assert a.shape[1] == b.shape[0] and a.shape[2] == 8 and b.shape[2] == 8
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    if get_device() == 'cuda' and _cuda_native_available():
        try:
            from rune.bindings.mjolnir_cuda_ffi import gpu_geom_matmul
            return gpu_geom_matmul(a, b)
        except Exception:
            pass
    if get_device() == 'cpu' and _native_available():
        try:
            from rune.bindings.mjolnir_ffi import native_geom_matmul
            return native_geom_matmul(a, b)
        except Exception:
            pass
    # Fallback: broadcast pairwise GP then sum over the inner dimension.
    a_exp = a[:, :, xp.newaxis, :]
    b_exp = b[xp.newaxis, :, :, :]
    return xp.sum(batched_geom_prod(a_exp, b_exp), axis=1)

def bivector_exp_from_components(bv_components):
    """exp(B) where B: (..., 3) -> (..., 8) rotor."""
    xp = _xp()
    bv_components = _ensure_f32(bv_components)
    bv = xp.zeros((*bv_components.shape[:-1], 8), dtype=xp.float32)
    bv[..., 4:7] = bv_components
    return batched_bivector_exp(bv)
