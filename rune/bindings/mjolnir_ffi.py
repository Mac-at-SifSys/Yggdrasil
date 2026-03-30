"""
mjolnir_ffi.py — FFI bindings to mjolnir C kernels

Uses ctypes to call into the compiled mjolnir DLL/SO.
Falls back to pure NumPy implementation if library not available.

All functions accept and return numpy arrays of shape (8,) float32.
"""

import ctypes
import numpy as np
import os
import sys
from typing import Optional

# ============================================================
# Library loading
# ============================================================

_lib: Optional[ctypes.CDLL] = None

# Search paths for the compiled library
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mjolnir'))
_lib_paths = [
    os.path.join(_base, 'build', 'Release', 'mjolnir_cpu.dll'),    # MSVC Release
    os.path.join(_base, 'build', 'Debug', 'mjolnir_cpu.dll'),      # MSVC Debug
    os.path.join(_base, 'build', 'mjolnir_cpu.dll'),               # Single-config
    os.path.join(_base, 'build', 'libmjolnir_cpu.so'),             # Linux
    os.path.join(_base, 'build', 'libmjolnir_cpu.dylib'),          # macOS
]


def _load_library() -> bool:
    global _lib
    for path in _lib_paths:
        path = os.path.normpath(path)
        if os.path.exists(path):
            try:
                _lib = ctypes.CDLL(path)
                _setup_prototypes()
                return True
            except OSError as e:
                print(f"[mjolnir_ffi] Failed to load {path}: {e}", file=sys.stderr)
                continue
    return False


def is_native_available() -> bool:
    """Check if the native mjolnir library is loaded."""
    if _lib is None:
        _load_library()
    return _lib is not None


def get_library_path() -> Optional[str]:
    """Return the path of the loaded library, or None."""
    if is_native_available():
        for path in _lib_paths:
            if os.path.exists(path):
                return os.path.normpath(path)
    return None


# ============================================================
# C struct mapping
# ============================================================

class Cl3MultivectorC(ctypes.Structure):
    """C-compatible Cl3Multivector struct.
    Matches: { float s; float v[3]; float b[3]; float t; }
    """
    _fields_ = [
        ('s', ctypes.c_float),
        ('v', ctypes.c_float * 3),
        ('b', ctypes.c_float * 3),
        ('t', ctypes.c_float),
    ]


_MV_PTR = ctypes.POINTER(Cl3MultivectorC)


def _mv_to_c(data: np.ndarray) -> Cl3MultivectorC:
    """Convert numpy array (8,) float32 → C struct."""
    c = Cl3MultivectorC()
    d = data.astype(np.float32).ravel()
    c.s = float(d[0])
    c.v[0] = float(d[1]); c.v[1] = float(d[2]); c.v[2] = float(d[3])
    c.b[0] = float(d[4]); c.b[1] = float(d[5]); c.b[2] = float(d[6])
    c.t = float(d[7])
    return c


def _c_to_np(c: Cl3MultivectorC) -> np.ndarray:
    """Convert C struct → numpy array (8,) float32."""
    return np.array([c.s, c.v[0], c.v[1], c.v[2],
                     c.b[0], c.b[1], c.b[2], c.t], dtype=np.float32)


# ============================================================
# Function prototypes
# ============================================================

def _setup_prototypes():
    """Set up ctypes function signatures for all mjolnir operations."""
    if _lib is None:
        return

    # Geometric product: Cl3Multivector cl3_geometric_product(const Cl3Multivector*, const Cl3Multivector*)
    _lib.cl3_geometric_product.restype = Cl3MultivectorC
    _lib.cl3_geometric_product.argtypes = [_MV_PTR, _MV_PTR]

    # Outer product
    _lib.cl3_outer_product.restype = Cl3MultivectorC
    _lib.cl3_outer_product.argtypes = [_MV_PTR, _MV_PTR]

    # Inner product
    _lib.cl3_inner_product.restype = Cl3MultivectorC
    _lib.cl3_inner_product.argtypes = [_MV_PTR, _MV_PTR]

    # Scalar product
    _lib.cl3_scalar_product.restype = ctypes.c_float
    _lib.cl3_scalar_product.argtypes = [_MV_PTR, _MV_PTR]

    # Grade projection
    _lib.cl3_grade_project.restype = Cl3MultivectorC
    _lib.cl3_grade_project.argtypes = [_MV_PTR, ctypes.c_int]

    # Reverse
    _lib.cl3_reverse.restype = Cl3MultivectorC
    _lib.cl3_reverse.argtypes = [_MV_PTR]

    # Involution
    _lib.cl3_involution.restype = Cl3MultivectorC
    _lib.cl3_involution.argtypes = [_MV_PTR]

    # Conjugate
    _lib.cl3_conjugate.restype = Cl3MultivectorC
    _lib.cl3_conjugate.argtypes = [_MV_PTR]

    # Norms
    _lib.cl3_norm_squared.restype = ctypes.c_float
    _lib.cl3_norm_squared.argtypes = [_MV_PTR]

    _lib.cl3_norm.restype = ctypes.c_float
    _lib.cl3_norm.argtypes = [_MV_PTR]

    _lib.cl3_normalize.restype = Cl3MultivectorC
    _lib.cl3_normalize.argtypes = [_MV_PTR]

    # Sandwich
    _lib.cl3_sandwich.restype = Cl3MultivectorC
    _lib.cl3_sandwich.argtypes = [_MV_PTR, _MV_PTR]

    # Exponential / Log
    _lib.cl3_bivector_exp.restype = Cl3MultivectorC
    _lib.cl3_bivector_exp.argtypes = [_MV_PTR]

    _lib.cl3_rotor_log.restype = Cl3MultivectorC
    _lib.cl3_rotor_log.argtypes = [_MV_PTR]

    # Arithmetic
    _lib.cl3_add.restype = Cl3MultivectorC
    _lib.cl3_add.argtypes = [_MV_PTR, _MV_PTR]

    _lib.cl3_sub.restype = Cl3MultivectorC
    _lib.cl3_sub.argtypes = [_MV_PTR, _MV_PTR]

    _lib.cl3_scale.restype = Cl3MultivectorC
    _lib.cl3_scale.argtypes = [_MV_PTR, ctypes.c_float]

    _lib.cl3_negate.restype = Cl3MultivectorC
    _lib.cl3_negate.argtypes = [_MV_PTR]

    # --- Flat-layout batch operations ---
    _FPTR = ctypes.POINTER(ctypes.c_float)

    _lib.cl3_batch_gp_flat.restype = None
    _lib.cl3_batch_gp_flat.argtypes = [_FPTR, _FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_reverse_flat.restype = None
    _lib.cl3_batch_reverse_flat.argtypes = [_FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_sandwich_flat.restype = None
    _lib.cl3_batch_sandwich_flat.argtypes = [_FPTR, _FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_bivector_exp_flat.restype = None
    _lib.cl3_batch_bivector_exp_flat.argtypes = [_FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_norm_flat.restype = None
    _lib.cl3_batch_norm_flat.argtypes = [_FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_normalize_flat.restype = None
    _lib.cl3_batch_normalize_flat.argtypes = [_FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_add_flat.restype = None
    _lib.cl3_batch_add_flat.argtypes = [_FPTR, _FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_scale_flat.restype = None
    _lib.cl3_batch_scale_flat.argtypes = [_FPTR, ctypes.c_float, _FPTR, ctypes.c_size_t]

    _lib.cl3_batch_grade_project_flat.restype = None
    _lib.cl3_batch_grade_project_flat.argtypes = [_FPTR, _FPTR, ctypes.c_int, ctypes.c_size_t]

    _lib.cl3_batch_scalar_product_flat.restype = None
    _lib.cl3_batch_scalar_product_flat.argtypes = [_FPTR, _FPTR, _FPTR, ctypes.c_size_t]

    _lib.cl3_geom_matmul_flat.restype = None
    _lib.cl3_geom_matmul_flat.argtypes = [_FPTR, _FPTR, _FPTR, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]


# ============================================================
# Python wrapper functions
# ============================================================

def native_geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric product via native mjolnir kernel."""
    ca = _mv_to_c(a)
    cb = _mv_to_c(b)
    result = _lib.cl3_geometric_product(ctypes.byref(ca), ctypes.byref(cb))
    return _c_to_np(result)


def native_outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca = _mv_to_c(a); cb = _mv_to_c(b)
    return _c_to_np(_lib.cl3_outer_product(ctypes.byref(ca), ctypes.byref(cb)))


def native_inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca = _mv_to_c(a); cb = _mv_to_c(b)
    return _c_to_np(_lib.cl3_inner_product(ctypes.byref(ca), ctypes.byref(cb)))


def native_reverse(a: np.ndarray) -> np.ndarray:
    ca = _mv_to_c(a)
    return _c_to_np(_lib.cl3_reverse(ctypes.byref(ca)))


def native_sandwich(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    cr = _mv_to_c(r); cx = _mv_to_c(x)
    return _c_to_np(_lib.cl3_sandwich(ctypes.byref(cr), ctypes.byref(cx)))


def native_bivector_exp(bv: np.ndarray) -> np.ndarray:
    cb = _mv_to_c(bv)
    return _c_to_np(_lib.cl3_bivector_exp(ctypes.byref(cb)))


def native_rotor_log(r: np.ndarray) -> np.ndarray:
    cr = _mv_to_c(r)
    return _c_to_np(_lib.cl3_rotor_log(ctypes.byref(cr)))


def native_norm(a: np.ndarray) -> float:
    ca = _mv_to_c(a)
    return float(_lib.cl3_norm(ctypes.byref(ca)))


def native_normalize(a: np.ndarray) -> np.ndarray:
    ca = _mv_to_c(a)
    return _c_to_np(_lib.cl3_normalize(ctypes.byref(ca)))


def native_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca = _mv_to_c(a); cb = _mv_to_c(b)
    return _c_to_np(_lib.cl3_add(ctypes.byref(ca), ctypes.byref(cb)))


def native_scale(a: np.ndarray, s: float) -> np.ndarray:
    ca = _mv_to_c(a)
    return _c_to_np(_lib.cl3_scale(ctypes.byref(ca), ctypes.c_float(s)))


def native_grade_project(a: np.ndarray, grade: int) -> np.ndarray:
    ca = _mv_to_c(a)
    return _c_to_np(_lib.cl3_grade_project(ctypes.byref(ca), ctypes.c_int(grade)))


# ============================================================
# Batched Python wrapper functions (flat-layout)
# ============================================================

_FPTR = ctypes.POINTER(ctypes.c_float)


def _as_flat(arr: np.ndarray, components: int = 8) -> np.ndarray:
    """Ensure contiguous float32 array reshaped to (-1, components)."""
    return np.ascontiguousarray(arr.reshape(-1, components), dtype=np.float32)


def _ensure_lib():
    """Ensure the native library is loaded before calling batch functions."""
    global _lib
    if _lib is None:
        _load_library()
    if _lib is None:
        raise RuntimeError("mjolnir_cpu library not available")
    return _lib


def native_batch_gp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched geometric product on flat (..., 8) arrays."""
    lib = _ensure_lib()
    shape = a.shape
    a_flat = _as_flat(a)
    b_flat = _as_flat(b)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_gp_flat(
        a_flat.ctypes.data_as(_FPTR),
        b_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_reverse(a: np.ndarray) -> np.ndarray:
    """Batched reverse on flat (..., 8) arrays."""
    shape = a.shape
    a_flat = _as_flat(a)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_reverse_flat(
        a_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_sandwich(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Batched sandwich product on flat (..., 8) arrays."""
    shape = r.shape
    r_flat = _as_flat(r)
    x_flat = _as_flat(x)
    n = r_flat.shape[0]
    out = np.empty_like(r_flat)
    _ensure_lib().cl3_batch_sandwich_flat(
        r_flat.ctypes.data_as(_FPTR),
        x_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_bvexp(bv: np.ndarray) -> np.ndarray:
    """Batched bivector exponential on flat (..., 8) arrays."""
    shape = bv.shape
    bv_flat = _as_flat(bv)
    n = bv_flat.shape[0]
    out = np.empty_like(bv_flat)
    _ensure_lib().cl3_batch_bivector_exp_flat(
        bv_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_norm(a: np.ndarray) -> np.ndarray:
    """Batched norm on flat (..., 8) arrays. Returns float array of shape (...)."""
    orig_shape = a.shape[:-1]  # drop last dim (8)
    a_flat = _as_flat(a)
    n = a_flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    _ensure_lib().cl3_batch_norm_flat(
        a_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(orig_shape) if orig_shape else out


def native_batch_normalize(a: np.ndarray) -> np.ndarray:
    """Batched normalize on flat (..., 8) arrays."""
    shape = a.shape
    a_flat = _as_flat(a)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_normalize_flat(
        a_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched addition on flat (..., 8) arrays."""
    shape = a.shape
    a_flat = _as_flat(a)
    b_flat = _as_flat(b)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_add_flat(
        a_flat.ctypes.data_as(_FPTR),
        b_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_scale(a: np.ndarray, s: float) -> np.ndarray:
    """Batched scalar multiply on flat (..., 8) arrays."""
    shape = a.shape
    a_flat = _as_flat(a)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_scale_flat(
        a_flat.ctypes.data_as(_FPTR),
        ctypes.c_float(s),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_grade_project(a: np.ndarray, grade: int) -> np.ndarray:
    """Batched grade projection on flat (..., 8) arrays."""
    shape = a.shape
    a_flat = _as_flat(a)
    n = a_flat.shape[0]
    out = np.empty_like(a_flat)
    _ensure_lib().cl3_batch_grade_project_flat(
        a_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_int(grade),
        ctypes.c_size_t(n))
    return out.reshape(shape)


def native_batch_scalar_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched scalar product on flat (..., 8) arrays. Returns float array of shape (...)."""
    orig_shape = a.shape[:-1]
    a_flat = _as_flat(a)
    b_flat = _as_flat(b)
    n = a_flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    _ensure_lib().cl3_batch_scalar_product_flat(
        a_flat.ctypes.data_as(_FPTR),
        b_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(n))
    return out.reshape(orig_shape) if orig_shape else out


def native_geom_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric matrix multiply: out[i,j] = sum_k gp(a[i,k], b[k,j]).
    a: (M, K, 8), b: (K, N, 8) -> out: (M, N, 8)
    """
    assert a.ndim == 3 and b.ndim == 3 and a.shape[-1] == 8 and b.shape[-1] == 8
    M, K = a.shape[0], a.shape[1]
    K2, N = b.shape[0], b.shape[1]
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    a_flat = np.ascontiguousarray(a, dtype=np.float32)
    b_flat = np.ascontiguousarray(b, dtype=np.float32)
    out = np.empty((M, N, 8), dtype=np.float32)
    _ensure_lib().cl3_geom_matmul_flat(
        a_flat.ctypes.data_as(_FPTR),
        b_flat.ctypes.data_as(_FPTR),
        out.ctypes.data_as(_FPTR),
        ctypes.c_size_t(M),
        ctypes.c_size_t(K),
        ctypes.c_size_t(N))
    return out


# ============================================================
# Backend dispatch: use native if available, else NumPy
# ============================================================

def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dispatch to native kernels, with batched support."""
    if not is_native_available():
        return _numpy_fallback_gp(a, b)
    if a.ndim == 1 and b.ndim == 1:
        return native_geometric_product(a, b)
    # Batched path
    return native_batch_gp(a, b)


def _numpy_fallback_gp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure NumPy geometric product fallback (handles batched)."""
    from rune.types.multivector import PRODUCT_IDX, PRODUCT_SIGN
    result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            idx = PRODUCT_IDX[i, j]
            sign = PRODUCT_SIGN[i, j]
            result[..., idx] += sign * a[..., i] * b[..., j]
    return result
