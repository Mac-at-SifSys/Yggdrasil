"""
mjolnir_cuda_ffi.py — Python bindings to mjolnir CUDA kernels.

All kernel functions operate on device pointers (GPU memory).
Use cuda_malloc/cuda_free/cuda_memcpy for memory management,
or the high-level wrappers that accept numpy arrays and handle transfers.
"""

import ctypes
import os
import glob
import numpy as np
from typing import Optional

_cuda_lib: Optional[ctypes.CDLL] = None  # mjolnir_cuda.dll
_cudart: Optional[ctypes.CDLL] = None    # CUDA runtime

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mjolnir'))


def _maybe_import_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def _is_cupy_array(arr) -> bool:
    cp = _maybe_import_cupy()
    return cp is not None and isinstance(arr, cp.ndarray)

def _load_cuda_library() -> bool:
    """Load mjolnir_cuda shared library."""
    global _cuda_lib
    search_paths = [
        os.path.join(_base, 'build_cuda', 'Release', 'mjolnir_cuda.dll'),
        os.path.join(_base, 'build_cuda', 'Debug', 'mjolnir_cuda.dll'),
        os.path.join(_base, 'build_cuda', 'mjolnir_cuda.dll'),
        os.path.join(_base, 'build_cuda', 'libmjolnir_cuda.so'),
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                _cuda_lib = ctypes.CDLL(path)
                _setup_cuda_prototypes()
                return True
            except OSError:
                continue
    return False

def _load_cudart() -> bool:
    """Load CUDA runtime library for memory management."""
    global _cudart
    import platform
    is_windows = platform.system() == 'Windows'

    search = []

    if is_windows:
        cuda_path = os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1')
        search += glob.glob(os.path.join(cuda_path, 'bin', 'cudart64_*.dll'))
        search += glob.glob(os.path.join(cuda_path, 'bin', 'x64', 'cudart64_*.dll'))
        search += ['cudart64_131.dll', 'cudart64_13.dll', 'cudart64_12.dll']
    else:
        # Linux: libcudart.so is typically in /usr/local/cuda/lib64/ or linked by the driver
        cuda_homes = [
            os.environ.get('CUDA_HOME', '/usr/local/cuda'),
            '/usr/local/cuda',
            '/usr/local/cuda-12',
            '/usr/local/cuda-11',
        ]
        for cuda_home in cuda_homes:
            search += glob.glob(os.path.join(cuda_home, 'lib64', 'libcudart.so*'))
            search += glob.glob(os.path.join(cuda_home, 'lib', 'libcudart.so*'))
        search += glob.glob('/usr/lib/x86_64-linux-gnu/libcudart.so*')
        search += glob.glob('/usr/lib64/libcudart.so*')
        # Try bare name — let ldconfig find it
        search += ['libcudart.so', 'libcudart.so.12', 'libcudart.so.11']

    for path in search:
        try:
            _cudart = ctypes.CDLL(path)
            _setup_cudart_prototypes()
            return True
        except OSError:
            continue
    return False

def _setup_cudart_prototypes():
    """Set up CUDA runtime function signatures."""
    # cudaMalloc(void **devPtr, size_t size)
    _cudart.cudaMalloc.restype = ctypes.c_int
    _cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    # cudaFree(void *devPtr)
    _cudart.cudaFree.restype = ctypes.c_int
    _cudart.cudaFree.argtypes = [ctypes.c_void_p]

    # cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    _cudart.cudaMemcpy.restype = ctypes.c_int
    _cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

_FP = ctypes.POINTER(ctypes.c_float)  # float*

def _setup_cuda_prototypes():
    """Set up mjolnir CUDA kernel function signatures."""
    if _cuda_lib is None:
        return

    # cl3_batch_gp_launch(const float*, const float*, float*, int, cudaStream_t)
    _cuda_lib.cl3_batch_gp_launch.restype = None
    _cuda_lib.cl3_batch_gp_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_batch_sandwich_launch.restype = None
    _cuda_lib.cl3_batch_sandwich_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_batch_bvexp_launch.restype = None
    _cuda_lib.cl3_batch_bvexp_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_batch_grade_proj_launch.restype = None
    _cuda_lib.cl3_batch_grade_proj_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_batch_norm_launch.restype = None
    _cuda_lib.cl3_batch_norm_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_geom_matmul_launch.restype = None
    _cuda_lib.cl3_geom_matmul_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_normalize_launch.restype = None
    _cuda_lib.cl3_normalize_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    _cuda_lib.cl3_fused_gp_grade_launch.restype = None
    _cuda_lib.cl3_fused_gp_grade_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

# ========================================================================
# CUDA memory management constants
# ========================================================================
CUDA_MEMCPY_H2D = 1  # cudaMemcpyHostToDevice
CUDA_MEMCPY_D2H = 2  # cudaMemcpyDeviceToHost
CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice

# ========================================================================
# Public API
# ========================================================================

def is_cuda_available() -> bool:
    """Check if CUDA mjolnir kernels are available."""
    if _cuda_lib is None:
        _load_cuda_library()
    if _cudart is None:
        _load_cudart()
    return _cuda_lib is not None and _cudart is not None

def cuda_malloc(nbytes: int) -> int:
    """Allocate GPU memory. Returns device pointer as int."""
    if not is_cuda_available():
        raise RuntimeError("CUDA not available")
    ptr = ctypes.c_void_p()
    err = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed with error {err}")
    return ptr.value

def cuda_free(ptr: int):
    """Free GPU memory."""
    if _cudart is not None:
        _cudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_memcpy_h2d(dst_ptr: int, src_array: np.ndarray):
    """Copy numpy array to GPU."""
    src = np.ascontiguousarray(src_array, dtype=np.float32)
    nbytes = src.nbytes
    err = _cudart.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(nbytes),
        CUDA_MEMCPY_H2D)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed: {err}")

def cuda_memcpy_d2h(dst_array: np.ndarray, src_ptr: int):
    """Copy GPU memory to numpy array."""
    nbytes = dst_array.nbytes
    err = _cudart.cudaMemcpy(
        dst_array.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes),
        CUDA_MEMCPY_D2H)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed: {err}")

def to_device(arr: np.ndarray) -> int:
    """Upload numpy array to GPU. Returns device pointer."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ptr = cuda_malloc(arr.nbytes)
    cuda_memcpy_h2d(ptr, arr)
    return ptr

def to_host(ptr: int, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Download GPU array to numpy. Does NOT free the device memory."""
    arr = np.empty(shape, dtype=dtype)
    cuda_memcpy_d2h(arr, ptr)
    return arr


def _stream_ptr(stream=None):
    if stream is None:
        return NULL_STREAM
    if isinstance(stream, ctypes.c_void_p):
        return stream
    return ctypes.c_void_p(int(stream))


def _as_cupy_f32(arr, cp):
    if not isinstance(arr, cp.ndarray):
        return cp.asarray(arr, dtype=cp.float32)
    if arr.dtype != cp.float32:
        arr = arr.astype(cp.float32)
    if not arr.flags.c_contiguous:
        arr = cp.ascontiguousarray(arr)
    return arr

# ========================================================================
# Kernel wrappers (device pointer level)
# ========================================================================

NULL_STREAM = ctypes.c_void_p(0)

def cuda_batch_gp(d_a: int, d_b: int, d_out: int, N: int, stream=None):
    _cuda_lib.cl3_batch_gp_launch(
        ctypes.c_void_p(d_a),
        ctypes.c_void_p(d_b),
        ctypes.c_void_p(d_out),
        N,
        _stream_ptr(stream),
    )

def cuda_batch_sandwich(d_r: int, d_x: int, d_out: int, N: int, stream=None):
    _cuda_lib.cl3_batch_sandwich_launch(
        ctypes.c_void_p(d_r),
        ctypes.c_void_p(d_x),
        ctypes.c_void_p(d_out),
        N,
        _stream_ptr(stream),
    )

def cuda_batch_bvexp(d_bv: int, d_out: int, N: int, stream=None):
    _cuda_lib.cl3_batch_bvexp_launch(
        ctypes.c_void_p(d_bv),
        ctypes.c_void_p(d_out),
        N,
        _stream_ptr(stream),
    )

def cuda_batch_grade_proj(d_in: int, d_out: int, grade: int, N: int, stream=None):
    _cuda_lib.cl3_batch_grade_proj_launch(
        ctypes.c_void_p(d_in),
        ctypes.c_void_p(d_out),
        grade,
        N,
        _stream_ptr(stream),
    )

def cuda_batch_norm(d_in: int, d_out: int, N: int, stream=None):
    _cuda_lib.cl3_batch_norm_launch(
        ctypes.c_void_p(d_in),
        ctypes.c_void_p(d_out),
        N,
        _stream_ptr(stream),
    )

def cuda_geom_matmul(d_a: int, d_b: int, d_out: int, M: int, K: int, N: int, stream=None):
    _cuda_lib.cl3_geom_matmul_launch(
        ctypes.c_void_p(d_a),
        ctypes.c_void_p(d_b),
        ctypes.c_void_p(d_out),
        M,
        K,
        N,
        _stream_ptr(stream),
    )

def cuda_normalize(d_in: int, d_out: int, N: int, stream=None):
    _cuda_lib.cl3_normalize_launch(
        ctypes.c_void_p(d_in),
        ctypes.c_void_p(d_out),
        N,
        _stream_ptr(stream),
    )

def cuda_fused_gp_grade(d_a: int, d_b: int, d_out: int, grade: int, N: int, stream=None):
    _cuda_lib.cl3_fused_gp_grade_launch(
        ctypes.c_void_p(d_a),
        ctypes.c_void_p(d_b),
        ctypes.c_void_p(d_out),
        grade,
        N,
        _stream_ptr(stream),
    )

# ========================================================================
# High-level wrappers (numpy in, numpy out, handles GPU transfers)
# ========================================================================

def gpu_batch_gp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched geometric product on GPU. a, b: (..., 8) -> (..., 8)."""
    if _is_cupy_array(a) or _is_cupy_array(b):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        b = _as_cupy_f32(b, cp)
        orig_shape = a.shape
        a_flat = cp.ascontiguousarray(a.reshape(-1, 8))
        b_flat = cp.ascontiguousarray(b.reshape(-1, 8))
        N = a_flat.shape[0]
        out = cp.empty((N, 8), dtype=cp.float32)
        cuda_batch_gp(
            a_flat.data.ptr,
            b_flat.data.ptr,
            out.data.ptr,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = a.shape
    a_flat = np.ascontiguousarray(a.reshape(-1, 8), dtype=np.float32)
    b_flat = np.ascontiguousarray(b.reshape(-1, 8), dtype=np.float32)
    N = a_flat.shape[0]

    d_a = to_device(a_flat)
    d_b = to_device(b_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_batch_gp(d_a, d_b, d_out, N)

    out = to_host(d_out, (N, 8))
    cuda_free(d_a)
    cuda_free(d_b)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_batch_sandwich(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Batched sandwich product on GPU. r, x: (..., 8) -> (..., 8)."""
    if _is_cupy_array(r) or _is_cupy_array(x):
        cp = _maybe_import_cupy()
        r = _as_cupy_f32(r, cp)
        x = _as_cupy_f32(x, cp)
        orig_shape = r.shape
        r_flat = cp.ascontiguousarray(r.reshape(-1, 8))
        x_flat = cp.ascontiguousarray(x.reshape(-1, 8))
        N = r_flat.shape[0]
        out = cp.empty((N, 8), dtype=cp.float32)
        cuda_batch_sandwich(
            r_flat.data.ptr,
            x_flat.data.ptr,
            out.data.ptr,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = r.shape
    r_flat = np.ascontiguousarray(r.reshape(-1, 8), dtype=np.float32)
    x_flat = np.ascontiguousarray(x.reshape(-1, 8), dtype=np.float32)
    N = r_flat.shape[0]

    d_r = to_device(r_flat)
    d_x = to_device(x_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_batch_sandwich(d_r, d_x, d_out, N)

    out = to_host(d_out, (N, 8))
    cuda_free(d_r)
    cuda_free(d_x)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_batch_bvexp(bv: np.ndarray) -> np.ndarray:
    """Batched bivector exp on GPU. bv: (..., 8) -> (..., 8)."""
    if _is_cupy_array(bv):
        cp = _maybe_import_cupy()
        bv = _as_cupy_f32(bv, cp)
        orig_shape = bv.shape
        bv_flat = cp.ascontiguousarray(bv.reshape(-1, 8))
        N = bv_flat.shape[0]
        out = cp.empty((N, 8), dtype=cp.float32)
        cuda_batch_bvexp(
            bv_flat.data.ptr,
            out.data.ptr,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = bv.shape
    bv_flat = np.ascontiguousarray(bv.reshape(-1, 8), dtype=np.float32)
    N = bv_flat.shape[0]

    d_bv = to_device(bv_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_batch_bvexp(d_bv, d_out, N)

    out = to_host(d_out, (N, 8))
    cuda_free(d_bv)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_batch_grade_proj(a: np.ndarray, grade: int) -> np.ndarray:
    """Batched grade projection on GPU. a: (..., 8), grade: 0-3 -> (..., 8)."""
    if _is_cupy_array(a):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        orig_shape = a.shape
        a_flat = cp.ascontiguousarray(a.reshape(-1, 8))
        N = a_flat.shape[0]
        out = cp.empty((N, 8), dtype=cp.float32)
        cuda_batch_grade_proj(
            a_flat.data.ptr,
            out.data.ptr,
            grade,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = a.shape
    a_flat = np.ascontiguousarray(a.reshape(-1, 8), dtype=np.float32)
    N = a_flat.shape[0]

    d_in = to_device(a_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_batch_grade_proj(d_in, d_out, grade, N)

    out = to_host(d_out, (N, 8))
    cuda_free(d_in)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_batch_norm(a: np.ndarray) -> np.ndarray:
    """Batched norm on GPU. a: (..., 8) -> (...) float."""
    if _is_cupy_array(a):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        orig_shape = a.shape[:-1]
        a_flat = cp.ascontiguousarray(a.reshape(-1, 8))
        N = a_flat.shape[0]
        out = cp.empty((N,), dtype=cp.float32)
        cuda_batch_norm(
            a_flat.data.ptr,
            out.data.ptr,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = a.shape[:-1]
    a_flat = np.ascontiguousarray(a.reshape(-1, 8), dtype=np.float32)
    N = a_flat.shape[0]

    d_a = to_device(a_flat)
    d_out = cuda_malloc(N * 4)

    cuda_batch_norm(d_a, d_out, N)

    out = to_host(d_out, (N,))
    cuda_free(d_a)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_batch_normalize(a: np.ndarray) -> np.ndarray:
    """Batched normalize on GPU. a: (..., 8) -> (..., 8) unit multivectors."""
    if _is_cupy_array(a):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        orig_shape = a.shape
        a_flat = cp.ascontiguousarray(a.reshape(-1, 8))
        N = a_flat.shape[0]
        out = cp.empty((N, 8), dtype=cp.float32)
        cuda_normalize(
            a_flat.data.ptr,
            out.data.ptr,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out.reshape(orig_shape)

    orig_shape = a.shape
    a_flat = np.ascontiguousarray(a.reshape(-1, 8), dtype=np.float32)
    N = a_flat.shape[0]

    d_in = to_device(a_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_normalize(d_in, d_out, N)

    out = to_host(d_out, (N, 8))
    cuda_free(d_in)
    cuda_free(d_out)
    return out.reshape(orig_shape)

def gpu_geom_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric matmul on GPU. a: (M,K,8), b: (K,N,8) -> (M,N,8)."""
    if _is_cupy_array(a) or _is_cupy_array(b):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        b = _as_cupy_f32(b, cp)
        M, K = a.shape[0], a.shape[1]
        N = b.shape[1]
        a_flat = cp.ascontiguousarray(a)
        b_flat = cp.ascontiguousarray(b)
        out = cp.empty((M, N, 8), dtype=cp.float32)
        cuda_geom_matmul(
            a_flat.data.ptr,
            b_flat.data.ptr,
            out.data.ptr,
            M,
            K,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out

    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    a_flat = np.ascontiguousarray(a, dtype=np.float32)
    b_flat = np.ascontiguousarray(b, dtype=np.float32)

    d_a = to_device(a_flat)
    d_b = to_device(b_flat)
    d_out = cuda_malloc(M * N * 8 * 4)

    cuda_geom_matmul(d_a, d_b, d_out, M, K, N)

    out = to_host(d_out, (M, N, 8))
    cuda_free(d_a)
    cuda_free(d_b)
    cuda_free(d_out)
    return out


def gpu_batch_scalar_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scalar part of the geometric product on GPU. a, b: (..., 8) -> (...)."""
    if _is_cupy_array(a) or _is_cupy_array(b):
        cp = _maybe_import_cupy()
        a = _as_cupy_f32(a, cp)
        b = _as_cupy_f32(b, cp)
        orig_shape = a.shape[:-1]
        a_flat = cp.ascontiguousarray(a.reshape(-1, 8))
        b_flat = cp.ascontiguousarray(b.reshape(-1, 8))
        N = a_flat.shape[0]
        out_mv = cp.empty((N, 8), dtype=cp.float32)
        cuda_fused_gp_grade(
            a_flat.data.ptr,
            b_flat.data.ptr,
            out_mv.data.ptr,
            0,
            N,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out_mv[:, 0].reshape(orig_shape)

    orig_shape = a.shape[:-1]
    a_flat = np.ascontiguousarray(a.reshape(-1, 8), dtype=np.float32)
    b_flat = np.ascontiguousarray(b.reshape(-1, 8), dtype=np.float32)
    N = a_flat.shape[0]
    d_a = to_device(a_flat)
    d_b = to_device(b_flat)
    d_out = cuda_malloc(N * 8 * 4)

    cuda_fused_gp_grade(d_a, d_b, d_out, 0, N)

    out_mv = to_host(d_out, (N, 8))
    cuda_free(d_a)
    cuda_free(d_b)
    cuda_free(d_out)
    return out_mv[:, 0].reshape(orig_shape)
