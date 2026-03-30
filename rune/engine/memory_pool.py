"""
memory_pool.py — GPU memory pool for the persistent CUDA engine.

Pre-allocates all buffers needed for a model: parameters, activations,
gradients, optimizer state, and scratch space. Each buffer gets a name
and a GPU pointer. The ProgramBuilder uses these pointers to construct
command sequences.
"""

import ctypes
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict


# CUDA memory management constants
CUDA_MEMCPY_H2D = 1
CUDA_MEMCPY_D2H = 2
CUDA_MEMCPY_D2D = 3

# Alignment for all allocations (16 bytes for float4 loads)
ALIGNMENT = 256  # Use 256 for cache-line + texture alignment


def _align_up(size: int, alignment: int) -> int:
    """Round up size to the next multiple of alignment."""
    return (size + alignment - 1) & ~(alignment - 1)


class BufferInfo:
    """Metadata for a named GPU buffer."""
    __slots__ = ('name', 'offset', 'size_bytes', 'shape', 'dtype', 'n_floats')

    def __init__(self, name: str, offset: int, size_bytes: int,
                 shape: tuple, dtype: np.dtype, n_floats: int):
        self.name = name
        self.offset = offset
        self.size_bytes = size_bytes
        self.shape = shape
        self.dtype = dtype
        self.n_floats = n_floats

    def __repr__(self):
        size_kb = self.size_bytes / 1024
        return (f"Buffer('{self.name}', shape={self.shape}, "
                f"offset=0x{self.offset:x}, size={size_kb:.1f}KB)")


class MemoryPool:
    """
    GPU memory pool with named buffer allocation.

    Usage:
        pool = MemoryPool(cudart_lib)
        pool.alloc('embed_weight', (vocab_size, d_model, 8))
        pool.alloc('layer0.ln.gamma', (d_model, 8))
        pool.alloc('act.hidden', (batch_size, seq_len, d_model, 8))
        pool.finalize()  # Actually allocates on GPU

        # Get device pointers
        ptr = pool.ptr('embed_weight')

        # Upload/download
        pool.upload('embed_weight', numpy_array)
        array = pool.download('embed_weight')
    """

    def __init__(self, cudart=None):
        """
        Args:
            cudart: CUDA runtime library (ctypes.CDLL). If None, will try
                    to load it from the standard locations.
        """
        self._cudart = cudart
        self._buffers: OrderedDict[str, BufferInfo] = OrderedDict()
        self._base_ptr: int = 0
        self._total_bytes: int = 0
        self._finalized: bool = False
        self._current_offset: int = 0

        if self._cudart is None:
            self._load_cudart()

    def _load_cudart(self):
        """Load CUDA runtime for memory management."""
        import os, glob, platform

        is_windows = platform.system() == 'Windows'
        search = []

        if is_windows:
            cuda_path = os.environ.get(
                'CUDA_PATH',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
            )
            search += glob.glob(os.path.join(cuda_path, 'bin', 'cudart64_*.dll'))
            search += ['cudart64_131.dll', 'cudart64_13.dll', 'cudart64_12.dll']
        else:
            cuda_homes = [
                os.environ.get('CUDA_HOME', '/usr/local/cuda'),
                '/usr/local/cuda',
            ]
            for ch in cuda_homes:
                search += glob.glob(os.path.join(ch, 'lib64', 'libcudart.so*'))
            search += ['libcudart.so']

        for path in search:
            try:
                self._cudart = ctypes.CDLL(path)
                self._setup_cudart()
                return
            except OSError:
                continue

        raise RuntimeError("Could not load CUDA runtime library")

    def _setup_cudart(self):
        """Set up CUDA runtime function signatures."""
        self._cudart.cudaMalloc.restype = ctypes.c_int
        self._cudart.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
        ]
        self._cudart.cudaFree.restype = ctypes.c_int
        self._cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self._cudart.cudaMemcpy.restype = ctypes.c_int
        self._cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self._cudart.cudaMemset.restype = ctypes.c_int
        self._cudart.cudaMemset.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        ]

    # ---- Allocation API ----

    def alloc(self, name: str, shape: tuple, dtype=np.float32) -> 'MemoryPool':
        """
        Register a named buffer. Does NOT allocate GPU memory yet.
        Call finalize() after all alloc() calls.

        Args:
            name: Unique buffer name.
            shape: Tensor shape (e.g. (batch, seq, d_model, 8)).
            dtype: numpy dtype (default float32).

        Returns self for chaining.
        """
        if self._finalized:
            raise RuntimeError("Pool already finalized. Cannot add more buffers.")
        if name in self._buffers:
            raise ValueError(f"Buffer '{name}' already exists")

        n_elements = int(np.prod(shape))
        element_size = np.dtype(dtype).itemsize
        size_bytes = n_elements * element_size
        aligned_size = _align_up(size_bytes, ALIGNMENT)

        n_floats = n_elements if dtype == np.float32 else size_bytes // 4

        info = BufferInfo(
            name=name,
            offset=self._current_offset,
            size_bytes=aligned_size,
            shape=shape,
            dtype=np.dtype(dtype),
            n_floats=n_floats,
        )
        self._buffers[name] = info
        self._current_offset += aligned_size
        return self

    def alloc_like(self, name: str, source_name: str) -> 'MemoryPool':
        """Allocate a buffer with the same shape/dtype as an existing one."""
        src = self._buffers[source_name]
        return self.alloc(name, src.shape, src.dtype)

    def alloc_int(self, name: str, shape: tuple) -> 'MemoryPool':
        """Allocate a buffer of int32 (for token IDs, targets, etc.)."""
        return self.alloc(name, shape, dtype=np.int32)

    def alloc_scalar(self, name: str, n: int = 1) -> 'MemoryPool':
        """Allocate a small float buffer (for loss, grad_norm, etc.)."""
        return self.alloc(name, (n,), dtype=np.float32)

    # ---- Finalization ----

    def finalize(self) -> int:
        """
        Allocate the entire pool as one contiguous GPU buffer.
        Returns total bytes allocated.
        """
        if self._finalized:
            return self._total_bytes

        self._total_bytes = self._current_offset
        if self._total_bytes == 0:
            self._finalized = True
            return 0

        # Allocate on GPU
        ptr = ctypes.c_void_p()
        err = self._cudart.cudaMalloc(
            ctypes.byref(ptr), ctypes.c_size_t(self._total_bytes)
        )
        if err != 0:
            raise RuntimeError(
                f"cudaMalloc failed (error {err}) for {self._total_bytes} bytes "
                f"({self._total_bytes / 1024**2:.1f} MB)"
            )
        self._base_ptr = ptr.value

        # Zero the entire pool
        self._cudart.cudaMemset(
            ctypes.c_void_p(self._base_ptr), 0, ctypes.c_size_t(self._total_bytes)
        )

        self._finalized = True
        return self._total_bytes

    # ---- Pointer access ----

    def ptr(self, name: str) -> int:
        """Get the device pointer for a named buffer."""
        if not self._finalized:
            raise RuntimeError("Pool not finalized. Call finalize() first.")
        info = self._buffers[name]
        return self._base_ptr + info.offset

    def info(self, name: str) -> BufferInfo:
        """Get metadata for a named buffer."""
        return self._buffers[name]

    def n_floats(self, name: str) -> int:
        """Get the number of float32 elements in a buffer."""
        return self._buffers[name].n_floats

    def shape(self, name: str) -> tuple:
        """Get the shape of a named buffer."""
        return self._buffers[name].shape

    # ---- Data transfer ----

    def upload(self, name: str, data: np.ndarray):
        """Upload a numpy array to the named GPU buffer."""
        if not self._finalized:
            raise RuntimeError("Pool not finalized")
        info = self._buffers[name]
        data = np.ascontiguousarray(data, dtype=info.dtype)
        if data.nbytes > info.size_bytes:
            raise ValueError(
                f"Data ({data.nbytes} bytes) exceeds buffer '{name}' "
                f"({info.size_bytes} bytes)"
            )
        dst = self._base_ptr + info.offset
        err = self._cudart.cudaMemcpy(
            ctypes.c_void_p(dst),
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(data.nbytes),
            CUDA_MEMCPY_H2D,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed for '{name}': error {err}")

    def upload_int(self, name: str, data: np.ndarray):
        """Upload int32 data to a named GPU buffer."""
        if not self._finalized:
            raise RuntimeError("Pool not finalized")
        info = self._buffers[name]
        data = np.ascontiguousarray(data, dtype=np.int32)
        dst = self._base_ptr + info.offset
        err = self._cudart.cudaMemcpy(
            ctypes.c_void_p(dst),
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(data.nbytes),
            CUDA_MEMCPY_H2D,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D (int) failed for '{name}': error {err}")

    def download(self, name: str) -> np.ndarray:
        """Download a named GPU buffer to numpy."""
        if not self._finalized:
            raise RuntimeError("Pool not finalized")
        info = self._buffers[name]
        data = np.empty(info.shape, dtype=info.dtype)
        src = self._base_ptr + info.offset
        err = self._cudart.cudaMemcpy(
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(src),
            ctypes.c_size_t(data.nbytes),
            CUDA_MEMCPY_D2H,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed for '{name}': error {err}")
        return data

    def download_scalar(self, name: str) -> float:
        """Download a single float from GPU."""
        arr = self.download(name)
        return float(arr.flat[0])

    def zero_buffer(self, name: str):
        """Zero out a named GPU buffer."""
        if not self._finalized:
            raise RuntimeError("Pool not finalized")
        info = self._buffers[name]
        self._cudart.cudaMemset(
            ctypes.c_void_p(self._base_ptr + info.offset),
            0,
            ctypes.c_size_t(info.size_bytes),
        )

    # ---- Query ----

    def has(self, name: str) -> bool:
        return name in self._buffers

    def names(self) -> List[str]:
        return list(self._buffers.keys())

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def total_mb(self) -> float:
        return self._total_bytes / (1024 * 1024)

    def summary(self) -> str:
        """Human-readable summary of all buffers."""
        lines = [f"MemoryPool: {self.total_mb:.2f} MB total, "
                 f"{len(self._buffers)} buffers"]
        for name, info in self._buffers.items():
            size_kb = info.size_bytes / 1024
            lines.append(f"  {name:40s} {str(info.shape):30s} {size_kb:10.1f} KB")
        return "\n".join(lines)

    # ---- Cleanup ----

    def free(self):
        """Free the GPU memory pool."""
        if self._base_ptr and self._cudart is not None:
            self._cudart.cudaFree(ctypes.c_void_p(self._base_ptr))
            self._base_ptr = 0
            self._finalized = False

    def __del__(self):
        self.free()

    def __repr__(self):
        status = "finalized" if self._finalized else "not finalized"
        return f"MemoryPool({len(self._buffers)} buffers, {self.total_mb:.2f} MB, {status})"
