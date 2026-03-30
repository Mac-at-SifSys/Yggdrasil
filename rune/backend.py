"""
backend.py — Array backend switcher for YGGDRASIL.

Provides `xp` — the active array library.
- CPU mode: xp = numpy
- GPU mode: xp = cupy

All holograph/forge code uses `from rune.backend import xp` instead of
`import numpy as np`. This keeps tensors on GPU when CuPy is active.

Clifford algebra ops (geometric product, reverse, etc.) still dispatch
through rune.ops.batched -> mjolnir kernels. The xp module handles
everything else: allocation, slicing, einsum, softmax, reductions.
"""

import numpy as _np

# The active array module — starts as numpy
xp = _np

# Track state
_device = 'cpu'


def set_backend(device: str):
    """
    Switch the array backend.

    Args:
        device: 'cpu' for numpy, 'cuda' for cupy
    """
    global xp, _device

    if device == 'cuda':
        try:
            import cupy as cp
            xp = cp
            _device = 'cuda'
        except ImportError:
            raise RuntimeError(
                "CuPy not installed. Install with: pip install cupy-cuda12x"
            )
    elif device == 'cpu':
        xp = _np
        _device = 'cpu'
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'.")


def get_backend() -> str:
    """Return current backend: 'cpu' or 'cuda'."""
    return _device


def to_numpy(arr) -> _np.ndarray:
    """Convert any array (numpy or cupy) to numpy on CPU."""
    if _device == 'cuda':
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    if isinstance(arr, _np.ndarray):
        return arr
    return _np.asarray(arr)


def to_device(arr):
    """Convert numpy array to the active backend."""
    if _device == 'cuda':
        import cupy as cp
        if isinstance(arr, _np.ndarray):
            return cp.asarray(arr)
        return arr  # already cupy
    return arr  # already numpy
