"""
param_utils.py -- Helpers for working with mixed parameter representations.

The original Forge optimizer/training code assumed every parameter was a
``{"mv": Multivector}`` wrapper. Higher-level HLM models in this repo often
expose raw ndarray parameters instead, with gradients stored in Holograph's
global gradient store. These helpers normalize access so Forge can work with
both styles.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from rune.backend import to_numpy, xp
from rune.types.multivector import GRADE_SLICES, Multivector


def _raw_grad_helpers():
    from holograph.layers.clifford_linear import (
        _accumulate_grad,
        _get_grad,
        _zero_grad_params,
    )

    return _accumulate_grad, _get_grad, _zero_grad_params


def get_param_name(param: Any, index: int) -> str:
    if isinstance(param, dict):
        return param.get("name", f"param_{index}")
    return f"param_{index}"


def _param_target(param: Any) -> Any:
    if isinstance(param, dict):
        return param["mv"]
    return param


def get_param_data(param: Any):
    target = _param_target(param)
    if hasattr(target, "_data"):
        return target._data
    return target


def get_param_grad(param: Any):
    target = _param_target(param)
    if hasattr(target, "_grad"):
        grad = target._grad
        if grad is None:
            return None
        return grad._data if hasattr(grad, "_data") else grad

    _, get_grad, _ = _raw_grad_helpers()
    return get_grad(target)


def clear_param_grad(param: Any) -> None:
    target = _param_target(param)
    if hasattr(target, "_grad"):
        target._grad = None
        return

    _, _, zero_grad_params = _raw_grad_helpers()
    zero_grad_params([target])


def assign_param_grad(param: Any, grad) -> None:
    target = _param_target(param)

    if hasattr(target, "_grad"):
        target._grad = Multivector(np.asarray(to_numpy(grad), dtype=np.float32))
        return

    accumulate_grad, _, zero_grad_params = _raw_grad_helpers()
    zero_grad_params([target])
    accumulate_grad(target, xp.asarray(to_numpy(grad), dtype=get_param_data(param).dtype))


def zero_state_like(data):
    state = data.copy()
    state[...] = 0
    return state


def infer_param_grade(data) -> Optional[int]:
    shape = getattr(data, "shape", ())
    if not shape:
        return 0
    if shape[-1] == 8:
        return None
    if shape[-1] == 3:
        return 2
    if shape[-1] == 1:
        return 0
    return None


def grade_scale_for_data(data, grade_values):
    grade = infer_param_grade(data)
    if grade is None and getattr(data, "shape", ()) and data.shape[-1] == 8:
        scale = xp.ones(8, dtype=xp.float32)
        for g, slc in GRADE_SLICES.items():
            scale[slc] = grade_values[g]
        return scale
    if grade is not None:
        return xp.asarray(grade_values[grade], dtype=xp.float32)
    return xp.asarray(1.0, dtype=xp.float32)


def copy_array_(dst, src):
    src_np = np.asarray(to_numpy(src), dtype=dst.dtype)
    if isinstance(dst, np.ndarray):
        dst[...] = src_np
    else:
        dst[...] = xp.asarray(src_np, dtype=dst.dtype)
    return dst
