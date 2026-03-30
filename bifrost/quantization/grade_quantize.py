"""
Grade-aware quantization for Cl(3,0) multivectors.

Each grade is quantized to a different bit-width so that the scalar
(grade 0) keeps high fidelity while the trivector (grade 3) is stored
more cheaply.  Both symmetric and asymmetric modes are supported.

Multivector layout assumed throughout bifrost:
    index 0        -> grade 0  (scalar, 1 component)
    indices 1-3    -> grade 1  (vector, 3 components: e1 e2 e3)
    indices 4-6    -> grade 2  (bivector, 3 components: e12 e13 e23)
    index 7        -> grade 3  (trivector / pseudoscalar, 1 component)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grade-index mapping for Cl(3,0)
# ---------------------------------------------------------------------------
GRADE_SLICES: Dict[int, slice] = {
    0: slice(0, 1),
    1: slice(1, 4),
    2: slice(4, 7),
    3: slice(7, 8),
}

GRADE_WIDTHS: Dict[int, int] = {0: 1, 1: 3, 2: 3, 3: 1}

NUM_COMPONENTS = 8  # total components in Cl(3,0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CliffordQuantConfig:
    """Per-grade bit-width configuration.

    Parameters
    ----------
    grade_bits : dict[int, int]
        Mapping from grade (0-3) to the number of quantization bits.
    symmetric : bool
        If True use symmetric quantization (zero-point = 0).
        If False use asymmetric (zero-point is computed from data range).
    """

    grade_bits: Dict[int, int] = field(
        default_factory=lambda: {0: 8, 1: 6, 2: 6, 3: 4}
    )
    symmetric: bool = True

    def __post_init__(self) -> None:
        for g in range(4):
            if g not in self.grade_bits:
                raise ValueError(f"Missing bit-width for grade {g}")
            b = self.grade_bits[g]
            if not (2 <= b <= 16):
                raise ValueError(
                    f"Bit-width for grade {g} must be in [2, 16], got {b}"
                )


# ---------------------------------------------------------------------------
# Per-grade quantization parameters
# ---------------------------------------------------------------------------
@dataclass
class _GradeQuantParams:
    """Quantization parameters for one grade."""

    scale: np.ndarray       # float64, broadcastable
    zero_point: np.ndarray  # int,     broadcastable
    bits: int
    qmin: int
    qmax: int


def _compute_params(
    data: np.ndarray,
    bits: int,
    symmetric: bool,
    scale_override: Optional[np.ndarray] = None,
    zero_point_override: Optional[np.ndarray] = None,
) -> _GradeQuantParams:
    """Compute scale and zero-point for a grade slice."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    if scale_override is not None:
        scale = scale_override
        zp = zero_point_override if zero_point_override is not None else np.zeros_like(scale, dtype=np.int64)
        return _GradeQuantParams(scale=scale, zero_point=zp, bits=bits, qmin=qmin, qmax=qmax)

    if symmetric:
        abs_max = np.max(np.abs(data)) if data.size > 0 else 1.0
        abs_max = max(abs_max, 1e-12)
        scale = np.array(abs_max / qmax, dtype=np.float64)
        zero_point = np.zeros(1, dtype=np.int64)
    else:
        dmin = np.min(data) if data.size > 0 else 0.0
        dmax = np.max(data) if data.size > 0 else 1.0
        drange = max(dmax - dmin, 1e-12)
        scale = np.array(drange / (qmax - qmin), dtype=np.float64)
        zero_point = np.array(
            np.clip(np.round(qmin - dmin / scale), qmin, qmax), dtype=np.int64
        )

    return _GradeQuantParams(
        scale=scale, zero_point=zero_point, bits=bits, qmin=qmin, qmax=qmax
    )


# ---------------------------------------------------------------------------
# Quantized representation
# ---------------------------------------------------------------------------
@dataclass
class QuantizedMultivector:
    """Quantized multivector with per-grade parameters.

    Attributes
    ----------
    grade_data : dict[int, np.ndarray]
        Integer arrays for each grade.
    grade_params : dict[int, _GradeQuantParams]
        Quantization parameters that allow dequantization.
    original_shape : tuple
        Shape of the original float multivector array.
    """

    grade_data: Dict[int, np.ndarray]
    grade_params: Dict[int, _GradeQuantParams]
    original_shape: Tuple[int, ...]

    # convenience --------------------------------------------------------
    def total_bits(self) -> int:
        """Total storage in bits (payload only, ignoring params overhead)."""
        total = 0
        for g in range(4):
            total += self.grade_data[g].size * self.grade_params[g].bits
        return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def quantize_multivector(
    mv: np.ndarray,
    config: CliffordQuantConfig,
    *,
    calibration_scales: Optional[Dict[int, np.ndarray]] = None,
    calibration_zero_points: Optional[Dict[int, np.ndarray]] = None,
) -> QuantizedMultivector:
    """Quantize a multivector array with per-grade bit-widths.

    Parameters
    ----------
    mv : np.ndarray
        Float array whose *last* axis has size 8 (the Cl(3,0) components).
    config : CliffordQuantConfig
        Bit-width and symmetry settings.
    calibration_scales / calibration_zero_points : optional
        Pre-computed scales (from ``determine_scales``).  If provided the
        quantization parameters are *not* recomputed from ``mv``.

    Returns
    -------
    QuantizedMultivector
    """
    if mv.shape[-1] != NUM_COMPONENTS:
        raise ValueError(
            f"Last axis must be {NUM_COMPONENTS}, got {mv.shape[-1]}"
        )

    mv = np.asarray(mv, dtype=np.float64)

    grade_data: Dict[int, np.ndarray] = {}
    grade_params: Dict[int, _GradeQuantParams] = {}

    for grade, slc in GRADE_SLICES.items():
        grade_slice = mv[..., slc]
        bits = config.grade_bits[grade]

        sc_over = calibration_scales.get(grade) if calibration_scales else None
        zp_over = calibration_zero_points.get(grade) if calibration_zero_points else None

        params = _compute_params(
            grade_slice, bits, config.symmetric,
            scale_override=sc_over, zero_point_override=zp_over,
        )

        # quantize: q = clamp(round(x / scale) + zero_point, qmin, qmax)
        q = np.clip(
            np.round(grade_slice / params.scale) + params.zero_point,
            params.qmin,
            params.qmax,
        ).astype(np.int64)

        grade_data[grade] = q
        grade_params[grade] = params

    return QuantizedMultivector(
        grade_data=grade_data,
        grade_params=grade_params,
        original_shape=mv.shape,
    )


def dequantize_multivector(
    quantized: QuantizedMultivector,
    config: Optional[CliffordQuantConfig] = None,
) -> np.ndarray:
    """Reconstruct a float multivector from its quantized form.

    Parameters
    ----------
    quantized : QuantizedMultivector
    config : CliffordQuantConfig, optional
        Unused but accepted for API symmetry.

    Returns
    -------
    np.ndarray  — float64, same shape as the original.
    """
    out = np.empty(quantized.original_shape, dtype=np.float64)

    for grade, slc in GRADE_SLICES.items():
        p = quantized.grade_params[grade]
        q = quantized.grade_data[grade].astype(np.float64)
        out[..., slc] = (q - p.zero_point) * p.scale

    return out
