"""
Quantization calibration for Clifford multivectors.

Collects activation statistics over a calibration dataset then computes
per-grade quantization scales that minimise reconstruction error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grade_quantize import (
    GRADE_SLICES,
    GRADE_WIDTHS,
    NUM_COMPONENTS,
    CliffordQuantConfig,
)


# ---------------------------------------------------------------------------
# Per-grade running statistics
# ---------------------------------------------------------------------------
@dataclass
class _GradeStats:
    """Running min / max / Welford mean+var for one grade."""

    count: int = 0
    running_min: float = float("inf")
    running_max: float = float("-inf")
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations (for Welford)

    def update(self, values: np.ndarray) -> None:
        flat = values.ravel().astype(np.float64)
        for v in flat:
            self.count += 1
            self.running_min = min(self.running_min, v)
            self.running_max = max(self.running_max, v)
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def update_batch(self, values: np.ndarray) -> None:
        """Fast batch update — preferred over ``update`` for large arrays."""
        flat = values.ravel().astype(np.float64)
        n = flat.size
        if n == 0:
            return
        batch_min = float(np.min(flat))
        batch_max = float(np.max(flat))
        self.running_min = min(self.running_min, batch_min)
        self.running_max = max(self.running_max, batch_max)

        batch_mean = float(np.mean(flat))
        batch_var = float(np.var(flat))
        batch_m2 = batch_var * n

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + n
            delta = batch_mean - self.mean
            self.mean = (self.mean * self.count + batch_mean * n) / total
            self.m2 += batch_m2 + delta * delta * (self.count * n) / total
            self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def summary(self) -> Dict[str, float]:
        return {
            "min": self.running_min,
            "max": self.running_max,
            "mean": self.mean,
            "std": self.std,
            "count": float(self.count),
        }


# ---------------------------------------------------------------------------
# Public collector
# ---------------------------------------------------------------------------
class CalibrationCollector:
    """Collect per-grade activation statistics during forward passes.

    Usage
    -----
    >>> collector = CalibrationCollector()
    >>> for batch in calibration_loader:
    ...     activations = model.forward(batch)   # (..., 8)
    ...     collector.observe(activations)
    >>> stats = collector.stats()
    >>> scales = determine_scales(stats, config)
    """

    def __init__(self) -> None:
        self._grade_stats: Dict[int, _GradeStats] = {
            g: _GradeStats() for g in range(4)
        }
        self._num_observations: int = 0

    # ---- collect ---------------------------------------------------------
    def observe(self, activations: np.ndarray) -> None:
        """Feed an activation tensor whose last axis is size 8."""
        activations = np.asarray(activations, dtype=np.float64)
        if activations.shape[-1] != NUM_COMPONENTS:
            raise ValueError(
                f"Last axis must be {NUM_COMPONENTS}, got {activations.shape[-1]}"
            )

        for grade, slc in GRADE_SLICES.items():
            grade_data = activations[..., slc]
            self._grade_stats[grade].update_batch(grade_data)

        self._num_observations += 1

    # ---- query -----------------------------------------------------------
    def stats(self) -> Dict[int, Dict[str, float]]:
        """Return per-grade summary statistics."""
        return {g: s.summary() for g, s in self._grade_stats.items()}

    @property
    def num_observations(self) -> int:
        return self._num_observations

    def reset(self) -> None:
        self._grade_stats = {g: _GradeStats() for g in range(4)}
        self._num_observations = 0


# ---------------------------------------------------------------------------
# Scale determination
# ---------------------------------------------------------------------------
def determine_scales(
    stats: Dict[int, Dict[str, float]],
    config: Optional[CliffordQuantConfig] = None,
    *,
    method: str = "minmax",
    percentile_clip: float = 1.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute per-grade quantization scales from collected statistics.

    Parameters
    ----------
    stats : dict
        Output of ``CalibrationCollector.stats()``.
    config : CliffordQuantConfig, optional
        Defaults to ``CliffordQuantConfig()`` if not provided.
    method : str
        ``"minmax"`` — use observed min/max.
        ``"mean_std"`` — use mean +/- 3*std for the range (clipped).
    percentile_clip : float
        Fraction of the full range to keep (1.0 = no clipping).

    Returns
    -------
    (scales, zero_points) — both ``Dict[int, np.ndarray]``.
    """
    if config is None:
        config = CliffordQuantConfig()

    scales: Dict[int, np.ndarray] = {}
    zero_points: Dict[int, np.ndarray] = {}

    for grade in range(4):
        gs = stats[grade]
        bits = config.grade_bits[grade]
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        if method == "minmax":
            dmin = gs["min"]
            dmax = gs["max"]
        elif method == "mean_std":
            dmin = gs["mean"] - 3.0 * gs["std"]
            dmax = gs["mean"] + 3.0 * gs["std"]
        else:
            raise ValueError(f"Unknown method: {method}")

        # percentile clipping
        if percentile_clip < 1.0:
            centre = (dmax + dmin) / 2.0
            half = (dmax - dmin) / 2.0 * percentile_clip
            dmin = centre - half
            dmax = centre + half

        if config.symmetric:
            abs_max = max(abs(dmin), abs(dmax), 1e-12)
            scale = abs_max / qmax
            zp = np.zeros(1, dtype=np.int64)
        else:
            drange = max(dmax - dmin, 1e-12)
            scale = drange / (qmax - qmin)
            zp = np.array(
                np.clip(np.round(qmin - dmin / scale), qmin, qmax),
                dtype=np.int64,
            )

        scales[grade] = np.array(scale, dtype=np.float64)
        zero_points[grade] = zp

    return scales, zero_points
