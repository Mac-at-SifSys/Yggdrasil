"""
Clifford KV Cache for serving Clifford-algebra transformers.

Key insight: Clifford attention computes scores from specific grades of
the key multivector (e.g. scalar + bivector).  We therefore only need to
cache those grades for keys, while values are typically consumed in full.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Grade slices (Cl(3,0), 8 components)
GRADE_SLICES: Dict[int, slice] = {
    0: slice(0, 1),
    1: slice(1, 4),
    2: slice(4, 7),
    3: slice(7, 8),
}
GRADE_WIDTHS: Dict[int, int] = {0: 1, 1: 3, 2: 3, 3: 1}
NUM_COMPONENTS = 8


# ---------------------------------------------------------------------------
# Grade-stratified storage
# ---------------------------------------------------------------------------
class GradeStratifiedCache:
    """Store multivector data split by grade for memory efficiency.

    Only the grades listed in *active_grades* are materialised.
    """

    def __init__(
        self,
        active_grades: Set[int],
        num_heads: int,
        head_dim_mv: int,
        max_seq: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.active_grades = active_grades
        self.num_heads = num_heads
        self.head_dim_mv = head_dim_mv  # number of multivectors per head
        self.max_seq = max_seq
        self.dtype = dtype

        # storage: grade -> (max_seq, num_heads, head_dim_mv, grade_width)
        self._buffers: Dict[int, np.ndarray] = {}
        for g in active_grades:
            w = GRADE_WIDTHS[g]
            self._buffers[g] = np.zeros(
                (max_seq, num_heads, head_dim_mv, w), dtype=dtype
            )

        self._length = 0  # current sequence length stored

    # ---- write -----------------------------------------------------------
    def append(self, mv: np.ndarray) -> None:
        """Append a (seq_new, num_heads, head_dim_mv, 8) tensor.

        Only the active grades are extracted and stored.
        """
        seq_new = mv.shape[0]
        if self._length + seq_new > self.max_seq:
            raise RuntimeError(
                f"Cache overflow: {self._length}+{seq_new} > {self.max_seq}"
            )

        for g in self.active_grades:
            slc = GRADE_SLICES[g]
            self._buffers[g][self._length : self._length + seq_new] = mv[
                ..., slc
            ]

        self._length += seq_new

    # ---- read ------------------------------------------------------------
    def get(
        self, seq_range: Optional[slice] = None
    ) -> Dict[int, np.ndarray]:
        """Return cached grades as {grade: array}.

        If *seq_range* is None, returns all cached positions.
        """
        if seq_range is None:
            seq_range = slice(0, self._length)
        return {
            g: self._buffers[g][seq_range].copy() for g in self.active_grades
        }

    def get_full(self, seq_range: Optional[slice] = None) -> np.ndarray:
        """Reconstruct a full 8-component multivector (zeros for missing grades)."""
        if seq_range is None:
            seq_range = slice(0, self._length)
        start, stop, _ = seq_range.indices(self._length)
        length = stop - start
        out = np.zeros(
            (length, self.num_heads, self.head_dim_mv, NUM_COMPONENTS),
            dtype=self.dtype,
        )
        for g in self.active_grades:
            slc = GRADE_SLICES[g]
            out[..., slc] = self._buffers[g][seq_range]
        return out

    # ---- info ------------------------------------------------------------
    @property
    def length(self) -> int:
        return self._length

    def memory_bytes(self) -> int:
        total = 0
        for buf in self._buffers.values():
            total += buf.nbytes
        return total

    def reset(self) -> None:
        self._length = 0
        for g in self._buffers:
            self._buffers[g][:] = 0


# ---------------------------------------------------------------------------
# Main KV Cache
# ---------------------------------------------------------------------------
@dataclass
class _LayerCache:
    keys: GradeStratifiedCache
    values: GradeStratifiedCache


class CliffordKVCache:
    """Multi-layer Clifford KV cache.

    Parameters
    ----------
    num_layers : int
    num_heads : int
    head_dim_mv : int
        Number of multivectors per attention head.
    max_seq : int
    key_active_grades : set[int]
        Grades cached for keys.  Default {0, 2} (scalar + bivector).
    value_active_grades : set[int] or None
        Grades cached for values.  Default all (0-3).
    dtype : np.dtype
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim_mv: int,
        max_seq: int = 2048,
        key_active_grades: Optional[Set[int]] = None,
        value_active_grades: Optional[Set[int]] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim_mv = head_dim_mv
        self.max_seq = max_seq
        self.key_grades = key_active_grades if key_active_grades is not None else {0, 2}
        self.value_grades = value_active_grades if value_active_grades is not None else {0, 1, 2, 3}
        self.dtype = dtype

        self._caches: Dict[int, _LayerCache] = {}
        for layer in range(num_layers):
            k_cache = GradeStratifiedCache(
                self.key_grades, num_heads, head_dim_mv, max_seq, dtype
            )
            v_cache = GradeStratifiedCache(
                self.value_grades, num_heads, head_dim_mv, max_seq, dtype
            )
            self._caches[layer] = _LayerCache(keys=k_cache, values=v_cache)

    # ---- update ----------------------------------------------------------
    def update(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append new key/value tokens and return the full cached K, V.

        Parameters
        ----------
        keys : (seq_new, num_heads, head_dim_mv, 8)
        values : (seq_new, num_heads, head_dim_mv, 8)
        layer_idx : int

        Returns
        -------
        (cached_keys, cached_values) each of shape
        (total_seq, num_heads, head_dim_mv, 8).
        """
        lc = self._caches[layer_idx]
        lc.keys.append(keys)
        lc.values.append(values)
        return lc.keys.get_full(), lc.values.get_full()

    # ---- get -------------------------------------------------------------
    def get(
        self,
        layer_idx: int,
        seq_range: Optional[slice] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve cached K, V for a layer (optionally a subsequence)."""
        lc = self._caches[layer_idx]
        return lc.keys.get_full(seq_range), lc.values.get_full(seq_range)

    # ---- info / reset ----------------------------------------------------
    def seq_length(self, layer_idx: int = 0) -> int:
        return self._caches[layer_idx].keys.length

    def memory_bytes(self) -> int:
        total = 0
        for lc in self._caches.values():
            total += lc.keys.memory_bytes() + lc.values.memory_bytes()
        return total

    def full_grade_memory_bytes(self) -> int:
        """Hypothetical memory if all grades were cached for both K and V."""
        per_position = self.num_heads * self.head_dim_mv * NUM_COMPONENTS
        per_position_bytes = per_position * np.dtype(self.dtype).itemsize
        # 2 for K + V, times all layers
        return self.num_layers * 2 * self.max_seq * per_position_bytes

    def memory_savings_ratio(self) -> float:
        full = self.full_grade_memory_bytes()
        actual = self.memory_bytes()
        if full == 0:
            return 0.0
        return 1.0 - actual / full

    def reset(self) -> None:
        for lc in self._caches.values():
            lc.keys.reset()
            lc.values.reset()
