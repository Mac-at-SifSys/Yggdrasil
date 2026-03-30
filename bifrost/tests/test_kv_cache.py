"""Tests for Clifford KV cache."""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bifrost.cache.clifford_kv_cache import (
    CliffordKVCache,
    GradeStratifiedCache,
    GRADE_SLICES,
    GRADE_WIDTHS,
    NUM_COMPONENTS,
)


class TestGradeStratifiedCache(unittest.TestCase):
    def test_append_and_get(self):
        cache = GradeStratifiedCache(
            active_grades={0, 2}, num_heads=2, head_dim_mv=4, max_seq=16
        )
        rng = np.random.RandomState(0)
        mv = rng.randn(3, 2, 4, 8).astype(np.float32)
        cache.append(mv)
        self.assertEqual(cache.length, 3)

        # retrieved grades should match input
        grades = cache.get()
        np.testing.assert_allclose(grades[0], mv[..., 0:1], atol=1e-6)
        np.testing.assert_allclose(grades[2], mv[..., 4:7], atol=1e-6)
        self.assertNotIn(1, grades)
        self.assertNotIn(3, grades)

    def test_get_full_reconstructs_zeros_for_missing(self):
        cache = GradeStratifiedCache(
            active_grades={0}, num_heads=1, head_dim_mv=2, max_seq=8
        )
        mv = np.ones((2, 1, 2, 8), dtype=np.float32)
        cache.append(mv)
        full = cache.get_full()
        # grade 0 present
        np.testing.assert_allclose(full[..., 0:1], 1.0)
        # all other grades zero
        np.testing.assert_allclose(full[..., 1:8], 0.0)

    def test_overflow_raises(self):
        cache = GradeStratifiedCache(
            active_grades={0, 1, 2, 3}, num_heads=1, head_dim_mv=1, max_seq=4
        )
        mv = np.zeros((5, 1, 1, 8), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            cache.append(mv)

    def test_reset(self):
        cache = GradeStratifiedCache(
            active_grades={0, 2}, num_heads=1, head_dim_mv=1, max_seq=8
        )
        cache.append(np.ones((2, 1, 1, 8), dtype=np.float32))
        cache.reset()
        self.assertEqual(cache.length, 0)

    def test_memory_bytes(self):
        cache = GradeStratifiedCache(
            active_grades={0, 2}, num_heads=2, head_dim_mv=4, max_seq=16
        )
        # grade 0: 16*2*4*1*4 bytes, grade 2: 16*2*4*3*4 bytes
        expected = 16 * 2 * 4 * 4 * (1 + 3)  # float32 = 4 bytes
        self.assertEqual(cache.memory_bytes(), expected)


class TestCliffordKVCache(unittest.TestCase):
    def _make_cache(self, **kw):
        defaults = dict(
            num_layers=2, num_heads=2, head_dim_mv=4, max_seq=32
        )
        defaults.update(kw)
        return CliffordKVCache(**defaults)

    def test_update_and_get(self):
        cache = self._make_cache()
        rng = np.random.RandomState(1)
        k = rng.randn(5, 2, 4, 8).astype(np.float32)
        v = rng.randn(5, 2, 4, 8).astype(np.float32)

        cached_k, cached_v = cache.update(k, v, layer_idx=0)
        self.assertEqual(cached_k.shape, (5, 2, 4, 8))
        self.assertEqual(cached_v.shape, (5, 2, 4, 8))

        # Values should be fully stored (all grades active)
        np.testing.assert_allclose(cached_v, v, atol=1e-6)

        # Keys: only grades 0 and 2 should be nonzero by default
        np.testing.assert_allclose(cached_k[..., 0:1], k[..., 0:1], atol=1e-6)
        np.testing.assert_allclose(cached_k[..., 4:7], k[..., 4:7], atol=1e-6)
        # grades 1 and 3 should be zero (not cached)
        np.testing.assert_allclose(cached_k[..., 1:4], 0.0)
        np.testing.assert_allclose(cached_k[..., 7:8], 0.0)

    def test_incremental_update(self):
        cache = self._make_cache()
        rng = np.random.RandomState(2)
        k1 = rng.randn(3, 2, 4, 8).astype(np.float32)
        v1 = rng.randn(3, 2, 4, 8).astype(np.float32)
        cache.update(k1, v1, layer_idx=0)

        k2 = rng.randn(2, 2, 4, 8).astype(np.float32)
        v2 = rng.randn(2, 2, 4, 8).astype(np.float32)
        cached_k, cached_v = cache.update(k2, v2, layer_idx=0)

        self.assertEqual(cached_k.shape[0], 5)  # 3 + 2
        self.assertEqual(cached_v.shape[0], 5)
        self.assertEqual(cache.seq_length(0), 5)

    def test_get_with_range(self):
        cache = self._make_cache()
        rng = np.random.RandomState(3)
        k = rng.randn(10, 2, 4, 8).astype(np.float32)
        v = rng.randn(10, 2, 4, 8).astype(np.float32)
        cache.update(k, v, layer_idx=0)

        k_sub, v_sub = cache.get(0, slice(2, 5))
        self.assertEqual(k_sub.shape[0], 3)
        self.assertEqual(v_sub.shape[0], 3)

    def test_multi_layer(self):
        cache = self._make_cache(num_layers=3)
        rng = np.random.RandomState(4)
        for layer in range(3):
            k = rng.randn(4, 2, 4, 8).astype(np.float32)
            v = rng.randn(4, 2, 4, 8).astype(np.float32)
            cache.update(k, v, layer_idx=layer)

        for layer in range(3):
            self.assertEqual(cache.seq_length(layer), 4)

    def test_memory_savings(self):
        cache = self._make_cache()
        savings = cache.memory_savings_ratio()
        # default: keys cache {0,2} = 4 components, values cache all 8
        # full would be 2*8 = 16 per position; actual is 4+8 = 12
        # savings = 1 - 12/16 = 0.25
        self.assertGreater(savings, 0.0)
        self.assertLess(savings, 1.0)

    def test_custom_key_grades(self):
        # Cache only scalar for keys
        cache = self._make_cache(key_active_grades={0})
        savings = cache.memory_savings_ratio()
        # keys: 1 component, values: 8 -> total 9 vs full 16
        self.assertGreater(savings, 0.4)

    def test_reset(self):
        cache = self._make_cache()
        rng = np.random.RandomState(5)
        k = rng.randn(4, 2, 4, 8).astype(np.float32)
        v = rng.randn(4, 2, 4, 8).astype(np.float32)
        cache.update(k, v, layer_idx=0)
        cache.reset()
        self.assertEqual(cache.seq_length(0), 0)


if __name__ == "__main__":
    unittest.main()
