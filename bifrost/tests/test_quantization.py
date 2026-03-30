"""Tests for grade-aware quantization, calibration, and mixed precision."""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np

# Ensure bifrost is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bifrost.quantization.grade_quantize import (
    CliffordQuantConfig,
    QuantizedMultivector,
    quantize_multivector,
    dequantize_multivector,
    GRADE_SLICES,
    NUM_COMPONENTS,
)
from bifrost.quantization.calibration import (
    CalibrationCollector,
    determine_scales,
)
from bifrost.quantization.mixed_precision import (
    MixedPrecisionConfig,
    apply_mixed_precision,
    estimate_memory_savings,
)


class TestCliffordQuantConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CliffordQuantConfig()
        self.assertEqual(cfg.grade_bits, {0: 8, 1: 6, 2: 6, 3: 4})
        self.assertTrue(cfg.symmetric)

    def test_missing_grade_raises(self):
        with self.assertRaises(ValueError):
            CliffordQuantConfig(grade_bits={0: 8, 1: 6, 2: 6})

    def test_invalid_bits_raises(self):
        with self.assertRaises(ValueError):
            CliffordQuantConfig(grade_bits={0: 1, 1: 6, 2: 6, 3: 4})


class TestQuantizeDequantize(unittest.TestCase):
    def _roundtrip_error(self, mv, config):
        q = quantize_multivector(mv, config)
        recon = dequantize_multivector(q)
        return np.max(np.abs(mv - recon))

    def test_roundtrip_symmetric_default(self):
        rng = np.random.RandomState(42)
        mv = rng.randn(10, 8)
        cfg = CliffordQuantConfig()
        err = self._roundtrip_error(mv, cfg)
        # 4-bit grade-3 is the bottleneck; error should be bounded
        self.assertLess(err, 0.5, f"Roundtrip error too large: {err}")

    def test_roundtrip_asymmetric(self):
        rng = np.random.RandomState(7)
        mv = rng.randn(5, 8) + 2.0  # shifted distribution
        cfg = CliffordQuantConfig(symmetric=False)
        err = self._roundtrip_error(mv, cfg)
        # 4-bit grade-3 on shifted data gives coarser quantization
        self.assertLess(err, 1.0)

    def test_high_precision_scalar(self):
        """Scalar (8-bit) should have smaller error than trivector (4-bit)."""
        rng = np.random.RandomState(99)
        mv = rng.randn(100, 8)
        cfg = CliffordQuantConfig()
        q = quantize_multivector(mv, cfg)
        recon = dequantize_multivector(q)

        scalar_err = np.max(np.abs(mv[..., 0:1] - recon[..., 0:1]))
        trivec_err = np.max(np.abs(mv[..., 7:8] - recon[..., 7:8]))
        # With equal range, 8-bit should be more precise than 4-bit
        self.assertLessEqual(scalar_err, trivec_err + 1e-9)

    def test_quantized_bits_count(self):
        mv = np.ones((4, 8))
        cfg = CliffordQuantConfig()
        q = quantize_multivector(mv, cfg)
        # 4 MVs: grade0=4*1*8, grade1=4*3*6, grade2=4*3*6, grade3=4*1*4
        expected = 4 * (1 * 8 + 3 * 6 + 3 * 6 + 1 * 4)
        self.assertEqual(q.total_bits(), expected)

    def test_batch_shape_preserved(self):
        mv = np.zeros((3, 5, 8))
        cfg = CliffordQuantConfig()
        q = quantize_multivector(mv, cfg)
        recon = dequantize_multivector(q)
        self.assertEqual(recon.shape, (3, 5, 8))

    def test_bad_last_axis_raises(self):
        with self.assertRaises(ValueError):
            quantize_multivector(np.zeros((4, 7)), CliffordQuantConfig())


class TestCalibration(unittest.TestCase):
    def test_collect_and_stats(self):
        collector = CalibrationCollector()
        rng = np.random.RandomState(0)
        for _ in range(10):
            collector.observe(rng.randn(4, 8))
        stats = collector.stats()
        self.assertEqual(set(stats.keys()), {0, 1, 2, 3})
        for g in range(4):
            self.assertIn("min", stats[g])
            self.assertIn("max", stats[g])
            self.assertIn("mean", stats[g])
            self.assertIn("std", stats[g])
            self.assertGreater(stats[g]["count"], 0)

    def test_determine_scales_minmax(self):
        collector = CalibrationCollector()
        rng = np.random.RandomState(1)
        for _ in range(20):
            collector.observe(rng.randn(8, 8))
        stats = collector.stats()
        scales, zps = determine_scales(stats)
        self.assertEqual(set(scales.keys()), {0, 1, 2, 3})
        for g in range(4):
            self.assertGreater(float(scales[g]), 0)

    def test_determine_scales_mean_std(self):
        collector = CalibrationCollector()
        rng = np.random.RandomState(2)
        for _ in range(20):
            collector.observe(rng.randn(8, 8))
        stats = collector.stats()
        scales, _ = determine_scales(stats, method="mean_std")
        for g in range(4):
            self.assertGreater(float(scales[g]), 0)

    def test_calibrated_quantization_improves_error(self):
        """Using calibration scales should give comparable or better error."""
        rng = np.random.RandomState(3)
        data = rng.randn(50, 8)
        cfg = CliffordQuantConfig()

        # Uncalibrated
        q1 = quantize_multivector(data, cfg)
        r1 = dequantize_multivector(q1)
        err1 = np.mean((data - r1) ** 2)

        # Calibrated
        collector = CalibrationCollector()
        collector.observe(data)
        stats = collector.stats()
        scales, zps = determine_scales(stats, cfg)
        q2 = quantize_multivector(
            data, cfg, calibration_scales=scales, calibration_zero_points=zps
        )
        r2 = dequantize_multivector(q2)
        err2 = np.mean((data - r2) ** 2)

        # Calibrated error should be close to or better than uncalibrated
        self.assertLess(err2, err1 * 1.5)

    def test_reset(self):
        collector = CalibrationCollector()
        collector.observe(np.ones((2, 8)))
        collector.reset()
        self.assertEqual(collector.num_observations, 0)


class TestMixedPrecision(unittest.TestCase):
    def _make_weights(self):
        rng = np.random.RandomState(10)
        return {
            "embed": rng.randn(32, 8),
            "attn.q": rng.randn(32, 8),
            "attn.k": rng.randn(32, 8),
            "head": rng.randn(16, 8),
        }

    def test_apply_default(self):
        weights = self._make_weights()
        cfg = MixedPrecisionConfig()
        qmodel = apply_mixed_precision(weights, cfg)
        for name in weights:
            w = qmodel.get_weight(name)
            self.assertEqual(w.shape, weights[name].shape)

    def test_keep_fp32(self):
        weights = self._make_weights()
        cfg = MixedPrecisionConfig(keep_fp32_layers=["head"])
        qmodel = apply_mixed_precision(weights, cfg)
        # head should be exact
        np.testing.assert_array_equal(qmodel.get_weight("head"), weights["head"])

    def test_per_layer_config(self):
        weights = self._make_weights()
        high = CliffordQuantConfig(grade_bits={0: 8, 1: 8, 2: 8, 3: 8})
        cfg = MixedPrecisionConfig(layer_configs={"embed": high})
        qmodel = apply_mixed_precision(weights, cfg)
        # embed uses 8-bit everywhere -> lower error
        embed_err = np.max(np.abs(weights["embed"] - qmodel.get_weight("embed")))
        self.assertLess(embed_err, 0.05)

    def test_estimate_memory_savings(self):
        weights = self._make_weights()
        cfg = MixedPrecisionConfig()
        report = estimate_memory_savings(weights, cfg)
        self.assertGreater(report["compression_ratio"], 1.0)
        self.assertGreater(report["saved_bytes"], 0)
        self.assertEqual(len(report["per_layer"]), len(weights))


if __name__ == "__main__":
    unittest.main()
