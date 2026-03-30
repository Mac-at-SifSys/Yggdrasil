"""
Mixed-precision inference for Clifford HLM models.

Allows different layers and grades to use different quantization
bit-widths, and provides utilities to estimate the memory savings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grade_quantize import (
    GRADE_SLICES,
    GRADE_WIDTHS,
    NUM_COMPONENTS,
    CliffordQuantConfig,
    QuantizedMultivector,
    quantize_multivector,
    dequantize_multivector,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MixedPrecisionConfig:
    """Per-layer, per-grade precision map.

    Parameters
    ----------
    default_config : CliffordQuantConfig
        Fallback config for layers not explicitly listed.
    layer_configs : dict[str, CliffordQuantConfig]
        Layer-name -> grade-bits override.
    keep_fp32_layers : list[str]
        Layer names that should remain in full float32 precision.
    """

    default_config: CliffordQuantConfig = field(
        default_factory=CliffordQuantConfig
    )
    layer_configs: Dict[str, CliffordQuantConfig] = field(default_factory=dict)
    keep_fp32_layers: List[str] = field(default_factory=list)

    def config_for(self, layer_name: str) -> Optional[CliffordQuantConfig]:
        """Return the quant config for *layer_name*, or None if fp32."""
        if layer_name in self.keep_fp32_layers:
            return None
        return self.layer_configs.get(layer_name, self.default_config)


# ---------------------------------------------------------------------------
# Simulated model wrapper
# ---------------------------------------------------------------------------
class _QuantizedLayer:
    """Wraps a single weight matrix as a quantized multivector."""

    def __init__(
        self,
        name: str,
        weight: np.ndarray,
        config: CliffordQuantConfig,
    ) -> None:
        self.name = name
        self.config = config
        self._quantized: QuantizedMultivector = quantize_multivector(weight, config)
        self._weight_shape = weight.shape

    def dequantized_weight(self) -> np.ndarray:
        return dequantize_multivector(self._quantized)

    @property
    def bits_used(self) -> int:
        return self._quantized.total_bits()

    @property
    def original_bits(self) -> int:
        """Bits that would be used at float32."""
        total_elems = 1
        for d in self._weight_shape:
            total_elems *= d
        return total_elems * 32


class QuantizedModel:
    """A lightweight model container with mixed-precision quantized layers."""

    def __init__(self) -> None:
        self.layers: Dict[str, Any] = {}
        self._fp32_layers: Dict[str, np.ndarray] = {}

    def add_quantized_layer(self, name: str, qlayer: _QuantizedLayer) -> None:
        self.layers[name] = qlayer

    def add_fp32_layer(self, name: str, weight: np.ndarray) -> None:
        self._fp32_layers[name] = weight

    def get_weight(self, name: str) -> np.ndarray:
        if name in self._fp32_layers:
            return self._fp32_layers[name]
        ql = self.layers.get(name)
        if ql is not None:
            return ql.dequantized_weight()
        raise KeyError(f"Unknown layer: {name}")

    def total_quantized_bits(self) -> int:
        total = 0
        for ql in self.layers.values():
            total += ql.bits_used
        for w in self._fp32_layers.values():
            total += int(np.prod(w.shape)) * 32
        return total

    def total_fp32_bits(self) -> int:
        total = 0
        for ql in self.layers.values():
            total += ql.original_bits
        for w in self._fp32_layers.values():
            total += int(np.prod(w.shape)) * 32
        return total


# ---------------------------------------------------------------------------
# apply_mixed_precision
# ---------------------------------------------------------------------------
def apply_mixed_precision(
    model_weights: Dict[str, np.ndarray],
    config: MixedPrecisionConfig,
) -> QuantizedModel:
    """Quantize a dict of named weight arrays according to *config*.

    Parameters
    ----------
    model_weights : dict[str, np.ndarray]
        Mapping from layer name to float weight array (last axis = 8).
    config : MixedPrecisionConfig

    Returns
    -------
    QuantizedModel
    """
    qmodel = QuantizedModel()

    for name, weight in model_weights.items():
        layer_cfg = config.config_for(name)
        if layer_cfg is None:
            qmodel.add_fp32_layer(name, weight.copy())
        else:
            ql = _QuantizedLayer(name, weight, layer_cfg)
            qmodel.add_quantized_layer(name, ql)

    return qmodel


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------
def estimate_memory_savings(
    model_weights: Dict[str, np.ndarray],
    config: MixedPrecisionConfig,
) -> Dict[str, Any]:
    """Estimate memory savings from grade-aware mixed-precision quantization.

    Returns a dict with keys:
        fp32_bytes, quantized_bytes, saved_bytes, compression_ratio,
        per_layer (list of per-layer dicts).
    """
    per_layer: List[Dict[str, Any]] = []
    total_fp32_bits = 0
    total_quant_bits = 0

    for name, weight in model_weights.items():
        num_elems = int(np.prod(weight.shape))
        fp32_bits = num_elems * 32
        total_fp32_bits += fp32_bits

        layer_cfg = config.config_for(name)
        if layer_cfg is None:
            quant_bits = fp32_bits  # stays fp32
        else:
            # estimate using per-grade bit-widths
            quant_bits = 0
            for grade, slc in GRADE_SLICES.items():
                grade_count = GRADE_WIDTHS[grade]
                # number of multivectors times components-in-this-grade
                n_mv = num_elems // NUM_COMPONENTS
                quant_bits += n_mv * grade_count * layer_cfg.grade_bits[grade]

        total_quant_bits += quant_bits
        per_layer.append(
            {
                "name": name,
                "fp32_bits": fp32_bits,
                "quantized_bits": quant_bits,
                "compression": fp32_bits / max(quant_bits, 1),
            }
        )

    fp32_bytes = total_fp32_bits / 8
    quant_bytes = total_quant_bits / 8
    return {
        "fp32_bytes": fp32_bytes,
        "quantized_bytes": quant_bytes,
        "saved_bytes": fp32_bytes - quant_bytes,
        "compression_ratio": fp32_bytes / max(quant_bytes, 1),
        "per_layer": per_layer,
    }
