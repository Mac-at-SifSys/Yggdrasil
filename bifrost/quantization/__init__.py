"""
Bifrost quantization — grade-aware quantization for Clifford multivectors.

In Cl(3,0) every multivector has 8 components spread across 4 grades:
  grade 0 (scalar):    1 component   — high precision needed
  grade 1 (vector):    3 components  — medium precision
  grade 2 (bivector):  3 components  — medium precision
  grade 3 (trivector): 1 component   — lower precision acceptable

This sub-package exploits that hierarchy to compress serving weights and
activations more aggressively than uniform quantization.
"""

from .grade_quantize import (
    CliffordQuantConfig,
    quantize_multivector,
    dequantize_multivector,
)
from .calibration import CalibrationCollector, determine_scales
from .mixed_precision import MixedPrecisionConfig, apply_mixed_precision, estimate_memory_savings
