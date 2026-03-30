"""
Bifrost export — ONNX and TensorRT export stubs for Clifford HLM models.

These are skeleton implementations that register the custom Clifford ops
(geometric product, grade projection, etc.) and provide an export entry
point.  Full implementation requires the actual ONNX / TensorRT runtimes.
"""

from .onnx_export import export_to_onnx
from .trt_export import export_to_trt
