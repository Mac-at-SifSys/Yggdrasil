"""
tou — Tensor of Understanding v2.0

Defines the 1,486 primitives and 9 blades of the ToU framework,
all represented as Cl(3,0) multivectors with learnable routing.
"""

from holograph.tou.primitives import ToUPrimitives
from holograph.tou.blades import ToUBlades
from holograph.tou.tou_v2 import ToUV2

__all__ = ['ToUPrimitives', 'ToUBlades', 'ToUV2']
