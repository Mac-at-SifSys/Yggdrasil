"""
Rune Compiler Optimization Passes.
"""

from rune.compiler.passes.grade_pruning import GradePruningPass
from rune.compiler.passes.fusion import FusionPass
from rune.compiler.passes.memory_plan import MemoryPlanPass
from rune.compiler.passes.lower_fused_ops import LowerFusedOpsPass

__all__ = ['GradePruningPass', 'FusionPass', 'MemoryPlanPass', 'LowerFusedOpsPass']
