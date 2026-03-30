"""
YGGDRASIL L2 Runtime — Python bindings
Pure-Python fallback for the Clifford algebra HLM runtime layer.

Provides:
  - EagerRuntime: immediate execution mode
  - GraphRuntime: trace-and-optimize execution
  - GradeMemoryPool: grade-stratified allocator
  - AutodiffTape: records multivector ops, replays backward
"""

from .runtime import (
    MVValue,
    EagerRuntime,
    GraphRuntime,
    GradeMemoryPool,
    AutodiffTape,
    # Constants
    NUM_GRADES,
    MV_DIM,
    GRADE_DIM,
    GRADE_0,
    GRADE_1,
    GRADE_2,
    GRADE_3,
    ALL_GRADES,
    # Op types
    OpType,
)

__version__ = "0.1.0"
__all__ = [
    "MVValue",
    "EagerRuntime",
    "GraphRuntime",
    "GradeMemoryPool",
    "AutodiffTape",
    "NUM_GRADES",
    "MV_DIM",
    "GRADE_DIM",
    "GRADE_0",
    "GRADE_1",
    "GRADE_2",
    "GRADE_3",
    "ALL_GRADES",
    "OpType",
]
