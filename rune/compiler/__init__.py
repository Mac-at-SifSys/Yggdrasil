"""
Rune Compiler — Phase 1b standalone compiler for HLM models.

Traces HLM computation, optimizes the IR, and emits execution plans
for the persistent CUDA engine.

Usage:
    from rune.compiler import CompiledModel
    compiled = CompiledModel.from_model(model, batch_size=16, seq_len=2048)
    loss = compiled.train_step(input_ids, target_ids, lr=1e-4)
"""

from rune.compiler.ir import IRGraph, IRNode, IRType, OpCode
from rune.compiler.ir import (
    GRADE_SCALAR, GRADE_VECTOR, GRADE_BIVECTOR, GRADE_TRIVECTOR,
    GRADE_EVEN, GRADE_ODD, GRADE_FULL,
)
from rune.compiler.tracer import Tracer
from rune.compiler.codegen import Codegen, EngineCommand, ExecutionPlan
from rune.compiler.compiled_model import CompiledModel

__all__ = [
    'IRGraph', 'IRNode', 'IRType', 'OpCode',
    'GRADE_SCALAR', 'GRADE_VECTOR', 'GRADE_BIVECTOR', 'GRADE_TRIVECTOR',
    'GRADE_EVEN', 'GRADE_ODD', 'GRADE_FULL',
    'Tracer', 'Codegen', 'EngineCommand', 'ExecutionPlan',
    'CompiledModel',
]
