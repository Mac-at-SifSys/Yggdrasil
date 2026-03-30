"""
rune.engine — Persistent CUDA training engine for the YGGDRASIL stack.

Eliminates all Python overhead from the training loop by executing
the full forward-backward-update cycle on GPU.
"""

from rune.engine.program import ProgramBuilder, EngineOp, EngineCommand
from rune.engine.memory_pool import MemoryPool
from rune.engine.cuda_engine import CUDAEngine
from rune.engine.hlm_adapter import (
    EngineSupportReport,
    compile_persistent_engine_for_hlm,
    pack_hlm_core_params,
    persistent_engine_support_report,
    sync_hlm_core_params_from_engine,
    can_attach,
    can_train,
)

__all__ = [
    'ProgramBuilder', 'EngineOp', 'EngineCommand', 'MemoryPool', 'CUDAEngine',
    'EngineSupportReport', 'compile_persistent_engine_for_hlm',
    'pack_hlm_core_params', 'persistent_engine_support_report',
    'sync_hlm_core_params_from_engine',
    'can_attach', 'can_train',
]
