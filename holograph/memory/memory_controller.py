"""
memory_controller.py — Write gating and memory management.

Controls when and how chunks are written to the memory bank.
Provides the interface between the training loop and the bank.
"""

import numpy as np
from rune.backend import xp
from holograph.memory.holographic_memory_bank import HolographicMemoryBank


class MemoryController:
    """
    Manages the memory bank lifecycle during training.

    - Writes chunk outputs to the bank after backward pass
    - Applies periodic grade decay
    - Provides diagnostics
    """

    def __init__(self, memory_bank: HolographicMemoryBank):
        self.bank = memory_bank
        self.chunk_count = 0

    def write_chunk(self, chunk_output: np.ndarray):
        """
        Write a processed chunk to memory.
        Called AFTER backward pass — this is not differentiable.

        Args:
            chunk_output: (seq_len, d_model, 8) — detached final layer output
        """
        self.bank.write(chunk_output, self.chunk_count)
        self.chunk_count += 1

    def get_stats(self) -> dict:
        """Return memory bank statistics."""
        stats = self.bank.memory_stats()
        stats['total_chunks_processed'] = self.chunk_count
        return stats
