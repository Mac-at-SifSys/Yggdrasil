"""
memory_attention.py -- Memory-augmented attention layer.

Inserts between self-attention and FFN in designated HLM blocks.
Reads from the HolographicMemoryBank via query projection,
injects retrieved context via gated residual.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from holograph.layers.clifford_linear import CliffordLinear, _accumulate_grad, _zero_grad_params
from holograph.memory.holographic_memory_bank import HolographicMemoryBank


def _sigmoid(x):
    return xp.where(x >= 0,
                    1.0 / (1.0 + xp.exp(-x)),
                    xp.exp(x) / (1.0 + xp.exp(x)))


class MemoryAttentionLayer:
    """
    Memory read + gated injection into the residual stream.

    Pipeline:
    1. Mean-pool input x over sequence -> (d_model, 8)
    2. Project to single query MV via CliffordLinear(d_model, 1) -> (1, 8) -> (8,)
    3. Read from memory bank -> (8,) retrieved context
    4. Project back via CliffordLinear(1, d_model) -> (d_model, 8)
    5. Gated residual: x = x + sigmoid(gate) * memory_context

    Gate starts at -10.0 so sigmoid(-10) ~= 0.0000454, effectively off.
    """

    def __init__(self, d_model: int, memory_bank: HolographicMemoryBank,
                 top_k: int = 64, gate_init: float = -10.0):
        self.d_model = d_model
        self.memory_bank = memory_bank
        self.top_k = top_k

        self.query_proj = CliffordLinear(d_model, 1, bias=True)
        self.memory_gate_proj = CliffordLinear(1, d_model, bias=True)
        self.gate_scalar = xp.array([gate_init], dtype=xp.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Read from memory and inject into residual stream.

        Args:
            x: (batch, seq, d_model, 8)
        Returns:
            (batch, seq, d_model, 8)
        """
        batch, seq, d_model, _ = x.shape

        x_pooled = xp.mean(x, axis=1)  # (batch, d_model, 8)

        # Batch the learnable projections and bank lookup together.
        query_mv = self.query_proj.forward(x_pooled)  # (batch, 1, 8)
        queries = query_mv[:, 0, :]  # (batch, 8)
        retrieved = self.memory_bank.read_many(queries, top_k=self.top_k)  # (batch, 8)
        memory_contexts = self.memory_gate_proj.forward(retrieved[:, xp.newaxis, :])  # (batch, d_model, 8)

        gate = float(_sigmoid(self.gate_scalar[0]))
        memory_broadcast = memory_contexts[:, xp.newaxis, :, :]

        self._cache = {
            'x': x,
            'gate': gate,
            'memory_contexts': memory_contexts,
        }

        output = x + gate * xp.broadcast_to(memory_broadcast, x.shape)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward through memory attention.

        Gradients do not flow into the memory bank contents.
        """
        gate = self._cache['gate']
        memory_contexts = self._cache['memory_contexts']

        grad_x = grad_output.copy()

        batch, seq, d_model, _ = grad_output.shape
        memory_broadcast = memory_contexts[:, xp.newaxis, :, :]
        memory_broadcast = xp.broadcast_to(memory_broadcast, grad_output.shape)
        grad_gate_raw = xp.sum(grad_output * memory_broadcast)
        gate_deriv = gate * (1.0 - gate)
        grad_gate_scalar = grad_gate_raw * gate_deriv
        _accumulate_grad(self.gate_scalar, xp.array([grad_gate_scalar], dtype=xp.float32))

        grad_memory = grad_output * gate
        grad_memory_summed = xp.sum(grad_memory, axis=1)  # (batch, d_model, 8)
        self.memory_gate_proj.backward(grad_memory_summed)

        return grad_x

    def zero_grad(self):
        self.query_proj.zero_grad()
        self.memory_gate_proj.zero_grad()
        _zero_grad_params([self.gate_scalar])

    def parameters(self):
        params = []
        params.extend(self.query_proj.parameters())
        params.extend(self.memory_gate_proj.parameters())
        params.append(self.gate_scalar)
        return params
