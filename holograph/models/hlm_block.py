"""
hlm_block.py — Single HLM transformer block.

Architecture:
    x -> CliffordLayerNorm -> CliffordAttention -> residual
      -> CliffordLayerNorm -> CliffordFFN       -> residual
      -> (optional) CliffordRouter + MoE experts

CliffordFFN = CliffordLinear(d_model, d_ff) -> activation -> CliffordLinear(d_ff, d_model)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from holograph.layers.clifford_linear import CliffordLinear
from holograph.layers.clifford_attention import CliffordAttention
from holograph.layers.normalization import CliffordLayerNorm
from holograph.layers.clifford_router import CliffordRouter
from holograph.layers.activations import (
    clifford_gelu, clifford_relu, clifford_sigmoid,
    clifford_gelu_backward, clifford_relu_backward, clifford_sigmoid_backward,
)


ACTIVATION_MAP = {
    'gelu': clifford_gelu,
    'relu': clifford_relu,
    'sigmoid': clifford_sigmoid,
}

ACTIVATION_BACKWARD_MAP = {
    'gelu': clifford_gelu_backward,
    'relu': clifford_relu_backward,
    'sigmoid': clifford_sigmoid_backward,
}


class CliffordFFN:
    """
    Feed-forward network using CliffordLinear layers.

    x -> CliffordLinear(d_model, d_ff) -> activation -> CliffordLinear(d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = 'gelu'):
        self.up = CliffordLinear(d_model, d_ff, bias=True)
        self.down = CliffordLinear(d_ff, d_model, bias=True)
        self.activation_name = activation
        self.activation_fn = ACTIVATION_MAP.get(activation, clifford_gelu)
        self.activation_backward_fn = ACTIVATION_BACKWARD_MAP.get(activation, clifford_gelu_backward)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., d_model, 8) -> (..., d_model, 8)"""
        h = self.up.forward(x)          # (..., d_ff, 8)
        self._cache = {'h_pre_act': h}
        h = self.activation_fn(h)       # (..., d_ff, 8) with grade-aware activation
        h = self.down.forward(h)        # (..., d_model, 8)
        return h

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward through CliffordFFN: down -> activation -> up.

        Args:
            grad_output: (..., d_model, 8)
        Returns:
            grad_input: (..., d_model, 8)
        """
        h_pre_act = self._cache['h_pre_act']

        # Backward through down projection
        grad_act_out = self.down.backward(grad_output)  # (..., d_ff, 8)

        # Backward through activation
        grad_act_in = self.activation_backward_fn(h_pre_act, grad_act_out)  # (..., d_ff, 8)

        # Backward through up projection
        grad_input = self.up.backward(grad_act_in)  # (..., d_model, 8)

        return grad_input

    def zero_grad(self):
        self.up.zero_grad()
        self.down.zero_grad()

    def parameters(self):
        return self.up.parameters() + self.down.parameters()


class HLMBlock:
    """
    Single HLM transformer block.

    Pre-norm architecture:
        x' = x + Attention(LayerNorm(x))
        x'' = x' + FFN(LayerNorm(x'))

    Optional MoE: replaces single FFN with routed experts.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, activation: str = 'gelu',
                 use_moe: bool = False, n_experts: int = 4, moe_top_k: int = 2):
        self.d_model = d_model
        self.use_moe = use_moe

        # Attention sub-layer
        self.attn_norm = CliffordLayerNorm(d_model)
        self.attn = CliffordAttention(d_model, n_heads, dropout=dropout)

        # FFN sub-layer
        self.ffn_norm = CliffordLayerNorm(d_model)

        if use_moe:
            # MoE: multiple FFN experts with router
            self.router = CliffordRouter(d_model, n_experts, top_k=moe_top_k)
            self.experts = [CliffordFFN(d_model, d_ff, activation) for _ in range(n_experts)]
        else:
            self.ffn = CliffordFFN(d_model, d_ff, activation)
            self.router = None
            self.experts = None

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq, d_model, 8)
            mask: optional attention mask
        Returns:
            (batch, seq, d_model, 8)
        """
        # Cache for backward
        self._cache = {'x_in': x, 'mask': mask}

        # Attention sub-layer with residual
        x_norm = self.attn_norm.forward(x)
        attn_out = self.attn.forward(x_norm, mask=mask)
        x = x + attn_out
        self._cache['x_after_attn'] = x

        # FFN sub-layer with residual
        x_norm = self.ffn_norm.forward(x)

        if self.use_moe and self.router is not None:
            ffn_out = self._moe_forward(x_norm)
        else:
            ffn_out = self.ffn.forward(x_norm)

        x = x + ffn_out
        return x

    def _moe_forward(self, x: np.ndarray) -> np.ndarray:
        """
        MoE forward: batch-route tokens, then run each expert once.
        x: (batch, seq, d_model, 8)
        """
        batch, seq, d_model, _ = x.shape
        n_tokens = batch * seq
        flat_x = x.reshape(n_tokens, d_model, 8)
        output = xp.zeros_like(flat_x)

        expert_indices, expert_weights, _ = self.router.forward(flat_x)
        flat_expert_indices = expert_indices.reshape(-1)
        flat_expert_weights = expert_weights.reshape(-1)
        flat_token_ids = xp.broadcast_to(
            xp.arange(n_tokens, dtype=xp.int32)[:, xp.newaxis],
            expert_indices.shape,
        ).reshape(-1)

        for expert_idx, expert in enumerate(self.experts):
            mask = flat_expert_indices == expert_idx
            if not xp.any(mask):
                continue

            token_ids = flat_token_ids[mask]
            expert_input = flat_x[token_ids]
            expert_output = expert.forward(expert_input)
            weighted = flat_expert_weights[mask][:, xp.newaxis, xp.newaxis] * expert_output
            xp.add.at(output, token_ids, weighted)

        return output.reshape(batch, seq, d_model, 8)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward through HLMBlock.

        x' = x + attn(norm1(x))
        x'' = x' + ffn(norm2(x'))

        Args:
            grad_output: (batch, seq, d_model, 8) — grad w.r.t. x''
        Returns:
            grad_input: (batch, seq, d_model, 8) — grad w.r.t. x
        """
        # ---- FFN sub-layer backward ----
        # x'' = x' + ffn(norm2(x'))
        # grad_x' = grad_x'' + grad through ffn path
        grad_x_after_attn = grad_output.copy()  # residual path

        if not self.use_moe:
            # Backward through ffn
            grad_ffn_in = self.ffn.backward(grad_output)  # (..., d_model, 8)
            # Backward through ffn_norm
            grad_x_after_attn += self.ffn_norm.backward(grad_ffn_in)

        # ---- Attention sub-layer backward ----
        # x' = x + attn(norm1(x))
        # grad_x = grad_x' + grad through attn path
        grad_x = grad_x_after_attn.copy()  # residual path

        # Backward through attention
        grad_attn_in = self.attn.backward(grad_x_after_attn)
        # Backward through attn_norm
        grad_x += self.attn_norm.backward(grad_attn_in)

        return grad_x

    def zero_grad(self):
        """Reset gradient accumulators for all parameters."""
        self.attn_norm.zero_grad()
        self.attn.zero_grad()
        self.ffn_norm.zero_grad()
        if self.use_moe:
            pass  # MoE backward not implemented
        else:
            self.ffn.zero_grad()

    def parameters(self):
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ffn_norm.parameters())
        if self.use_moe:
            params.extend(self.router.parameters())
            for expert in self.experts:
                params.extend(expert.parameters())
        else:
            params.extend(self.ffn.parameters())
        return params

    def __repr__(self):
        moe_str = f", moe={self.use_moe}" if self.use_moe else ""
        return f"HLMBlock(d_model={self.d_model}{moe_str})"
