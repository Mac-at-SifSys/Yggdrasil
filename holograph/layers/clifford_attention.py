"""
clifford_attention.py — Multi-head Clifford attention.

Attention score = grade_0(Q * ~K) — scalar part of geometric product with reverse.
This captures rotational similarity: two multivectors that differ by a rotation
will have a high scalar product of Q*~K (cos of rotation angle).

Multi-head: different heads can specialize by grade via grade-projected Q/K/V.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from holograph.layers.clifford_linear import CliffordLinear


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / (xp.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


class CliffordAttention:
    """
    Multi-head Clifford attention.

    For each head h:
        Q_h = proj_q_h(x)                     shape: (batch, seq, d_head, 8)
        K_h = proj_k_h(x)                     shape: (batch, seq, d_head, 8)
        V_h = proj_v_h(x)                     shape: (batch, seq, d_head, 8)

        score[b,h,i,j] = sum_d grade_0( Q_h[b,i,d] * ~K_h[b,j,d] )
        attn = softmax(score / sqrt(d_head))
        out_h[b,i,d] = sum_j attn[b,h,i,j] * V_h[b,j,d]     (scalar * MV)

    Output = concat heads, project back.
    """

    @staticmethod
    def _flatten_head_features(x: np.ndarray) -> np.ndarray:
        """Flatten the multivector head dim so score computation is a dense dot product."""
        batch, seq_len, d_head, _ = x.shape
        return x.reshape(batch, seq_len, d_head * 8)

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.scale = 1.0 / xp.sqrt(self.d_head)

        # Q, K, V projections: one CliffordLinear per head
        self.proj_q = [CliffordLinear(d_model, self.d_head, bias=False) for _ in range(n_heads)]
        self.proj_k = [CliffordLinear(d_model, self.d_head, bias=False) for _ in range(n_heads)]
        self.proj_v = [CliffordLinear(d_model, self.d_head, bias=False) for _ in range(n_heads)]

        # Output projection: concatenated heads -> d_model
        self.proj_out = CliffordLinear(d_model, d_model, bias=True)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq, d_model, 8) — input multivector sequence
            mask: (batch, seq, seq) or (seq, seq) — attention mask (True = attend)
        Returns:
            (batch, seq, d_model, 8)
        """
        batch, seq_len, d_model, _ = x.shape
        assert d_model == self.d_model

        # Per-head projections still use separate parameters, but score/value math
        # now runs as a batched tensor program across all heads.
        Q = xp.stack([proj.forward(x) for proj in self.proj_q], axis=1)  # (batch, heads, seq, d_head, 8)
        K = xp.stack([proj.forward(x) for proj in self.proj_k], axis=1)
        V = xp.stack([proj.forward(x) for proj in self.proj_v], axis=1)

        Q_flat = Q.reshape(batch, self.n_heads, seq_len, self.d_head * 8)
        K_flat = K.reshape(batch, self.n_heads, seq_len, self.d_head * 8)
        scores = xp.matmul(Q_flat, xp.swapaxes(K_flat, -1, -2)) * self.scale

        attn_mask = mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[xp.newaxis, xp.newaxis, :, :]
            else:
                attn_mask = attn_mask[:, xp.newaxis, :, :]
            scores = xp.where(attn_mask, scores, -1e9)

        attn_weights = _softmax(scores, axis=-1)

        if self.dropout > 0:
            drop_mask = (xp.random.rand(*attn_weights.shape) > self.dropout).astype(xp.float32)
            attn_weights = attn_weights * drop_mask / (1.0 - self.dropout + 1e-12)

        head_outputs = xp.einsum('bhqk,bhkdm->bhqdm', attn_weights, V)
        concat = xp.swapaxes(head_outputs, 1, 2).reshape(batch, seq_len, d_model, 8)

        self._cache = {
            'x': x,
            'mask': mask,
            'Q': Q,
            'K': K,
            'V': V,
            'Q_flat': Q_flat,
            'K_flat': K_flat,
            'attn_weights': attn_weights,
            # Preserve legacy shape for attention-map introspection.
            'heads': [{'attn_weights': attn_weights[:, h]} for h in range(self.n_heads)],
        }

        # Final projection
        output = self.proj_out.forward(concat)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through CliffordAttention.

        Args:
            grad_output: (batch, seq, d_model, 8)
        Returns:
            grad_input: (batch, seq, d_model, 8)
        """
        x = self._cache['x']
        batch, seq_len, d_model, _ = x.shape

        # Backward through output projection
        grad_concat = self.proj_out.backward(grad_output)  # (batch, seq, d_model, 8)

        # Split gradient across heads and batch the attention backward math.
        grad_heads = grad_concat.reshape(batch, seq_len, self.n_heads, self.d_head, 8)
        grad_heads = xp.swapaxes(grad_heads, 1, 2)  # (batch, heads, seq, d_head, 8)

        grad_input = xp.zeros_like(x)
        V = self._cache['V']
        attn_weights = self._cache['attn_weights']
        Q_flat = self._cache['Q_flat']
        K_flat = self._cache['K_flat']

        grad_attn = xp.einsum('bhqdm,bhkdm->bhqk', grad_heads, V)
        grad_V = xp.einsum('bhqk,bhqdm->bhkdm', attn_weights, grad_heads)

        dot = xp.sum(grad_attn * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn - dot)
        grad_scores_unscaled = grad_scores * self.scale

        grad_Q_flat = xp.matmul(grad_scores_unscaled, K_flat)
        grad_K_flat = xp.matmul(xp.swapaxes(grad_scores_unscaled, -1, -2), Q_flat)
        grad_Q = grad_Q_flat.reshape(batch, self.n_heads, seq_len, self.d_head, 8)
        grad_K = grad_K_flat.reshape(batch, self.n_heads, seq_len, self.d_head, 8)

        for h in range(self.n_heads):
            grad_input += self.proj_q[h].backward(grad_Q[:, h])
            grad_input += self.proj_k[h].backward(grad_K[:, h])
            grad_input += self.proj_v[h].backward(grad_V[:, h])

        return grad_input

    def zero_grad(self):
        """Reset gradient accumulators for all sub-layer parameters."""
        for h in range(self.n_heads):
            self.proj_q[h].zero_grad()
            self.proj_k[h].zero_grad()
            self.proj_v[h].zero_grad()
        self.proj_out.zero_grad()

    def parameters(self):
        params = []
        for h in range(self.n_heads):
            params.extend(self.proj_q[h].parameters())
            params.extend(self.proj_k[h].parameters())
            params.extend(self.proj_v[h].parameters())
        params.extend(self.proj_out.parameters())
        return params

    def __repr__(self):
        return f"CliffordAttention(d_model={self.d_model}, n_heads={self.n_heads})"
