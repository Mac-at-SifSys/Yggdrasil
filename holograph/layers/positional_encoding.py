"""
positional_encoding.py — Rotor-based positional encoding for Cl(3,0).

Each position is encoded as a rotor: R(pos) = exp(pos * B_learnable)
where B_learnable is a set of learnable bivectors (one per d_model channel).

Key property: relative position is captured by geometric product:
    R(i) * ~R(j) = exp((i-j) * B) = R(i-j)

This gives rotational relative position encoding for free.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from rune.ops.batched import batched_geom_prod, batched_reverse, bivector_exp_from_components
from rune.autodiff.clifford_rules import CliffordDerivativeRules
from holograph.layers.clifford_linear import _accumulate_grad, _zero_grad_params


class RotorPositionalEncoding:
    """
    Rotor-based positional encoding.

    For each of the d_model channels, we have a learnable bivector B_d.
    Position p is encoded as:
        R(p, d) = exp(p * B_d)

    The encoding is applied via geometric product:
        x_encoded[..., d, :] = R(pos, d) * x[..., d, :]

    Relative position between positions i and j:
        R(i, d) * ~R(j, d) = exp((i-j) * B_d)
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learnable bivector frequencies: (d_model, 3)
        # Initialize with logarithmically spaced frequencies
        # inspired by sinusoidal positional encoding
        freq_base = xp.logspace(-1, -4, d_model, dtype=xp.float32)
        self.bivector_freqs = xp.zeros((d_model, 3), dtype=xp.float32)

        # Distribute across the three bivector planes
        for d in range(d_model):
            plane = d % 3  # cycle through e12, e13, e23
            self.bivector_freqs[d, plane] = freq_base[d]

        # Pre-compute rotor table: (max_seq_len, d_model, 8)
        self._rotor_table = None
        self._build_table()

    def _build_table(self):
        """Pre-compute rotors for all positions."""
        positions = xp.arange(self.max_seq_len, dtype=xp.float32)  # (L,)

        # Scaled bivectors: pos * B_d -> (L, d_model, 3)
        # positions: (L, 1, 1), bivector_freqs: (1, d_model, 3)
        scaled_bv = positions[:, xp.newaxis, xp.newaxis] * self.bivector_freqs[xp.newaxis, :, :]

        # Exponentiate to get rotors: (L, d_model, 8)
        self._rotor_table = bivector_exp_from_components(scaled_bv)

    def forward(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Apply rotor positional encoding.

        Args:
            x: (batch, seq_len, d_model, 8) — input multivectors
            offset: starting position (for cached generation)
        Returns:
            (batch, seq_len, d_model, 8) — positionally encoded
        """
        seq_len = x.shape[1]
        assert offset + seq_len <= self.max_seq_len, \
            f"Position {offset + seq_len} exceeds max {self.max_seq_len}"

        # Get rotors for this sequence: (seq_len, d_model, 8)
        rotors = self._rotor_table[offset:offset + seq_len]

        # Apply via geometric product: R * x
        # rotors: (1, seq_len, d_model, 8), x: (batch, seq_len, d_model, 8)
        encoded = batched_geom_prod(rotors[xp.newaxis], x)

        # Cache for backward
        self._cache = {'x': x.copy(), 'rotors': rotors, 'offset': offset}
        return encoded

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through RotorPositionalEncoding.

        encoded = GP(rotors, x)
        Gradient flows to x. Rotors are derived from bivector_freqs (learnable).

        Args:
            grad_output: (batch, seq_len, d_model, 8)
        Returns:
            grad_input: (batch, seq_len, d_model, 8)
        """
        x = self._cache['x']
        rotors = self._cache['rotors']

        rotors_exp = rotors[xp.newaxis]  # (1, seq, d_model, 8)
        rotors_bc = xp.broadcast_to(rotors_exp, x.shape).copy()

        # GP backward: encoded = GP(rotors, x)
        grad_rotors, grad_x = CliffordDerivativeRules.geometric_product_backward(
            grad_output, rotors_bc, x
        )

        # Accumulate gradient for bivector_freqs through the rotor table
        # This is complex (requires chain through exp), so we accumulate into rotors
        # and use the bivector_exp_backward rule.
        # For now, sum grad_rotors over batch to get (seq_len, d_model, 8)
        grad_rotors_summed = grad_rotors.sum(axis=0)  # (seq_len, d_model, 8)

        # Chain through exp(pos * B_d): use bivector_exp_backward
        offset = self._cache['offset']
        seq_len = x.shape[1]
        positions = xp.arange(offset, offset + seq_len, dtype=xp.float32)

        # scaled_bv shape: (seq_len, d_model, 3)
        scaled_bv_full = xp.zeros((seq_len, self.d_model, 8), dtype=xp.float32)
        scaled_bv_full[..., 4:7] = positions[:, xp.newaxis, xp.newaxis] * self.bivector_freqs[xp.newaxis, :, :]

        grad_scaled_bv = CliffordDerivativeRules.bivector_exp_backward(
            grad_rotors_summed, scaled_bv_full
        )
        # grad_scaled_bv: (seq_len, d_model, 8), only [4:7] meaningful
        # scaled_bv = pos * B_d, so grad_B_d = sum_pos pos * grad_scaled_bv
        grad_bv_freqs = xp.sum(
            positions[:, xp.newaxis, xp.newaxis] * grad_scaled_bv[..., 4:7],
            axis=0
        )  # (d_model, 3)
        _accumulate_grad(self.bivector_freqs, grad_bv_freqs)

        return grad_x

    def zero_grad(self):
        """Reset gradient accumulators for all parameters."""
        _zero_grad_params(self.parameters())

    def relative_rotor(self, pos_i: int, pos_j: int) -> np.ndarray:
        """
        Get the relative position rotor between positions i and j.

        R_rel = R(i) * ~R(j) = exp((i-j) * B)

        Returns: (d_model, 8)
        """
        ri = self._rotor_table[pos_i]  # (d_model, 8)
        rj_rev = batched_reverse(self._rotor_table[pos_j])  # (d_model, 8)
        return batched_geom_prod(ri, rj_rev)

    def parameters(self):
        return [self.bivector_freqs]

    def __repr__(self):
        return f"RotorPositionalEncoding(d_model={self.d_model}, max_seq_len={self.max_seq_len})"
