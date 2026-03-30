"""
tou_layer.py — Tensor of Understanding integration layer.

Wraps the ToU v2.0 system into a layer that can be placed in the
HLM transformer stack. Routes input through 1,486 primitives and
9 blades, accumulates with residual connection.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from holograph.tou.tou_v2 import ToUV2
from holograph.layers.normalization import CliffordLayerNorm
from rune.ops.batched import batched_sandwich
from rune.autodiff.clifford_rules import CliffordDerivativeRules
from holograph.layers.clifford_linear import _accumulate_grad, _zero_grad_params


class ToULayer:
    """
    Tensor of Understanding integration layer.

    Pipeline:
    1. Pre-norm the input
    2. Route through ToU v2.0 (primitives + blades + sandwich rotors)
    3. Residual connection: output = input + alpha * tou_output
    4. Optional: grade projection on output

    The learnable alpha controls how strongly the ToU influences the residual stream.
    """

    def __init__(self, d_model: int, n_primitives: int = 256, n_blades: int = 9):
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.n_blades = n_blades

        # ToU v2.0 core
        self.tou = ToUV2(d_model, n_active_primitives=n_primitives)

        # Pre-normalization
        self.norm = CliffordLayerNorm(d_model)

        # Residual gate: learnable scalar (initialized small for stable training)
        self.alpha = xp.array([0.1], dtype=xp.float32)

        # Per-blade learnable rotors for additional transformation
        # (separate from ToU internal rotors — these are layer-level)
        self.layer_rotor = xp.zeros((d_model, 8), dtype=xp.float32)
        self.layer_rotor[:, 0] = 1.0  # identity

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq, d_model, 8)
        Returns:
            (batch, seq, d_model, 8) — with ToU integration
        """
        # Pre-norm
        x_norm = self.norm.forward(x)

        # Route through ToU
        # Process each (batch, seq) position: flatten batch*seq, process, reshape
        batch_shape = x_norm.shape[:-2]  # (batch, seq)
        flat = x_norm.reshape(-1, self.d_model, 8)  # (B*S, d_model, 8)

        tou_out = self.tou.forward(flat)  # (B*S, d_model, 8)

        # Apply layer-level rotor via sandwich product
        tou_transformed = batched_sandwich(self.layer_rotor, tou_out)

        # Reshape back
        tou_transformed = tou_transformed.reshape(*batch_shape, self.d_model, 8)

        # Residual connection with learnable gate
        output = x + self.alpha * tou_transformed

        # Cache for backward
        self._cache = {
            'x': x,
            'x_norm': x_norm,
            'tou_out_flat': tou_out,
            'tou_transformed': tou_transformed,
            'batch_shape': batch_shape,
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through ToULayer.

        output = x + alpha * tou_transformed
        Gradient flows through residual + ToU path.

        Args:
            grad_output: (batch, seq, d_model, 8)
        Returns:
            grad_input: (batch, seq, d_model, 8)
        """
        tou_transformed = self._cache['tou_transformed']
        batch_shape = self._cache['batch_shape']
        tou_out_flat = self._cache['tou_out_flat']
        # Residual: output = x + alpha * tou_transformed
        # grad_x_residual = grad_output
        grad_x = grad_output.copy()

        # grad through alpha * tou_transformed
        grad_tou_transformed = grad_output * self.alpha  # (batch, seq, d_model, 8)

        # grad_alpha = sum(grad_output * tou_transformed)
        grad_alpha = xp.sum(grad_output * tou_transformed)
        _accumulate_grad(self.alpha, xp.array([grad_alpha], dtype=xp.float32))

        # Backward through sandwich: tou_transformed = GP(GP(r, tou_out), r_rev)
        # Flatten grad to match tou_out_flat shape
        grad_tt_flat = grad_tou_transformed.reshape(-1, self.d_model, 8)

        # Sandwich backward: use CliffordDerivativeRules
        # Expand rotor to match batch dim for broadcasting
        n_flat = grad_tt_flat.shape[0]
        r_expanded = xp.broadcast_to(self.layer_rotor, (n_flat, self.d_model, 8)).copy()
        grad_r, grad_tou_out = CliffordDerivativeRules.sandwich_backward(
            grad_tt_flat, r_expanded, tou_out_flat
        )
        # Accumulate rotor gradient (sum over batch*seq)
        grad_layer_rotor = grad_r.sum(axis=0)  # sum over flat batch dim -> (d_model, 8)
        grad_layer_rotor[:, 1:4] = 0.0
        grad_layer_rotor[:, 7:8] = 0.0
        _accumulate_grad(self.layer_rotor, grad_layer_rotor)

        # ToU backward is complex — approximate by passing gradient through as identity
        # (ToU parameters get zero gradient for now; full ToU backward is a separate task)
        grad_x_norm_flat = grad_tou_out  # (B*S, d_model, 8)
        grad_x_norm = grad_x_norm_flat.reshape(*batch_shape, self.d_model, 8)

        # Backward through norm
        grad_x += self.norm.backward(grad_x_norm)

        return grad_x

    def zero_grad(self):
        """Reset gradient accumulators for all parameters."""
        _zero_grad_params(self.parameters())

    def parameters(self):
        params = self.tou.parameters()
        params.extend(self.norm.parameters())
        params.extend([self.alpha, self.layer_rotor])
        return params

    def __repr__(self):
        return f"ToULayer(d_model={self.d_model}, n_primitives={self.n_primitives})"
