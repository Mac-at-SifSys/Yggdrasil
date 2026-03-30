"""ToU Memory Attention: cross-attention to knowledge bank primitives.

The 1,486 primitives from the Tensor of Understanding knowledge bank
are embedded as learnable vectors. At designated layers (every 4th),
each blade's tokens attend to their blade's primitives via cross-attention.

This makes the knowledge bank differentiable — primitive embeddings learn
to encode useful semantic content through backprop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.config import GeoFormerConfig


class ToUMemoryAttention(nn.Module):
    """Cross-attention from token representations to ToU primitive embeddings.

    Each blade independently attends to its own subset of primitives.
    A learned gate controls how much of the retrieved information is injected
    into the residual stream.

    Architecture per blade:
        Q = W_q(token_blade_repr)     # From the current blade channel
        K = W_k(primitive_embeddings)  # From the primitive bank
        V = W_v(primitive_embeddings)
        attn_out = softmax(Q @ K^T / sqrt(d)) @ V
        output = gate * attn_out      # Gated injection
    """

    def __init__(self, config: GeoFormerConfig, blade_primitive_counts: dict = None):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.n_primitives = config.tou_n_primitives

        # Per-blade Q, K, V projections for cross-attention
        # Using batched linear: (n_blades, d_blade, d_blade) each
        self.W_q = nn.Parameter(
            torch.randn(config.n_blades, config.d_blade, config.d_blade) * config.init_std
        )
        self.W_k = nn.Parameter(
            torch.randn(config.n_blades, config.d_blade, config.d_blade) * config.init_std
        )
        self.W_v = nn.Parameter(
            torch.randn(config.n_blades, config.d_blade, config.d_blade) * config.init_std
        )

        # Gating: learned per-blade gate from the full multivector
        self.gate_proj = nn.Linear(config.d_model, config.n_blades, bias=True)
        nn.init.zeros_(self.gate_proj.bias)  # Start with gates near 0.5

    def forward(
        self,
        mv: torch.Tensor,
        primitive_embeddings: torch.Tensor,
        blade_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mv: (batch, seq, n_blades, d_blade) — current multivector
            primitive_embeddings: (n_primitives, d_blade) — learnable bank
            blade_masks: (n_blades, n_primitives) — binary mask, which blade owns which primitive

        Returns:
            (batch, seq, n_blades, d_blade) — gated contribution to add to residual
        """
        B, T, N, D = mv.shape
        device = mv.device

        # Compute gate from full multivector
        mv_flat = mv.reshape(B, T, N * D)
        gate = self.gate_proj(mv_flat).sigmoid()  # (B, T, N)
        gate = gate.unsqueeze(-1)  # (B, T, N, 1)

        blade_masks = blade_masks.to(device=device, dtype=torch.bool)
        valid_blades = blade_masks.any(dim=-1)  # (N,)
        safe_mask = blade_masks | (~valid_blades).unsqueeze(-1)

        # Batch all blade-specific projections in one tensor program.
        q = torch.einsum("btnd,ndh->btnh", mv, self.W_q)  # (B, T, N, D)
        k = torch.einsum("pd,ndh->nph", primitive_embeddings, self.W_k)  # (N, P, D)
        v = torch.einsum("pd,ndh->nph", primitive_embeddings, self.W_v)  # (N, P, D)

        scale = math.sqrt(D)
        attn_scores = torch.einsum("btnd,npd->btnp", q, k) / scale  # (B, T, N, P)
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(
            ~safe_mask.unsqueeze(0).unsqueeze(0),
            mask_value,
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        retrieved = torch.einsum("btnp,npd->btnd", attn_weights, v)
        retrieved = retrieved * valid_blades.to(retrieved.dtype).view(1, 1, N, 1)

        return gate * retrieved
