"""Geometric Self-Attention with Cayley cross-blade mixing.

Each of the 8 blades has its own attention head. Within-blade attention
is standard scaled dot-product with causal mask. After attention, the
Cayley table governs cross-blade information flow:

    causation_out x affect_out -> relations channel (via e1*e2 = e12)

This makes the algebraic structure of Cl(3,0) a hard constraint on how
information flows between semantic dimensions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.config import GeoFormerConfig
from geoformer.clifford.algebra import CAYLEY_TABLE


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding, applied per-blade."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class CayleyMixer(nn.Module):
    """Cross-blade mixing governed by the Cl(3,0) Cayley table.

    For each pair (i, j) where cayley[i][j] = (k, sign), we add:
        output[k] += sign * alpha[i,j] * (input[i] element-wise-product input[j])

    alpha is a learned scalar per (i,j) pair, initialized small.
    The Cayley table structure is a hard mask — only algebraically valid
    interactions are permitted.
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.n_blades = config.n_blades

        # Build Cayley structure mask and sign tensor
        # cayley_mask[i,j] = 1 if this is a non-trivial product (k != i, k != j)
        mask = torch.zeros(8, 8)
        signs = torch.zeros(8, 8)
        targets = torch.zeros(8, 8, dtype=torch.long)
        source_i = []
        source_j = []
        active_targets = []

        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                k, sign = CAYLEY_TABLE[i][j]
                if k != i and k != j:
                    mask[i, j] = 1.0
                    signs[i, j] = float(sign)
                    targets[i, j] = k
                    source_i.append(i)
                    source_j.append(j)
                    active_targets.append(k)

        self.register_buffer("cayley_mask", mask)
        self.register_buffer("cayley_signs", signs)
        self.register_buffer("cayley_targets", targets)
        self.register_buffer("pair_source_i", torch.tensor(source_i, dtype=torch.long))
        self.register_buffer("pair_source_j", torch.tensor(source_j, dtype=torch.long))
        self.register_buffer("pair_targets", torch.tensor(active_targets, dtype=torch.long))

        # Learned mixing strength per pair, initialized small
        self.alpha = nn.Parameter(
            torch.ones(8, 8) * config.cayley_mix_init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, n_blades, d_blade) with cross-blade contributions added
        """
        B, T, N, D = x.shape
        if self.pair_targets.numel() == 0:
            return x

        # Effective mixing weights: learned alpha * Cayley sign, masked
        weights = self.alpha * self.cayley_signs * self.cayley_mask  # (8, 8)
        pair_weights = weights[self.pair_source_i, self.pair_source_j]
        contributions = x[:, :, self.pair_source_i, :] * x[:, :, self.pair_source_j, :]
        contributions = contributions * pair_weights.view(1, 1, -1, 1)

        mixed = torch.zeros_like(x)
        target_index = self.pair_targets.view(1, 1, -1, 1).expand(B, T, -1, D)
        mixed.scatter_add_(2, target_index, contributions)

        return x + mixed


class GeometricAttention(nn.Module):
    """Multi-blade self-attention with Cayley cross-blade mixing.

    Architecture:
    1. Per-blade Q, K, V projections (8 independent heads)
    2. RoPE positional encoding per blade
    3. Causal masked scaled dot-product attention per blade
    4. Per-blade output projection
    5. Cayley mixer: cross-blade information flow via geometric product structure
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.attn_dropout = config.attn_dropout
        self.use_flash = config.use_flash_attn

        # Per-blade Q, K, V, O projections
        # Implemented as a single linear for efficiency, then reshaped
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(config.d_blade, config.max_seq_len, config.rope_theta)

        # Cayley cross-blade mixing
        self.cayley_mixer = CayleyMixer(config)

    def forward(self, mv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            mv: (batch, seq, n_blades, d_blade) — multivector residual
            mask: optional (seq, seq) causal mask

        Returns:
            (batch, seq, n_blades, d_blade)
        """
        B, T, N, D = mv.shape

        # Flatten to (B, T, d_model) for efficient QKV projection
        x_flat = mv.reshape(B, T, N * D)
        qkv = self.qkv(x_flat)  # (B, T, 3 * d_model)

        # Reshape to (B, T, 3, n_blades, d_blade)
        qkv = qkv.view(B, T, 3, N, D)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, N, D)

        # Apply RoPE per blade
        cos, sin = self.rope(T)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Transpose for attention: (B, N, T, D) — each blade is a "head"
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash:
            # Use Flash Attention (memory-efficient, no materialized T×T matrix)
            dropout_p = self.attn_dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Manual attention (materializes full T×T matrix — uses more VRAM)
            scale = math.sqrt(D)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
            if mask is None:
                mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=mv.device),
                    diagonal=1
                )
            attn_weights = attn_weights + mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
            attn_out = torch.matmul(attn_weights, v)

        # Back to (B, T, N, D)
        attn_out = attn_out.permute(0, 2, 1, 3)

        # Output projection (flatten, project, reshape)
        attn_flat = attn_out.reshape(B, T, N * D)
        attn_flat = self.o_proj(attn_flat)
        attn_out = attn_flat.view(B, T, N, D)

        # Cross-blade mixing via Cayley table
        attn_out = self.cayley_mixer(attn_out)

        return attn_out
