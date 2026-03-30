"""Geometric FFN: shared SwiGLU on full d_model + Clifford cross-product mixing.

The FFN operates on the flattened d_model space (all blades concatenated) for
parameter efficiency, then the output is reshaped back to blade form for
Clifford cross-product mixing.

This hybrid approach gives:
- Standard transformer-scale FFN parameters (d_model -> d_ffn -> d_model)
- Geometric structure via Clifford mixing on the blade-reshaped output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.config import GeoFormerConfig
from geoformer.clifford.algebra import CAYLEY_TABLE


class SharedSwiGLU(nn.Module):
    """SwiGLU FFN on the full d_model (flattened blade representation).

    Architecture:
        flat(mv) -> gate_proj, up_proj -> SiLU(gate) * up -> down_proj -> reshape(mv)

    This is a standard SwiGLU like in Llama/Qwen, operating on the
    concatenated blade channels.
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ffn = config.d_ffn
        self.dropout = config.ffn_dropout

        # SwiGLU: gate and up projections combined
        self.gate_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.down_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            (batch, seq, d_model)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up

        if self.training and self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout)

        return self.down_proj(hidden)


class CliffordFFNMixer(nn.Module):
    """Lightweight cross-blade mixing after FFN.

    For each Cayley product (i, j) -> (k, sign):
        output[k] += sign * alpha[i,j] * (ffn_out[i] * ffn_out[j])

    Only non-trivial products (where k != i and k != j) are included.
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()

        # Collect non-trivial Cayley entries
        entries = []
        for i in range(8):
            for j in range(i + 1, 8):  # Upper triangle only
                k, sign = CAYLEY_TABLE[i][j]
                if k != i and k != j:
                    entries.append((i, j, k, sign))

        self.register_buffer(
            "entry_ijk",
            torch.tensor([[e[0], e[1], e[2]] for e in entries], dtype=torch.long)
        )
        self.register_buffer(
            "entry_signs",
            torch.tensor([e[3] for e in entries], dtype=torch.float32)
        )

        # One learned scalar per Cayley product pair
        self.alpha = nn.Parameter(
            torch.ones(len(entries)) * config.cayley_mix_init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, n_blades, d_blade) with cross-blade contributions
        """
        if self.entry_ijk.numel() == 0:
            return x

        B, T, _, D = x.shape
        source_i = self.entry_ijk[:, 0]
        source_j = self.entry_ijk[:, 1]
        targets = self.entry_ijk[:, 2]
        scales = (self.entry_signs * self.alpha).view(1, 1, -1, 1)

        contributions = x[:, :, source_i, :] * x[:, :, source_j, :]
        contributions = contributions * scales

        mixed = torch.zeros_like(x)
        target_index = targets.view(1, 1, -1, 1).expand(B, T, -1, D)
        mixed.scatter_add_(2, target_index, contributions)

        return x + mixed


class GeometricFFN(nn.Module):
    """Complete Geometric FFN: shared SwiGLU + Clifford blade mixing.

    1. Flatten multivector to d_model
    2. SwiGLU FFN (standard transformer-scale)
    3. Reshape back to (n_blades, d_blade)
    4. Clifford cross-product mixing via Cayley table
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.swiglu = SharedSwiGLU(config)
        self.clifford_mixer = CliffordFFNMixer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, n_blades, d_blade)
        """
        B, T, N, D = x.shape

        # Flatten blade dimensions for shared FFN
        flat = x.reshape(B, T, N * D)

        # SwiGLU on full d_model
        flat = self.swiglu(flat)

        # Reshape back to blade form
        out = flat.reshape(B, T, N, D)

        # Clifford cross-product mixing
        out = self.clifford_mixer(out)

        return out
