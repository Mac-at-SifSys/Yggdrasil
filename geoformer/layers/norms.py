"""Normalization layers for GeoFormer."""

import torch
import torch.nn as nn


class BladeRMSNorm(nn.Module):
    """RMSNorm applied independently to each blade channel.

    Input shape:  (batch, seq, n_blades, d_blade)
    Output shape: (batch, seq, n_blades, d_blade)

    Each blade gets its own learned scale parameter.
    """

    def __init__(self, n_blades: int, d_blade: int, eps: float = 1e-6):
        super().__init__()
        self.n_blades = n_blades
        self.d_blade = d_blade
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_blades, d_blade))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8, d_blade)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
