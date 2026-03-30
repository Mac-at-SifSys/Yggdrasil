"""Blade Projector: projects flat embeddings into multivector space."""

import torch
import torch.nn as nn

from geoformer.config import GeoFormerConfig


class BladeProjector(nn.Module):
    """Projects flat d_model vectors into (n_blades, d_blade) multivector form.

    This is the entry point to the geometric residual stream.
    Each token's embedding is decomposed into 8 blade channels.
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            (batch, seq, n_blades, d_blade)
        """
        B, T, D = x.shape
        projected = self.proj(x)  # (B, T, d_model)
        return projected.view(B, T, self.n_blades, self.d_blade)
