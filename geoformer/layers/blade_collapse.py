"""Blade Collapse: project multivector back to flat vector for output head."""

import torch
import torch.nn as nn

from geoformer.config import GeoFormerConfig
from geoformer.layers.norms import BladeRMSNorm


class BladeCollapse(nn.Module):
    """Collapse 8-blade multivector back to flat d_model vector.

    Simply reshapes (B, T, 8, d_blade) -> (B, T, d_model), applies
    final RMSNorm, and optional learned projection.
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.d_model = config.d_model

        self.final_norm = nn.RMSNorm(config.d_model)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mv: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, d_model)
        """
        B, T, N, D = mv.shape
        flat = mv.reshape(B, T, self.d_model)
        return self.final_norm(flat)
