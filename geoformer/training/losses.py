"""Multi-task losses for GeoFormer training.

Loss components:
1. Causal LM loss (primary) — standard next-token prediction
2. Blade activation loss (auxiliary) — encourage correct blade specialization
3. ToU retrieval loss (auxiliary) — encourage attention to correct primitives
4. Cayley consistency loss (auxiliary) — respect the algebra structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from geoformer.clifford.algebra import CAYLEY_TABLE


class GeoFormerLoss(nn.Module):
    """Combined multi-task loss for GeoFormer training."""

    def __init__(
        self,
        lm_weight: float = 1.0,
        blade_weight: float = 0.1,
        cayley_weight: float = 0.01,
        vocab_size: int = 50_304,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.blade_weight = blade_weight
        self.cayley_weight = cayley_weight
        self.vocab_size = vocab_size

        # Precompute Cayley consistency targets
        # If blade i and blade j are active, blade cayley[i][j] should also be active
        cayley_pairs = []
        for i in range(8):
            for j in range(i + 1, 8):
                k, _ = CAYLEY_TABLE[i][j]
                if k != i and k != j:
                    cayley_pairs.append((i, j, k))
        self.cayley_pairs = cayley_pairs

    def lm_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard causal LM cross-entropy loss."""
        return F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )

    def blade_activation_loss(
        self,
        blade_logits: torch.Tensor,
        blade_targets: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss on blade activation predictions.

        Args:
            blade_logits: (batch, seq, n_blades) predicted
            blade_targets: (batch, n_blades) ground truth activations [0,1]
        """
        # Expand targets across sequence
        if blade_targets.dim() == 2:
            blade_targets = blade_targets.unsqueeze(1).expand_as(blade_logits)
        return F.binary_cross_entropy_with_logits(blade_logits, blade_targets)

    def cayley_consistency_loss(
        self,
        blade_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage blade activations to respect Cayley algebra structure.

        If blades i and j are both active, then blade cayley[i][j] = k
        should also be active. Soft MSE constraint.

        Args:
            blade_activations: (batch, seq, n_blades) activation magnitudes
        """
        loss = torch.tensor(0.0, device=blade_activations.device)

        for i, j, k in self.cayley_pairs:
            # Expected activation of blade k given blades i and j
            expected_k = blade_activations[:, :, i] * blade_activations[:, :, j]
            actual_k = blade_activations[:, :, k]

            # Only penalize when the product should be active but isn't
            # (don't penalize when neither input is active)
            mask = expected_k > 0.1
            if mask.any():
                loss = loss + F.mse_loss(actual_k[mask], expected_k[mask])

        return loss / max(len(self.cayley_pairs), 1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        blade_logits: Optional[torch.Tensor] = None,
        blade_targets: Optional[torch.Tensor] = None,
        blade_activations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Returns dict with total loss and per-component losses.
        """
        losses = {}

        # Primary: LM loss
        lm = self.lm_loss(logits, targets)
        losses["lm"] = lm
        total = self.lm_weight * lm

        # Auxiliary: blade activation
        if blade_logits is not None and blade_targets is not None:
            blade = self.blade_activation_loss(blade_logits, blade_targets)
            losses["blade"] = blade
            total = total + self.blade_weight * blade

        # Auxiliary: Cayley consistency
        if blade_activations is not None:
            cayley = self.cayley_consistency_loss(blade_activations)
            losses["cayley"] = cayley
            total = total + self.cayley_weight * cayley

        losses["total"] = total
        return losses
