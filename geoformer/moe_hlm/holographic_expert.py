"""HLM-8^3: Holographic Expert with 3 rounds of geometric products.

Each expert is a full holographic processor:
1. Project d_model → 8 blades × d_blade
2. Three rounds of geometric product (full Cayley table, all 64 interactions)
3. Optional ToU bank cross-attention
4. Collapse 8 × d_blade → d_model

The "8^3" means 8 blades processed through 3 geometric rounds —
after 3 rounds of Cl(3,0) products, you've reached full algebraic closure
since the algebra has maximum grade 3.

Between each geometric round, a small FFN refines per-blade representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.moe_hlm.config import MoEHLMConfig
from geoformer.clifford.algebra import CAYLEY_TABLE, cayley_sign_tensor


class GeometricRound(nn.Module):
    """One round of full geometric product + per-blade FFN.

    Computes ALL 64 pairwise blade interactions via the Cayley table,
    then refines with a small per-blade FFN.

    This is the expensive operation that makes HLMs slow —
    but in MoE, only 2 of 8 experts run it per token.
    """

    def __init__(self, config: MoEHLMConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade

        # Precompute Cayley sign tensor for fast geometric product
        self.register_buffer("cayley_signs", self._build_cayley_signs())

        # Learned scaling per blade pair (which interactions matter)
        self.interaction_weights = nn.Parameter(
            torch.ones(config.n_blades, config.n_blades) * 0.1
        )

        # Per-blade FFN to refine after geometric mixing
        # Small: d_blade → 2*d_blade → d_blade
        self.blade_ffn_gate = nn.Linear(config.d_blade, config.expert_d_ffn, bias=False)
        self.blade_ffn_up = nn.Linear(config.d_blade, config.expert_d_ffn, bias=False)
        self.blade_ffn_down = nn.Linear(config.expert_d_ffn, config.d_blade, bias=False)

        # Layer norm per blade
        self.norm = nn.LayerNorm(config.d_blade)

        # Mix ratio: how much geometric product vs residual
        self.geo_gate = nn.Parameter(torch.tensor(0.5))

    def _build_cayley_signs(self) -> torch.Tensor:
        """Build sparse sign tensor: signs[i, j, k] for geometric product."""
        signs = torch.zeros(8, 8, 8)
        for i in range(8):
            for j in range(8):
                k, s = CAYLEY_TABLE[i][j]
                signs[i, j, k] = s
        return signs

    def geometric_product(self, x: torch.Tensor) -> torch.Tensor:
        """Full geometric self-product: x * x via Cayley table.

        For each blade pair (i, j), computes:
            result[k] += cayley_sign[i,j,k] * weight[i,j] * x[i] * x[j]

        This is the FULL geometric product — all 64 interactions,
        not the lightweight Cayley mixing from GeoFormer.

        Args:
            x: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, n_blades, d_blade)
        """
        B, T, N, D = x.shape

        # Outer product of blade channels: x_i * x_j for all i,j
        # x: (B, T, N, D) → expand for pairwise products
        x_i = x.unsqueeze(3)  # (B, T, N, 1, D)
        x_j = x.unsqueeze(2)  # (B, T, 1, N, D)
        outer = x_i * x_j     # (B, T, N, N, D) — all pairwise element products

        # Apply interaction weights (learned per blade pair)
        weights = self.interaction_weights.sigmoid()  # (N, N)
        outer = outer * weights.unsqueeze(-1)  # (B, T, N, N, D)

        # Contract with Cayley signs to get result blades
        # cayley_signs: (N, N, N) — signs[i, j, k]
        # We want: result[k] = sum_ij signs[i,j,k] * outer[i,j]
        # Reshape for einsum: outer is (B, T, i, j, D), signs is (i, j, k)
        result = torch.einsum("btijd,ijk->btkd", outer, self.cayley_signs)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, n_blades, d_blade)
        Returns:
            (batch, seq, n_blades, d_blade)
        """
        B, T, N, D = x.shape

        # Geometric product (the holographic part)
        geo = self.geometric_product(x)

        # Gated residual: mix geometric product with input
        gate = self.geo_gate.sigmoid()
        mixed = gate * geo + (1.0 - gate) * x

        # Per-blade FFN (applied to all blades at once via reshape)
        # Reshape: (B, T, N, D) → (B*T*N, D)
        flat = mixed.reshape(B * T * N, D)
        flat = self.norm(flat)

        # SwiGLU per blade
        g = self.blade_ffn_gate(flat)
        u = self.blade_ffn_up(flat)
        h = F.silu(g) * u
        out = self.blade_ffn_down(h)

        # Reshape back and residual
        out = out.reshape(B, T, N, D)
        return x + out  # Residual from input (pre-geometric)


class HolographicExpert(nn.Module):
    """Full HLM-8^3 expert: project → 3 geometric rounds → collapse.

    This replaces the standard FFN in a MoE layer. Instead of:
        d_model → 4*d_model → d_model (standard FFN)

    It does:
        d_model → 8×d_blade → [geo_round × 3] → 8×d_blade → d_model

    Three geometric rounds ensure full algebraic closure of Cl(3,0).
    """

    def __init__(self, config: MoEHLMConfig):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.d_model = config.d_model
        expert_dim = config.n_blades * config.d_blade  # 8 * 96 = 768

        # Project into blade space
        self.proj_in = nn.Linear(config.d_model, expert_dim, bias=False)

        # Three rounds of geometric product
        self.geo_rounds = nn.ModuleList([
            GeometricRound(config)
            for _ in range(config.n_geometric_rounds)
        ])

        # Project back to model space
        self.proj_out = nn.Linear(expert_dim, config.d_model, bias=False)

        # Final norm
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        tou_embeds: torch.Tensor = None,
        blade_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model) — flat hidden state
            tou_embeds: optional (n_primitives, d_blade) — shared bank
            blade_masks: optional (n_blades, n_primitives) — blade assignment

        Returns:
            (batch, seq, d_model)
        """
        B, T, D = x.shape

        # Project to blade space
        h = self.proj_in(x)  # (B, T, n_blades * d_blade)
        mv = h.reshape(B, T, self.n_blades, self.d_blade)  # (B, T, 8, 96)

        # Three rounds of geometric product
        for geo_round in self.geo_rounds:
            mv = geo_round(mv)

        # Collapse back to flat
        flat = mv.reshape(B, T, self.n_blades * self.d_blade)
        out = self.proj_out(flat)  # (B, T, d_model)
        out = self.out_norm(out)

        return out
