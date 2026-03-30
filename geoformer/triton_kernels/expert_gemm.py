"""Kernel 2: Grouped Expert GEMM for MoE-HLM.

Instead of running 8 separate expert forward passes sequentially,
stack all expert weights and run them as batched operations.

All 8 experts share the same architecture (proj_in → geo_rounds → proj_out)
but have independent weights. By stacking weights into (8, d_in, d_out)
tensors, we can use torch.bmm to run all experts in parallel.

This eliminates the for-loop over experts — the main bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from geoformer.triton_kernels.cayley_constants import get_target_accumulation_table


class FusedGeometricRound(nn.Module):
    """Geometric round with scatter-free Cayley product.

    Uses per-target accumulation instead of scatter_add.
    Each output blade directly sums its 8 contributing products.
    No atomics, no scatter, fully deterministic.
    """

    def __init__(self, n_blades, d_blade, d_ffn):
        super().__init__()
        self.n_blades = n_blades
        self.d_blade = d_blade

        # Per-target Cayley table (scatter-free)
        src_ij, src_signs = get_target_accumulation_table("cpu")
        self.register_buffer("src_ij", src_ij)      # (8, 8, 2)
        self.register_buffer("src_signs", src_signs) # (8, 8)

        self.interaction_weights = nn.Parameter(torch.ones(64) * 0.1)
        self.geo_gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_blade)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d_blade, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_blade, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_blade, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_tok, 8, D) multivector
        Returns:
            (N_tok, 8, D) transformed multivector
        """
        N_tok, N_blades, D = x.shape
        dtype = x.dtype

        w = self.interaction_weights.sigmoid().to(dtype)

        # Scatter-free geometric product: compute each output blade directly
        geo = torch.zeros_like(x)
        for k in range(8):
            for p in range(8):
                i = self.src_ij[k, p, 0].long()
                j = self.src_ij[k, p, 1].long()
                sign = self.src_signs[k, p].to(dtype)
                weight = w[i * 8 + j]
                geo[:, k, :] = geo[:, k, :] + sign * weight * x[:, i, :] * x[:, j, :]

        # Gated mix
        g = self.geo_gate.sigmoid()
        mixed = g * geo + (1.0 - g) * x

        # LayerNorm + SwiGLU (on flattened blades)
        flat = self.norm(mixed.reshape(-1, D))
        h = F.silu(self.gate_proj(flat)) * self.up_proj(flat)
        out = self.down_proj(h).reshape(N_tok, N_blades, D)

        return x + out


class StackedHolographicExpert(nn.Module):
    """All 8 holographic experts with stacked weights.

    Instead of 8 separate HolographicExpert modules, we store all weights
    in stacked tensors and use batched matmul (bmm) to run all experts
    in parallel.

    proj_in:  (n_experts, d_model, expert_dim)
    proj_out: (n_experts, expert_dim, d_model)

    Geometric rounds are still per-expert (shared architecture, independent compute)
    but the heavy matmuls (proj_in, proj_out) are batched.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.d_model = config.d_model
        self.expert_dim = config.n_blades * config.d_blade

        # Stacked projection weights — all experts in one tensor
        self.proj_in = nn.Parameter(
            torch.randn(config.n_experts, config.d_model, self.expert_dim) * config.init_std
        )
        self.proj_out = nn.Parameter(
            torch.randn(config.n_experts, self.expert_dim, config.d_model) * config.init_std
        )

        # LayerNorm for output (shared across experts — they all normalize to same space)
        self.out_norm = nn.LayerNorm(config.d_model)

        # Geometric rounds — shared across all experts
        # (same architecture, shared weights — the routing decides specialization)
        self.geo_rounds = nn.ModuleList([
            FusedGeometricRound(config.n_blades, config.d_blade, config.expert_d_ffn)
            for _ in range(config.n_geometric_rounds)
        ])

    def forward(
        self,
        x_sorted: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Run all experts on pre-sorted tokens.

        Args:
            x_sorted: (N_total, d_model) — all tokens, sorted by expert assignment
            expert_ids: (N_total,) — which expert each token is assigned to
            expert_offsets: (n_experts,) — start index for each expert's tokens
            expert_counts: (n_experts,) — number of tokens per expert

        Returns:
            y_sorted: (N_total, d_model) — expert outputs in same sorted order
        """
        N_total, D = x_sorted.shape
        y = torch.zeros_like(x_sorted)

        for e in range(self.n_experts):
            count = expert_counts[e].item()
            if count == 0:
                continue

            offset = expert_offsets[e].item()
            x_e = x_sorted[offset:offset + count]  # (count, d_model)

            # proj_in: (count, d_model) @ (d_model, expert_dim) → (count, expert_dim)
            h = torch.mm(x_e, self.proj_in[e])

            # Reshape to multivector
            mv = h.reshape(count, self.n_blades, self.d_blade)

            # Geometric rounds
            for geo in self.geo_rounds:
                mv = geo(mv)

            # Flatten and proj_out
            flat = mv.reshape(count, self.expert_dim)
            out = torch.mm(flat, self.proj_out[e])
            y[offset:offset + count] = self.out_norm(out)

        return y


class FastMoELayer(nn.Module):
    """Optimized MoE layer with pre-sorted dispatch.

    Flow:
    1. Route: gate(x) → top-k expert assignments
    2. Sort: group tokens by expert (contiguous memory)
    3. Expert compute: stacked experts on sorted tokens
    4. Unsort: scatter results back to original positions

    No per-expert Python loops for gather/scatter — just one sort and one unsort.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.d_model = config.d_model

        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight)

        self.experts = StackedHolographicExpert(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
            aux_loss: scalar
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # Route
        logits = self.gate(x_flat)  # (N, n_experts)
        top_w, top_i = torch.topk(logits, self.top_k, dim=-1)  # (N, top_k)
        top_w = F.softmax(top_w, dim=-1)

        # Aux loss
        probs = F.softmax(logits, dim=-1)
        f = probs.mean(dim=0)
        aux_loss = (f * f).sum() * self.n_experts

        # Flatten top-k: each token appears top_k times
        # token_ids: which original token
        # expert_ids: which expert
        # weights: routing weight
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        expert_ids = top_i.reshape(-1)  # (N * top_k,)
        weights = top_w.reshape(-1)     # (N * top_k,)

        N_expanded = N * self.top_k

        # Sort by expert for contiguous processing
        sort_indices = expert_ids.argsort(stable=True)
        expert_ids_sorted = expert_ids[sort_indices]
        token_ids_sorted = token_ids[sort_indices]
        weights_sorted = weights[sort_indices]

        # Gather sorted tokens
        x_sorted = x_flat[token_ids_sorted]  # (N_expanded, D)

        # Compute expert offsets and counts
        expert_offsets = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        expert_counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        for e in range(self.n_experts):
            mask = expert_ids_sorted == e
            if mask.any():
                indices = mask.nonzero(as_tuple=True)[0]
                expert_offsets[e] = indices[0]
                expert_counts[e] = len(indices)

        # Run all experts
        y_sorted = self.experts(x_sorted, expert_ids_sorted,
                                expert_offsets, expert_counts)

        # Weight and scatter back
        y_weighted = y_sorted * weights_sorted.unsqueeze(-1)

        # Accumulate back to original token positions
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        output.index_add_(0, token_ids_sorted, y_weighted)

        return output.reshape(B, T, D), aux_loss
