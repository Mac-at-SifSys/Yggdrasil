"""Flat tensor router for MoE-HLM.

Standard top-k expert routing with:
- Linear projection to expert logits
- Top-k selection with softmax normalization
- Load balancing auxiliary loss to prevent expert collapse
- Optional noise for exploration during training

This is intentionally simple â€” the intelligence lives in the experts.
The router just needs to learn "which type of content benefits from
which expert's geometric specialization."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class FlatTensorRouter(nn.Module):
    """Standard MoE router: Linear(d_model, n_experts) â†’ top-k.

    Returns expert indices and normalized weights for the top-k experts
    per token, plus load balancing loss.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, d_model)

        Returns:
            expert_weights: (batch, seq, top_k) â€” normalized routing weights
            expert_indices: (batch, seq, top_k) â€” which experts to use
            aux_loss: scalar â€” load balancing loss
        """
        B, T, D = x.shape

        # Route: (B, T, n_experts)
        logits = self.gate(x)

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # Normalize among selected

        # Load balancing auxiliary loss (Switch Transformer style)
        # Encourages uniform expert utilization
        routing_probs = F.softmax(logits, dim=-1)  # (B, T, n_experts)
        # fraction of tokens routed to each expert
        tokens_per_expert = routing_probs.mean(dim=[0, 1])  # (n_experts,)
        # fraction of router probability allocated to each expert
        prob_per_expert = routing_probs.mean(dim=[0, 1])  # (n_experts,)

        # f_i * P_i should be uniform â†’ minimize sum(f_i * P_i) * n_experts
        aux_loss = (tokens_per_expert * prob_per_expert).sum() * self.n_experts

        return top_k_weights, top_k_indices, aux_loss


class MoELayer(nn.Module):
    """Mixture of Experts layer with Holographic experts.

    Dispatches tokens to top-k experts, runs each expert independently,
    then combines outputs with routing weights.

    For efficiency, this uses the "token choice" approach:
    - Route all tokens
    - Group tokens by expert
    - Batch-process each expert's tokens
    - Scatter results back
    """

    def __init__(self, config):
        super().__init__()
        from geoformer.moe_hlm.holographic_expert import HolographicExpert

        self.n_experts = config.n_experts
        self.top_k = config.top_k

        self.router = FlatTensorRouter(config.d_model, config.n_experts, config.top_k)
        self.experts = nn.ModuleList([
            HolographicExpert(config)
            for _ in range(config.n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        tou_embeds: torch.Tensor = None,
        blade_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, d_model)
            tou_embeds: optional shared ToU bank
            blade_masks: optional blade masks

        Returns:
            output: (batch, seq, d_model)
            aux_loss: scalar load balancing loss
        """
        B, T, D = x.shape

        # Route
        weights, indices, aux_loss = self.router(x)
        # weights: (B, T, top_k), indices: (B, T, top_k)

        # Flatten token assignments so each expert runs once on its full token batch.
        flat_x = x.reshape(B * T, D)
        flat_output = torch.zeros_like(flat_x)
        flat_indices = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_token_ids = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        for expert_id, expert in enumerate(self.experts):
            mask = flat_indices == expert_id
            if not mask.any():
                continue

            token_ids = flat_token_ids[mask]
            expert_input = flat_x.index_select(0, token_ids).unsqueeze(0)  # (1, n_tokens, D)
            expert_output = expert(
                expert_input,
                tou_embeds=tou_embeds,
                blade_masks=blade_masks,
            ).squeeze(0)  # (n_tokens, D)

            weighted = expert_output * flat_weights[mask].unsqueeze(-1)
            flat_output.index_add_(0, token_ids, weighted)

        return flat_output.view(B, T, D), aux_loss
