"""MoE-HLM: Full model with standard attention + holographic expert MoE.

Architecture per layer:
    1. Standard multi-head self-attention (flat, fast, tensor-core optimized)
    2. MoE layer: flat router → top-2 holographic experts (HLM-8^3)
    3. [Optional] Shared ToU bank cross-attention (at designated layers)

The attention is completely standard — no blade decomposition.
All geometric processing is confined to the experts, which only
fire for 2/8 tokens' worth of compute per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from geoformer.moe_hlm.config import MoEHLMConfig
from geoformer.moe_hlm.router import MoELayer
from geoformer.tou.bank import ToUBank


class RoPEAttention(nn.Module):
    """Standard multi-head self-attention with RoPE. No blade structure."""

    def __init__(self, config: MoEHLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # RoPE
        inv_freq = 1.0 / (config.rope_theta **
                          (torch.arange(0, config.d_head, 2).float() / config.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def _build_rope(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, n_heads, d_head)

        # RoPE
        cos, sin = self._build_rope(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Transpose for attention: (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class ToUCrossAttention(nn.Module):
    """Lightweight cross-attention to shared ToU primitive bank.

    Applied at designated layers. All experts share this bank.
    """

    def __init__(self, config: MoEHLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_prim = config.tou_d_prim

        self.q_proj = nn.Linear(config.d_model, config.tou_d_prim, bias=False)
        self.k_proj = nn.Linear(config.tou_d_prim, config.tou_d_prim, bias=False)
        self.v_proj = nn.Linear(config.tou_d_prim, config.tou_d_prim, bias=False)
        self.o_proj = nn.Linear(config.tou_d_prim, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, 1, bias=True)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        prim_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
            prim_embeds: (n_primitives, d_prim)
        Returns:
            (batch, seq, d_model)
        """
        B, T, D = x.shape

        q = self.q_proj(x)         # (B, T, d_prim)
        k = self.k_proj(prim_embeds)  # (N_prim, d_prim)
        v = self.v_proj(prim_embeds)  # (N_prim, d_prim)

        # Cross-attention
        scale = math.sqrt(self.d_prim)
        attn = torch.matmul(q, k.t()) / scale  # (B, T, N_prim)
        attn = F.softmax(attn, dim=-1)

        retrieved = torch.matmul(attn, v)  # (B, T, d_prim)
        out = self.o_proj(retrieved)        # (B, T, d_model)

        # Gated injection
        g = self.gate(x).sigmoid()  # (B, T, 1)
        return g * out


class MoEHLMBlock(nn.Module):
    """Single MoE-HLM transformer block.

    Pre-norm architecture:
        x = x + attn(norm1(x))           # Standard attention (fast)
        x = x + moe(norm2(x))            # MoE with holographic experts
        [if tou_layer]:
        x = x + tou_attn(norm3(x), bank) # Shared ToU bank lookup
    """

    def __init__(self, config: MoEHLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_tou = layer_idx in config.tou_attn_layers

        # Standard attention
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = RoPEAttention(config)

        # MoE with holographic experts
        self.norm2 = nn.RMSNorm(config.d_model)
        self.moe = MoELayer(config)

        # Optional ToU cross-attention
        if self.has_tou:
            self.norm3 = nn.RMSNorm(config.d_model)
            self.tou_attn = ToUCrossAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        tou_embeds: torch.Tensor = None,
        blade_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (batch, seq, d_model)
            aux_loss: scalar MoE load balancing loss
        """
        # Standard attention
        x = x + self.attn(self.norm1(x), mask)

        # MoE with holographic experts
        moe_out, aux_loss = self.moe(self.norm2(x), tou_embeds, blade_masks)
        x = x + moe_out

        # ToU cross-attention (at designated layers)
        if self.has_tou and tou_embeds is not None:
            x = x + self.tou_attn(self.norm3(x), tou_embeds)

        return x, aux_loss


class MoEHLM(nn.Module):
    """MoE-HLM: Mixture of Experts with Holographic Language Model experts.

    Standard transformer attention + MoE routing to HLM-8^3 experts.
    Shared ToU knowledge bank across all experts.
    """

    def __init__(self, config: MoEHLMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # Transformer blocks with MoE
        self.blocks = nn.ModuleList([
            MoEHLMBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = nn.RMSNorm(config.d_model)

        # Shared ToU bank
        self.tou_bank = ToUBank(
            n_primitives=config.tou_n_primitives,
            d_blade=config.tou_d_prim,
        )

        # Output head (weight-tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            logits: (batch, seq, vocab_size)
            loss: scalar (if targets provided, includes aux loss)
            lm_loss: scalar (pure LM loss)
            aux_loss: scalar (MoE load balancing)
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_embed(input_ids)
        x = self.embed_dropout(x)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

        # ToU bank
        prim_embeds, blade_masks = self.tou_bank()

        # Forward through blocks, accumulate aux loss
        total_aux_loss = torch.tensor(0.0, device=device)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                # Gradient checkpointing for MoE blocks
                x_out, aux = torch.utils.checkpoint.checkpoint(
                    block, x, causal_mask, prim_embeds, blade_masks,
                    use_reentrant=False,
                )
            else:
                x_out, aux = block(x, causal_mask, prim_embeds, blade_masks)

            x = x_out
            total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits, "aux_loss": total_aux_loss}

        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["lm_loss"] = lm_loss
            result["loss"] = lm_loss + self.config.router_aux_loss_weight * total_aux_loss

        return result

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["token_embed"] = sum(p.numel() for p in self.token_embed.parameters())
        counts["tou_bank"] = sum(p.numel() for p in self.tou_bank.parameters())

        attn_params = 0
        moe_router_params = 0
        moe_expert_params = 0
        tou_attn_params = 0
        norm_params = 0

        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            moe_router_params += sum(p.numel() for p in block.moe.router.parameters())
            moe_expert_params += sum(p.numel() for p in block.moe.experts.parameters())
            norm_params += sum(p.numel() for p in block.norm1.parameters())
            norm_params += sum(p.numel() for p in block.norm2.parameters())
            if block.has_tou:
                tou_attn_params += sum(p.numel() for p in block.tou_attn.parameters())
                norm_params += sum(p.numel() for p in block.norm3.parameters())

        counts["attention"] = attn_params
        counts["moe_routers"] = moe_router_params
        counts["moe_experts_total"] = moe_expert_params
        counts["moe_experts_per_expert"] = moe_expert_params // self.config.n_experts
        counts["tou_cross_attn"] = tou_attn_params
        counts["norms"] = norm_params
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        total = sum(p.numel() for p in self.parameters())
        counts["total"] = total
        counts["total_unique"] = total - self.lm_head.weight.numel()

        # Active params per forward pass (attention + top_k experts + overhead)
        active_expert_params = moe_expert_params * self.config.top_k // self.config.n_experts
        counts["active_per_forward"] = (
            counts["token_embed"] + attn_params + moe_router_params +
            active_expert_params + tou_attn_params + norm_params
        )

        return counts
