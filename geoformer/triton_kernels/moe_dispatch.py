"""Kernel 3: MoE Dispatch — route, sort, scatter.

Provides the top-level MoE-HLM block that uses the optimized
expert dispatch from expert_gemm.py.

This module ties everything together into a drop-in replacement
for the original MoEHLMBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from geoformer.triton_kernels.expert_gemm import FastMoELayer


class FastRoPEAttention(nn.Module):
    """Standard attention with SDPA (uses flash attention on H100)."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = 1.0 / (config.rope_theta **
                          (torch.arange(0, config.d_head, 2).float() / config.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        # RoPE
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        q1, q2 = q.chunk(2, dim=-1)
        q = q * cos + torch.cat([-q2, q1], dim=-1) * sin
        k1, k2 = k.chunk(2, dim=-1)
        k = k * cos + torch.cat([-k2, k1], dim=-1) * sin

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


class FastToUCrossAttention(nn.Module):
    """ToU cross-attention (unchanged — already small)."""

    def __init__(self, config):
        super().__init__()
        dp = config.tou_d_prim
        dm = config.d_model
        self.d_prim = dp
        self.q_proj = nn.Linear(dm, dp, bias=False)
        self.k_proj = nn.Linear(dp, dp, bias=False)
        self.v_proj = nn.Linear(dp, dp, bias=False)
        self.o_proj = nn.Linear(dp, dm, bias=False)
        self.gate = nn.Linear(dm, 1, bias=True)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, prim_embeds):
        q = self.q_proj(x)
        k = self.k_proj(prim_embeds)
        v = self.v_proj(prim_embeds)
        attn = F.softmax(torch.matmul(q, k.t()) / math.sqrt(self.d_prim), dim=-1)
        out = self.o_proj(torch.matmul(attn, v))
        return self.gate(x).sigmoid() * out


class FastMoEHLMBlock(nn.Module):
    """Optimized MoE-HLM block with sorted dispatch."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.has_tou = layer_idx in config.tou_attn_layers
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = FastRoPEAttention(config)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.moe = FastMoELayer(config)
        if self.has_tou:
            self.norm3 = nn.RMSNorm(config.d_model)
            self.tou_attn = FastToUCrossAttention(config)

    def forward(self, x, mask=None, tou_embeds=None, blade_masks=None):
        x = x + self.attn(self.norm1(x), mask)
        moe_out, aux = self.moe(self.norm2(x))
        x = x + moe_out
        if self.has_tou and tou_embeds is not None:
            x = x + self.tou_attn(self.norm3(x), tou_embeds)
        return x, aux
