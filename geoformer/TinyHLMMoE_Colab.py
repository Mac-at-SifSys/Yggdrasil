#!/usr/bin/env python3
"""TinyHLMMoE: Tiny Holographic Language Model with Mixture of Experts.

~76M total params, ~48M active params per forward pass.
Blade-routed MoE: 8 blade groups x 8 experts, top-2 active per blade.
No memory banks. No ToU injection. Clean, fast, iterable.

Training: ~150K tok/s on A100 -> 10K steps in ~2.5 hours.
Goal: prove SFT works before scaling up.
"""

# ============================================================================
# Cell 1: Imports, GPU Setup, Drive Mount
# ============================================================================

import math
import time
import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f}GB")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_CKPT_DIR = '/content/drive/MyDrive/tiny_hlm_moe'
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    print(f"Checkpoints: {DRIVE_CKPT_DIR}")
except ImportError:
    DRIVE_CKPT_DIR = './tiny_hlm_moe'
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    print(f"Checkpoints: {DRIVE_CKPT_DIR}")


# ============================================================================
# Cell 2: Configuration
# ============================================================================

@dataclass
class TinyHLMMoEConfig:
    # Vocabulary
    vocab_size: int = 50_304
    max_seq_len: int = 1024

    # Core dimensions — 8 blades x 64 = 512 d_model
    d_model:   int = 512
    n_layers:  int = 12
    n_heads:   int = 8
    d_head:    int = 64        # d_model // n_heads
    n_blades:  int = 8
    d_blade:   int = 64        # d_model // n_blades

    # Mixture of Experts
    n_experts:     int = 8     # experts per blade group
    top_k:         int = 2     # active experts per blade per token
    d_expert_ffn:  int = 256   # hidden dim inside each expert (4 x d_blade)
    moe_aux_coeff: float = 0.01  # load balancing loss coefficient

    # Attention
    rope_theta: float = 10_000.0

    # Regularization
    embed_dropout: float = 0.1
    gradient_checkpointing: bool = True
    init_std: float = 0.02


CONFIG = TinyHLMMoEConfig()

# ============================================================================
# Cell 3: Clifford Algebra Cl(3,0) — Cayley Table
# ============================================================================

def build_cayley_table():
    """Build Cl(3,0) geometric product Cayley table.

    Basis: {1, e1, e2, e3, e12, e13, e23, e123}
    Returns: result_blade (8,8), sign (8,8) as tensors.
    """
    # Blade composition rules for Cl(3,0)
    # Each basis blade encoded as a frozenset of generator indices (1-indexed)
    basis = [
        frozenset(),          # 0: scalar (1)
        frozenset([1]),       # 1: e1
        frozenset([2]),       # 2: e2
        frozenset([3]),       # 3: e3
        frozenset([1,2]),     # 4: e12
        frozenset([1,3]),     # 5: e13
        frozenset([2,3]),     # 6: e23
        frozenset([1,2,3]),   # 7: e123
    ]
    blade_to_idx = {b: i for i, b in enumerate(basis)}

    def blade_product(a_set, b_set):
        """Compute geometric product of two basis blades. Returns (index, sign)."""
        # Combine generators
        combined = list(a_set) + list(b_set)
        sign = 1
        # Bubble sort to canonical order, counting swaps
        arr = combined[:]
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
                    sign *= -1
        # Cancel pairs (e_i * e_i = +1 in Cl(3,0))
        result = []
        i = 0
        while i < len(arr):
            if i+1 < len(arr) and arr[i] == arr[i+1]:
                i += 2  # cancel, sign stays same (+1 metric)
            else:
                result.append(arr[i])
                i += 1
        result_blade = blade_to_idx[frozenset(result)]
        return result_blade, sign

    n = 8
    result_blade = torch.zeros(n, n, dtype=torch.long)
    sign_tensor  = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            rb, s = blade_product(basis[i], basis[j])
            result_blade[i, j] = rb
            sign_tensor[i, j]  = float(s)
    return result_blade, sign_tensor


_CAYLEY_RESULT, _CAYLEY_SIGN = build_cayley_table()


def _build_cayley_signs_3d() -> torch.Tensor:
    """Convert (8,8) Cayley tables to (8,8,8) sparse sign tensor.

    signs[i,j,k] = ±1 if blade_i * blade_j = ±blade_k in Cl(3,0), else 0.
    This is the native-stack format (geoformer.clifford.algebra.cayley_sign_tensor)
    enabling a single einsum geometric product with no Python loops.
    """
    signs = torch.zeros(8, 8, 8)
    signs.scatter_(2, _CAYLEY_RESULT.unsqueeze(2), _CAYLEY_SIGN.unsqueeze(2))
    return signs


_CAYLEY_SIGNS_3D = _build_cayley_signs_3d()   # (8,8,8) — module-level constant


class GeometricProduct(nn.Module):
    """Cl(3,0) geometric self-product — fully vectorized via native sign tensor.

    Uses the same approach as geoformer.clifford.ops.geometric_product_fast:
        outer[B,T,i,j,D] = x[B,T,i,D] * x[B,T,j,D]
        result = einsum("BTijD,ijk->BTkD", outer, signs_w)

    Single fused einsum replaces the previous 8-iteration Python loop,
    allowing torch.compile to emit one GPU kernel for the entire product.
    """

    def __init__(self, n_blades: int = 8):
        super().__init__()
        self.n_blades = n_blades
        self.interaction_weights = nn.Parameter(torch.zeros(n_blades, n_blades))
        # Fixed Cl(3,0) sign tensor — not learned, registered as buffer
        self.register_buffer("signs", _CAYLEY_SIGNS_3D.clone())  # (8,8,8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_blades, d_blade) -> (B, T, n_blades, d_blade)"""
        weights = self.interaction_weights.sigmoid()          # (8,8)
        # Weighted sign tensor: signs_w[i,j,k] = sign[i,j->k] * weight[i,j]
        signs_w = self.signs * weights.unsqueeze(-1)          # (8,8,8)
        # Outer product across blade dim: (B,T,8,8,D)
        outer = x.unsqueeze(3) * x.unsqueeze(2)
        # Contract blade pairs -> output blades: single GPU kernel
        result = torch.einsum("BTijD,ijk->BTkD", outer, signs_w)
        return result.to(x.dtype)


# ============================================================================
# Cell 4: Blade-Routed Mixture of Experts
# ============================================================================

class GeometricMoEFFN(nn.Module):
    """Cl(3,0) geometric product + blade-routed MoE — fully parallelized.

    All 8 blades × 8 experts are computed simultaneously using batched
    einsum operations, with no Python loops over blades or experts.
    This replaces 64 serial module calls with 3 fused einsum kernels,
    enabling torch.compile to emit a minimal set of GPU kernels.

    Flow:
      x (B,T,d_model)
      -> reshape to (B,T,8,d_blade)
      -> GeometricProduct  : vectorized Cayley einsum (single kernel)
      -> all 8 routers     : einsum "BTbD,bED->BTbE" (all blades at once)
      -> all 64 experts    : einsum "BTbD,bEFD->BTbEF" x2 + silu + einsum (batched SwiGLU)
      -> top-k gather      : gather + weighted sum (all blades in parallel)
      -> load balance loss : scatter + mean (all blades at once)
      -> reshape back to (B,T,d_model)
    """

    def __init__(self, config: TinyHLMMoEConfig):
        super().__init__()
        Nb  = config.n_blades      # 8
        Ne  = config.n_experts     # 8
        D   = config.d_blade       # 64
        Ff  = config.d_expert_ffn  # 256

        self.n_blades  = Nb
        self.n_experts = Ne
        self.d_blade   = D
        self.top_k     = config.top_k
        self.aux_coeff = config.moe_aux_coeff

        # Clifford geometric product (vectorized single-einsum)
        self.geo_product = GeometricProduct(config.n_blades)

        # Input/output norms
        self.norm_in  = nn.RMSNorm(config.d_model)
        self.norm_out = nn.RMSNorm(config.d_model)

        # Batched router weights: (n_blades, n_experts, d_blade)
        # Replaces 8 × Linear(d_blade, n_experts) modules
        self.router_w = nn.Parameter(torch.empty(Nb, Ne, D))

        # Batched expert weights (SwiGLU):
        #   gate/up : (n_blades, n_experts, d_expert_ffn, d_blade)
        #   down    : (n_blades, n_experts, d_blade, d_expert_ffn)
        # Replaces 8 blades × 8 experts × 3 Linear layers = 192 Linear modules
        self.expert_gate = nn.Parameter(torch.empty(Nb, Ne, Ff, D))
        self.expert_up   = nn.Parameter(torch.empty(Nb, Ne, Ff, D))
        self.expert_down = nn.Parameter(torch.empty(Nb, Ne, D,  Ff))

        std = config.init_std
        for p in (self.router_w, self.expert_gate, self.expert_up, self.expert_down):
            nn.init.normal_(p, std=std)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_model) -> (output: same shape, aux_loss: scalar)"""
        B, T, _ = x.shape
        Nb, Ne, D = self.n_blades, self.n_experts, self.d_blade

        normed = self.norm_in(x)
        blades = normed.view(B, T, Nb, D)                          # (B,T,8,64)

        # ── Clifford geometric product (vectorized, single GPU kernel) ────────
        blades = blades + self.geo_product(blades)

        # ── All 8 routers in parallel ─────────────────────────────────────────
        # router_w: (8,8,64) — (n_blades, n_experts, d_blade)
        scores    = torch.einsum("BTbD,bED->BTbE", blades, self.router_w)   # (B,T,8,8)
        all_probs = F.softmax(scores, dim=-1)                               # (B,T,8,8)
        top_w, top_idx = torch.topk(all_probs, self.top_k, dim=-1)          # (B,T,8,k)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)            # renorm

        # ── All 64 experts in parallel (batched SwiGLU) ───────────────────────
        # gate/up: (8,8,256,64)  →  (B,T,8,8,256)
        gate_h   = torch.einsum("BTbD,bEFD->BTbEF", blades, self.expert_gate)
        up_h     = torch.einsum("BTbD,bEFD->BTbEF", blades, self.expert_up)
        hidden   = F.silu(gate_h) * up_h                                    # (B,T,8,8,256)
        # down: (8,8,64,256)  →  (B,T,8,8,64)
        all_outs = torch.einsum("BTbEF,bEDF->BTbED", hidden, self.expert_down)

        # ── Top-k selection per blade (all blades in parallel) ────────────────
        idx_exp    = top_idx.unsqueeze(-1).expand(B, T, Nb, self.top_k, D)  # (B,T,8,k,64)
        selected   = all_outs.gather(3, idx_exp)                             # (B,T,8,k,64)
        expert_out = (selected * top_w.unsqueeze(-1)).sum(dim=3)             # (B,T,8,64)
        output_blades = blades + expert_out                                  # residual

        # ── Load balancing loss (all blades at once) ──────────────────────────
        f = all_probs.mean(dim=(0, 1))                                       # (8,8)
        expert_mask = torch.zeros(B, T, Nb, Ne, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(3, top_idx, 1.0)
        p = expert_mask.mean(dim=(0, 1))                                     # (8,8)
        aux_loss = self.aux_coeff * Ne * (f * p).sum()

        output = self.norm_out(output_blades.view(B, T, -1))
        return x + output, aux_loss


# ============================================================================
# Cell 5: RoPE Attention
# ============================================================================

class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding."""

    def __init__(self, config: TinyHLMMoEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm   = nn.RMSNorm(config.d_model)

        # RoPE cache
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._rope_len   = 0
        self.theta       = config.rope_theta

    def _build_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._rope_len and self._rope_cache is not None:
            cos, sin = self._rope_cache
            return cos[:seq_len].to(device, dtype), sin[:seq_len].to(device, dtype)
        half = self.d_head // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half, device=device).float() / half))
        t   = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        cos  = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin  = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self._rope_cache = (cos.cpu(), sin.cpu())
        self._rope_len   = seq_len
        return cos.to(dtype), sin.to(dtype)

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return x * cos.unsqueeze(0).unsqueeze(0) + self._rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self._build_rope(T, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out  = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return x + self.o_proj(out)


# ============================================================================
# Cell 6: Full Model
# ============================================================================

class TinyHLMMoEBlock(nn.Module):
    """Single transformer block: RoPE attention + blade-routed GeometricMoE."""

    def __init__(self, config: TinyHLMMoEConfig):
        super().__init__()
        self.attn   = RoPEAttention(config)
        self.moe    = GeometricMoEFFN(config)

    def forward(self, x: torch.Tensor):
        x = self.attn(x)
        x, aux = self.moe(x)
        return x, aux


class TinyHLMMoE(nn.Module):
    """TinyHLMMoE: ~76M total params, ~48M active.

    Architecture:
      - RoPE multi-head self-attention
      - Cl(3,0) geometric product + blade-routed MoE FFN
      - 8 blades x 8 experts per blade, top-2 active
      - Weight-tied input/output embeddings
    """

    def __init__(self, config: TinyHLMMoEConfig):
        super().__init__()
        self.config = config

        self.token_embed  = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_drop   = nn.Dropout(config.embed_dropout)
        self.blocks       = nn.ModuleList([TinyHLMMoEBlock(config) for _ in range(config.n_layers)])
        self.final_norm   = nn.RMSNorm(config.d_model)
        self.lm_head      = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        x = self.embed_drop(self.token_embed(input_ids))

        aux_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                def _ckpt_fn(x_in, block=block):
                    return block(x_in)
                out = grad_checkpoint(_ckpt_fn, x, use_reentrant=False)
                x, aux = out if isinstance(out, tuple) else (out, torch.tensor(0.0))
            else:
                x, aux = block(x)
            aux_total = aux_total + aux

        x      = self.final_norm(x)
        logits = self.lm_head(x)
        result = {"logits": logits, "aux_loss": aux_total / self.config.n_layers}

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = ce_loss + aux_total / self.config.n_layers

        return result

    def count_parameters(self) -> Dict[str, int]:
        total   = sum(p.numel() for p in self.parameters())
        attn    = sum(p.numel() for b in self.blocks for p in b.attn.parameters())
        experts = sum(p.numel() for b in self.blocks
                      for n, p in b.moe.named_parameters() if 'expert' in n)
        routers = sum(p.numel() for b in self.blocks
                      for n, p in b.moe.named_parameters() if 'router' in n)
        embed   = sum(p.numel() for p in self.token_embed.parameters())
        active_expert_fraction = self.config.top_k / self.config.n_experts
        active  = int(attn * self.config.n_layers
                      + experts * active_expert_fraction * self.config.n_layers + embed)
        print(f"  Total params:  {total/1e6:.1f}M")
        print(f"  Active params: {active/1e6:.1f}M ({100*active/total:.0f}% utilization)")
        print(f"  Embedding:     {embed/1e6:.1f}M")
        print(f"  Attention:     {attn/1e6:.2f}M/layer")
        print(f"  Experts:       {experts/1e6:.1f}M total ({experts*active_expert_fraction/1e6:.1f}M active)")
        print(f"  Routers:       {routers/1e3:.0f}K")
        return {"total": total, "active": active}


# ============================================================================
# Cell 7: Data Pipeline
# ============================================================================

CHAT_USER_PREFIX = "\n<|user|>\n"
CHAT_ASST_PREFIX = "\n<|assistant|>\n"
CHAT_END         = "\n<|end|>\n"

def get_pretrain_data(tokenizer, seq_len: int = 1024):
    """Streaming pretrain data: 60% FineWeb-Edu, 20% OpenWebMath, 20% CodeParrot."""
    from datasets import load_dataset

    def extract(ex):
        for field in ['text', 'content', 'code']:
            if field in ex and ex[field]:
                return ex[field]
        return ""

    sources = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.60),
        ("open-web-math/open-web-math", None,          0.20),
        ("transformersbook/codeparrot", None,           0.20),
    ]

    gens = []
    for name, cfg, _ in sources:
        try:
            if cfg:
                ds = load_dataset(name, cfg, split="train", streaming=True)
            else:
                ds = load_dataset(name, split="train", streaming=True)
            gens.append((ds, _))
        except Exception as e:
            print(f"  Warning: could not load {name}: {e}")

    token_buffer = []
    vocab_size = CONFIG.vocab_size

    while True:
        ds, weight = random.choices([(g, w) for g, w in gens],
                                    weights=[w for _, w in gens], k=1)[0]
        try:
            for ex in ds:
                text = extract(ex)
                if not text.strip():
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids = [t for t in ids if t < vocab_size]
                token_buffer.extend(ids)

                while len(token_buffer) >= seq_len + 1:
                    chunk = token_buffer[:seq_len + 1]
                    token_buffer = token_buffer[seq_len:]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    targets   = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield input_ids, targets
        except Exception as e:
            print(f"  Data error: {e}, retrying...")
            continue


def get_sft_data(tokenizer, seq_len: int = 512):
    """Streaming SFT data: OASST2 + Dolly + MetaMathQA + CodeAlpaca.
    Only plain-text English datasets — no Qwen extended vocab tokens.
    """
    from datasets import load_dataset
    VOCAB_SIZE = CONFIG.vocab_size

    def encode_example(user_text, assistant_text):
        if not user_text.strip() or not assistant_text.strip():
            return None
        prompt  = CHAT_USER_PREFIX + user_text.strip() + CHAT_END + CHAT_ASST_PREFIX
        full    = prompt + assistant_text.strip() + CHAT_END
        p_ids   = tokenizer.encode(prompt, add_special_tokens=False)
        f_ids   = tokenizer.encode(full,   add_special_tokens=False)
        # Skip examples with >2% OOV tokens
        oov = sum(1 for t in f_ids if t >= VOCAB_SIZE)
        if oov > max(1, len(f_ids) * 0.02):
            return None
        f_ids = [t if t < VOCAB_SIZE else 0 for t in f_ids]
        p_ids = [t if t < VOCAB_SIZE else 0 for t in p_ids]
        if len(f_ids) > seq_len:
            f_ids = f_ids[:seq_len]
        inp = f_ids[:-1]
        tgt = f_ids[1:]
        n_p = min(len(p_ids) - 1, len(tgt))
        tgt[:n_p] = [-100] * n_p
        pad = (seq_len - 1) - len(inp)
        if pad > 0:
            inp += [0] * pad
            tgt += [-100] * pad
        return (torch.tensor(inp[:seq_len-1], dtype=torch.long),
                torch.tensor(tgt[:seq_len-1], dtype=torch.long))

    def gen_oasst2():
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        for ex in ds:
            if ex.get("role") == "assistant" and ex.get("lang", "en") == "en":
                r = encode_example(ex.get("parent_text", ""), ex.get("text", ""))
                if r: yield r

    def gen_dolly():
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        for ex in ds:
            inst = ex.get("instruction", "")
            ctx  = ex.get("context", "")
            user = inst + (" Context: " + ctx if ctx.strip() else "")
            r = encode_example(user, ex.get("response", ""))
            if r: yield r

    def gen_math():
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        for ex in ds:
            r = encode_example(ex.get("query", ""), ex.get("response", ""))
            if r: yield r

    def gen_code():
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        for ex in ds:
            user = ex.get("instruction", "")
            if ex.get("input", "").strip():
                user += "\n" + ex["input"]
            r = encode_example(user, ex.get("output", ""))
            if r: yield r

    generators = [gen_oasst2, gen_dolly, gen_math, gen_code]
    weights    = [0.40,        0.30,      0.20,     0.10]

    while True:
        gen_fn = random.choices(generators, weights=weights, k=1)[0]
        try:
            for item in gen_fn():
                if item: yield item
        except Exception as e:
            print(f"  SFT data error: {e}, retrying...")
            continue


# ============================================================================
# Cell 8: Checkpoint Utilities
# ============================================================================

def save_checkpoint(model, optimizer, step, loss, path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
    }, path)
    try:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    print(f"  Saved: {path} (fsynced)")


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        return 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = ckpt['model_state_dict']
    ms   = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in ms and ms[k].shape == v.shape}
    skipped  = [k for k in sd if k not in filtered]
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched keys")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  {len(missing)} new keys initialized fresh")
    if optimizer is not None and 'optimizer_state_dict' in ckpt and not skipped:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            print("  Fresh optimizer state")
    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', 0.0)
    print(f"  Resumed step {step}, loss={loss:.4f}")
    return step


def get_lr(step, max_steps, base_lr, warmup=500):
    min_lr = base_lr * 0.1
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))


# ============================================================================
# Cell 9: Pretrain Loop
# ============================================================================

def train():
    from transformers import AutoTokenizer

    # Hyperparameters
    BATCH_SIZE     = 32
    GRAD_ACCUM     = 4
    BASE_LR        = 3e-4
    WARMUP         = 500
    TOTAL_STEPS    = 10_000    # ~1.3B tokens, ~27x model size
    SEQ_LEN        = 1024
    LOG_INTERVAL   = 50
    EVAL_INTERVAL  = 500
    SAVE_INTERVAL  = 1_000

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    config = CONFIG
    print(f"\nBuilding TinyHLMMoE...")
    model = TinyHLMMoE(config).to(DEVICE)
    model.count_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR,
        weight_decay=0.1, betas=(0.9, 0.95)
    )

    # Resume if checkpoint exists
    resume_path = os.path.join(DRIVE_CKPT_DIR, 'latest.pt')
    start_step  = load_checkpoint(model, optimizer, resume_path)

    # torch.compile
    try:
        compiled_model = torch.compile(model)
        print("torch.compile() enabled")
    except Exception:
        compiled_model = model

    print(f"\n{'='*60}")
    print(f"  TinyHLMMoE Pretraining")
    print(f"  Steps: {TOTAL_STEPS:,} | Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum")
    print(f"  LR: {BASE_LR:.1e} | Seq: {SEQ_LEN}")
    print(f"  Expected: ~150K tok/s -> {TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / 1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    data_gen = get_pretrain_data(tokenizer, SEQ_LEN)

    def get_batch():
        items = [next(data_gen) for _ in range(BATCH_SIZE)]
        return (torch.stack([it[0] for it in items]).to(DEVICE),
                torch.stack([it[1] for it in items]).to(DEVICE))

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    t0 = time.time()
    best_loss = float('inf')
    step = start_step

    while step < TOTAL_STEPS:
        for _ in range(GRAD_ACCUM):
            inp, tgt = get_batch()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = compiled_model(inp, tgt)
                loss   = result["loss"] / GRAD_ACCUM
            loss.backward()
            loss_accum += result["loss"].item()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, TOTAL_STEPS, BASE_LR, WARMUP)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % LOG_INTERVAL == 0:
            elapsed  = time.time() - t0
            avg_loss = loss_accum / (LOG_INTERVAL * GRAD_ACCUM)
            ppl      = math.exp(min(avg_loss, 20))
            tokens   = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
            tps      = tokens / elapsed
            eta_h    = (TOTAL_STEPS - step) / max(step, 1) * elapsed / 3600
            print(f"[step {step:>6d}/{TOTAL_STEPS}] loss={avg_loss:.4f} ppl={ppl:.1f} | "
                  f"{tps/1000:.0f}K tok/s | lr={lr:.2e} | {tokens/1e9:.2f}B tok | ETA {eta_h:.1f}h")
            loss_accum = 0.0

        if step % EVAL_INTERVAL == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(20):
                    inp, tgt = get_batch()
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        r = compiled_model(inp, tgt)
                    eval_losses.append(r["loss"].item())
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_ppl  = math.exp(min(eval_loss, 20))
            print(f"  [EVAL {step}] loss={eval_loss:.4f} ppl={eval_ppl:.1f}")
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(model, optimizer, step, eval_loss,
                                os.path.join(DRIVE_CKPT_DIR, 'best.pt'))
            model.train()

        if step % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, step, avg_loss,
                            os.path.join(DRIVE_CKPT_DIR, f'step_{step:06d}.pt'))
            save_checkpoint(model, optimizer, step, avg_loss,
                            os.path.join(DRIVE_CKPT_DIR, 'latest.pt'))

    save_checkpoint(model, optimizer, step, best_loss,
                    os.path.join(DRIVE_CKPT_DIR, 'final.pt'))
    print(f"\n{'='*60}")
    print(f"  Pretraining complete! Steps: {step:,} | Best eval loss: {best_loss:.4f}")
    print(f"{'='*60}")
    return model, tokenizer


# ============================================================================
# Cell 10: LoRA SFT (built-in)
# ============================================================================

class LoRALinear(nn.Module):
    """Low-rank adapter for a frozen linear layer."""
    def __init__(self, linear, rank=16, alpha=32.0):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        for p in self.linear.parameters():
            p.requires_grad = False
        dev = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features, device=dev) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=dev))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model, rank=16, alpha=32.0):
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for module in model.modules():
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Linear):
                setattr(module, attr, LoRALinear(getattr(module, attr), rank, alpha))
                count += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {count} adapters, {trainable:,} trainable ({100*trainable/total:.2f}%)")
    return model


def save_lora(model, optimizer, step, loss, path):
    lora_state = {}
    for name, module in model.named_modules():
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr) and isinstance(getattr(module, attr), LoRALinear):
                layer = getattr(module, attr)
                lora_state[f"{name}.{attr}.A"] = layer.lora_A.data.cpu()
                lora_state[f"{name}.{attr}.B"] = layer.lora_B.data.cpu()
    torch.save({'step': step, 'loss': loss, 'lora': lora_state,
                'opt': optimizer.state_dict()}, path)
    try:
        with open(path, 'rb') as f: os.fsync(f.fileno())
    except OSError: pass
    print(f"  Saved LoRA: {path} ({os.path.getsize(path)/1024**2:.1f}MB)")


def load_lora(model, optimizer, path):
    if not os.path.exists(path): return 0
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    for name, module in model.named_modules():
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr) and isinstance(getattr(module, attr), LoRALinear):
                layer = getattr(module, attr)
                ka = f"{name}.{attr}.A"
                kb = f"{name}.{attr}.B"
                if ka in ckpt['lora']:
                    layer.lora_A.data = ckpt['lora'][ka].to(DEVICE)
                    layer.lora_B.data = ckpt['lora'][kb].to(DEVICE)
    if optimizer and 'opt' in ckpt:
        try: optimizer.load_state_dict(ckpt['opt'])
        except Exception: pass
    step = ckpt.get('step', 0)
    print(f"  Resumed LoRA step {step}, loss={ckpt.get('loss',0):.4f}")
    return step


def sft_finetune(model=None, tokenizer=None):
    """LoRA SFT: loads best.pt if no model passed, trains 2000 steps."""
    from transformers import AutoTokenizer
    import glob

    SFT_DIR        = os.path.join(DRIVE_CKPT_DIR, 'sft')
    os.makedirs(SFT_DIR, exist_ok=True)
    LORA_LR        = 2e-4
    LORA_WARMUP    = 100
    LORA_STEPS     = 2_000
    LORA_BATCH     = 4
    LORA_ACCUM     = 8
    LORA_SEQ       = 512
    LORA_LOG       = 25
    LORA_SAVE      = 500

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    if model is None:
        config = CONFIG
        model  = TinyHLMMoE(config).to(DEVICE)
        ckpt   = None
        for name in ['best.pt', 'final.pt', 'latest.pt']:
            p = os.path.join(DRIVE_CKPT_DIR, name)
            if os.path.exists(p):
                ckpt = p; break
        if ckpt:
            load_checkpoint(model, None, ckpt)
        else:
            print("WARNING: no pretrained checkpoint found, SFT from scratch")

    # Apply LoRA
    model = apply_lora(model, rank=16, alpha=32.0)
    model.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    # Delete old corrupted LoRA checkpoints if any exist
    resume_path = os.path.join(SFT_DIR, 'lora_latest.pt')
    start_step  = load_lora(model, optimizer, resume_path)

    print(f"\n{'='*60}")
    print(f"  LoRA SFT | Steps: {LORA_STEPS} | LR: {LORA_LR:.1e}")
    print(f"  Batch: {LORA_BATCH} x {LORA_ACCUM} | Seq: {LORA_SEQ}")
    print(f"{'='*60}\n")

    data_gen = get_sft_data(tokenizer, LORA_SEQ)

    def get_sft_batch():
        items = [next(data_gen) for _ in range(LORA_BATCH)]
        return (torch.stack([it[0] for it in items]).to(DEVICE),
                torch.stack([it[1] for it in items]).to(DEVICE))

    loss_accum = 0.0
    best_loss  = float('inf')
    t0 = time.time()
    step = start_step
    optimizer.zero_grad()

    while step < LORA_STEPS:
        for _ in range(LORA_ACCUM):
            inp, tgt = get_sft_batch()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model(inp, tgt)
                loss   = result["loss"] / LORA_ACCUM
            loss.backward()
            loss_accum += result["loss"].item()

        nn.utils.clip_grad_norm_(lora_params, 1.0)
        lr = get_lr(step, LORA_STEPS, LORA_LR, LORA_WARMUP)
        for pg in optimizer.param_groups: pg['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % LORA_LOG == 0:
            elapsed  = time.time() - t0
            avg_loss = loss_accum / (LORA_LOG * LORA_ACCUM)
            ppl      = math.exp(min(avg_loss, 20))
            tps      = (step * LORA_BATCH * LORA_ACCUM * LORA_SEQ) / elapsed
            eta_h    = (LORA_STEPS - step) / max(step, 1) * elapsed / 3600
            print(f"[SFT {step:>5d}/{LORA_STEPS}] loss={avg_loss:.4f} ppl={ppl:.1f} | "
                  f"{tps/1000:.0f}K tok/s | lr={lr:.2e} | ETA {eta_h:.1f}h")
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_lora(model, optimizer, step, avg_loss,
                          os.path.join(SFT_DIR, 'lora_best.pt'))
            loss_accum = 0.0

        if step % LORA_SAVE == 0:
            save_lora(model, optimizer, step, best_loss,
                      os.path.join(SFT_DIR, f'lora_step_{step:06d}.pt'))
            save_lora(model, optimizer, step, best_loss, resume_path)

    save_lora(model, optimizer, step, best_loss,
              os.path.join(SFT_DIR, 'lora_final.pt'))
    print(f"\nLoRA SFT complete! Best loss: {best_loss:.4f}")
    return model, tokenizer


# ============================================================================
# Cell 11: Chat Interface
# ============================================================================

def chat(model=None, tokenizer=None):
    """Interactive chat with the fine-tuned model."""
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    if model is None:
        config = CONFIG
        model  = TinyHLMMoE(config).to(DEVICE)
        # Try SFT checkpoint first, then pretrain
        for path in [
            os.path.join(DRIVE_CKPT_DIR, 'sft', 'lora_final.pt'),
            os.path.join(DRIVE_CKPT_DIR, 'best.pt'),
            os.path.join(DRIVE_CKPT_DIR, 'final.pt'),
        ]:
            if os.path.exists(path):
                if 'lora' in path:
                    model = apply_lora(model)
                    load_lora(model, None, path)
                else:
                    load_checkpoint(model, None, path)
                break

    model.eval()
    VOCAB_SIZE = CONFIG.vocab_size

    @torch.no_grad()
    def generate(input_ids, max_new=256, temperature=0.7, top_k=50, rep_penalty=1.15):
        for _ in range(max_new):
            idx = input_ids[:, -CONFIG.max_seq_len:]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model(idx)
            logits = result["logits"][:, -1, :] / temperature

            if rep_penalty != 1.0:
                for tid in set(input_ids[0].tolist()):
                    logits[0, tid] = logits[0, tid] / rep_penalty if logits[0, tid] > 0 else logits[0, tid] * rep_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            tok_str = tokenizer.decode(next_tok[0].tolist(), skip_special_tokens=False)
            if any(s in tok_str for s in ['<|end|>', '<|user|>', '<|system|>']):
                break
            print(tok_str, end='', flush=True)
        return input_ids

    print("\n" + "="*60)
    print("  TinyHLMMoE Chat")
    print("  'quit' to exit | 'reset' to clear history")
    print("="*60 + "\n")

    history  = []
    MAX_HIST = 2
    system   = "You are a helpful, accurate, and concise assistant."

    while True:
        try:
            user = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break

        if user.lower() in ('quit', 'exit', 'q'): break
        if user.lower() == 'reset':
            history = []
            print("[History cleared]\n")
            continue
        if not user: continue

        recent = history[-MAX_HIST:]
        parts  = [f"{CHAT_USER_PREFIX}{system}{CHAT_END}"]
        for u, a in recent:
            parts += [f"{CHAT_USER_PREFIX}{u}{CHAT_END}", f"{CHAT_ASST_PREFIX}{a}{CHAT_END}"]
        parts += [f"{CHAT_USER_PREFIX}{user}{CHAT_END}", CHAT_ASST_PREFIX]

        text = "".join(parts)
        ids  = [t for t in tokenizer.encode(text, add_special_tokens=False) if t < VOCAB_SIZE]
        input_ids = torch.tensor([ids], device=DEVICE)

        print(f"\n[{len(ids)} tokens]\nHLM-v9: ", end='', flush=True)
        out_ids = generate(input_ids)
        new_toks = out_ids[0][len(ids):].tolist()
        response = tokenizer.decode(new_toks, skip_special_tokens=False)
        for stop in ['\n<|end|>', '<|end|>', '\n<|user|>', '<|user|>']:
            if stop in response:
                response = response[:response.index(stop)]
        print("\n")
        history.append((user, response.strip()))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sft',   action='store_true')
    parser.add_argument('--full',  action='store_true')
    parser.add_argument('--chat',  action='store_true')
    args = parser.parse_args()

    if args.full:
        model, tok = train()
        model, tok = sft_finetune(model, tok)
        chat(model, tok)
    elif args.train:
        model, tok = train()
        chat(model, tok)
    elif args.sft:
        model, tok = sft_finetune()
        chat(model, tok)
    elif args.chat:
        chat()
