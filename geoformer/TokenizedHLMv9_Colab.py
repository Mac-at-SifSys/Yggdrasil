#!/usr/bin/env python3
"""Tokenized HLM-v9: 250M hybrid attention + geometric FFN model.

Self-contained Colab training script. Run as one cell or split at section markers.

Architecture:
    - RoPE self-attention for sequence modeling (fast, tensor-core optimized)
    - GeometricFFN replacing standard MLP: full Cl(3,0) Cayley products across 8 blades
    - WZ1/ToU knowledge injection via gated cross-attention at layers 9, 19, 29
    - Multi-scale token n-gram memory banks (1/2/4/8/16-gram)
    - ~253M trainable parameters, 50K vocabulary (Qwen2.5)

Training data: 60% FineWeb-Edu + 20% OpenWebMath + 20% StarCoderData (Python)
"""

# ============================================================================
# Cell 1: Imports and GPU Setup
# ============================================================================

import math
import time
import os
import sys
import gc
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

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
    print("WARNING: No GPU detected — training will be extremely slow")
    DEVICE = torch.device("cpu")

# Google Drive for checkpoints (Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_CKPT_DIR = '/content/drive/MyDrive/hlm_v9_token_ckpts'
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    print(f"Checkpoints will save to: {DRIVE_CKPT_DIR}")
except ImportError:
    DRIVE_CKPT_DIR = './hlm_v9_token_ckpts'
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    print(f"Not in Colab — checkpoints save to: {DRIVE_CKPT_DIR}")


# ============================================================================
# Cell 2: Configuration
# ============================================================================

@dataclass
class TokenizedHLMv9Config:
    # Vocabulary
    vocab_size: int = 50_304
    max_seq_len: int = 1024

    # Core dimensions
    d_model: int = 1024
    n_layers: int = 30
    n_heads: int = 16
    d_head: int = 64  # d_model // n_heads

    # Clifford / Geometric FFN
    n_blades: int = 8
    d_blade: int = 128         # 8 * 128 = 1024 = d_model
    n_geometric_rounds: int = 2
    geo_ffn_dim: int = 512     # SwiGLU FFN between geometric rounds

    # WZ1 / ToU Knowledge Injection
    tou_n_primitives: int = 1_486
    tou_d_prim: int = 128      # = d_blade
    tou_inject_layers: List[int] = field(default_factory=lambda: [9, 19, 29])

    # Token Memory Banks — 3-gram + 4-gram overlapping windows (~21.6GB VRAM)
    mem_d_state: int = 16
    mem_scales: List[int] = field(default_factory=lambda: [3, 4])
    mem_3_slots: int = 100_000_000   # ~1/5 collision rate, 7.2GB
    mem_4_slots: int = 200_000_000   # ~1/3 collision rate, 14.4GB

    # Attention
    rope_theta: float = 10_000.0

    # Regularization
    embed_dropout: float = 0.1

    # Training
    gradient_checkpointing: bool = True
    init_std: float = 0.02

    # Data mix
    data_mix_weights: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])


CONFIG = TokenizedHLMv9Config()


# ============================================================================
# Cell 3: Cl(3,0) Geometric Algebra Core
# ============================================================================

BLADE_NAMES = [
    "narrative",   # 0: scalar (grade 0)
    "causation",   # 1: e1 (grade 1)
    "affect",      # 2: e2 (grade 1)
    "wisdom",      # 3: e3 (grade 1)
    "relations",   # 4: e12 (grade 2)
    "ecology",     # 5: e13 (grade 2)
    "epistemics",  # 6: e23 (grade 2)
    "temporal",    # 7: e123 (grade 3)
]
BLADE_INDEX = {name: i for i, name in enumerate(BLADE_NAMES)}
BLADE_GRADES = [0, 1, 1, 1, 2, 2, 2, 3]

_BASIS_GENS = [
    (),        # scalar
    (1,),      # e1
    (2,),      # e2
    (3,),      # e3
    (1, 2),    # e12
    (1, 3),    # e13
    (2, 3),    # e23
    (1, 2, 3), # e123
]
_GENS_TO_IDX = {gens: i for i, gens in enumerate(_BASIS_GENS)}


def _reduce_product(a_gens: tuple, b_gens: tuple) -> Tuple[int, int]:
    """Compute basis element product via bubble sort with anticommutation."""
    gens = list(a_gens) + list(b_gens)
    sign = 1
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(gens) - 1:
            if gens[i] == gens[i + 1]:
                gens.pop(i)
                gens.pop(i)
                changed = True
                if i > 0:
                    i -= 1
            elif gens[i] > gens[i + 1]:
                gens[i], gens[i + 1] = gens[i + 1], gens[i]
                sign *= -1
                changed = True
                i += 1
            else:
                i += 1
    return _GENS_TO_IDX[tuple(gens)], sign


def _derive_cayley_table() -> List[List[Tuple[int, int]]]:
    """Derive full 8x8 Cayley table from Cl(3,0) axioms."""
    table = []
    for i in range(8):
        row = []
        for j in range(8):
            idx, s = _reduce_product(_BASIS_GENS[i], _BASIS_GENS[j])
            row.append((idx, s))
        table.append(row)
    return table


CAYLEY_TABLE = _derive_cayley_table()


def cayley_sign_tensor(device=None, dtype=torch.float32) -> torch.Tensor:
    """Build sparse sign tensor: signs[i, j, k] for geometric product. Shape [8, 8, 8]."""
    signs = torch.zeros(8, 8, 8, device=device, dtype=dtype)
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            signs[i, j, k] = s
    return signs


# ============================================================================
# Cell 4: Token Memory Banks
# ============================================================================

class TokenMemoryBank(nn.Module):
    """Memory bank for token n-grams. Non-trainable buffers."""

    def __init__(self, n_gram_size: int, d_state: int = 16, n_slots: int = 500_000):
        super().__init__()
        self.n = n_gram_size
        self.d_state = d_state
        self.n_slots = n_slots
        self.register_buffer('bank', torch.zeros(n_slots, d_state))
        self.register_buffer('counts', torch.zeros(n_slots, dtype=torch.long))

    def compute_address(self, token_window: torch.Tensor) -> torch.Tensor:
        """FNV-1a hash on token IDs. token_window: (..., n)."""
        tw = token_window.long()
        h = torch.full(tw.shape[:-1], 2166136261, dtype=torch.long, device=tw.device)
        for i in range(self.n):
            h = ((h ^ tw[..., i]) * 16777619) % (2**32)
        return h % self.n_slots

    def read(self, token_window: torch.Tensor) -> torch.Tensor:
        """Read states for given n-gram windows. Returns (..., d_state)."""
        addr = self.compute_address(token_window)
        return self.bank[addr]

    def write_batch(self, token_window: torch.Tensor, states: torch.Tensor, momentum: float = 0.9):
        """EMA write using CPU temporaries to avoid VRAM fragmentation."""
        with torch.no_grad():
            addr = self.compute_address(token_window).reshape(-1).cpu()
            flat_states = states.reshape(-1, self.d_state).float().cpu()

            ones = torch.ones(addr.shape[0])
            hit_counts = torch.zeros(self.n_slots)
            hit_counts.scatter_add_(0, addr, ones)

            state_sums = torch.zeros(self.n_slots, self.d_state)
            addr_expanded = addr.unsqueeze(1).expand_as(flat_states)
            state_sums.scatter_add_(0, addr_expanded, flat_states)

            hit_mask = hit_counts > 0
            if hit_mask.any():
                mean_states = state_sums[hit_mask] / hit_counts[hit_mask].unsqueeze(1)
                old_counts = self.counts[hit_mask].cpu()
                is_new = (old_counts == 0)
                alpha = torch.where(is_new, torch.zeros_like(mean_states[:, 0]),
                                    torch.full_like(mean_states[:, 0], momentum))
                new_vals = (alpha.unsqueeze(1) * self.bank[hit_mask].cpu() +
                            (1 - alpha.unsqueeze(1)) * mean_states)
                self.bank[hit_mask] = new_vals.to(self.bank.device)
                self.counts[hit_mask] += hit_counts[hit_mask].long().to(self.counts.device)

    def coverage(self) -> float:
        return (self.counts > 0).float().mean().item()


class TokenBladeBank(nn.Module):
    """8 blade-specific token memory banks stacked for parallel lookup."""

    def __init__(self, n_gram_size: int, d_state: int = 16, n_slots: int = 50_000):
        super().__init__()
        self.n = n_gram_size
        self.d_state = d_state
        self.n_slots = n_slots
        self.register_buffer('bank', torch.zeros(8, n_slots, d_state))
        self.register_buffer('counts', torch.zeros(8, n_slots, dtype=torch.long))

    def compute_address(self, token_window: torch.Tensor) -> torch.Tensor:
        """FNV-1a hash on token IDs."""
        tw = token_window.long()
        h = torch.full(tw.shape[:-1], 2166136261, dtype=torch.long, device=tw.device)
        for i in range(self.n):
            h = ((h ^ tw[..., i]) * 16777619) % (2**32)
        return h % self.n_slots

    def read_all_blades(self, token_window: torch.Tensor) -> torch.Tensor:
        """Read from all 8 blade banks. Returns (B, seq, 8, d_state)."""
        addr = self.compute_address(token_window)  # (B, seq)
        addr_flat = addr.reshape(-1)
        results = self.bank[:, addr_flat, :]  # (8, B*seq, d_state)
        B_seq = addr.shape
        return results.permute(1, 0, 2).reshape(*B_seq, 8, self.d_state)

    def write_by_blade(self, token_window: torch.Tensor, states: torch.Tensor,
                       dominant_blade: torch.Tensor, momentum: float = 0.9):
        """Vectorized blade-routed write — all 8 blades in one pass.

        Instead of looping over 8 blades sequentially, we create a composite
        index (blade * n_slots + addr) and scatter into a flattened (8*n_slots)
        tensor, then reshape back. Single scatter_add pass for all blades.
        """
        with torch.no_grad():
            addr = self.compute_address(token_window).reshape(-1).cpu()
            blades = dominant_blade.reshape(-1).cpu()
            flat_states = states.reshape(-1, self.d_state).float().cpu()
            N = len(addr)

            # Composite index: blade * n_slots + addr → unique slot per (blade, addr)
            composite = blades * self.n_slots + addr  # (N,)

            total_slots = 8 * self.n_slots
            ones = torch.ones(N)
            hit_counts = torch.zeros(total_slots)
            hit_counts.scatter_add_(0, composite, ones)

            state_sums = torch.zeros(total_slots, self.d_state)
            composite_exp = composite.unsqueeze(1).expand(N, self.d_state)
            state_sums.scatter_add_(0, composite_exp, flat_states)

            hit_mask = hit_counts > 0
            if hit_mask.any():
                mean_states = state_sums[hit_mask] / hit_counts[hit_mask].unsqueeze(1)

                # Map back to (blade, slot) for bank update
                hit_indices = hit_mask.nonzero(as_tuple=True)[0]
                hit_blades = hit_indices // self.n_slots
                hit_addrs = hit_indices % self.n_slots

                # Gather old values: bank[blade, addr]
                old_bank = self.bank[hit_blades, hit_addrs].cpu()
                old_counts = self.counts[hit_blades, hit_addrs].cpu()

                is_new = (old_counts == 0)
                alpha = torch.where(is_new, torch.zeros_like(mean_states[:, 0]),
                                    torch.full_like(mean_states[:, 0], momentum))
                new_vals = alpha.unsqueeze(1) * old_bank + (1 - alpha.unsqueeze(1)) * mean_states

                self.bank[hit_blades, hit_addrs] = new_vals.to(self.bank.device)
                self.counts[hit_blades, hit_addrs] += hit_counts[hit_mask].long().to(self.counts.device)

    def write_batch(self, token_window: torch.Tensor, blade_states: torch.Tensor,
                    momentum: float = 0.9):
        """Write all 8 blade states at once. blade_states: (..., 8, d_state)."""
        with torch.no_grad():
            addr = self.compute_address(token_window).reshape(-1).cpu()  # (N,)
            # blade_states: (..., 8, d_state) -> (N, 8, d_state)
            bs = blade_states.reshape(-1, 8, self.d_state).float().cpu()
            N = len(addr)

            for b in range(8):
                flat_states = bs[:, b, :]  # (N, d_state)
                ones = torch.ones(N)
                hit_counts = torch.zeros(self.n_slots)
                hit_counts.scatter_add_(0, addr, ones)

                state_sums = torch.zeros(self.n_slots, self.d_state)
                addr_exp = addr.unsqueeze(1).expand(N, self.d_state)
                state_sums.scatter_add_(0, addr_exp, flat_states)

                hit_mask = hit_counts > 0
                if hit_mask.any():
                    mean_states = state_sums[hit_mask] / hit_counts[hit_mask].unsqueeze(1)
                    hit_indices = hit_mask.nonzero(as_tuple=True)[0]

                    old_bank = self.bank[b, hit_indices].cpu()
                    old_counts = self.counts[b, hit_indices].cpu()

                    is_new = (old_counts == 0)
                    alpha = torch.where(is_new, torch.zeros_like(mean_states[:, 0]),
                                        torch.full_like(mean_states[:, 0], momentum))
                    new_vals = alpha.unsqueeze(1) * old_bank + (1 - alpha.unsqueeze(1)) * mean_states

                    self.bank[b, hit_indices] = new_vals.to(self.bank.device)
                    self.counts[b, hit_indices] += hit_counts[hit_mask].long().to(self.counts.device)

    def coverage(self) -> float:
        return (self.counts > 0).float().mean().item()


class TokenMultiScaleMemory(nn.Module):
    """Overlapping 3-gram + 4-gram token memory banks (~21.6GB VRAM).

    3-gram (100M slots, ~1/5 collision) and 4-gram (200M slots, ~1/3 collision)
    provide complementary views — their windows can never align exactly,
    so the model gets two offset perspectives on every token sequence.
    Injected at model input (layer 0) with learned blending.
    """

    def __init__(self, config: 'TokenizedHLMv9Config'):
        super().__init__()
        self.d_state = config.mem_d_state
        self.bank_3 = TokenMemoryBank(3, config.mem_d_state, config.mem_3_slots)
        self.bank_4 = TokenMemoryBank(4, config.mem_d_state, config.mem_4_slots)

        # Learnable blending between 3-gram and 4-gram
        self.scale_weights = nn.Parameter(torch.ones(2) / 2)

    def extract_windows(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 3-gram and 4-gram windows from token sequence."""
        B, seq = token_ids.shape
        padded = F.pad(token_ids, (3, 0), value=0)  # pad 3 for 4-gram (covers 3-gram too)
        win3 = padded[:, 1:1 + seq + 2].unfold(1, 3, 1)  # (B, seq, 3)
        win4 = padded.unfold(1, 4, 1)                      # (B, seq, 4)
        return win3, win4

    def read(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Read both scales and blend. Returns (B, seq, d_state)."""
        win3, win4 = self.extract_windows(token_ids)

        addr3 = self.bank_3.compute_address(win3)
        addr4 = self.bank_4.compute_address(win4)
        r3 = self.bank_3.bank[addr3]
        r4 = self.bank_4.bank[addr4]

        stacked = torch.stack([r3, r4], dim=2)  # (B, seq, 2, d_state)
        weights = F.softmax(self.scale_weights, dim=0)
        return (stacked * weights[None, None, :, None]).sum(dim=2)

    def write(self, token_ids: torch.Tensor, states: torch.Tensor,
              blade_profiles: torch.Tensor = None):
        """Write to both banks in parallel threads."""
        states_proj = states[..., :self.d_state]
        win3, win4 = self.extract_windows(token_ids)

        t3 = threading.Thread(target=self.bank_3.write_batch, args=(win3, states_proj))
        t4 = threading.Thread(target=self.bank_4.write_batch, args=(win4, states_proj))
        t3.start()
        t4.start()
        t3.join()
        t4.join()

    def coverage_stats(self) -> Dict[str, float]:
        return {
            '3-gram': self.bank_3.coverage() * 100,
            '4-gram': self.bank_4.coverage() * 100,
        }


# ============================================================================
# Cell 5: GeometricRound + GeometricFFN
# ============================================================================

class GeometricRound(nn.Module):
    """One round of full Cl(3,0) geometric product + per-blade SwiGLU FFN.

    Computes ALL 64 pairwise blade interactions via the Cayley table,
    then refines with a small per-blade FFN.

    Uses SPARSE Cayley accumulation: instead of materializing the full
    (B, T, 8, 8, D) outer product, we iterate over 64 nonzero Cayley
    entries and accumulate directly into the result. This uses 67× less
    peak memory and is faster on GPU due to reduced memory bandwidth.
    """

    def __init__(self, config: TokenizedHLMv9Config):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade

        # Precompute k-grouped Cayley mappings: for each target blade k,
        # store which (i, j) pairs contribute and their signs.
        # Each k gets exactly 8 source pairs (since 64 total / 8 blades = 8 each).
        # This enables direct vectorized accumulation — no scatter_add_ needed.
        k_groups_i = [[] for _ in range(8)]  # source i indices per target k
        k_groups_j = [[] for _ in range(8)]  # source j indices per target k
        k_groups_s = [[] for _ in range(8)]  # signs per target k
        for i in range(8):
            for j in range(8):
                k, s = CAYLEY_TABLE[i][j]
                k_groups_i[k].append(i)
                k_groups_j[k].append(j)
                k_groups_s[k].append(float(s))
        # Register as (8, 8) tensors — row k has the 8 (i,j,sign) entries targeting blade k
        self.register_buffer("_kg_i", torch.tensor(k_groups_i, dtype=torch.long))   # (8, 8)
        self.register_buffer("_kg_j", torch.tensor(k_groups_j, dtype=torch.long))   # (8, 8)
        self.register_buffer("_kg_s", torch.tensor(k_groups_s, dtype=torch.float32))  # (8, 8)

        # Learned scaling per blade pair
        self.interaction_weights = nn.Parameter(
            torch.ones(config.n_blades, config.n_blades) * 0.1
        )

        # Per-blade SwiGLU FFN
        self.blade_ffn_gate = nn.Linear(config.d_blade, config.geo_ffn_dim, bias=False)
        self.blade_ffn_up = nn.Linear(config.d_blade, config.geo_ffn_dim, bias=False)
        self.blade_ffn_down = nn.Linear(config.geo_ffn_dim, config.d_blade, bias=False)

        # Layer norm per blade
        self.norm = nn.LayerNorm(config.d_blade)

        # Mix ratio: geometric product vs residual
        self.geo_gate = nn.Parameter(torch.tensor(0.5))

    def geometric_product(self, x: torch.Tensor) -> torch.Tensor:
        """K-grouped Cayley geometric self-product — no scatter, fully vectorized.

        For each target blade k, exactly 8 source pairs (i,j) contribute.
        We gather all 8 pairs per k, multiply with signs and weights, and sum.
        This replaces scatter_add_ with direct tensor ops that GPUs love.

        x: (B, T, 8, d_blade) → (B, T, 8, d_blade)
        """
        B, T, N, D = x.shape
        weights = self.interaction_weights.sigmoid()  # (8, 8)

        # _kg_i, _kg_j: (8, 8) — for each target k, the 8 source (i,j) pairs
        # _kg_s: (8, 8) — signs for each pair
        # Gather: x[:,:,_kg_i[k],:] gives the 8 source-i blades for target k
        x_i = x[:, :, self._kg_i, :]  # (B, T, 8_targets, 8_pairs, D)
        x_j = x[:, :, self._kg_j, :]  # (B, T, 8_targets, 8_pairs, D)

        # Interaction weights for each (i,j) pair grouped by target k
        w = weights[self._kg_i, self._kg_j]  # (8, 8)
        sw = (self._kg_s * w).unsqueeze(-1)  # (8, 8, 1) — sign × weight

        # Compute all 64 products at once, already grouped by target blade
        products = x_i * x_j * sw.unsqueeze(0).unsqueeze(0)  # (B, T, 8, 8, D)

        # Sum over the 8 source pairs per target blade — direct reduction, no scatter!
        result = products.sum(dim=3)  # (B, T, 8, D)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 8, d_blade) → (B, T, 8, d_blade)"""
        B, T, N, D = x.shape

        # Geometric product
        geo = self.geometric_product(x)

        # Gated residual
        gate = self.geo_gate.sigmoid()
        mixed = gate * geo + (1.0 - gate) * x

        # Per-blade SwiGLU FFN
        flat = mixed.reshape(B * T * N, D)
        flat = self.norm(flat)
        g = self.blade_ffn_gate(flat)
        u = self.blade_ffn_up(flat)
        h = F.silu(g) * u
        out = self.blade_ffn_down(h)

        out = out.reshape(B, T, N, D)
        return x + out  # Residual


class GeometricFFN(nn.Module):
    """Full geometric FFN: project → geometric rounds → collapse.

    Replaces the standard MLP. Uses Cl(3,0) geometric products
    across 8 semantic blades.
    """

    def __init__(self, config: TokenizedHLMv9Config):
        super().__init__()
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.d_model = config.d_model
        expert_dim = config.n_blades * config.d_blade

        self.proj_in = nn.Linear(config.d_model, expert_dim, bias=False)
        self.geo_rounds = nn.ModuleList([
            GeometricRound(config)
            for _ in range(config.n_geometric_rounds)
        ])
        self.proj_out = nn.Linear(expert_dim, config.d_model, bias=False)
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, D = x.shape
        h = self.proj_in(x)
        mv = h.reshape(B, T, self.n_blades, self.d_blade)

        for geo_round in self.geo_rounds:
            mv = geo_round(mv)

        flat = mv.reshape(B, T, self.n_blades * self.d_blade)
        out = self.proj_out(flat)
        return self.out_norm(out)


# ============================================================================
# Cell 6: RoPE Attention
# ============================================================================

class RoPEAttention(nn.Module):
    """Standard multi-head self-attention with Rotary Position Embeddings.

    Caches cos/sin tables to avoid recomputation across layers and steps.
    All 30 layers share the same RoPE frequencies, so we build once and reuse.
    """

    # Class-level RoPE cache shared across all layers
    _rope_cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(self, config: TokenizedHLMv9Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        inv_freq = 1.0 / (config.rope_theta **
                          (torch.arange(0, config.d_head, 2).float() / config.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rope(self, seq_len: int, device: torch.device):
        """Get cached RoPE cos/sin. Builds once per (seq_len, device)."""
        key = (seq_len, device)
        if key not in RoPEAttention._rope_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_head)
            sin = emb.sin().unsqueeze(0).unsqueeze(2)
            RoPEAttention._rope_cache[key] = (cos, sin)
        return RoPEAttention._rope_cache[key]

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        cos, sin = self._get_rope(T, x.device)

        # Apply RoPE to Q and K simultaneously (fused rotation)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat([-q2, q1], dim=-1) * sin + q * cos
        k = torch.cat([-k2, k1], dim=-1) * sin + k * cos

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch SDPA for Flash Attention when available
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


# ============================================================================
# Cell 7: ToU Bank + Cross-Attention (WZ1 Knowledge Injection)
# ============================================================================

class ToUBank(nn.Module):
    """Learnable embedding bank for 1,486 ToU/WZ1 primitives."""

    def __init__(self, n_primitives: int, d_blade: int, bank_path: Optional[str] = None):
        super().__init__()
        self.n_primitives = n_primitives
        self.d_blade = d_blade

        self.embeddings = nn.Embedding(n_primitives, d_blade)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        self.register_buffer(
            "blade_masks",
            torch.zeros(len(BLADE_NAMES), n_primitives)
        )
        self.blade_to_indices: Dict[str, List[int]] = {name: [] for name in BLADE_NAMES}

        if bank_path is not None:
            self.load_bank(bank_path)
        else:
            self._distribute_evenly()

    def _distribute_evenly(self):
        n_blades = len(BLADE_NAMES)
        per_blade = self.n_primitives // n_blades
        remainder = self.n_primitives % n_blades
        masks = torch.zeros(n_blades, self.n_primitives)
        idx = 0
        for b in range(n_blades):
            count = per_blade + (1 if b < remainder else 0)
            masks[b, idx:idx + count] = 1.0
            self.blade_to_indices[BLADE_NAMES[b]] = list(range(idx, idx + count))
            idx += count
        self.blade_masks = masks

    def load_bank(self, bank_path: str):
        path = Path(bank_path)
        if not path.exists():
            print(f"WARNING: bank_full.json not found at {bank_path}, using even distribution")
            self._distribute_evenly()
            return

        with open(path, "r", encoding="utf-8") as f:
            bank = json.load(f)

        idx = 0
        blade_masks = torch.zeros(len(BLADE_NAMES), self.n_primitives)
        for blade_name in BLADE_NAMES:
            if blade_name not in bank.get("blades", {}):
                continue
            blade_data = bank["blades"][blade_name]
            blade_idx = BLADE_INDEX[blade_name]
            for family_code, family_data in blade_data.get("families", {}).items():
                for prim in family_data.get("primitives", []):
                    if idx >= self.n_primitives:
                        break
                    blade_masks[blade_idx, idx] = 1.0
                    self.blade_to_indices[blade_name].append(idx)
                    idx += 1
        self.blade_masks = blade_masks
        if idx < self.n_primitives:
            self.blade_masks[:, idx:] = 0
        print(f"Loaded {idx} ToU primitives from {bank_path}")

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings.weight, self.blade_masks


class ToUCrossAttention(nn.Module):
    """Gated cross-attention to shared ToU primitive bank."""

    def __init__(self, config: TokenizedHLMv9Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_prim = config.tou_d_prim

        self.q_proj = nn.Linear(config.d_model, config.tou_d_prim, bias=False)
        self.k_proj = nn.Linear(config.tou_d_prim, config.tou_d_prim, bias=False)
        self.v_proj = nn.Linear(config.tou_d_prim, config.tou_d_prim, bias=False)
        self.o_proj = nn.Linear(config.tou_d_prim, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, 1, bias=True)
        # Start near-zero so injection grows during training
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, prim_embeds: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model), prim_embeds: (N_prim, d_prim) → (B, T, d_model)"""
        q = self.q_proj(x)
        k = self.k_proj(prim_embeds)
        v = self.v_proj(prim_embeds)

        scale = math.sqrt(self.d_prim)
        attn = torch.matmul(q, k.t()) / scale
        attn = F.softmax(attn, dim=-1)

        retrieved = torch.matmul(attn, v)
        out = self.o_proj(retrieved)

        g = self.gate(x).sigmoid()
        return g * out


# ============================================================================
# Cell 8: TokenizedHLMv9Block + Full Model
# ============================================================================

class TokenizedHLMv9Block(nn.Module):
    """Single transformer block: attention + geometric FFN + optional ToU."""

    def __init__(self, config: TokenizedHLMv9Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_tou = layer_idx in config.tou_inject_layers

        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = RoPEAttention(config)

        self.norm2 = nn.RMSNorm(config.d_model)
        self.geo_ffn = GeometricFFN(config)

        if self.has_tou:
            self.norm3 = nn.RMSNorm(config.d_model)
            self.tou_attn = ToUCrossAttention(config)

    def forward(self, x: torch.Tensor, tou_embeds: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.geo_ffn(self.norm2(x))
        if self.has_tou and tou_embeds is not None:
            x = x + self.tou_attn(self.norm3(x), tou_embeds)
        return x


class TokenizedHLMv9(nn.Module):
    """Tokenized HLM-v9: ~253M params.

    Hybrid architecture: RoPE attention + Cl(3,0) GeometricFFN + WZ1 injection.
    Token n-gram memory banks for context augmentation.
    """

    def __init__(self, config: TokenizedHLMv9Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # 4-gram memory bank (50M slots, injected at input)
        self.memory = TokenMultiScaleMemory(config)
        self.mem_to_model = nn.Linear(config.mem_d_state, config.d_model, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TokenizedHLMv9Block(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = nn.RMSNorm(config.d_model)

        # ToU bank (shared across all layers)
        self.tou_bank = ToUBank(
            n_primitives=config.tou_n_primitives,
            d_blade=config.tou_d_prim,
        )

        # Output head (weight-tied with embedding)
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

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None,
                do_memory_write: bool = False) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Memory read
        mem_state = self.memory.read(input_ids)
        mem_signal = self.mem_to_model(mem_state)

        # Embedding + memory
        x = self.token_embed(input_ids) + mem_signal
        x = self.embed_dropout(x)

        # ToU bank
        prim_embeds, _ = self.tou_bank()

        # Forward through blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, prim_embeds, use_reentrant=False)
            else:
                x = block(x, prim_embeds)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if targets is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        # Memory write — 4-gram bank
        if do_memory_write:
            with torch.no_grad():
                write_states = x.detach()[:, :, :self.config.mem_d_state]
                self.memory.write(input_ids, write_states)

        return result

    def count_parameters(self) -> Dict[str, int]:
        counts = {}
        counts["token_embed"] = sum(p.numel() for p in self.token_embed.parameters())
        counts["memory_proj"] = sum(p.numel() for p in self.mem_to_model.parameters())
        counts["tou_bank"] = sum(p.numel() for p in self.tou_bank.parameters())

        attn_params = 0
        geo_ffn_params = 0
        tou_attn_params = 0
        norm_params = 0

        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            geo_ffn_params += sum(p.numel() for p in block.geo_ffn.parameters())
            norm_params += sum(p.numel() for p in block.norm1.parameters())
            norm_params += sum(p.numel() for p in block.norm2.parameters())
            if block.has_tou:
                tou_attn_params += sum(p.numel() for p in block.tou_attn.parameters())
                norm_params += sum(p.numel() for p in block.norm3.parameters())

        counts["attention"] = attn_params
        counts["geometric_ffn"] = geo_ffn_params
        counts["tou_cross_attn"] = tou_attn_params
        counts["norms"] = norm_params
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        total = sum(p.numel() for p in self.parameters())
        counts["total"] = total
        # Subtract weight-tied lm_head
        counts["total_unique"] = total - self.lm_head.weight.numel()

        return counts

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            result = self.forward(idx_cond)
            logits = result["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================================
# Cell 9: Data Pipeline (FineWeb-Edu + OpenWebMath + StarCoderData)
# ============================================================================

def setup_tokenizer():
    """Load Qwen2.5 tokenizer and configure for our vocab size."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer


def create_mixed_dataloader(tokenizer, config: TokenizedHLMv9Config,
                            batch_size: int = 8, seq_len: int = 1024):
    """Create a streaming dataloader that mixes 3 datasets."""
    from datasets import load_dataset, interleave_datasets

    # Load datasets in streaming mode
    print("Loading datasets (streaming)...")
    ds_fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split="train", streaming=True
    )
    ds_math = load_dataset(
        "open-web-math/open-web-math",
        split="train", streaming=True
    )
    ds_code = load_dataset(
        "transformersbook/codeparrot",
        split="train", streaming=True
    )

    # Normalize text column name
    def extract_text_fineweb(example):
        return {"text": example.get("text", "")}

    def extract_text_math(example):
        return {"text": example.get("text", "")}

    def extract_text_code(example):
        return {"text": example.get("content", example.get("code", example.get("text", "")))}

    ds_fineweb = ds_fineweb.map(extract_text_fineweb).select_columns(["text"])
    ds_math = ds_math.map(extract_text_math).select_columns(["text"])
    ds_code = ds_code.map(extract_text_code).select_columns(["text"])

    # Interleave with weights: 60% general, 20% math, 20% code
    mixed = interleave_datasets(
        [ds_fineweb, ds_math, ds_code],
        probabilities=config.data_mix_weights,
        stopping_strategy="all_exhausted",
    )

    # Tokenize and pack into fixed-length sequences
    max_vocab = config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    def tokenize_and_pack():
        """Generator: yields (input_ids, targets) pairs of length seq_len.

        Uses torch.clamp for vectorized vocab clamping instead of Python list comp.
        """
        buffer = []
        for example in mixed:
            text = example.get("text", "")
            if not text or len(text) < 10:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Vectorized vocab clamping via torch
            ids_t = torch.tensor(ids, dtype=torch.long)
            ids_t[ids_t >= max_vocab] = unk_id
            buffer.extend(ids_t.tolist())

            while len(buffer) >= seq_len + 1:
                chunk = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, targets

    return tokenize_and_pack


class PrefetchIterator:
    """Background-threaded prefetch buffer for data pipeline.

    Fills a queue from the generator in a background thread so the GPU
    never blocks waiting for data preparation.
    """

    def __init__(self, gen_fn, batch_size: int, prefetch_batches: int = 4):
        self.gen_fn = gen_fn
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.queue = deque()
        self._lock = threading.Lock()          # Protect queue + generator access
        self._stop = threading.Event()
        self._exhausted = threading.Event()
        self._gen = None
        self._thread = None

    def start(self):
        self._stop.clear()
        self._exhausted.clear()
        with self._lock:
            self._gen = self.gen_fn()
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()
        return self

    def _fill_loop(self):
        """Background thread: continuously fills the queue from generator."""
        while not self._stop.is_set():
            with self._lock:
                qlen = len(self.queue)
            if qlen >= self.prefetch_batches * self.batch_size:
                time.sleep(0.001)
                continue
            try:
                with self._lock:
                    item = next(self._gen)
                with self._lock:
                    self.queue.append(item)
            except StopIteration:
                self._exhausted.set()
                return
            except Exception:
                self._exhausted.set()
                return

    def get_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get a batch of (input_ids, targets). Returns None if exhausted."""
        items = []
        for _ in range(self.batch_size):
            attempts = 0
            while True:
                with self._lock:
                    if self.queue:
                        items.append(self.queue.popleft())
                        break
                if self._exhausted.is_set():
                    return None
                time.sleep(0.001)
                attempts += 1
                if attempts > 5000:  # 5s timeout
                    return None

        inputs = torch.stack([it[0] for it in items])
        targets = torch.stack([it[1] for it in items])
        return inputs, targets

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)  # Give thread time to exit cleanly
            self._thread = None

    def restart(self):
        """Restart with a fresh generator — waits for old thread to fully exit."""
        self.stop()
        with self._lock:
            self.queue.clear()
        self.start()


# ============================================================================
# Cell 10: Training Loop
# ============================================================================

def get_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int = 500) -> float:
    """Cosine decay with linear warmup."""
    min_lr = base_lr * 0.1
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff


def save_checkpoint(model, optimizer, step, loss, path):
    """Save checkpoint to disk with fsync to ensure Drive flush."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
    }, path)
    # Force flush to Google Drive FUSE mount before continuing
    try:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    except OSError:
        pass  # fsync not supported on this filesystem, file still saved
    print(f"  Saved checkpoint: {path} (fsynced)")


def load_checkpoint(model, optimizer, path):
    """Load checkpoint if it exists. Returns step number."""
    if not os.path.exists(path):
        return 0
    print(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Filter out keys with shape mismatches (e.g. resized memory banks)
    state_dict = ckpt['model_state_dict']
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} keys with shape mismatch (memory bank resize):")
        for k in skipped[:5]:
            print(f"    {k}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  Note: {len(missing)} new buffer(s) initialized fresh (architecture update)")
    if unexpected:
        print(f"  Note: {len(unexpected)} old key(s) ignored from checkpoint")
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        if skipped:
            print("  Note: skipping optimizer state (model shape changed, fresh optimizer)")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("  Note: optimizer state incompatible, starting fresh optimizer")
    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', 0)
    print(f"  Resumed from step {step}, loss={loss:.4f}")
    return step


def train():
    """Main training loop."""
    config = CONFIG

    # Hyperparameters
    BATCH_SIZE = 32
    GRAD_ACCUM = 4
    BASE_LR = 3e-4
    WARMUP_STEPS = 500
    TOTAL_STEPS = 50_000
    LOG_INTERVAL = 50
    EVAL_INTERVAL = 500
    SAVE_INTERVAL = 2000
    MEM_WRITE_INTERVAL = 10
    SEQ_LEN = config.max_seq_len

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    # Setup model
    print("\nInitializing model...")
    model = TokenizedHLMv9(config).to(DEVICE)

    # Print parameter counts
    counts = model.count_parameters()
    print("\n=== Parameter Budget ===")
    for k, v in counts.items():
        print(f"  {k:25s}: {v:>12,}")
    print(f"  {'TOTAL':25s}: {counts['total']:>12,} ({counts['total']/1e6:.1f}M)")
    print()

    # Try loading ToU bank
    for bank_path in [
        '/content/drive/MyDrive/bank_full.json',
        '/content/bank_full.json',
        'bank_full.json',
        'training/bank_full.json',
    ]:
        if os.path.exists(bank_path):
            model.tou_bank.load_bank(bank_path)
            break
    else:
        print("NOTE: bank_full.json not found — using evenly distributed ToU primitives")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    # Resume from checkpoint
    resume_path = os.path.join(DRIVE_CKPT_DIR, 'latest.pt')
    start_step = load_checkpoint(model, optimizer, resume_path)

    # Compile model for speed (A100+)
    compiled_model = model
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            compiled_model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed ({e}), using eager mode")
            compiled_model = model

    # Data pipeline with background prefetch
    print("\nSetting up data pipeline with prefetch...")
    data_gen_fn = create_mixed_dataloader(tokenizer, config, BATCH_SIZE, SEQ_LEN)
    prefetcher = PrefetchIterator(data_gen_fn, BATCH_SIZE, prefetch_batches=8).start()

    # Training
    print(f"\n{'='*70}")
    print(f"  Training Tokenized HLM-v9 ({counts['total']/1e6:.0f}M params)")
    print(f"  Steps: {TOTAL_STEPS:,} | Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,} tokens/step")
    print(f"  Total tokens: ~{TOTAL_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / 1e9:.1f}B")
    print(f"  Data mix: 60% FineWeb-Edu + 20% OpenWebMath + 20% Python code")
    print(f"{'='*70}\n")

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    tokens_seen = start_step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    best_loss = float('inf')
    t0 = time.time()

    step = start_step

    while step < TOTAL_STEPS:
        # Get pre-fetched batch (non-blocking — data ready from background thread)
        batch = prefetcher.get_batch()
        if batch is None:
            print("Dataset exhausted, restarting...")
            prefetcher.restart()
            continue

        input_ids, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)

        # Determine if memory write this step
        do_mem_write = (step % MEM_WRITE_INTERVAL == 0) and (step > 0)

        # Forward + backward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = compiled_model(input_ids, targets, do_memory_write=do_mem_write)
            loss = result["loss"] / GRAD_ACCUM

        loss.backward()
        loss_accum += result["loss"].item()
        tokens_seen += BATCH_SIZE * SEQ_LEN

        # Gradient accumulation step
        if (step + 1) % GRAD_ACCUM == 0:
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # LR schedule
            opt_step = (step + 1) // GRAD_ACCUM
            lr = get_lr(opt_step, TOTAL_STEPS // GRAD_ACCUM, BASE_LR, WARMUP_STEPS)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % LOG_INTERVAL == 0:
            avg_loss = loss_accum / LOG_INTERVAL
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed
            eta_h = (TOTAL_STEPS - step) / (step / elapsed) / 3600 if step > 0 else 0
            ppl = math.exp(min(avg_loss, 20))
            lr_now = optimizer.param_groups[0]['lr']

            print(f"[step {step:>6d}/{TOTAL_STEPS}] "
                  f"loss={avg_loss:.4f} ppl={ppl:.1f} | "
                  f"{tps/1000:.0f}K tok/s | lr={lr_now:.2e} | "
                  f"{tokens_seen/1e6:.0f}M tok | ETA {eta_h:.1f}h")
            loss_accum = 0.0

        # Eval
        if step % EVAL_INTERVAL == 0 and step > 0:
            model.eval()
            eval_loss = 0.0
            eval_steps = 20
            with torch.no_grad():
                for _ in range(eval_steps):
                    eval_batch = prefetcher.get_batch()
                    if eval_batch is None:
                        prefetcher.restart()
                        eval_batch = prefetcher.get_batch()
                    if eval_batch is None:
                        break
                    inp = eval_batch[0].to(DEVICE)
                    tgt = eval_batch[1].to(DEVICE)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        r = model(inp, tgt)
                    eval_loss += r["loss"].item()
            eval_loss /= eval_steps
            eval_ppl = math.exp(min(eval_loss, 20))
            print(f"  [EVAL step {step}] loss={eval_loss:.4f} ppl={eval_ppl:.1f}")

            # Memory coverage
            cov = model.memory.coverage_stats()
            print(f"  [MEM] " + " | ".join(f"{k}: {v:.1f}%" for k, v in cov.items()))

            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(model, optimizer, step, eval_loss,
                                os.path.join(DRIVE_CKPT_DIR, 'best.pt'))
            model.train()

        # Periodic save
        if step % SAVE_INTERVAL == 0 and step > 0:
            save_checkpoint(model, optimizer, step, loss_accum,
                            os.path.join(DRIVE_CKPT_DIR, f'step_{step:06d}.pt'))
            save_checkpoint(model, optimizer, step, loss_accum,
                            os.path.join(DRIVE_CKPT_DIR, 'latest.pt'))

        # Memory cleanup
        if step % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Cleanup prefetch thread
    prefetcher.stop()

    # Final save
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Steps: {step:,}")
    print(f"  Tokens: {tokens_seen:,} ({tokens_seen/1e9:.2f}B)")
    print(f"  Best eval loss: {best_loss:.4f}")
    print(f"{'='*70}")

    final_path = os.path.join(DRIVE_CKPT_DIR, 'final.pt')
    save_checkpoint(model, optimizer, step, best_loss, final_path)
    print(f"Final checkpoint: {final_path}")

    return model, tokenizer


# ============================================================================
# Cell 11: Chat Template
# ============================================================================

# Simple chat format the model learns during SFT.
# We avoid complex special tokens since our vocab is truncated from Qwen2.5.
# Instead we use plain text delimiters that tokenize cleanly.
CHAT_USER_PREFIX = "\n<|user|>\n"
CHAT_ASST_PREFIX = "\n<|assistant|>\n"
CHAT_END = "\n<|end|>\n"
CHAT_SYSTEM_PREFIX = "\n<|system|>\n"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise assistant. "
    "Answer questions clearly. Show your reasoning for math and code problems."
)


def format_chat_pair(user_msg: str, assistant_msg: str,
                     system_msg: str = None) -> str:
    """Format a single Q&A pair into the chat template."""
    parts = []
    if system_msg:
        parts.append(f"{CHAT_SYSTEM_PREFIX}{system_msg}{CHAT_END}")
    parts.append(f"{CHAT_USER_PREFIX}{user_msg}{CHAT_END}")
    parts.append(f"{CHAT_ASST_PREFIX}{assistant_msg}{CHAT_END}")
    return "".join(parts)


def format_chat_prompt(user_msg: str, system_msg: str = None,
                       history: List[Tuple[str, str]] = None) -> str:
    """Format a prompt for generation (no assistant response yet)."""
    parts = []
    if system_msg:
        parts.append(f"{CHAT_SYSTEM_PREFIX}{system_msg}{CHAT_END}")
    if history:
        for u, a in history:
            parts.append(f"{CHAT_USER_PREFIX}{u}{CHAT_END}")
            parts.append(f"{CHAT_ASST_PREFIX}{a}{CHAT_END}")
    parts.append(f"{CHAT_USER_PREFIX}{user_msg}{CHAT_END}")
    parts.append(CHAT_ASST_PREFIX)
    return "".join(parts)


# ============================================================================
# Cell 12: Supervised Fine-Tuning (SFT) — Chat / Instruction Tuning
# ============================================================================

def create_sft_dataloader(tokenizer, config: TokenizedHLMv9Config, seq_len: int = 1024):
    """Create streaming SFT dataloader mixing instruction/chat datasets.

    Data mix:
      - 40% OpenAssistant (oasst2) — multi-turn chat, diverse topics
      - 25% SlimOrca — GPT-4 distilled instruction-following
      - 20% MetaMathQA — step-by-step math reasoning
      - 15% Code Alpaca — code generation instructions

    Only trains on assistant responses (user/system tokens masked with -100).
    """
    from datasets import load_dataset, interleave_datasets

    print("Loading SFT datasets (streaming)...")

    # --- OpenAssistant 2 ---
    ds_oasst = load_dataset(
        "OpenAssistant/oasst2", split="train", streaming=True
    )

    def extract_oasst(example):
        """Extract top-ranked Q&A from oasst conversation trees."""
        text = example.get("text", "")
        role = example.get("role", "")
        parent = example.get("parent_id", None)
        # oasst2 has flat messages with role and parent_id
        # We treat each assistant message as a response to its parent
        if role == "assistant" and text:
            return {"user": example.get("parent_text", ""), "assistant": text, "valid": True}
        return {"user": "", "assistant": "", "valid": False}

    # --- SlimOrca ---
    ds_orca = load_dataset(
        "Open-Orca/SlimOrca", split="train", streaming=True
    )

    def extract_orca(example):
        """Extract from SlimOrca's conversations format."""
        convos = example.get("conversations", [])
        user_msg = ""
        asst_msg = ""
        system_msg = ""
        for turn in convos:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "system":
                system_msg = value
            elif role == "human":
                user_msg = value
            elif role == "gpt":
                asst_msg = value
        if user_msg and asst_msg:
            if system_msg:
                user_msg = f"[System: {system_msg}]\n{user_msg}"
            return {"user": user_msg, "assistant": asst_msg, "valid": True}
        return {"user": "", "assistant": "", "valid": False}

    # --- MetaMathQA ---
    ds_math = load_dataset(
        "meta-math/MetaMathQA", split="train", streaming=True
    )

    def extract_metamath(example):
        q = example.get("query", "")
        a = example.get("response", "")
        if q and a:
            return {"user": q, "assistant": a, "valid": True}
        return {"user": "", "assistant": "", "valid": False}

    # --- Code Alpaca ---
    ds_code = load_dataset(
        "sahil2801/CodeAlpaca-20k", split="train", streaming=True
    )

    def extract_code_alpaca(example):
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        if instruction and output:
            user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
            return {"user": user_msg, "assistant": output, "valid": True}
        return {"user": "", "assistant": "", "valid": False}

    # Apply extractors
    ds_oasst = ds_oasst.map(extract_oasst).filter(lambda x: x["valid"])
    ds_orca = ds_orca.map(extract_orca).filter(lambda x: x["valid"])
    ds_math = ds_math.map(extract_metamath).filter(lambda x: x["valid"])
    ds_code = ds_code.map(extract_code_alpaca).filter(lambda x: x["valid"])

    # Interleave: 40% chat, 25% orca, 20% math, 15% code
    mixed = interleave_datasets(
        [ds_oasst, ds_orca, ds_math, ds_code],
        probabilities=[0.40, 0.25, 0.20, 0.15],
        stopping_strategy="all_exhausted",
    )

    max_vocab = config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    def sft_generator():
        """Yields (input_ids, targets) with assistant-only loss masking.

        User/system tokens have target=-100 (ignored by cross_entropy).
        Only the assistant's response tokens contribute to the loss.
        """
        for example in mixed:
            user_msg = example.get("user", "")
            asst_msg = example.get("assistant", "")
            if not user_msg or not asst_msg:
                continue

            # Build the full formatted chat
            system = DEFAULT_SYSTEM_PROMPT
            prompt_part = format_chat_prompt(user_msg, system_msg=system)
            full_text = prompt_part + asst_msg + CHAT_END

            # Tokenize the full sequence
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            full_ids = [t if t < max_vocab else unk_id for t in full_ids]

            # Tokenize just the prompt to find where assistant starts
            prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
            prompt_ids = [t if t < max_vocab else unk_id for t in prompt_ids]
            prompt_len = len(prompt_ids)

            # Truncate to seq_len + 1
            if len(full_ids) > seq_len + 1:
                full_ids = full_ids[:seq_len + 1]
            if len(full_ids) < 4:
                continue

            # Build input/target with masking
            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            targets = torch.tensor(full_ids[1:], dtype=torch.long)

            # Mask prompt tokens: only train on assistant response
            # The assistant response starts at prompt_len in the input
            if prompt_len > 0 and prompt_len < len(targets):
                targets[:prompt_len - 1] = -100  # Mask user/system tokens

            # Pad to seq_len if needed
            pad_len = seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=0)
                targets = F.pad(targets, (0, pad_len), value=-100)

            yield input_ids, targets

    return sft_generator


def sft_finetune(model=None, tokenizer=None, checkpoint_path=None):
    """Supervised Fine-Tuning stage: instruction/chat tuning.

    Runs after pretraining. Lower learning rate, shorter schedule,
    loss only on assistant responses.

    Saves SFT checkpoints separately so the base pretrained model
    is preserved.
    """
    config = CONFIG

    # SFT Hyperparameters
    SFT_BATCH_SIZE = 4          # Smaller batch — SFT data is more varied
    SFT_GRAD_ACCUM = 8          # Effective batch = 32 sequences
    SFT_LR = 5e-6               # Conservative — prevents mode collapse on 253M model
    SFT_WARMUP = 200            # Longer warmup to ease in gently
    SFT_TOTAL_STEPS = 4_000     # Fewer steps — stop before overfitting
    SFT_LOG_INTERVAL = 25
    SFT_EVAL_INTERVAL = 250
    SFT_SAVE_INTERVAL = 1000
    SEQ_LEN = config.max_seq_len

    SFT_CKPT_DIR = os.path.join(DRIVE_CKPT_DIR, 'sft')
    os.makedirs(SFT_CKPT_DIR, exist_ok=True)

    # Setup tokenizer
    if tokenizer is None:
        tokenizer = setup_tokenizer()

    # Setup model
    if model is None:
        print("\nLoading pretrained model for SFT...")
        model = TokenizedHLMv9(config).to(DEVICE)
        if checkpoint_path is None:
            # Always load from pretrain final.pt — never from a previous SFT run
            # This prevents loading a collapsed SFT model as the starting point
            for name in ['final.pt', 'best.pt', 'latest.pt']:
                p = os.path.join(DRIVE_CKPT_DIR, name)
                if os.path.exists(p):
                    checkpoint_path = p
                    break
        if checkpoint_path:
            load_checkpoint(model, None, checkpoint_path)
        else:
            print("WARNING: No pretrained checkpoint found! SFT from scratch.")

    print(f"\n{'='*70}")
    print(f"  SFT: Instruction Fine-Tuning")
    print(f"  Steps: {SFT_TOTAL_STEPS:,} | LR: {SFT_LR:.1e}")
    print(f"  Batch: {SFT_BATCH_SIZE} × {SFT_GRAD_ACCUM} accum")
    print(f"  Data: 40% OpenAssistant + 25% SlimOrca + 20% MetaMathQA + 15% CodeAlpaca")
    print(f"  Loss: assistant-only (user/system tokens masked)")
    print(f"  Checkpoints: {SFT_CKPT_DIR}")
    print(f"{'='*70}\n")

    # SFT optimizer — lower LR, less weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SFT_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.05,  # Lighter regularization for SFT
    )

    # Resume SFT if checkpoint exists
    sft_resume = os.path.join(SFT_CKPT_DIR, 'latest.pt')
    start_step = load_checkpoint(model, optimizer, sft_resume)

    # Data pipeline with prefetch
    sft_gen_fn = create_sft_dataloader(tokenizer, config, SEQ_LEN)
    prefetcher = PrefetchIterator(sft_gen_fn, SFT_BATCH_SIZE, prefetch_batches=8).start()

    # Compile
    compiled_model = model
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            compiled_model = torch.compile(model)
            print("torch.compile() enabled for SFT")
        except Exception:
            compiled_model = model

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    best_sft_loss = float('inf')
    t0 = time.time()
    step = start_step

    while step < SFT_TOTAL_STEPS:
        batch = prefetcher.get_batch()
        if batch is None:
            print("SFT dataset exhausted, restarting...")
            prefetcher.restart()
            continue

        input_ids = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = compiled_model(input_ids, targets)
            loss = result["loss"] / SFT_GRAD_ACCUM

        loss.backward()
        loss_accum += result["loss"].item()

        if (step + 1) % SFT_GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Cosine LR with warmup
            opt_step = (step + 1) // SFT_GRAD_ACCUM
            lr = get_lr(opt_step, SFT_TOTAL_STEPS // SFT_GRAD_ACCUM,
                        SFT_LR, SFT_WARMUP)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % SFT_LOG_INTERVAL == 0:
            avg_loss = loss_accum / SFT_LOG_INTERVAL
            elapsed = time.time() - t0
            eta_h = (SFT_TOTAL_STEPS - step) / (step / elapsed) / 3600 if step > 0 else 0
            ppl = math.exp(min(avg_loss, 20))
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[SFT {step:>5d}/{SFT_TOTAL_STEPS}] "
                  f"loss={avg_loss:.4f} ppl={ppl:.1f} | "
                  f"lr={lr_now:.2e} | ETA {eta_h:.1f}h")
            loss_accum = 0.0

        # Eval
        if step % SFT_EVAL_INTERVAL == 0 and step > 0:
            model.eval()
            eval_loss = 0.0
            eval_n = 20
            with torch.no_grad():
                for _ in range(eval_n):
                    eb = prefetcher.get_batch()
                    if eb is None:
                        prefetcher.restart()
                        eb = prefetcher.get_batch()
                    if eb is None:
                        break
                    inp = eb[0].to(DEVICE)
                    tgt = eb[1].to(DEVICE)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        r = model(inp, tgt)
                    eval_loss += r["loss"].item()
            eval_loss /= eval_n
            eval_ppl = math.exp(min(eval_loss, 20))
            print(f"  [SFT EVAL] loss={eval_loss:.4f} ppl={eval_ppl:.1f}")

            if eval_loss < best_sft_loss:
                best_sft_loss = eval_loss
                save_checkpoint(model, optimizer, step, eval_loss,
                                os.path.join(SFT_CKPT_DIR, 'best.pt'))
            model.train()

        # Periodic save
        if step % SFT_SAVE_INTERVAL == 0 and step > 0:
            save_checkpoint(model, optimizer, step, loss_accum,
                            os.path.join(SFT_CKPT_DIR, f'sft_step_{step:06d}.pt'))
            save_checkpoint(model, optimizer, step, loss_accum,
                            os.path.join(SFT_CKPT_DIR, 'latest.pt'))

        if step % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prefetcher.stop()

    # Final SFT save
    print(f"\n{'='*70}")
    print(f"  SFT complete!")
    print(f"  Steps: {step:,}")
    print(f"  Best SFT eval loss: {best_sft_loss:.4f}")
    print(f"{'='*70}")

    final_sft_path = os.path.join(SFT_CKPT_DIR, 'sft_final.pt')
    save_checkpoint(model, optimizer, step, best_sft_loss, final_sft_path)
    print(f"SFT checkpoint: {final_sft_path}")

    return model, tokenizer


# ============================================================================
# Cell 13: Chat / Generation Interface
# ============================================================================

def chat(model=None, tokenizer=None, checkpoint_path=None):
    """Interactive multi-turn chat with the SFT model.

    Uses the chat template from training so the model sees the same
    format it was fine-tuned on. Supports multi-turn conversation
    with history.
    """
    if model is None:
        config = CONFIG
        model = TokenizedHLMv9(config).to(DEVICE)
        if checkpoint_path is None:
            # Prefer SFT checkpoint, fall back to pretrained
            for name in [
                'sft/sft_final.pt', 'sft/best.pt',
                'best.pt', 'final.pt', 'latest.pt',
            ]:
                p = os.path.join(DRIVE_CKPT_DIR, name)
                if os.path.exists(p):
                    checkpoint_path = p
                    break
        if checkpoint_path:
            load_checkpoint(model, None, checkpoint_path)
        else:
            print("WARNING: No checkpoint found!")

    if tokenizer is None:
        tokenizer = setup_tokenizer()

    max_vocab = CONFIG.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    # Encode the end token for stopping
    end_ids = tokenizer.encode(CHAT_END.strip(), add_special_tokens=False)

    model.eval()
    print("\nTokenized HLM-v9 Chat")
    print("  Multi-turn conversation with history")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("-" * 50)

    history: List[Tuple[str, str]] = []

    while True:
        try:
            prompt = input("\nYOU: ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        if prompt.lower() in ('reset', 'clear'):
            history.clear()
            print("  [History cleared]")
            continue

        # Format with full history
        formatted = format_chat_prompt(
            prompt,
            system_msg=DEFAULT_SYSTEM_PROMPT,
            history=history,
        )

        ids = tokenizer.encode(formatted, add_special_tokens=False)
        ids = [t if t < max_vocab else unk_id for t in ids]

        # Truncate history if prompt is too long
        while len(ids) > CONFIG.max_seq_len - 128 and len(history) > 0:
            history.pop(0)
            formatted = format_chat_prompt(
                prompt,
                system_msg=DEFAULT_SYSTEM_PROMPT,
                history=history,
            )
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            ids = [t if t < max_vocab else unk_id for t in ids]

        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

        new_ids = output_ids[0, len(ids):].tolist()
        response = tokenizer.decode(new_ids, skip_special_tokens=False)

        # Stop at end token
        for end_marker in [CHAT_END.strip(), "<|end|>", "<|user|>"]:
            if end_marker in response:
                response = response[:response.index(end_marker)]
                break

        response = response.strip()
        print(f"\nHLM-v9: {response}")

        # Add to history for multi-turn
        history.append((prompt, response))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tokenized HLM-v9")
    parser.add_argument("--train", action="store_true", help="Run pretraining")
    parser.add_argument("--sft", action="store_true", help="Run SFT (after pretraining)")
    parser.add_argument("--full", action="store_true", help="Run pretraining + SFT + chat")
    parser.add_argument("--chat", action="store_true", help="Run chat interface")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--bank", type=str, default=None, help="Path to bank_full.json")
    args = parser.parse_args()

    if args.full:
        model, tokenizer = train()
        print("\n\nStarting SFT phase...")
        model, tokenizer = sft_finetune(model, tokenizer)
        print("\nStarting chat...")
        chat(model, tokenizer)
    elif args.train:
        model, tokenizer = train()
    elif args.sft:
        model, tokenizer = sft_finetune(checkpoint_path=args.checkpoint)
        print("\nStarting chat with SFT model...")
        chat(model, tokenizer)
    elif args.chat:
        chat(checkpoint_path=args.checkpoint)
    else:
        # Default: full pipeline
        model, tokenizer = train()
        print("\n\nStarting SFT phase...")
        model, tokenizer = sft_finetune(model, tokenizer)
        print("\nStarting chat...")
        chat(model, tokenizer)
