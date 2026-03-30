"""
HLM Kernel Benchmark — Kaggle H100
====================================

Paste into a Kaggle notebook cell with H100 accelerator.
Compares original PyTorch MoE-HLM vs optimized kernel stack.

Measures:
1. Geometric product: scatter_add vs scatter-free
2. Full geometric round
3. MoE layer: loop dispatch vs sorted dispatch
4. Estimated full-model throughput
"""

import time, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# Clifford Algebra (shared)
# ============================================================

def _derive_cayley_table():
    basis_gens = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
    gens_to_idx = {g: i for i, g in enumerate(basis_gens)}
    def reduce(a, b):
        gens = list(a) + list(b)
        sign = 1
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(gens) - 1:
                if gens[i] == gens[i+1]:
                    gens.pop(i); gens.pop(i); changed = True
                    if i > 0: i -= 1
                elif gens[i] > gens[i+1]:
                    gens[i], gens[i+1] = gens[i+1], gens[i]
                    sign *= -1; changed = True; i += 1
                else: i += 1
        return gens_to_idx[tuple(gens)], sign
    return [[reduce(basis_gens[i], basis_gens[j]) for j in range(8)] for i in range(8)]

CAYLEY_TABLE = _derive_cayley_table()

def get_sparse_cayley(device):
    si, sj, tk, sg = [], [], [], []
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            si.append(i); sj.append(j); tk.append(k); sg.append(float(s))
    return (torch.tensor(si, device=device), torch.tensor(sj, device=device),
            torch.tensor(tk, device=device, dtype=torch.long),
            torch.tensor(sg, device=device, dtype=torch.float32))

def get_target_table(device):
    """Per-target accumulation: for each output blade k, the 8 (i,j,sign) products."""
    by_target = {}
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            if k not in by_target: by_target[k] = []
            by_target[k].append((i, j, s))
    src_ij = torch.zeros(8, 8, 2, dtype=torch.int32, device=device)
    src_signs = torch.zeros(8, 8, dtype=torch.float32, device=device)
    for k in range(8):
        for p, (i, j, sign) in enumerate(by_target[k]):
            src_ij[k, p, 0] = i
            src_ij[k, p, 1] = j
            src_signs[k, p] = sign
    return src_ij, src_signs


# ============================================================
# Config
# ============================================================

@dataclass
class BenchConfig:
    d_model: int = 1536
    n_layers: int = 16
    n_heads: int = 24
    d_head: int = 64
    n_experts: int = 8
    top_k: int = 2
    n_blades: int = 8
    d_blade: int = 128
    n_geometric_rounds: int = 2
    expert_d_ffn: int = 640
    init_std: float = 0.02
    tou_every_n_layers: int = 4
    tou_n_primitives: int = 1486
    tou_d_prim: int = 128
    rope_theta: float = 10000.0
    router_aux_loss_weight: float = 0.01

    @property
    def tou_attn_layers(self):
        return list(range(self.tou_every_n_layers - 1, self.n_layers, self.tou_every_n_layers))


# ============================================================
# ORIGINAL: scatter_add geometric round (baseline)
# ============================================================

class OriginalGeometricRound(nn.Module):
    def __init__(self, d_blade, d_ffn, device):
        super().__init__()
        si, sj, tk, sg = get_sparse_cayley(device)
        self.register_buffer("cayley_si", si)
        self.register_buffer("cayley_sj", sj)
        self.register_buffer("cayley_tk", tk)
        self.register_buffer("cayley_sg", sg)
        self.interaction_weights = nn.Parameter(torch.ones(64, device=device) * 0.1)
        self.geo_gate = nn.Parameter(torch.tensor(0.5, device=device))
        self.norm = nn.LayerNorm(d_blade, device=device)
        self.gate_proj = nn.Linear(d_blade, d_ffn, bias=False, device=device)
        self.up_proj = nn.Linear(d_blade, d_ffn, bias=False, device=device)
        self.down_proj = nn.Linear(d_ffn, d_blade, bias=False, device=device)

    def forward(self, x):
        N_tok, N_blade, D = x.shape
        w = self.interaction_weights.sigmoid().to(x.dtype)
        sg = self.cayley_sg.to(x.dtype)
        xi = x[:, self.cayley_si, :]
        xj = x[:, self.cayley_sj, :]
        products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
        geo = torch.zeros_like(x)
        tk_exp = self.cayley_tk.unsqueeze(0).unsqueeze(-1).expand(N_tok, 64, D)
        geo.scatter_add_(1, tk_exp, products)
        g = self.geo_gate.sigmoid()
        mixed = g * geo + (1.0 - g) * x
        flat = self.norm(mixed.reshape(-1, D))
        h = F.silu(self.gate_proj(flat)) * self.up_proj(flat)
        out = self.down_proj(h).reshape(N_tok, N_blade, D)
        return x + out


# ============================================================
# OPTIMIZED: scatter-free geometric round
# ============================================================

class OptimizedGeometricRound(nn.Module):
    def __init__(self, d_blade, d_ffn, device):
        super().__init__()
        src_ij, src_signs = get_target_table(device)
        self.register_buffer("src_ij", src_ij)
        self.register_buffer("src_signs", src_signs)
        self.interaction_weights = nn.Parameter(torch.ones(64, device=device) * 0.1)
        self.geo_gate = nn.Parameter(torch.tensor(0.5, device=device))
        self.norm = nn.LayerNorm(d_blade, device=device)
        self.gate_proj = nn.Linear(d_blade, d_ffn, bias=False, device=device)
        self.up_proj = nn.Linear(d_blade, d_ffn, bias=False, device=device)
        self.down_proj = nn.Linear(d_ffn, d_blade, bias=False, device=device)

    def forward(self, x):
        N_tok, N_blade, D = x.shape
        dtype = x.dtype
        w = self.interaction_weights.sigmoid().to(dtype)
        geo = torch.zeros_like(x)
        for k in range(8):
            for p in range(8):
                i = self.src_ij[k, p, 0].long()
                j = self.src_ij[k, p, 1].long()
                sign = self.src_signs[k, p].to(dtype)
                weight = w[i * 8 + j]
                geo[:, k, :] = geo[:, k, :] + sign * weight * x[:, i, :] * x[:, j, :]
        g = self.geo_gate.sigmoid()
        mixed = g * geo + (1.0 - g) * x
        flat = self.norm(mixed.reshape(-1, D))
        h = F.silu(self.gate_proj(flat)) * self.up_proj(flat)
        out = self.down_proj(h).reshape(N_tok, N_blade, D)
        return x + out


# ============================================================
# ORIGINAL: loop-based MoE expert dispatch
# ============================================================

class OriginalExpert(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        expert_dim = config.n_blades * config.d_blade
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.proj_in = nn.Linear(config.d_model, expert_dim, bias=False, device=device)
        self.geo_rounds = nn.ModuleList([
            OriginalGeometricRound(config.d_blade, config.expert_d_ffn, device)
            for _ in range(config.n_geometric_rounds)
        ])
        self.proj_out = nn.Linear(expert_dim, config.d_model, bias=False, device=device)
        self.out_norm = nn.LayerNorm(config.d_model, device=device)

    def forward(self, x):
        N, D = x.shape
        h = self.proj_in(x)
        mv = h.reshape(N, self.n_blades, self.d_blade)
        for geo in self.geo_rounds:
            mv = geo(mv)
        flat = mv.reshape(N, self.n_blades * self.d_blade)
        return self.out_norm(self.proj_out(flat))


class OriginalMoELayer(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False, device=device)
        self.experts = nn.ModuleList([
            OriginalExpert(config, device) for _ in range(config.n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        logits = self.gate(x_flat)
        top_w, top_i = torch.topk(logits, self.top_k, dim=-1)
        top_w = F.softmax(top_w, dim=-1)
        probs = F.softmax(logits, dim=-1)
        f = probs.mean(dim=0)
        aux_loss = (f * f).sum() * self.n_experts
        output = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top_i == e)
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            unique_tokens = token_indices.unique()
            expert_input = x_flat[unique_tokens]
            expert_output = self.experts[e](expert_input)
            token_to_idx = torch.zeros(N, dtype=torch.long, device=x.device)
            token_to_idx[unique_tokens] = torch.arange(len(unique_tokens), device=x.device)
            weights = top_w[token_indices, k_indices]
            out_vals = expert_output[token_to_idx[token_indices]]
            output.index_add_(0, token_indices, out_vals * weights.unsqueeze(-1))
        return output.reshape(B, T, D), aux_loss


# ============================================================
# OPTIMIZED: sorted dispatch MoE
# ============================================================

class OptimizedExpert(nn.Module):
    """Single expert using optimized geometric round."""
    def __init__(self, config, device):
        super().__init__()
        expert_dim = config.n_blades * config.d_blade
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.proj_in = nn.Linear(config.d_model, expert_dim, bias=False, device=device)
        self.geo_rounds = nn.ModuleList([
            OptimizedGeometricRound(config.d_blade, config.expert_d_ffn, device)
            for _ in range(config.n_geometric_rounds)
        ])
        self.proj_out = nn.Linear(expert_dim, config.d_model, bias=False, device=device)
        self.out_norm = nn.LayerNorm(config.d_model, device=device)

    def forward(self, x):
        N, D = x.shape
        h = self.proj_in(x)
        mv = h.reshape(N, self.n_blades, self.d_blade)
        for geo in self.geo_rounds:
            mv = geo(mv)
        flat = mv.reshape(N, self.n_blades * self.d_blade)
        return self.out_norm(self.proj_out(flat))


class OptimizedMoELayer(nn.Module):
    """Sorted dispatch MoE with optimized geo rounds."""
    def __init__(self, config, device):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.d_model = config.d_model
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False, device=device)
        self.experts = nn.ModuleList([
            OptimizedExpert(config, device) for _ in range(config.n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        logits = self.gate(x_flat)
        top_w, top_i = torch.topk(logits, self.top_k, dim=-1)
        top_w = F.softmax(top_w, dim=-1)
        probs = F.softmax(logits, dim=-1)
        f = probs.mean(dim=0)
        aux_loss = (f * f).sum() * self.n_experts

        # Flatten top-k assignments
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        expert_ids = top_i.reshape(-1)
        weights = top_w.reshape(-1)

        # Sort by expert
        sort_idx = expert_ids.argsort(stable=True)
        expert_ids_sorted = expert_ids[sort_idx]
        token_ids_sorted = token_ids[sort_idx]
        weights_sorted = weights[sort_idx]
        x_sorted = x_flat[token_ids_sorted]

        # Process each expert on its contiguous slice
        output = torch.zeros_like(x_flat)
        N_expanded = N * self.top_k

        pos = 0
        for e in range(self.n_experts):
            # Find contiguous range for this expert
            count = (expert_ids_sorted == e).sum().item()
            if count == 0:
                continue
            x_e = x_sorted[pos:pos + count]
            y_e = self.experts[e](x_e)
            y_weighted = y_e * weights_sorted[pos:pos + count].unsqueeze(-1)
            output.index_add_(0, token_ids_sorted[pos:pos + count], y_weighted)
            pos += count

        return output.reshape(B, T, D), aux_loss


# ============================================================
# Benchmark runner
# ============================================================

def bench(fn, warmup=3, iters=20, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        times.append(t)
    times.sort()
    med = times[len(times) // 2]
    print(f"  {label:<50} {med:>8.2f} ms")
    return med


def main():
    device = torch.device("cuda")
    print("=" * 70)
    print("  HLM KERNEL BENCHMARKS — H100")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    config = BenchConfig()

    # ---- Benchmark 1: Geometric Round ----
    print("=" * 70)
    print("BENCH 1: Geometric Round (geo product + SwiGLU FFN)")
    print("=" * 70)

    N_tok = 2048  # Typical tokens per expert call
    x_geo = torch.randn(N_tok, 8, 128, device=device, dtype=torch.bfloat16)

    orig_geo = OriginalGeometricRound(128, 640, device).bfloat16()
    opt_geo = OptimizedGeometricRound(128, 640, device).bfloat16()
    # Copy weights for fair comparison
    opt_geo.load_state_dict(orig_geo.state_dict(), strict=False)

    t_orig = bench(lambda: orig_geo(x_geo), label="ORIGINAL (scatter_add)")
    t_opt = bench(lambda: opt_geo(x_geo), label="OPTIMIZED (scatter-free)")
    print(f"  {'SPEEDUP:':<50} {t_orig/t_opt:>8.2f}x")
    print()

    # ---- Benchmark 2: Single Expert ----
    print("=" * 70)
    print("BENCH 2: Single Expert Forward (proj + 2x geo round + proj)")
    print("=" * 70)

    N_expert_tokens = 512  # ~25% of 2048 tokens
    x_expert = torch.randn(N_expert_tokens, config.d_model, device=device, dtype=torch.bfloat16)

    orig_expert = OriginalExpert(config, device).bfloat16()
    opt_expert = OptimizedExpert(config, device).bfloat16()

    t_orig = bench(lambda: orig_expert(x_expert), label="ORIGINAL expert")
    t_opt = bench(lambda: opt_expert(x_expert), label="OPTIMIZED expert")
    print(f"  {'SPEEDUP:':<50} {t_orig/t_opt:>8.2f}x")
    print()

    # ---- Benchmark 3: Full MoE Layer ----
    print("=" * 70)
    print("BENCH 3: Full MoE Layer (route + 8 experts + scatter)")
    print("=" * 70)

    B, T = 2, 1024
    x_moe = torch.randn(B, T, config.d_model, device=device, dtype=torch.bfloat16)

    orig_moe = OriginalMoELayer(config, device).bfloat16()
    opt_moe = OptimizedMoELayer(config, device).bfloat16()

    t_orig = bench(lambda: orig_moe(x_moe), label="ORIGINAL MoE (loop dispatch)")
    t_opt = bench(lambda: opt_moe(x_moe), label="OPTIMIZED MoE (sorted dispatch)")
    print(f"  {'SPEEDUP:':<50} {t_orig/t_opt:>8.2f}x")

    tokens = B * T
    print(f"\n  Original:  {tokens / (t_orig / 1000):>10,.0f} tok/s per MoE layer")
    print(f"  Optimized: {tokens / (t_opt / 1000):>10,.0f} tok/s per MoE layer")
    print()

    # ---- Estimate full model throughput ----
    print("=" * 70)
    print("ESTIMATED FULL MODEL THROUGHPUT")
    print("=" * 70)

    # Attention is roughly same speed, estimate from standard SDPA
    # For 1536D, 24 heads, seq=1024: ~2-3ms per layer on H100
    est_attn_ms = 2.5  # conservative estimate per layer
    n_layers = config.n_layers

    orig_total = (t_orig + est_attn_ms) * n_layers
    opt_total = (t_opt + est_attn_ms) * n_layers

    orig_tps = tokens / (orig_total / 1000)
    opt_tps = tokens / (opt_total / 1000)

    print(f"  Attention per layer (est):   {est_attn_ms:.1f} ms")
    print(f"  MoE per layer (original):    {t_orig:.1f} ms")
    print(f"  MoE per layer (optimized):   {t_opt:.1f} ms")
    print(f"  {n_layers}-layer forward (original):  {orig_total:.0f} ms -> {orig_tps:,.0f} tok/s")
    print(f"  {n_layers}-layer forward (optimized): {opt_total:.0f} ms -> {opt_tps:,.0f} tok/s")
    print(f"\n  OVERALL SPEEDUP: {orig_total/opt_total:.2f}x")
    print()

    # ---- Benchmark 4: Forward + Backward (training step) ----
    print("=" * 70)
    print("BENCH 4: Forward + Backward (one MoE layer, training mode)")
    print("=" * 70)

    x_train = torch.randn(B, T, config.d_model, device=device, dtype=torch.bfloat16,
                           requires_grad=True)

    def orig_train_step():
        x_train.grad = None
        out, aux = orig_moe(x_train)
        loss = out.sum() + aux
        loss.backward()

    def opt_train_step():
        x_train.grad = None
        out, aux = opt_moe(x_train)
        loss = out.sum() + aux
        loss.backward()

    t_orig_train = bench(orig_train_step, warmup=2, iters=10, label="ORIGINAL fwd+bwd")
    t_opt_train = bench(opt_train_step, warmup=2, iters=10, label="OPTIMIZED fwd+bwd")
    print(f"  {'TRAINING SPEEDUP:':<50} {t_orig_train/t_opt_train:>8.2f}x")

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
