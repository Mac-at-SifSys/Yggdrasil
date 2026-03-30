"""
HLM Training Profiler — Kaggle H100
=====================================

Paste into Kaggle notebook with H100.
Times EVERY operation in the training loop to find the real bottleneck.

Uses the actual MoE-HLM model from kaggle_train.py (the original one
that was getting 2,100 tok/s) but instruments every step.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers", "datasets"])

import time, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List
from contextlib import contextmanager

# ============================================================
# Timer utility
# ============================================================

class Timer:
    """Accumulates timing for named operations."""
    def __init__(self):
        self.times = {}
        self.counts = {}

    @contextmanager
    def track(self, name):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000  # ms
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += dt
        self.counts[name] += 1

    def report(self):
        print("\n" + "=" * 70)
        print("  PROFILING RESULTS (sorted by total time)")
        print("=" * 70)
        items = sorted(self.times.items(), key=lambda x: -x[1])
        total = sum(v for v in self.times.values())
        for name, t in items:
            count = self.counts[name]
            pct = t / total * 100
            avg = t / count
            print(f"  {name:<45} {t:>8.1f} ms ({pct:>5.1f}%)  "
                  f"[{count}x, {avg:.2f} ms avg]")
        print(f"  {'TOTAL':<45} {total:>8.1f} ms")
        return items, total


# ============================================================
# Clifford Algebra
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

def build_sparse_cayley(device):
    si, sj, tk, sg = [], [], [], []
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            si.append(i); sj.append(j); tk.append(k); sg.append(float(s))
    return (torch.tensor(si, device=device), torch.tensor(sj, device=device),
            torch.tensor(tk, device=device, dtype=torch.long),
            torch.tensor(sg, device=device, dtype=torch.float32))


# ============================================================
# Model (same as kaggle_train.py original — the 2100 tok/s version)
# ============================================================

@dataclass
class Cfg:
    d_model: int = 1536
    n_layers: int = 16
    n_heads: int = 24
    d_head: int = 64
    n_experts: int = 8
    top_k: int = 2
    router_aux_loss_weight: float = 0.01
    n_blades: int = 8
    d_blade: int = 128
    n_geometric_rounds: int = 2
    expert_d_ffn: int = 640
    vocab_size: int = 151936
    max_seq_len: int = 1024
    tou_n_primitives: int = 1486
    tou_d_prim: int = 128
    tou_every_n_layers: int = 4
    embed_dropout: float = 0.1
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = True
    init_std: float = 0.02

    @property
    def tou_attn_layers(self):
        return list(range(self.tou_every_n_layers - 1, self.n_layers, self.tou_every_n_layers))


class ToUBank(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.embeddings = nn.Embedding(n, d)
        masks = torch.zeros(8, n)
        per = n // 8; rem = n % 8; idx = 0
        for b in range(8):
            c = per + (1 if b < rem else 0)
            masks[b, idx:idx+c] = 1.0; idx += c
        self.register_buffer("blade_masks", masks)
    def forward(self):
        return self.embeddings.weight, self.blade_masks


class GeoRound(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        si, sj, tk, sg = build_sparse_cayley(device)
        self.register_buffer("si", si); self.register_buffer("sj", sj)
        self.register_buffer("tk", tk); self.register_buffer("sg", sg)
        self.iw = nn.Parameter(torch.ones(64, device=device) * 0.1)
        self.gate_proj = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=device)
        self.up_proj = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=device)
        self.down_proj = nn.Linear(cfg.expert_d_ffn, cfg.d_blade, bias=False, device=device)
        self.norm = nn.LayerNorm(cfg.d_blade, device=device)
        self.gg = nn.Parameter(torch.tensor(0.5, device=device))

    def forward(self, x, timer=None):
        N, Nb, D = x.shape
        if timer:
            with timer.track("geo.sigmoid"):
                w = self.iw.sigmoid().to(x.dtype)
                sg = self.sg.to(x.dtype)
            with timer.track("geo.gather"):
                xi = x[:, self.si, :]
                xj = x[:, self.sj, :]
            with timer.track("geo.multiply"):
                products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
            with timer.track("geo.scatter_add"):
                geo = torch.zeros_like(x)
                tk_exp = self.tk.unsqueeze(0).unsqueeze(-1).expand(N, 64, D)
                geo.scatter_add_(1, tk_exp, products)
            with timer.track("geo.gate_mix"):
                g = self.gg.sigmoid()
                mixed = g * geo + (1.0 - g) * x
            with timer.track("geo.layernorm"):
                flat = self.norm(mixed.reshape(-1, D))
            with timer.track("geo.swiglu"):
                h = F.silu(self.gate_proj(flat)) * self.up_proj(flat)
                out = self.down_proj(h).reshape(N, Nb, D)
            with timer.track("geo.residual"):
                return x + out
        else:
            w = self.iw.sigmoid().to(x.dtype)
            sg = self.sg.to(x.dtype)
            xi = x[:, self.si, :]; xj = x[:, self.sj, :]
            products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
            geo = torch.zeros_like(x)
            tk_exp = self.tk.unsqueeze(0).unsqueeze(-1).expand(N, 64, D)
            geo.scatter_add_(1, tk_exp, products)
            g = self.gg.sigmoid()
            mixed = g * geo + (1.0 - g) * x
            flat = self.norm(mixed.reshape(-1, D))
            h = F.silu(self.gate_proj(flat)) * self.up_proj(flat)
            out = self.down_proj(h).reshape(N, Nb, D)
            return x + out


class Expert(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        ed = cfg.n_blades * cfg.d_blade
        self.nb = cfg.n_blades; self.db = cfg.d_blade
        self.pi = nn.Linear(cfg.d_model, ed, bias=False, device=device)
        self.geo = nn.ModuleList([GeoRound(cfg, device) for _ in range(cfg.n_geometric_rounds)])
        self.po = nn.Linear(ed, cfg.d_model, bias=False, device=device)
        self.on = nn.LayerNorm(cfg.d_model, device=device)

    def forward(self, x, timer=None):
        N, D = x.shape
        if timer:
            with timer.track("expert.proj_in"):
                h = self.pi(x)
            mv = h.reshape(N, self.nb, self.db)
            for i, g in enumerate(self.geo):
                with timer.track(f"expert.geo_round_{i}"):
                    mv = g(mv, timer=timer)
            with timer.track("expert.proj_out"):
                flat = mv.reshape(N, self.nb * self.db)
                return self.on(self.po(flat))
        else:
            h = self.pi(x)
            mv = h.reshape(N, self.nb, self.db)
            for g in self.geo: mv = g(mv)
            flat = mv.reshape(N, self.nb * self.db)
            return self.on(self.po(flat))


class MoE(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.ne = cfg.n_experts; self.tk = cfg.top_k
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False, device=device)
        self.experts = nn.ModuleList([Expert(cfg, device) for _ in range(cfg.n_experts)])

    def forward(self, x, timer=None):
        B, T, D = x.shape; N = B * T
        x_flat = x.reshape(N, D)

        if timer:
            with timer.track("moe.routing"):
                logits = self.gate(x_flat)
                top_w, top_i = torch.topk(logits, self.tk, dim=-1)
                top_w = F.softmax(top_w, dim=-1)
                probs = F.softmax(logits, dim=-1)
                f = probs.mean(dim=0)
                aux = (f * f).sum() * self.ne

            output = torch.zeros(N, D, device=x.device, dtype=torch.float32)
            for e in range(self.ne):
                with timer.track(f"moe.expert_{e}_dispatch"):
                    mask = (top_i == e)
                    if not mask.any(): continue
                    ti, ki = mask.nonzero(as_tuple=True)
                    ut = ti.unique()
                    ei = x_flat[ut]

                with timer.track(f"moe.expert_{e}_compute"):
                    eo = self.experts[e](ei, timer=timer if e == 0 else None)

                with timer.track(f"moe.expert_{e}_scatter"):
                    t2i = torch.zeros(N, dtype=torch.long, device=x.device)
                    t2i[ut] = torch.arange(len(ut), device=x.device)
                    ws = top_w[ti, ki]
                    ov = eo[t2i[ti]].float()
                    output.index_add_(0, ti, ov * ws.unsqueeze(-1))
        else:
            logits = self.gate(x_flat)
            top_w, top_i = torch.topk(logits, self.tk, dim=-1)
            top_w = F.softmax(top_w, dim=-1)
            probs = F.softmax(logits, dim=-1)
            f = probs.mean(dim=0)
            aux = (f * f).sum() * self.ne
            output = torch.zeros(N, D, device=x.device, dtype=torch.float32)
            for e in range(self.ne):
                mask = (top_i == e)
                if not mask.any(): continue
                ti, ki = mask.nonzero(as_tuple=True)
                ut = ti.unique()
                ei = x_flat[ut]
                eo = self.experts[e](ei)
                t2i = torch.zeros(N, dtype=torch.long, device=x.device)
                t2i[ut] = torch.arange(len(ut), device=x.device)
                ws = top_w[ti, ki]
                ov = eo[t2i[ti]].float()
                output.index_add_(0, ti, ov * ws.unsqueeze(-1))

        return output.to(x.dtype).reshape(B, T, D), aux


class Attn(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.nh = cfg.n_heads; self.dh = cfg.d_head; self.dm = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False, device=device)
        self.op = nn.Linear(cfg.d_model, cfg.d_model, bias=False, device=device)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2).float() / cfg.d_head))
        self.register_buffer("inv_freq", inv.to(device))

    def forward(self, x, timer=None):
        B, T, D = x.shape
        if timer:
            with timer.track("attn.qkv_proj"):
                qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
                q, k, v = qkv.unbind(dim=2)
            with timer.track("attn.rope"):
                t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat([freqs, freqs], dim=-1)
                cos = emb.cos().unsqueeze(0).unsqueeze(2)
                sin = emb.sin().unsqueeze(0).unsqueeze(2)
                q1, q2 = q.chunk(2, dim=-1)
                q = q * cos + torch.cat([-q2, q1], dim=-1) * sin
                k1, k2 = k.chunk(2, dim=-1)
                k = k * cos + torch.cat([-k2, k1], dim=-1) * sin
            with timer.track("attn.sdpa"):
                q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            with timer.track("attn.out_proj"):
                out = self.op(out.transpose(1, 2).reshape(B, T, D))
            return out
        else:
            qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
            q, k, v = qkv.unbind(dim=2)
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
            return self.op(out.transpose(1, 2).reshape(B, T, D))


class ToUCA(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        dp = cfg.tou_d_prim; dm = cfg.d_model; self.dp = dp
        self.qp = nn.Linear(dm, dp, bias=False, device=device)
        self.kp = nn.Linear(dp, dp, bias=False, device=device)
        self.vp = nn.Linear(dp, dp, bias=False, device=device)
        self.opr = nn.Linear(dp, dm, bias=False, device=device)
        self.g = nn.Linear(dm, 1, bias=True, device=device)
    def forward(self, x, pe):
        q = self.qp(x); k = self.kp(pe); v = self.vp(pe)
        a = F.softmax(torch.matmul(q, k.t()) / math.sqrt(self.dp), dim=-1)
        return self.g(x).sigmoid() * self.opr(torch.matmul(a, v))


class Block(nn.Module):
    def __init__(self, cfg, li, device):
        super().__init__()
        self.ht = li in cfg.tou_attn_layers
        self.n1 = nn.RMSNorm(cfg.d_model, device=device)
        self.attn = Attn(cfg, device)
        self.n2 = nn.RMSNorm(cfg.d_model, device=device)
        self.moe = MoE(cfg, device)
        if self.ht:
            self.n3 = nn.RMSNorm(cfg.d_model, device=device)
            self.tou = ToUCA(cfg, device)

    def forward(self, x, pe=None, bm=None, timer=None):
        if timer:
            with timer.track("block.norm1"):
                n1 = self.n1(x)
            with timer.track("block.attention"):
                x = x + self.attn(n1, timer=timer)
            with timer.track("block.norm2"):
                n2 = self.n2(x)
            with timer.track("block.moe"):
                mo, aux = self.moe(n2, timer=timer)
                x = x + mo
            if self.ht and pe is not None:
                with timer.track("block.tou_attn"):
                    x = x + self.tou(self.n3(x), pe)
        else:
            x = x + self.attn(self.n1(x))
            mo, aux = self.moe(self.n2(x))
            x = x + mo
            if self.ht and pe is not None:
                x = x + self.tou(self.n3(x), pe)
            aux = torch.tensor(0.0, device=x.device)
        return x, aux


class Model(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.te = nn.Embedding(cfg.vocab_size, cfg.d_model, device=device)
        self.ed = nn.Dropout(cfg.embed_dropout)
        self.blocks = nn.ModuleList([Block(cfg, i, device) for i in range(cfg.n_layers)])
        self.fn = nn.RMSNorm(cfg.d_model, device=device)
        self.tb = ToUBank(cfg.tou_n_primitives, cfg.tou_d_prim).to(device)
        self.lm = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, device=device)
        self.lm.weight = self.te.weight

    def forward(self, ids, tgt=None, timer=None):
        B, T = ids.shape
        ids = ids.clamp(max=self.cfg.vocab_size - 1)
        if tgt is not None:
            tgt = tgt.clone()
            v = tgt != -100
            tgt[v] = tgt[v].clamp(max=self.cfg.vocab_size - 1)

        if timer:
            with timer.track("model.embed"):
                x = self.ed(self.te(ids))
            with timer.track("model.tou_bank"):
                pe, bm = self.tb()
            ta = torch.tensor(0.0, device=ids.device)
            # Profile first 2 layers in detail, rest without
            for i, blk in enumerate(self.blocks):
                use_timer = timer if i < 2 else None
                with timer.track(f"model.layer_{i}"):
                    if self.cfg.gradient_checkpointing and self.training:
                        xo, a = torch.utils.checkpoint.checkpoint(
                            blk, x, pe, bm, None, use_reentrant=False)
                    else:
                        xo, a = blk(x, pe, bm, timer=use_timer)
                    x = xo; ta = ta + a
            with timer.track("model.final_norm"):
                x = self.fn(x)
            with timer.track("model.lm_head"):
                logits = self.lm(x)
            if tgt is not None:
                with timer.track("model.loss"):
                    lm = F.cross_entropy(logits.view(-1, self.cfg.vocab_size),
                                         tgt.view(-1), ignore_index=-100)
            else:
                lm = None
        else:
            x = self.ed(self.te(ids))
            pe, bm = self.tb()
            ta = torch.tensor(0.0, device=ids.device)
            for blk in self.blocks:
                if self.cfg.gradient_checkpointing and self.training:
                    xo, a = torch.utils.checkpoint.checkpoint(
                        blk, x, pe, bm, None, use_reentrant=False)
                else:
                    xo, a = blk(x, pe, bm)
                x = xo; ta = ta + a
            logits = self.lm(self.fn(x))
            lm = F.cross_entropy(logits.view(-1, self.cfg.vocab_size),
                                  tgt.view(-1), ignore_index=-100) if tgt is not None else None

        r = {"logits": logits, "aux_loss": ta}
        if lm is not None:
            r["lm_loss"] = lm
            r["loss"] = lm + self.cfg.router_aux_loss_weight * ta
        return r


# ============================================================
# Profile the full training step
# ============================================================

def profile():
    device = torch.device("cuda")
    print("=" * 70)
    print("  HLM TRAINING PROFILER — H100")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg = Cfg()

    # ---- Test multiple configs ----
    # All use grad_ckpt=True (required to fit in 80GB)
    # No GradScaler (not needed for bf16, causes errors)
    configs_to_test = [
        ("batch=2,  seq=1024, accum=4",  2, 1024, 4),
        ("batch=4,  seq=1024, accum=4",  4, 1024, 4),
        ("batch=8,  seq=512,  accum=4",  8, 512,  4),
        ("batch=2,  seq=2048, accum=4",  2, 2048, 4),
        ("batch=4,  seq=512,  accum=4",  4, 512,  4),
    ]

    for label, bs, sl, ga in configs_to_test:
        print(f"\n{'=' * 70}")
        print(f"  CONFIG: {label}")
        print(f"{'=' * 70}")

        cfg.gradient_checkpointing = True
        cfg.max_seq_len = sl

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = Model(cfg, device).bfloat16()
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

            # Dummy data
            ids = torch.randint(0, 1000, (bs, sl), device=device)
            tgt = torch.randint(0, 1000, (bs, sl), device=device)

            # Warmup (no profiling)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(ids, tgt)
                    loss = out["loss"] / ga
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Profile one full optimizer step (with grad accumulation)
            timer = Timer()
            torch.cuda.synchronize()
            step_start = time.perf_counter()

            with timer.track("optimizer.zero_grad"):
                optimizer.zero_grad(set_to_none=True)

            for micro in range(ga):
                with timer.track("forward_pass"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(ids, tgt, timer=timer if micro == 0 else None)
                        loss = out["loss"] / ga

                with timer.track("backward_pass"):
                    loss.backward()

            with timer.track("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with timer.track("optimizer.step"):
                optimizer.step()

            torch.cuda.synchronize()
            step_total = (time.perf_counter() - step_start) * 1000

            # Report
            timer.report()

            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            tokens_per_step = bs * sl * min(ga, 4)
            tps = tokens_per_step / (step_total / 1000)

            print(f"\n  Step total:     {step_total:.0f} ms")
            print(f"  Peak GPU mem:   {peak_mem:.1f} GB / 85 GB")
            print(f"  Tokens/step:    {tokens_per_step:,}")
            print(f"  Throughput:     {tps:,.0f} tok/s")
            print(f"  Eff. batch:     {bs} x {ga} = {bs * ga}")

            del model, optimizer, scaler
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"  FAILED: {e}")
            torch.cuda.empty_cache()
            continue

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    profile()
