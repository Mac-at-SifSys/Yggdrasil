"""
Fused Expert Benchmark — Kaggle H100
======================================

Paste into Kaggle notebook with H100 accelerator.
Compares original MoE (sequential loop) vs fused (bmm + stacked weights).
Takes ~2 minutes to run.
"""

import time, math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

# ============================================================
# Clifford Algebra
# ============================================================
def _ct():
    bg = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
    gi = {g: i for i, g in enumerate(bg)}
    def r(a, b):
        g = list(a) + list(b); s = 1; ch = True
        while ch:
            ch = False; i = 0
            while i < len(g) - 1:
                if g[i] == g[i+1]: g.pop(i); g.pop(i); ch = True; i = max(0,i-1)
                elif g[i] > g[i+1]: g[i],g[i+1] = g[i+1],g[i]; s *= -1; ch = True; i += 1
                else: i += 1
        return gi[tuple(g)], s
    return [[r(bg[i], bg[j]) for j in range(8)] for i in range(8)]
CT = _ct()
BY_TARGET = {}
for i in range(8):
    for j in range(8):
        k, s = CT[i][j]
        BY_TARGET.setdefault(k, []).append((i, j, s))

def sc(dev):
    si,sj,tk,sg = [],[],[],[]
    for i in range(8):
        for j in range(8):
            k,s = CT[i][j]; si.append(i); sj.append(j); tk.append(k); sg.append(float(s))
    return (torch.tensor(si,device=dev), torch.tensor(sj,device=dev),
            torch.tensor(tk,device=dev,dtype=torch.long), torch.tensor(sg,device=dev))

def target_table(dev):
    src_ij = torch.zeros(8, 8, 2, dtype=torch.int32, device=dev)
    src_signs = torch.zeros(8, 8, dtype=torch.float32, device=dev)
    for k in range(8):
        for p, (i, j, sign) in enumerate(BY_TARGET[k]):
            src_ij[k, p, 0] = i; src_ij[k, p, 1] = j; src_signs[k, p] = sign
    return src_ij, src_signs

# ============================================================
# Config
# ============================================================
@dataclass
class Cfg:
    d_model: int = 1536; n_experts: int = 8; top_k: int = 2
    n_blades: int = 8; d_blade: int = 128; n_geometric_rounds: int = 2
    expert_d_ffn: int = 640; init_std: float = 0.02

# ============================================================
# ORIGINAL: Sequential expert loop (what we had before)
# ============================================================
class OrigGeoRound(nn.Module):
    def __init__(self, cfg, dev):
        super().__init__()
        si,sj,tk,sg = sc(dev)
        self.register_buffer("si",si); self.register_buffer("sj",sj)
        self.register_buffer("tk",tk); self.register_buffer("sg",sg)
        self.iw = nn.Parameter(torch.ones(64,device=dev)*0.1)
        self.gp = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=dev)
        self.up = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=dev)
        self.dp = nn.Linear(cfg.expert_d_ffn, cfg.d_blade, bias=False, device=dev)
        self.ln = nn.LayerNorm(cfg.d_blade, device=dev)
        self.gg = nn.Parameter(torch.tensor(0.5, device=dev))
    def forward(self, x):
        N,Nb,D = x.shape
        w = self.iw.sigmoid().to(x.dtype); sg = self.sg.to(x.dtype)
        xi = x[:,self.si,:]; xj = x[:,self.sj,:]
        products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
        geo = torch.zeros_like(x)
        geo.scatter_add_(1, self.tk.unsqueeze(0).unsqueeze(-1).expand(N,64,D), products)
        g = self.gg.sigmoid(); mixed = g * geo + (1-g) * x
        flat = self.ln(mixed.reshape(-1, D))
        h = F.silu(self.gp(flat)) * self.up(flat)
        return x + self.dp(h).reshape(N,Nb,D)

class OrigExpert(nn.Module):
    def __init__(self, cfg, dev):
        super().__init__()
        ed = cfg.n_blades*cfg.d_blade; self.nb=cfg.n_blades; self.db=cfg.d_blade
        self.pi = nn.Linear(cfg.d_model, ed, bias=False, device=dev)
        self.geo = nn.ModuleList([OrigGeoRound(cfg,dev) for _ in range(cfg.n_geometric_rounds)])
        self.po = nn.Linear(ed, cfg.d_model, bias=False, device=dev)
        self.on = nn.LayerNorm(cfg.d_model, device=dev)
    def forward(self, x):
        N,D = x.shape; h = self.pi(x); mv = h.reshape(N,self.nb,self.db)
        for g in self.geo: mv = g(mv)
        return self.on(self.po(mv.reshape(N, self.nb*self.db)))

class OrigMoE(nn.Module):
    def __init__(self, cfg, dev):
        super().__init__()
        self.ne=cfg.n_experts; self.tk_=cfg.top_k
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False, device=dev)
        self.experts = nn.ModuleList([OrigExpert(cfg,dev) for _ in range(cfg.n_experts)])
    def forward(self, x):
        B,T,D = x.shape; N=B*T; xf=x.reshape(N,D)
        logits=self.gate(xf); tw,ti=torch.topk(logits,self.tk_,dim=-1)
        tw=F.softmax(tw,dim=-1)
        probs=F.softmax(logits,dim=-1); f=probs.mean(dim=0)
        aux=(f*f).sum()*self.ne
        output=torch.zeros(N,D,device=x.device,dtype=torch.float32)
        for e in range(self.ne):
            mask=(ti==e)
            if not mask.any(): continue
            tok_i,k_i=mask.nonzero(as_tuple=True); ut=tok_i.unique()
            eo=self.experts[e](xf[ut])
            t2i=torch.zeros(N,dtype=torch.long,device=x.device)
            t2i[ut]=torch.arange(len(ut),device=x.device)
            output.index_add_(0,tok_i,(eo[t2i[tok_i]]*tw[tok_i,k_i].unsqueeze(-1)).float())
        return output.to(x.dtype).reshape(B,T,D), aux

# ============================================================
# FUSED: bmm + stacked weights
# ============================================================
class FusedGeoRound(nn.Module):
    def __init__(self, cfg, dev):
        super().__init__()
        src_ij, src_signs = target_table(dev)
        self.register_buffer("src_ij", src_ij); self.register_buffer("src_signs", src_signs)
        self.iw = nn.Parameter(torch.ones(64, device=dev)*0.1)
        self.gg = nn.Parameter(torch.tensor(0.5, device=dev))
        self.ln = nn.LayerNorm(cfg.d_blade, device=dev)
        self.gp = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=dev)
        self.up = nn.Linear(cfg.d_blade, cfg.expert_d_ffn, bias=False, device=dev)
        self.dp = nn.Linear(cfg.expert_d_ffn, cfg.d_blade, bias=False, device=dev)
    def forward(self, x):
        N,Nb,D = x.shape; dtype = x.dtype
        w = self.iw.sigmoid().to(dtype)
        geo = torch.zeros_like(x)
        for k in range(8):
            for p in range(8):
                i=self.src_ij[k,p,0].long(); j=self.src_ij[k,p,1].long()
                sign=self.src_signs[k,p].to(dtype); wt=w[i*8+j]
                geo[:,k,:] = geo[:,k,:] + sign * wt * x[:,i,:] * x[:,j,:]
        g=self.gg.sigmoid(); mixed = g*geo + (1-g)*x
        flat=self.ln(mixed.reshape(-1,D))
        h=F.silu(self.gp(flat))*self.up(flat)
        return x + self.dp(h).reshape(N,Nb,D)

class FusedMoE(nn.Module):
    """Stacked weights + bmm for projections, sorted dispatch."""
    def __init__(self, cfg, dev):
        super().__init__()
        self.ne=cfg.n_experts; self.tk_=cfg.top_k; self.dm=cfg.d_model
        self.nb=cfg.n_blades; self.db=cfg.d_blade; self.ed=cfg.n_blades*cfg.d_blade
        self.nr=cfg.n_geometric_rounds
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False, device=dev)
        # Stacked projections
        self.proj_in = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, self.ed, device=dev)*cfg.init_std)
        self.proj_out = nn.Parameter(torch.randn(cfg.n_experts, self.ed, cfg.d_model, device=dev)*cfg.init_std)
        self.out_ln = nn.LayerNorm(cfg.d_model, device=dev)
        # Per-expert geo rounds (shared arch, independent weights)
        self.geo_rounds = nn.ModuleList([
            nn.ModuleList([FusedGeoRound(cfg, dev) for _ in range(cfg.n_geometric_rounds)])
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x):
        B,T,D = x.shape; N=B*T; xf=x.reshape(N,D); dev=x.device; dtype=x.dtype
        logits=self.gate(xf); tw,ti=torch.topk(logits,self.tk_,dim=-1)
        tw=F.softmax(tw,dim=-1)
        probs=F.softmax(logits,dim=-1); f=probs.mean(dim=0)
        aux=(f*f).sum()*self.ne

        # Flatten top-k, sort by expert
        tok_ids=torch.arange(N,device=dev).unsqueeze(1).expand(N,self.tk_).reshape(-1)
        exp_ids=ti.reshape(-1); weights=tw.reshape(-1)
        sort_idx=exp_ids.argsort(stable=True)
        exp_sorted=exp_ids[sort_idx]; tok_sorted=tok_ids[sort_idx]; w_sorted=weights[sort_idx]
        x_sorted=xf[tok_sorted]

        # Get expert offsets/counts
        NE=N*self.tk_
        offsets=torch.zeros(self.ne,dtype=torch.long,device=dev)
        counts=torch.zeros(self.ne,dtype=torch.long,device=dev)
        for e in range(self.ne):
            mask=(exp_sorted==e)
            if mask.any():
                idx=mask.nonzero(as_tuple=True)[0]
                offsets[e]=idx[0]; counts[e]=len(idx)

        # Grouped proj_in via bmm
        mc=counts.max().item()
        if mc == 0:
            return torch.zeros_like(x), aux

        x_batch=torch.zeros(self.ne, mc, D, device=dev, dtype=dtype)
        for e in range(self.ne):
            c=counts[e].item()
            if c>0: x_batch[e,:c]=x_sorted[offsets[e]:offsets[e]+c]

        # bmm: (8, mc, 1536) @ (8, 1536, 1024) = (8, mc, 1024)
        h_batch = torch.bmm(x_batch, self.proj_in.to(dtype))
        mv_batch = h_batch.reshape(self.ne, mc, self.nb, self.db)

        # Geo rounds per expert (still looped, but projections are batched)
        for e in range(self.ne):
            c=counts[e].item()
            if c==0: continue
            mv_e = mv_batch[e,:c]
            for r in range(self.nr):
                mv_e = self.geo_rounds[e][r](mv_e)
            mv_batch[e,:c] = mv_e

        # bmm proj_out: (8, mc, 1024) @ (8, 1024, 1536) = (8, mc, 1536)
        flat_batch = mv_batch.reshape(self.ne, mc, self.ed)
        y_batch = torch.bmm(flat_batch, self.proj_out.to(dtype))

        # Scatter back
        y_sorted = torch.zeros(NE, D, device=dev, dtype=torch.float32)
        for e in range(self.ne):
            c=counts[e].item()
            if c>0:
                y_sorted[offsets[e]:offsets[e]+c] = self.out_ln(y_batch[e,:c]).float()

        output=torch.zeros(N, D, device=dev, dtype=torch.float32)
        output.index_add_(0, tok_sorted, y_sorted * w_sorted.unsqueeze(-1))
        return output.to(dtype).reshape(B,T,D), aux


# ============================================================
# Benchmark
# ============================================================
def bench(fn, warmup=3, iters=20, label=""):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)
    times.sort()
    med = times[len(times)//2]
    print(f"  {label:<55} {med:>8.2f} ms")
    return med

def main():
    dev = torch.device("cuda")
    print("=" * 70)
    print("  FUSED vs ORIGINAL MoE — H100 BENCHMARK")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    cfg = Cfg()

    # ---- Bench 1: Just proj_in — bmm vs loop ----
    print("BENCH 1: proj_in only (bmm vs loop)")
    print("-" * 70)
    n_e, dm, ed = 8, 1536, 1024
    mc = 512
    W_stacked = torch.randn(n_e, dm, ed, device=dev, dtype=torch.bfloat16)
    W_separate = [torch.randn(dm, ed, device=dev, dtype=torch.bfloat16) for _ in range(n_e)]
    x_batch = torch.randn(n_e, mc, dm, device=dev, dtype=torch.bfloat16)

    def loop_proj():
        return [x_batch[e] @ W_separate[e] for e in range(n_e)]
    def bmm_proj():
        return torch.bmm(x_batch, W_stacked)

    t_loop = bench(loop_proj, label="Loop (8x matmul)")
    t_bmm = bench(bmm_proj, label="bmm  (1x batched matmul)")
    print(f"  {'SPEEDUP:':<55} {t_loop/t_bmm:>8.2f}x")
    print()

    # ---- Bench 2: Full MoE layer — forward only ----
    print("BENCH 2: Full MoE layer forward")
    print("-" * 70)
    B, T = 2, 1024
    x = torch.randn(B, T, cfg.d_model, device=dev, dtype=torch.bfloat16)

    orig_moe = OrigMoE(cfg, dev).bfloat16()
    fused_moe = FusedMoE(cfg, dev).bfloat16()

    t_orig = bench(lambda: orig_moe(x), label="ORIGINAL (sequential loop)")
    t_fused = bench(lambda: fused_moe(x), label="FUSED (bmm + sorted dispatch)")
    tokens = B * T
    print(f"  {'SPEEDUP:':<55} {t_orig/t_fused:>8.2f}x")
    print(f"  Original:  {tokens/(t_orig/1000):>10,.0f} tok/s per layer")
    print(f"  Fused:     {tokens/(t_fused/1000):>10,.0f} tok/s per layer")
    print()

    # ---- Bench 3: Forward + backward ----
    print("BENCH 3: Forward + backward (training step, 1 layer)")
    print("-" * 70)
    x_train = torch.randn(B, T, cfg.d_model, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def orig_train():
        x_train.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, aux = orig_moe(x_train)
        (out.sum() + aux).backward()
    def fused_train():
        x_train.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, aux = fused_moe(x_train)
        (out.sum() + aux).backward()

    t_orig_t = bench(orig_train, iters=10, label="ORIGINAL fwd+bwd")
    t_fused_t = bench(fused_train, iters=10, label="FUSED fwd+bwd")
    print(f"  {'TRAINING SPEEDUP:':<55} {t_orig_t/t_fused_t:>8.2f}x")
    print()

    # ---- Estimated full model ----
    print("ESTIMATED FULL MODEL (16 layers)")
    print("-" * 70)
    n_layers = 16
    attn_est = 2.5  # ms per layer (SDPA on H100)
    orig_full = (t_orig + attn_est) * n_layers
    fused_full = (t_fused + attn_est) * n_layers
    print(f"  Original:  {orig_full:.0f} ms -> {tokens/(orig_full/1000):>8,.0f} tok/s")
    print(f"  Fused:     {fused_full:.0f} ms -> {tokens/(fused_full/1000):>8,.0f} tok/s")
    print(f"  Speedup:   {orig_full/fused_full:.2f}x")
    print()

    print("=" * 70)
    print("  DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
