"""CPU profiler for MoE-HLM — find the real bottleneck."""
import time, math, torch, torch.nn as nn, torch.nn.functional as F, sys
from dataclasses import dataclass
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.times = {}
        self.counts = {}
    @contextmanager
    def track(self, name):
        t0 = time.perf_counter()
        yield
        dt = (time.perf_counter() - t0) * 1000
        self.times[name] = self.times.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1
    def report(self):
        items = sorted(self.times.items(), key=lambda x: -x[1])
        total = sum(v for v in self.times.values())
        for name, t in items[:30]:
            c = self.counts[name]
            pct = t / total * 100
            print(f"  {name:<45} {t:>8.1f} ms ({pct:>5.1f}%) [{c}x, {t/c:.1f} avg]")
        print(f"  {'TOTAL':<45} {total:>8.1f} ms")
        return total

# Cayley table
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
def sc():
    si,sj,tk,sg = [],[],[],[]
    for i in range(8):
        for j in range(8):
            k,s = CT[i][j]; si.append(i); sj.append(j); tk.append(k); sg.append(float(s))
    return (torch.tensor(si), torch.tensor(sj), torch.tensor(tk, dtype=torch.long), torch.tensor(sg))

@dataclass
class Cfg:
    d_model: int = 256; n_layers: int = 4; n_heads: int = 4; d_head: int = 64
    n_experts: int = 8; top_k: int = 2; n_blades: int = 8; d_blade: int = 32
    n_geometric_rounds: int = 2; expert_d_ffn: int = 128; vocab_size: int = 1000
    router_aux_loss_weight: float = 0.01

class GR(nn.Module):
    def __init__(self, c):
        super().__init__()
        si,sj,tk,sg = sc()
        self.register_buffer("si",si); self.register_buffer("sj",sj)
        self.register_buffer("tk",tk); self.register_buffer("sg",sg)
        self.iw = nn.Parameter(torch.ones(64)*0.1)
        self.gp = nn.Linear(c.d_blade, c.expert_d_ffn, bias=False)
        self.up = nn.Linear(c.d_blade, c.expert_d_ffn, bias=False)
        self.dp = nn.Linear(c.expert_d_ffn, c.d_blade, bias=False)
        self.ln = nn.LayerNorm(c.d_blade)
        self.gg = nn.Parameter(torch.tensor(0.5))
    def forward(self, x, timer=None):
        N,Nb,D = x.shape
        if timer:
            with timer.track("geo.weights_sigmoid"):
                w = self.iw.sigmoid(); sg = self.sg
            with timer.track("geo.gather_blades"):
                xi = x[:,self.si,:]; xj = x[:,self.sj,:]
            with timer.track("geo.elementwise_multiply"):
                products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
            with timer.track("geo.scatter_add"):
                geo = torch.zeros_like(x)
                tk_exp = self.tk.unsqueeze(0).unsqueeze(-1).expand(N,64,D)
                geo.scatter_add_(1, tk_exp, products)
            with timer.track("geo.gate_mix"):
                g = self.gg.sigmoid(); mixed = g * geo + (1-g) * x
            with timer.track("geo.layernorm"):
                flat = self.ln(mixed.reshape(-1, D))
            with timer.track("geo.swiglu_ffn"):
                h = F.silu(self.gp(flat)) * self.up(flat)
                out = self.dp(h).reshape(N,Nb,D)
            with timer.track("geo.residual_add"):
                return x + out
        else:
            w = self.iw.sigmoid(); sg = self.sg
            xi = x[:,self.si,:]; xj = x[:,self.sj,:]
            products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
            geo = torch.zeros_like(x)
            geo.scatter_add_(1, self.tk.unsqueeze(0).unsqueeze(-1).expand(N,64,D), products)
            g = self.gg.sigmoid(); mixed = g * geo + (1-g) * x
            flat = self.ln(mixed.reshape(-1, D))
            h = F.silu(self.gp(flat)) * self.up(flat)
            return x + self.dp(h).reshape(N,Nb,D)

class Exp(nn.Module):
    def __init__(self, c):
        super().__init__()
        ed = c.n_blades * c.d_blade; self.nb=c.n_blades; self.db=c.d_blade
        self.pi = nn.Linear(c.d_model, ed, bias=False)
        self.geo = nn.ModuleList([GR(c) for _ in range(c.n_geometric_rounds)])
        self.po = nn.Linear(ed, c.d_model, bias=False)
        self.on = nn.LayerNorm(c.d_model)
    def forward(self, x, timer=None):
        N,D = x.shape
        if timer:
            with timer.track("expert.proj_in"):
                h = self.pi(x)
            mv = h.reshape(N, self.nb, self.db)
            for i,g in enumerate(self.geo):
                with timer.track(f"expert.geo_round_{i}"):
                    mv = g(mv, timer=timer)
            with timer.track("expert.proj_out+norm"):
                return self.on(self.po(mv.reshape(N, self.nb*self.db)))
        else:
            h = self.pi(x); mv = h.reshape(N, self.nb, self.db)
            for g in self.geo: mv = g(mv)
            return self.on(self.po(mv.reshape(N, self.nb*self.db)))

class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ne=c.n_experts; self.tk_=c.top_k
        self.gate = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.experts = nn.ModuleList([Exp(c) for _ in range(c.n_experts)])
    def forward(self, x, timer=None):
        B,T,D = x.shape; N=B*T; xf=x.reshape(N,D)
        if timer:
            with timer.track("moe.gate_routing"):
                logits=self.gate(xf); tw,ti=torch.topk(logits,self.tk_,dim=-1)
                tw=F.softmax(tw,dim=-1); probs=F.softmax(logits,dim=-1)
                aux=(probs.mean(dim=0)**2).sum()*self.ne
            output = torch.zeros(N, D)
            for e in range(self.ne):
                with timer.track(f"moe.expert{e}_dispatch"):
                    mask=(ti==e)
                    if not mask.any(): continue
                    tok_i,k_i=mask.nonzero(as_tuple=True)
                    ut=tok_i.unique(); ei=xf[ut]
                with timer.track(f"moe.expert{e}_compute"):
                    eo = self.experts[e](ei, timer=timer if e==0 else None)
                with timer.track(f"moe.expert{e}_scatter"):
                    t2i=torch.zeros(N,dtype=torch.long); t2i[ut]=torch.arange(len(ut))
                    ws=tw[tok_i,k_i]; ov=eo[t2i[tok_i]].float()
                    output.index_add_(0, tok_i, ov*ws.unsqueeze(-1))
        else:
            logits=self.gate(xf); tw,ti=torch.topk(logits,self.tk_,dim=-1)
            tw=F.softmax(tw,dim=-1); aux=torch.tensor(0.0)
            output=torch.zeros(N,D)
            for e in range(self.ne):
                mask=(ti==e)
                if not mask.any(): continue
                tok_i,k_i=mask.nonzero(as_tuple=True); ut=tok_i.unique()
                eo=self.experts[e](xf[ut])
                t2i=torch.zeros(N,dtype=torch.long); t2i[ut]=torch.arange(len(ut))
                output.index_add_(0,tok_i,eo[t2i[tok_i]].float()*tw[tok_i,k_i].unsqueeze(-1))
        return output.to(x.dtype).reshape(B,T,D), aux

class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh=c.n_heads; self.dh=c.d_head; self.dm=c.d_model
        self.qkv=nn.Linear(c.d_model,3*c.d_model,bias=False)
        self.op=nn.Linear(c.d_model,c.d_model,bias=False)
    def forward(self, x, timer=None):
        B,T,D=x.shape
        if timer:
            with timer.track("attn.qkv_proj"):
                qkv=self.qkv(x).reshape(B,T,3,self.nh,self.dh); q,k,v=qkv.unbind(dim=2)
            with timer.track("attn.sdpa"):
                q,k,v=[t.transpose(1,2) for t in (q,k,v)]
                out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            with timer.track("attn.out_proj"):
                return self.op(out.transpose(1,2).reshape(B,T,D))
        else:
            qkv=self.qkv(x).reshape(B,T,3,self.nh,self.dh); q,k,v=qkv.unbind(dim=2)
            q,k,v=[t.transpose(1,2) for t in (q,k,v)]
            return self.op(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1=nn.RMSNorm(c.d_model); self.attn=Attn(c)
        self.n2=nn.RMSNorm(c.d_model); self.moe=MoE(c)
    def forward(self, x, timer=None):
        if timer:
            with timer.track("block.norm1"): n1=self.n1(x)
            with timer.track("block.attention"): x=x+self.attn(n1,timer=timer)
            with timer.track("block.norm2"): n2=self.n2(x)
            with timer.track("block.moe"): mo,aux=self.moe(n2,timer=timer); x=x+mo
        else:
            x=x+self.attn(self.n1(x)); mo,aux=self.moe(self.n2(x)); x=x+mo
        return x, aux

class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c=c; self.te=nn.Embedding(c.vocab_size,c.d_model)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.fn=nn.RMSNorm(c.d_model)
        self.lm=nn.Linear(c.d_model,c.vocab_size,bias=False)
        self.lm.weight=self.te.weight
    def forward(self, ids, tgt=None, timer=None):
        if timer:
            with timer.track("model.embed"): x=self.te(ids)
            for i,blk in enumerate(self.blocks):
                with timer.track(f"model.layer_{i}"):
                    x,a=blk(x, timer=timer if i==0 else None)
            with timer.track("model.final_norm+lm_head"):
                logits=self.lm(self.fn(x))
            if tgt is not None:
                with timer.track("model.loss"):
                    lm=F.cross_entropy(logits.view(-1,self.c.vocab_size),tgt.view(-1))
            else: lm=torch.tensor(0.0)
        else:
            x=self.te(ids)
            for blk in self.blocks: x,a=blk(x)
            logits=self.lm(self.fn(x))
            lm=F.cross_entropy(logits.view(-1,self.c.vocab_size),tgt.view(-1)) if tgt is not None else torch.tensor(0.0)
        return {"loss": lm, "logits": logits}

# ---- Run ----
c = Cfg(); m = Model(c); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
B, T, GA = 2, 256, 4
ids = torch.randint(0, c.vocab_size, (B, T))
tgt = torch.randint(0, c.vocab_size, (B, T))

# Warmup
for _ in range(2):
    opt.zero_grad(); out=m(ids,tgt); out["loss"].backward(); opt.step()

# Profile
timer = Timer()
with timer.track("full_step"):
    with timer.track("zero_grad"): opt.zero_grad(set_to_none=True)
    for micro in range(GA):
        with timer.track("forward"):
            out = m(ids, tgt, timer=timer if micro==0 else None)
            loss = out["loss"] / GA
        with timer.track("backward"):
            loss.backward()
    with timer.track("grad_clip"):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    with timer.track("optimizer_step"):
        opt.step()

print()
print("=" * 70)
print("  MoE-HLM TRAINING STEP PROFILE (CPU)")
print("  d_model=256, 4 layers, 8 experts, 2 geo rounds")
print("  batch=2, seq=256, grad_accum=4")
print("=" * 70)
timer.report()
tok = B * T * GA
total = timer.times["full_step"]
print(f"\n  Tokens/step: {tok:,}")
print(f"  Throughput: {tok/(total/1000):,.0f} tok/s (CPU)")
