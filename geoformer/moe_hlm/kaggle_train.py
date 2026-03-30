"""
MoE-HLM-1B Training — Kaggle H100 (v3 — fits in 80GB)
=======================================================

Paste this entire file into a single Kaggle notebook cell.
Set accelerator to GPU H100 before running.

v3 changes: reduced to 16 layers, 2 geo rounds, batch=2, seq=1024
to fit in H100 80GB with gradient checkpointing.
All architecture features preserved: flat router, HLM-8^2 experts,
Cayley geometric products, ToU bank.

Checkpoints + evals every 500 steps to /kaggle/working/moe_hlm_1b/
Auto-resume from latest checkpoint.
"""

# ============================================================
# Install dependencies
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers", "datasets", "tqdm"])

# ============================================================
# All model code (self-contained)
# ============================================================
import os, json, math, time, logging, random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("moe-hlm")

# ---- Clifford Algebra Cl(3,0) ----

BLADE_NAMES = ["narrative", "causation", "affect", "wisdom",
               "relations", "ecology", "epistemics", "temporal"]

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
    """Sparse Cayley: (src_i, src_j, tgt_k, signs) for all 64 products."""
    si, sj, tk, sg = [], [], [], []
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            si.append(i); sj.append(j); tk.append(k); sg.append(float(s))
    return (torch.tensor(si, device=device), torch.tensor(sj, device=device),
            torch.tensor(tk, device=device), torch.tensor(sg, device=device))


# ---- Config ----

@dataclass
class MoEHLMConfig:
    d_model: int = 1536
    n_layers: int = 16           # Reduced from 24 to fit memory
    n_heads: int = 24
    d_head: int = 64
    n_experts: int = 8
    top_k: int = 2
    router_aux_loss_weight: float = 0.01
    n_blades: int = 8
    d_blade: int = 128
    n_geometric_rounds: int = 2  # Reduced from 3 to fit memory
    expert_d_ffn: int = 640
    vocab_size: int = 151936
    max_seq_len: int = 1024
    tou_n_primitives: int = 1_486
    tou_d_prim: int = 128
    tou_every_n_layers: int = 4
    attn_dropout: float = 0.0
    embed_dropout: float = 0.1
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = True
    init_std: float = 0.02

    @property
    def tou_attn_layers(self):
        return list(range(self.tou_every_n_layers - 1, self.n_layers, self.tou_every_n_layers))


# ---- ToU Bank ----

class ToUBank(nn.Module):
    def __init__(self, n_primitives, d_prim):
        super().__init__()
        self.embeddings = nn.Embedding(n_primitives, d_prim)
        nn.init.normal_(self.embeddings.weight, std=0.02)
        masks = torch.zeros(8, n_primitives)
        per = n_primitives // 8
        rem = n_primitives % 8
        idx = 0
        for b in range(8):
            c = per + (1 if b < rem else 0)
            masks[b, idx:idx+c] = 1.0
            idx += c
        self.register_buffer("blade_masks", masks)

    def forward(self):
        return self.embeddings.weight, self.blade_masks


# ---- Geometric Round (sparse Cayley, no outer product) ----

class GeometricRound(nn.Module):
    def __init__(self, config):
        super().__init__()
        si, sj, tk, sg = build_sparse_cayley("cpu")
        self.register_buffer("cayley_si", si)
        self.register_buffer("cayley_sj", sj)
        self.register_buffer("cayley_tk", tk)
        self.register_buffer("cayley_sg", sg)
        self.interaction_weights = nn.Parameter(torch.ones(64) * 0.1)
        self.blade_ffn_gate = nn.Linear(config.d_blade, config.expert_d_ffn, bias=False)
        self.blade_ffn_up = nn.Linear(config.d_blade, config.expert_d_ffn, bias=False)
        self.blade_ffn_down = nn.Linear(config.expert_d_ffn, config.d_blade, bias=False)
        self.norm = nn.LayerNorm(config.d_blade)
        self.geo_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (N_tok, 8, D)
        N_tok, N_blade, D = x.shape
        w = self.interaction_weights.sigmoid().to(x.dtype)
        sg = self.cayley_sg.to(x.dtype)
        xi = x[:, self.cayley_si, :]      # (N_tok, 64, D)
        xj = x[:, self.cayley_sj, :]      # (N_tok, 64, D)
        products = xi * xj * (sg * w).unsqueeze(0).unsqueeze(-1)
        geo = torch.zeros_like(x)
        tk_exp = self.cayley_tk.unsqueeze(0).unsqueeze(-1).expand(N_tok, 64, D)
        geo.scatter_add_(1, tk_exp, products)
        g = self.geo_gate.sigmoid()
        mixed = g * geo + (1.0 - g) * x
        flat = self.norm(mixed.reshape(-1, D))
        h = F.silu(self.blade_ffn_gate(flat)) * self.blade_ffn_up(flat)
        out = self.blade_ffn_down(h).reshape(N_tok, N_blade, D)
        return x + out


# ---- Holographic Expert (HLM-8^2) ----

class HolographicExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        expert_dim = config.n_blades * config.d_blade
        self.n_blades = config.n_blades
        self.d_blade = config.d_blade
        self.proj_in = nn.Linear(config.d_model, expert_dim, bias=False)
        self.geo_rounds = nn.ModuleList([
            GeometricRound(config) for _ in range(config.n_geometric_rounds)
        ])
        self.proj_out = nn.Linear(expert_dim, config.d_model, bias=False)
        self.out_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # x: (N_tokens, D)
        N, D = x.shape
        h = self.proj_in(x)
        mv = h.reshape(N, self.n_blades, self.d_blade)
        for geo in self.geo_rounds:
            mv = geo(mv)
        flat = mv.reshape(N, self.n_blades * self.d_blade)
        return self.out_norm(self.proj_out(flat))


# ---- MoE Layer (batched dispatch) ----

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight)
        self.experts = nn.ModuleList([
            HolographicExpert(config) for _ in range(config.n_experts)
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


# ---- Standard Attention (SDPA flash attention) ----

class RoPEAttention(nn.Module):
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
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


# ---- ToU Cross-Attention ----

class ToUCrossAttention(nn.Module):
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


# ---- Transformer Block ----

class MoEHLMBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.has_tou = layer_idx in config.tou_attn_layers
        self.norm1 = nn.RMSNorm(config.d_model)
        self.attn = RoPEAttention(config)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.moe = MoELayer(config)
        if self.has_tou:
            self.norm3 = nn.RMSNorm(config.d_model)
            self.tou_attn = ToUCrossAttention(config)

    def forward(self, x, mask=None, tou_embeds=None, blade_masks=None):
        x = x + self.attn(self.norm1(x), mask)
        moe_out, aux = self.moe(self.norm2(x))
        x = x + moe_out
        if self.has_tou and tou_embeds is not None:
            x = x + self.tou_attn(self.norm3(x), tou_embeds)
        return x, aux


# ---- Full Model ----

class MoEHLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.ModuleList([
            MoEHLMBlock(config, i) for i in range(config.n_layers)
        ])
        self.final_norm = nn.RMSNorm(config.d_model)
        self.tou_bank = ToUBank(config.tou_n_primitives, config.tou_d_prim)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.config.init_std)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.config.init_std)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        input_ids = input_ids.clamp(max=self.config.vocab_size - 1)
        if targets is not None:
            targets = targets.clone()
            valid = targets != -100
            targets[valid] = targets[valid].clamp(max=self.config.vocab_size - 1)

        x = self.embed_dropout(self.token_embed(input_ids))
        prim_e, blade_m = self.tou_bank()
        total_aux = torch.tensor(0.0, device=input_ids.device)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x_out, aux = torch.utils.checkpoint.checkpoint(
                    block, x, None, prim_e, blade_m, use_reentrant=False)
            else:
                x_out, aux = block(x, None, prim_e, blade_m)
            x = x_out
            total_aux = total_aux + aux

        logits = self.lm_head(self.final_norm(x))
        result = {"logits": logits, "aux_loss": total_aux}
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size),
                                       targets.view(-1), ignore_index=-100)
            result["lm_loss"] = lm_loss
            result["loss"] = lm_loss + self.config.router_aux_loss_weight * total_aux
        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Streaming Dataset
# ============================================================

class StreamingFineWeb(IterableDataset):
    def __init__(self, tokenizer, seq_len=1024, max_token_id=None,
                 split="train", subset="CC-MAIN-2013-20"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_token_id = max_token_id
        self.split = split
        self.subset = subset

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", name=self.subset,
                          split=self.split, streaming=True, trust_remote_code=True)
        buffer = []
        for example in ds:
            text = example.get("text", "")
            if len(text) < 50: continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if self.max_token_id is not None:
                tokens = [min(t, self.max_token_id) for t in tokens]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                tgt = torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long)
                yield {"input_ids": ids, "targets": tgt}


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(model, tokenizer, device, max_token_id=None, max_batches=50, seq_len=512):
    model.eval()
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2014-23",
                      split="train", streaming=True, trust_remote_code=True)
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    buffer = []

    for example in ds:
        text = example.get("text", "")
        if len(text) < 50: continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if max_token_id is not None:
            tokens = [min(t, max_token_id) for t in tokens]
        buffer.extend(tokens)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]
            ids = torch.tensor(chunk[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(chunk[1:seq_len+1], dtype=torch.long, device=device).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(ids, targets=tgt)
            total_loss += out["lm_loss"].item() * seq_len
            total_tokens += seq_len
            n_batches += 1
            if n_batches >= max_batches: break
        if n_batches >= max_batches: break

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity, "eval_batches": n_batches}


# ============================================================
# Training Loop
# ============================================================

def get_lr(step, warmup=1000, max_lr=3e-4, min_lr=3e-5, total_steps=50000):
    if step < warmup:
        return max_lr * step / warmup
    decay = (step - warmup) / (total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(decay, 1.0)))


def train():
    OUT_DIR = Path("/kaggle/working/moe_hlm_1b")
    OUT_DIR.mkdir(exist_ok=True)

    # ---- Config (conservative for H100 80GB) ----
    BATCH_SIZE = 2
    GRAD_ACCUM = 64          # Effective batch = 2 * 64 = 128
    SEQ_LEN = 1024           # Reduced from 2048
    TOTAL_STEPS = 50_000
    WARMUP = 1000
    MAX_LR = 3e-4
    MIN_LR = 3e-5
    SAVE_EVERY = 500
    LOG_EVERY = 10
    GRAD_CLIP = 1.0

    device = torch.device("cuda")

    # ---- Tokenizer ----
    log.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    actual_vocab = len(tokenizer)
    padded_vocab = ((actual_vocab + 63) // 64) * 64
    log.info(f"Tokenizer vocab: {actual_vocab}, padded to: {padded_vocab}")
    max_token_id = padded_vocab - 1

    # ---- Model ----
    log.info("Building MoE-HLM (16L, 2 geo rounds, 8 experts)...")
    config = MoEHLMConfig()
    config.vocab_size = padded_vocab
    model = MoEHLM(config).to(device)

    total_params = model.count_parameters()
    log.info(f"Total parameters: {total_params:,} ({total_params/1e9:.3f}B)")

    # Log GPU memory after model load
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        log.info(f"GPU memory after model load: {mem:.1f} GB")

    # ---- Optimizer ----
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "norm" in name or "bias" in name or "embeddings" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume ----
    start_step = 0
    best_eval_loss = float("inf")
    checkpoint_path = OUT_DIR / "latest.pt"
    if checkpoint_path.exists():
        log.info(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_embed = ckpt["model_state_dict"].get("token_embed.weight")
        if saved_embed is not None and saved_embed.shape[0] != config.vocab_size:
            log.info(f"Vocab changed: {saved_embed.shape[0]} -> {config.vocab_size}")
            new_embed = torch.zeros(config.vocab_size, config.d_model)
            min_v = min(saved_embed.shape[0], config.vocab_size)
            new_embed[:min_v] = saved_embed[:min_v]
            ckpt["model_state_dict"]["token_embed.weight"] = new_embed
            if "lm_head.weight" in ckpt["model_state_dict"]:
                ckpt["model_state_dict"]["lm_head.weight"] = new_embed
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            log.info(f"Could not load optimizer state: {e}, starting fresh optimizer")
        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        start_step = ckpt.get("step", 0) + 1
        best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        log.info(f"Resumed at step {start_step}")

    # ---- Quick memory test ----
    log.info("Running memory test with dummy batch...")
    try:
        dummy_ids = torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN), device=device)
        dummy_tgt = torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dummy_out = model(dummy_ids, targets=dummy_tgt)
            dummy_loss = dummy_out["loss"]
        dummy_loss.backward()
        model.zero_grad(set_to_none=True)
        del dummy_ids, dummy_tgt, dummy_out, dummy_loss
        torch.cuda.empty_cache()
        mem = torch.cuda.max_memory_allocated() / 1e9
        log.info(f"Memory test PASSED. Peak GPU: {mem:.1f} GB / 80 GB")
    except RuntimeError as e:
        log.error(f"Memory test FAILED: {e}")
        log.error("Reduce BATCH_SIZE or SEQ_LEN")
        return

    # ---- Data ----
    log.info("Setting up streaming dataset...")
    dataset = StreamingFineWeb(tokenizer, seq_len=SEQ_LEN, max_token_id=max_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2,
                        pin_memory=True, prefetch_factor=4)

    # ---- Training ----
    log.info("=" * 60)
    log.info("  MoE-HLM Training (v3 — fits H100)")
    log.info("=" * 60)
    log.info(f"  Params:       {total_params:,}")
    log.info(f"  Vocab:        {config.vocab_size}")
    log.info(f"  Layers:       {config.n_layers}")
    log.info(f"  Geo rounds:   {config.n_geometric_rounds}")
    log.info(f"  Experts:      {config.n_experts} (top-{config.top_k})")
    log.info(f"  Batch:        {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    log.info(f"  Seq len:      {SEQ_LEN}")
    log.info(f"  Start step:   {start_step}")
    log.info(f"  Save/eval:    every {SAVE_EVERY}")
    log.info("=" * 60)

    model.train()
    data_iter = iter(loader)
    running_loss = 0.0
    running_lm = 0.0
    running_aux = 0.0
    t0 = time.time()
    tokens_seen = start_step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN

    for step in range(start_step, TOTAL_STEPS):
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, TOTAL_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm = 0.0
        step_aux = 0.0

        for micro in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, targets=targets)
                loss = outputs["loss"] / GRAD_ACCUM

            scaler.scale(loss).backward()

            step_loss += outputs["loss"].item() / GRAD_ACCUM
            step_lm += outputs["lm_loss"].item() / GRAD_ACCUM
            step_aux += outputs["aux_loss"].item() / GRAD_ACCUM
            tokens_seen += BATCH_SIZE * SEQ_LEN

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        scaler.step(optimizer)
        scaler.update()

        running_loss += step_loss
        running_lm += step_lm
        running_aux += step_aux

        if (step + 1) % LOG_EVERY == 0:
            dt = time.time() - t0
            tps = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN * LOG_EVERY / dt
            log.info(
                f"step {step+1:>6,} | loss {running_loss/LOG_EVERY:.4f} | "
                f"lm {running_lm/LOG_EVERY:.4f} | aux {running_aux/LOG_EVERY:.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{tps:,.0f} tok/s | {tokens_seen/1e6:.0f}M tok"
            )
            running_loss = 0.0
            running_lm = 0.0
            running_aux = 0.0
            t0 = time.time()

        if (step + 1) % SAVE_EVERY == 0:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "best_eval_loss": best_eval_loss,
                "tokens_seen": tokens_seen,
            }
            torch.save(ckpt, OUT_DIR / f"step_{step+1:06d}.pt")
            torch.save(ckpt, OUT_DIR / "latest.pt")
            log.info(f"Saved checkpoint step {step+1}")

            log.info("Running evaluation...")
            eval_results = evaluate(model, tokenizer, device, max_token_id=max_token_id)
            log.info(
                f"  EVAL: loss={eval_results['eval_loss']:.4f} | "
                f"ppl={eval_results['eval_perplexity']:.1f}"
            )
            if eval_results["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_results["eval_loss"]
                torch.save(ckpt, OUT_DIR / "best.pt")
                log.info(f"  New best! loss={best_eval_loss:.4f}")
            model.train()

    log.info(f"Training complete. Tokens: {tokens_seen:,}, Best eval: {best_eval_loss:.4f}")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    train()
