#!/usr/bin/env python3
"""LoRA fine-tuning for TokenizedHLMv9.

Loads final.pt from Drive, applies LoRA adapters to attention layers only,
trains ~4M params instead of 253M. Stable SFT that preserves pretrained knowledge.
"""

# ============================================================================
# Cell 1: Imports, GPU Setup, Drive Mount
# ============================================================================

import math
import time
import os
import sys
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f}GB")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_CKPT_DIR = '/content/drive/MyDrive/hlm_v9_token_ckpts'
    LORA_CKPT_DIR = '/content/drive/MyDrive/hlm_v9_token_ckpts/lora'
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)
    print(f"Pretrain checkpoint: {DRIVE_CKPT_DIR}")
    print(f"LoRA checkpoints: {LORA_CKPT_DIR}")
except ImportError:
    DRIVE_CKPT_DIR = './hlm_v9_token_ckpts'
    LORA_CKPT_DIR = './hlm_v9_token_ckpts/lora'
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

# Copy model file from Drive to local if needed
MODEL_FILE_SRC = os.path.join(DRIVE_CKPT_DIR, 'TokenizedHLMv9_Colab.py')
MODEL_FILE_DST = '/content/TokenizedHLMv9_Colab.py'
if os.path.exists(MODEL_FILE_SRC):
    import shutil
    shutil.copy(MODEL_FILE_SRC, MODEL_FILE_DST)
    print(f"Copied model file to {MODEL_FILE_DST}")
elif os.path.exists('TokenizedHLMv9_Colab.py'):
    MODEL_FILE_DST = 'TokenizedHLMv9_Colab.py'
    print(f"Using local model file")
else:
    raise FileNotFoundError(
        "TokenizedHLMv9_Colab.py not found!\n"
        f"Please upload it to Google Drive at: {MODEL_FILE_SRC}"
    )

# Import model classes from main file
sys.path.insert(0, os.path.dirname(MODEL_FILE_DST))
from TokenizedHLMv9_Colab import (
    TokenizedHLMv9Config, TokenizedHLMv9, CONFIG,
    CHAT_USER_PREFIX, CHAT_ASST_PREFIX, CHAT_END, DEFAULT_SYSTEM_PROMPT,
)
print("Model classes imported successfully")


# ============================================================================
# Cell 2: LoRA Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """Low-rank adaptation of a frozen linear layer.

    Adds trainable matrices A (r × in) and B (out × r) to a frozen linear.
    Output = W*x + scaling * B*(A*x)  where scaling = alpha / rank
    B is initialized to zero so LoRA starts as identity (no disruption).
    """

    def __init__(self, linear: nn.Linear, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features

        # A initialized with small random, B initialized to zero
        # Place on same device as the base linear layer
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA into base weights for efficient inference."""
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            merged = nn.Linear(
                self.linear.in_features,
                self.linear.out_features,
                bias=self.linear.bias is not None,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
            merged.weight.data = self.linear.weight.data + delta
            if self.linear.bias is not None:
                merged.bias.data = self.linear.bias.data.clone()
        return merged


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0,
               dropout: float = 0.05,
               target_modules: List[str] = None) -> nn.Module:
    """Freeze all model params, then inject LoRA into target attention projections."""
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: inject LoRA into attention projection layers
    lora_count = 0
    for module in model.modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                original = getattr(module, attr_name)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank,
                                           alpha=alpha, dropout=dropout)
                    setattr(module, attr_name, lora_layer)
                    lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {lora_count} linear layers")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.3f}% of {total:,})")
    return model


def save_lora(model: nn.Module, optimizer, step: int, loss: float, path: str):
    """Save only the LoRA adapter weights (tiny — ~32MB instead of 24GB)."""
    lora_state = {}
    for name, module in model.named_modules():
        for attr_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr_name):
                layer = getattr(module, attr_name)
                if isinstance(layer, LoRALinear):
                    lora_state[f"{name}.{attr_name}.lora_A"] = layer.lora_A.data.cpu()
                    lora_state[f"{name}.{attr_name}.lora_B"] = layer.lora_B.data.cpu()

    torch.save({
        'step': step,
        'loss': loss,
        'lora_state': lora_state,
        'optimizer_state': optimizer.state_dict(),
        'rank': 16,
        'alpha': 32.0,
    }, path)
    try:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    print(f"  Saved LoRA: {path} ({os.path.getsize(path)/1024**2:.1f}MB)")


def load_lora(model: nn.Module, optimizer, path: str) -> int:
    """Load LoRA adapter weights into an already-LoRA-patched model."""
    if not os.path.exists(path):
        return 0
    print(f"Resuming LoRA from {path}...")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    lora_state = ckpt['lora_state']

    for name, module in model.named_modules():
        for attr_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr_name):
                layer = getattr(module, attr_name)
                if isinstance(layer, LoRALinear):
                    key_A = f"{name}.{attr_name}.lora_A"
                    key_B = f"{name}.{attr_name}.lora_B"
                    if key_A in lora_state:
                        layer.lora_A.data = lora_state[key_A].to(DEVICE)
                        layer.lora_B.data = lora_state[key_B].to(DEVICE)

    if optimizer is not None and 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception:
            print("  Note: optimizer state incompatible, starting fresh")

    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', 0.0)
    print(f"  Resumed LoRA from step {step}, loss={loss:.4f}")
    return step


def merge_and_save(model: nn.Module, tokenizer, path: str):
    """Merge LoRA weights into base model and save as a full checkpoint."""
    print("Merging LoRA weights into base model...")
    for module in model.modules():
        for attr_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(module, attr_name):
                layer = getattr(module, attr_name)
                if isinstance(layer, LoRALinear):
                    setattr(module, attr_name, layer.merge())

    torch.save({
        'step': -1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'loss': 0.0,
        'config': model.config,
        'sft': True,
        'lora_merged': True,
    }, path)
    try:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    print(f"  Merged model saved: {path}")


# ============================================================================
# Cell 3: Load Pretrained Model and Apply LoRA
# ============================================================================

LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05

def load_base_model():
    """Load final.pt and apply LoRA adapters."""
    config = CONFIG
    print(f"Building model ({config.n_layers}L, d={config.d_model})...")
    model = TokenizedHLMv9(config).to(DEVICE)

    # Load pretrained weights — prefer best.pt (lowest eval loss checkpoint)
    base_ckpt_path = None
    for name in ['best.pt', 'final.pt', 'latest.pt']:
        p = os.path.join(DRIVE_CKPT_DIR, name)
        if os.path.exists(p):
            base_ckpt_path = p
            break

    if base_ckpt_path is None:
        raise FileNotFoundError(f"No pretrained checkpoint found in {DRIVE_CKPT_DIR}")

    print(f"Loading pretrained weights from {os.path.basename(base_ckpt_path)}...")
    ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict']
    model_state = model.state_dict()

    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched buffers (memory banks)")
    if missing:
        print(f"  {len(missing)} new keys initialized fresh")
    print(f"  Loaded from step {ckpt.get('step', 0)}, loss={ckpt.get('loss', 0):.4f}")

    # Apply LoRA — freeze everything, inject adapters into attention only
    print(f"\nApplying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    model = apply_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

    return model


# ============================================================================
# Cell 4: SFT Data Pipeline (LoRA version)
# ============================================================================

CHAT_USER_PREFIX = "\n<|user|>\n"
CHAT_ASST_PREFIX = "\n<|assistant|>\n"
CHAT_END = "\n<|end|>\n"

def load_sft_datasets_lora(tokenizer, max_seq_len: int = 512):
    """Load SFT datasets, yield (input_ids, targets) with assistant-only loss.

    Only uses plain-text English datasets — avoids any dataset that uses
    Qwen's extended vocab (IDs > 50303) in its special tokens.
    Examples with more than 2% out-of-vocab tokens are skipped entirely
    rather than clamped, so no garbage 'anyak' tokens corrupt training.
    """
    from datasets import load_dataset
    import random

    VOCAB_SIZE = CONFIG.vocab_size  # 50304

    def encode_example(user_text, assistant_text):
        """Encode one Q/A turn with assistant-only loss masking.
        Returns None if >2% of tokens are out-of-vocab (skip corrupted examples).
        """
        if not user_text.strip() or not assistant_text.strip():
            return None

        prompt = CHAT_USER_PREFIX + user_text.strip() + CHAT_END + CHAT_ASST_PREFIX
        full   = prompt + assistant_text.strip() + CHAT_END

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids   = tokenizer.encode(full,   add_special_tokens=False)

        # Skip examples with too many out-of-vocab tokens (>2%)
        oov = sum(1 for t in full_ids if t >= VOCAB_SIZE)
        if oov > max(1, len(full_ids) * 0.02):
            return None

        # Hard-clip to vocab range (the small residual OOV gets mapped to 0)
        full_ids   = [t if t < VOCAB_SIZE else 0 for t in full_ids]
        prompt_ids = [t if t < VOCAB_SIZE else 0 for t in prompt_ids]

        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        input_ids = full_ids[:-1]
        targets   = full_ids[1:]

        # Mask prompt tokens — only compute loss on assistant response
        n_prompt = min(len(prompt_ids) - 1, len(targets))
        targets[:n_prompt] = [-100] * n_prompt

        # Pad to max_seq_len - 1
        pad_len = (max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            targets   = targets   + [-100] * pad_len

        return (
            torch.tensor(input_ids[:max_seq_len - 1], dtype=torch.long),
            torch.tensor(targets[:max_seq_len - 1],   dtype=torch.long),
        )

    # --- Dataset generators (plain-text English only, no Qwen extended vocab) ---

    def gen_oasst2():
        """OpenAssistant OASST2 — multilingual Q&A, plain text."""
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        for ex in ds:
            if ex.get("role") == "assistant" and ex.get("lang", "en") == "en":
                result = encode_example(
                    ex.get("parent_text", ""),
                    ex.get("text", "")
                )
                if result is not None:
                    yield result

    def gen_dolly():
        """Databricks Dolly-15K — instruction following, plain English."""
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        for ex in ds:
            instruction = ex.get("instruction", "")
            context     = ex.get("context", "")
            response    = ex.get("response", "")
            user_text   = instruction + (" Context: " + context if context.strip() else "")
            result = encode_example(user_text, response)
            if result is not None:
                yield result

    def gen_math():
        """MetaMathQA — math reasoning, plain English + LaTeX."""
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        for ex in ds:
            result = encode_example(ex.get("query", ""), ex.get("response", ""))
            if result is not None:
                yield result

    def gen_code():
        """CodeAlpaca — code instruction following."""
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        for ex in ds:
            user = ex.get("instruction", "")
            if ex.get("input", "").strip():
                user += "\n" + ex["input"]
            result = encode_example(user, ex.get("output", ""))
            if result is not None:
                yield result

    # Weights: 40% general QA, 30% instruction, 20% math, 10% code
    generators = [gen_oasst2, gen_dolly, gen_math, gen_code]
    weights    = [0.40,        0.30,      0.20,     0.10]

    while True:
        gen_fn = random.choices(generators, weights=weights, k=1)[0]
        try:
            gen = gen_fn()
            for item in gen:
                if item is not None:
                    yield item
        except Exception as e:
            print(f"  Dataset error: {e}, retrying...")
            continue


# ============================================================================
# Cell 5: LoRA Training Loop
# ============================================================================

LORA_LR           = 2e-4    # LoRA can use higher LR — only 4M params moving
LORA_WARMUP       = 100
LORA_TOTAL_STEPS  = 3_000   # Conservative — stop before memorization
LORA_BATCH_SIZE   = 4
LORA_GRAD_ACCUM   = 8       # Effective batch = 32 sequences
LORA_LOG_INTERVAL = 25
LORA_EVAL_INTERVAL= 250
LORA_SAVE_INTERVAL= 500
LORA_SEQ_LEN      = 512     # Shorter seq for SFT


def get_lora_lr(step: int) -> float:
    """Cosine decay with warmup."""
    min_lr = LORA_LR * 0.1
    if step < LORA_WARMUP:
        return LORA_LR * step / max(1, LORA_WARMUP)
    progress = (step - LORA_WARMUP) / max(1, LORA_TOTAL_STEPS - LORA_WARMUP)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (LORA_LR - min_lr) * coeff


def lora_finetune():
    """Main LoRA fine-tuning function."""
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    # Build model + apply LoRA
    model = load_base_model()
    model.train()

    # Only optimize LoRA params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01,
                                   betas=(0.9, 0.95))

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-Tuning")
    print(f"  Steps: {LORA_TOTAL_STEPS:,} | LR: {LORA_LR:.1e}")
    print(f"  Batch: {LORA_BATCH_SIZE} × {LORA_GRAD_ACCUM} accum = {LORA_BATCH_SIZE*LORA_GRAD_ACCUM} seq/step")
    print(f"  Trainable: {sum(p.numel() for p in lora_params):,} params")
    print(f"  Checkpoints: {LORA_CKPT_DIR}")
    print(f"{'='*60}\n")

    # Resume LoRA if previous run exists
    resume_path = os.path.join(LORA_CKPT_DIR, 'lora_latest.pt')
    start_step = load_lora(model, optimizer, resume_path)

    # Data
    print("Loading SFT datasets...")
    data_gen = load_sft_datasets_lora(tokenizer, max_seq_len=LORA_SEQ_LEN)

    # Build a batch from the generator
    def get_batch():
        items = []
        for _ in range(LORA_BATCH_SIZE):
            items.append(next(data_gen))
        inputs = torch.stack([it[0] for it in items]).to(DEVICE)
        targets = torch.stack([it[1] for it in items]).to(DEVICE)
        return inputs, targets

    best_loss = float('inf')
    loss_accum = 0.0
    t0 = time.time()
    step = start_step
    optimizer.zero_grad()

    while step < LORA_TOTAL_STEPS:
        # Gradient accumulation
        for _ in range(LORA_GRAD_ACCUM):
            try:
                input_ids, targets = get_batch()
            except StopIteration:
                break

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model(input_ids, targets, do_memory_write=False)
                loss = result["loss"] / LORA_GRAD_ACCUM

            loss.backward()
            loss_accum += result["loss"].item()

        nn.utils.clip_grad_norm_(lora_params, 1.0)

        lr = get_lora_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % LORA_LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            avg_loss = loss_accum / LORA_LOG_INTERVAL
            ppl = math.exp(min(avg_loss, 20))
            tps = (step * LORA_BATCH_SIZE * LORA_GRAD_ACCUM * LORA_SEQ_LEN) / elapsed
            eta_h = (LORA_TOTAL_STEPS - step) / max(step, 1) * elapsed / 3600
            print(f"[LoRA {step:>5d}/{LORA_TOTAL_STEPS}] "
                  f"loss={avg_loss:.4f} ppl={ppl:.1f} | "
                  f"{tps/1000:.0f}K tok/s | lr={lr:.2e} | ETA {eta_h:.1f}h")
            loss_accum = 0.0

        if step % LORA_SAVE_INTERVAL == 0:
            path = os.path.join(LORA_CKPT_DIR, f'lora_step_{step:06d}.pt')
            save_lora(model, optimizer, step, avg_loss if step > 0 else 0.0, path)
            save_lora(model, optimizer, step, avg_loss if step > 0 else 0.0, resume_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_lora(model, optimizer, step, avg_loss,
                          os.path.join(LORA_CKPT_DIR, 'lora_best.pt'))

    # Save final LoRA weights
    final_lora_path = os.path.join(LORA_CKPT_DIR, 'lora_final.pt')
    save_lora(model, optimizer, step, best_loss, final_lora_path)
    print(f"\nLoRA training complete! Best loss: {best_loss:.4f}")

    # Merge LoRA into base model and save as full checkpoint
    merged_path = os.path.join(LORA_CKPT_DIR, 'lora_merged.pt')
    merge_and_save(model, tokenizer, merged_path)

    return model, tokenizer


# ============================================================================
# Cell 6: Chat Interface
# ============================================================================

def chat_lora(model, tokenizer):
    """Chat with the LoRA fine-tuned model."""
    model.eval()

    CHAT_USER_PREFIX_C = "\n<|user|>\n"
    CHAT_ASST_PREFIX_C = "\n<|assistant|>\n"
    CHAT_END_C = "\n<|end|>\n"

    @torch.no_grad()
    def generate(input_ids, max_new_tokens=256, temperature=0.7,
                 top_k=50, top_p=0.9, rep_penalty=1.15):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -CONFIG.max_seq_len:]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model(idx_cond, do_memory_write=False)
            logits = result["logits"][:, -1, :] / temperature

            # Repetition penalty
            if rep_penalty != 1.0:
                for tok_id in set(input_ids[0].tolist()):
                    if logits[0, tok_id] > 0:
                        logits[0, tok_id] /= rep_penalty
                    else:
                        logits[0, tok_id] *= rep_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                logits[remove.scatter(1, sorted_idx, remove)] = float('-inf')

            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode and check for stop tokens before printing
            tok_str = tokenizer.decode(next_token[0].tolist(),
                                       skip_special_tokens=False)
            stop = False
            for marker in ['<|end|>', '<|user|>', '<|system|>']:
                if marker in tok_str:
                    stop = True
                    break
            if stop:
                break
            print(tok_str, end='', flush=True)

        return input_ids

    print("\n" + "=" * 60)
    print("  TokenizedHLMv9 + LoRA Chat")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60 + "\n")

    history = []
    MAX_HISTORY = 2  # Keep last 2 turns — prevents context bleed
    system = "You are a helpful, accurate, and concise assistant. Answer questions clearly."

    while True:
        try:
            user_input = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if user_input.lower() == 'reset':
            history = []
            print("[History cleared]\n")
            continue
        if not user_input:
            continue

        # Keep only last MAX_HISTORY turns to prevent context overflow
        recent_history = history[-MAX_HISTORY:]

        # Build prompt
        parts = [f"{CHAT_USER_PREFIX_C}{system}{CHAT_END_C}"]
        for u, a in recent_history:
            parts.append(f"{CHAT_USER_PREFIX_C}{u}{CHAT_END_C}")
            parts.append(f"{CHAT_ASST_PREFIX_C}{a}{CHAT_END_C}")
        parts.append(f"{CHAT_USER_PREFIX_C}{user_input}{CHAT_END_C}")
        parts.append(CHAT_ASST_PREFIX_C)

        text = "".join(parts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Skip OOV tokens rather than clamping to anyak
        ids = [t for t in ids if t < CONFIG.vocab_size]
        input_ids = torch.tensor([ids], device=DEVICE)

        print(f"\n[{len(ids)} tokens]\nHLM-v9: ", end='', flush=True)
        output_ids = generate(input_ids)
        new_tokens = output_ids[0][len(ids):].tolist()
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)
        for stop in ['\n<|end|>', '<|end|>', '\n<|user|>', '<|user|>']:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()
        print("\n")
        history.append((user_input, response))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.train or not args.chat:
        model, tokenizer = lora_finetune()
        chat_lora(model, tokenizer)
    elif args.chat:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        model = load_base_model()
        ckpt_path = args.checkpoint or os.path.join(LORA_CKPT_DIR, 'lora_final.pt')
        load_lora(model, None, ckpt_path)
        chat_lora(model, tokenizer)
