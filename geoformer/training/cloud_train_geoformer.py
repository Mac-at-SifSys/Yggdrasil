#!/usr/bin/env python3
"""GeoFormer-250M Cloud Training Script — Self-contained for A100.

Upload this script + the geoformer/ package to your Lambda A100 instance
and run directly:

    # On Lambda A100:
    pip install torch transformers datasets tqdm wandb
    python cloud_train_geoformer.py \
        --output /workspace/geoformer-250m \
        --bank /workspace/bank_full.json \
        --labeled /workspace/wz1_labeled.jsonl \
        --wandb-project geoformer

Hardware requirements:
    - A100 80GB: batch_size=32, seq_len=2048, grad_accum=4 (~25GB VRAM)
    - A100 40GB: batch_size=16, seq_len=2048, grad_accum=8 (~18GB VRAM)

Training phases (3-phase curriculum):
    Phase 1 (60%): Language foundation on SlimPajama
    Phase 2 (25%): Blade alignment (mix general + WZ1 labeled)
    Phase 3 (15%): Knowledge grounding (heavy WZ1 + ToU retrieval)
"""

import os
import sys
import time
import math
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Add parent dir to path so geoformer package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from geoformer.config import GeoFormerConfig
from geoformer.model import GeoFormer
from geoformer.training.losses import GeoFormerLoss
from geoformer.training.data import CurriculumMixer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("geoformer")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Stream text from HuggingFace datasets for causal LM pretraining."""

    # Datasets to try in order (some get removed from HF)
    DATASET_CANDIDATES = [
        ("HuggingFaceFW/fineweb", {"name": "sample-10BT", "split": "train"}),
        ("allenai/c4", {"name": "en", "split": "train"}),
        ("togethercomputer/RedPajama-Data-V2", {"name": "sample", "split": "train"}),
    ]

    def __init__(self, tokenizer, seq_len: int = 2048, split: str = "train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split

    def __iter__(self):
        from datasets import load_dataset

        ds = None
        for dataset_path, kwargs in self.DATASET_CANDIDATES:
            try:
                log.info(f"Trying dataset: {dataset_path}")
                ds = load_dataset(
                    dataset_path,
                    streaming=True,
                    trust_remote_code=True,
                    **kwargs,
                )
                log.info(f"  Using: {dataset_path}")
                break
            except Exception as e:
                log.warning(f"  {dataset_path} unavailable: {e}")
                continue

        if ds is None:
            raise RuntimeError("No pretraining dataset available. Check HF access.")

        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 50:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}


class WZ1LabeledDataset(Dataset):
    """Loads WZ1 labeled data with blade activation targets."""

    def __init__(self, path: str, tokenizer, max_seq_len: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        log.info(f"Loading labeled data from {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        log.info(f"  Loaded {len(self.samples):,} labeled examples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build text from messages
        messages = sample.get("messages", [])
        text = ""
        for msg in messages:
            text += f"<|{msg.get('role', '')}|>\n{msg.get('content', '')}\n"

        # Fallback: use 'text' field directly
        if not text.strip():
            text = sample.get("text", "Hello")

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        if len(tokens) < 2:
            tokens = [0, 0]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        result = {"input_ids": input_ids, "targets": targets}

        if "blade_activations" in sample:
            result["blade_targets"] = torch.tensor(
                sample["blade_activations"], dtype=torch.float32
            )

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    targets = []
    blade_targets = []
    has_blades = any("blade_targets" in item for item in batch)

    for item in batch:
        T = item["input_ids"].shape[0]
        pad = max_len - T
        input_ids.append(F.pad(item["input_ids"], (0, pad), value=0))
        targets.append(F.pad(item["targets"], (0, pad), value=-100))
        if has_blades:
            bt = item.get("blade_targets", torch.zeros(8))
            blade_targets.append(bt)

    result = {
        "input_ids": torch.stack(input_ids),
        "targets": torch.stack(targets),
    }
    if blade_targets:
        result["blade_targets"] = torch.stack(blade_targets)

    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class CloudTrainer:
    """Self-contained training loop for A100 deployment."""

    def __init__(self, args):
        self.args = args

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            log.warning("No CUDA device — training on CPU (will be very slow)")

        # Tokenizer
        log.info("Loading Qwen2 tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            trust_remote_code=True,
        )
        log.info(f"  Vocab size: {len(self.tokenizer)}")

        # Model config
        actual_vocab = len(self.tokenizer)
        # Round up to nearest 64 for tensor core efficiency
        padded_vocab = ((actual_vocab + 63) // 64) * 64

        self.model_config = GeoFormerConfig(
            d_model=640,
            n_blades=8,
            d_blade=80,
            n_layers=18,
            n_heads=8,
            d_ffn=5120,
            vocab_size=padded_vocab,
            max_seq_len=args.seq_len,
            tou_n_primitives=1486,
            tou_attn_layers=[4, 8, 12, 16],
            tou_bank_path=args.bank if args.bank else None,
            embed_dropout=0.1,
            attn_dropout=0.0,
            residual_dropout=0.0,
            ffn_dropout=0.0,
            gradient_checkpointing=args.grad_ckpt,
            use_flash_attn=True,
        )

        # Model
        log.info("Initializing GeoFormer-250M...")
        self.model = GeoFormer(self.model_config).to(self.device)

        param_counts = self.model.count_parameters()
        log.info("Parameter breakdown:")
        for name, count in param_counts.items():
            log.info(f"  {name:<20} {count:>12,}")

        # Compile if PyTorch 2.0+
        if args.compile and hasattr(torch, "compile"):
            log.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["norm", "bias", "embeddings"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=args.lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())

        # Loss
        self.loss_fn = GeoFormerLoss(vocab_size=padded_vocab)

        # Curriculum
        self.curriculum = CurriculumMixer(
            total_steps=args.total_steps,
            phase_splits=[0.6, 0.25, 0.15],
        )

        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16))

        # W&B
        self.use_wandb = args.wandb_project is not None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=f"geoformer-250m-{time.strftime('%m%d-%H%M')}",
                    config=vars(args),
                )
                self.wandb = wandb
            except ImportError:
                log.warning("wandb not installed — logging disabled")
                self.use_wandb = False

        self.global_step = 0
        self.best_loss = float("inf")

    def get_lr(self, step: int) -> float:
        """Cosine schedule with warmup."""
        args = self.args
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        decay_ratio = (step - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    def save(self, tag: str):
        """Save checkpoint."""
        path = Path(self.args.output) / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap compiled model if needed
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "model_config": self.model_config,
            "args": vars(self.args),
        }, str(path))
        log.info(f"Saved checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    def train(self):
        args = self.args

        # Data loaders
        log.info("Setting up data loaders...")

        general_loader = DataLoader(
            StreamingTextDataset(self.tokenizer, seq_len=args.seq_len),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        labeled_loader = None
        labeled_iter = None
        if args.labeled and Path(args.labeled).exists():
            labeled_ds = WZ1LabeledDataset(args.labeled, self.tokenizer, args.seq_len)
            labeled_loader = DataLoader(
                labeled_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
            )
            labeled_iter = iter(labeled_loader)

        general_iter = iter(general_loader)

        log.info(f"Starting training: {args.total_steps:,} steps")
        log.info(f"  Effective batch: {args.batch_size * args.grad_accum} "
                 f"(bs={args.batch_size} x accum={args.grad_accum})")
        log.info(f"  Phase boundaries: {self.curriculum.phase_boundaries}")

        t0 = time.time()
        tokens_processed = 0
        running_losses = {"total": 0, "lm": 0, "blade": 0, "cayley": 0}

        for step in range(args.total_steps):
            self.global_step = step

            # LR schedule
            lr = self.get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Curriculum phase
            phase = self.curriculum.get_phase(step)
            loss_weights = self.curriculum.get_loss_weights(step)
            self.loss_fn.lm_weight = loss_weights["lm"]
            self.loss_fn.blade_weight = loss_weights["blade"]
            self.loss_fn.cayley_weight = loss_weights["cayley"]

            # Accumulation
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            for micro in range(args.grad_accum):
                # Choose data source
                use_labeled = (
                    self.curriculum.should_use_labeled(step)
                    and labeled_iter is not None
                )

                if use_labeled:
                    try:
                        batch = next(labeled_iter)
                    except StopIteration:
                        labeled_iter = iter(labeled_loader)
                        batch = next(labeled_iter)
                else:
                    try:
                        batch = next(general_iter)
                    except StopIteration:
                        general_iter = iter(general_loader)
                        batch = next(general_iter)

                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                blade_targets = batch.get("blade_targets")
                if blade_targets is not None:
                    blade_targets = blade_targets.to(self.device)

                with torch.autocast("cuda", dtype=self.dtype):
                    outputs = self.model(
                        input_ids,
                        return_blade_activations=(phase >= 2),
                    )

                    blade_acts = None
                    if outputs.get("blade_activations"):
                        blade_acts = outputs["blade_activations"][-1]

                    losses = self.loss_fn(
                        logits=outputs["logits"],
                        targets=targets,
                        blade_logits=outputs.get("blade_logits"),
                        blade_targets=blade_targets if (phase >= 2 and blade_targets is not None) else None,
                        blade_activations=blade_acts,
                    )

                loss = losses["total"] / args.grad_accum
                self.scaler.scale(loss).backward()

                tokens_processed += input_ids.numel()

            # Clip and step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), args.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            for k in running_losses:
                if k in losses:
                    running_losses[k] += losses[k].item()

            # Logging
            if step > 0 and step % args.log_interval == 0:
                dt = time.time() - t0
                tps = tokens_processed / dt

                avg = {k: v / args.log_interval for k, v in running_losses.items()}

                log.info(
                    f"step {step:>6,} | phase {phase} | "
                    f"loss {avg['total']:.4f} | lm {avg['lm']:.4f} | "
                    f"blade {avg.get('blade', 0):.4f} | "
                    f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                    f"{tps:,.0f} tok/s"
                )

                if self.use_wandb:
                    self.wandb.log({
                        "loss/total": avg["total"],
                        "loss/lm": avg["lm"],
                        "loss/blade": avg.get("blade", 0),
                        "loss/cayley": avg.get("cayley", 0),
                        "lr": lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "phase": phase,
                        "tokens_per_sec": tps,
                    }, step=step)

                running_losses = {k: 0 for k in running_losses}
                tokens_processed = 0
                t0 = time.time()

            # Checkpoint
            if step > 0 and step % args.save_interval == 0:
                self.save(f"step_{step:06d}")

                # Also keep a "latest" for easy resume
                self.save("latest")

        # Final
        self.save("final")
        log.info("Training complete!")

        if self.use_wandb:
            self.wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GeoFormer-250M Cloud Training")

    # Paths
    p.add_argument("--output", default="/workspace/geoformer-250m",
                   help="Output directory for checkpoints")
    p.add_argument("--bank", default=None,
                   help="Path to bank_full.json (ToU knowledge bank)")
    p.add_argument("--labeled", default=None,
                   help="Path to WZ1 labeled JSONL for phases 2-3")

    # Training
    p.add_argument("--total-steps", type=int, default=100_000,
                   help="Total training steps")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Micro batch size per step")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length")

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate")
    p.add_argument("--min-lr", type=float, default=3e-5,
                   help="Minimum learning rate (end of cosine)")
    p.add_argument("--warmup-steps", type=int, default=2000,
                   help="LR warmup steps")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm")

    # Performance
    p.add_argument("--compile", action="store_true", default=False,
                   help="Use torch.compile")
    p.add_argument("--no-compile", action="store_false", dest="compile")
    p.add_argument("--grad-ckpt", action="store_true", default=True,
                   help="Gradient checkpointing (saves VRAM, default: True)")
    p.add_argument("--no-grad-ckpt", action="store_false", dest="grad_ckpt")

    # Logging
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name (disabled if not set)")

    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  GeoFormer-250M — Cloud Training")
    log.info("=" * 60)
    log.info(f"  Output:     {args.output}")
    log.info(f"  Bank:       {args.bank or '(none — even distribution)'}")
    log.info(f"  Labeled:    {args.labeled or '(none — phase 1 only)'}")
    log.info(f"  Steps:      {args.total_steps:,}")
    log.info(f"  Eff. batch: {args.batch_size * args.grad_accum}")
    log.info(f"  Seq len:    {args.seq_len}")
    log.info(f"  LR:         {args.lr} -> {args.min_lr}")
    log.info("=" * 60)

    trainer = CloudTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
