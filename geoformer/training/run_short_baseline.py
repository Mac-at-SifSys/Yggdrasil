#!/usr/bin/env python3
"""Short flat-stack GeoFormer baseline run.

This is a minimal LM-only training loop for the GeoFormer model family.
It exists to answer one question cleanly:

    "Does the flat-stack geometric model learn on real text?"

It intentionally disables the extra curriculum/labeled-data machinery and
defaults to:
- no ToU layers
- no auxiliary heads
- pure cross-entropy LM training
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from geoformer.config import GeoFormerConfig
from geoformer.model import GeoFormer


def round_up_vocab(n: int, multiple: int = 64) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class StreamingTokenChunks(Iterable[dict[str, torch.Tensor]]):
    """Stream token chunks from HF datasets."""

    DATASET_CANDIDATES = (
        ("Skylion007/openwebtext", {}),
        ("HuggingFaceFW/fineweb", {"name": "sample-10BT"}),
        ("allenai/c4", {"name": "en"}),
    )

    def __init__(self, tokenizer, seq_len: int, dataset_name: Optional[str] = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name

    def _iter_dataset(self):
        if self.dataset_name:
            yield self.dataset_name, load_dataset(self.dataset_name, split="train", streaming=True)
            return

        last_error = None
        for name, kwargs in self.DATASET_CANDIDATES:
            try:
                ds = load_dataset(name, split="train", streaming=True, trust_remote_code=True, **kwargs)
                yield name, ds
                return
            except Exception as exc:  # pragma: no cover - depends on remote availability
                last_error = exc
                continue

        raise RuntimeError(f"no streaming dataset available: {last_error}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        _, ds = next(iter(self._iter_dataset()))
        buffer: list[int] = []

        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 32:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in items], dim=0)
    targets = torch.stack([item["targets"] for item in items], dim=0)
    return {"input_ids": input_ids, "targets": targets}


def make_model(args) -> tuple[GeoFormer, GeoFormerConfig]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    padded_vocab = round_up_vocab(len(tokenizer), 64)

    config = GeoFormerConfig(
        d_model=args.d_model,
        n_blades=args.n_blades,
        d_blade=args.d_blade,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        vocab_size=padded_vocab,
        max_seq_len=args.seq_len,
        tou_attn_layers=[],
        embed_dropout=args.embed_dropout,
        residual_dropout=0.0,
        ffn_dropout=0.0,
        gradient_checkpointing=args.grad_ckpt,
        use_flash_attn=True,
        use_blade_predictor=False,
        use_narrative_parse=False,
    )
    model = GeoFormer(config)
    return model, config


def get_lr(step: int, total_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Short GeoFormer LM-only baseline")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dataset", default=None, help="Optional HF dataset path")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)
    parser.add_argument("--d-model", type=int, default=640)
    parser.add_argument("--n-blades", type=int, default=8)
    parser.add_argument("--d-blade", type=int, default=80)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ffn", type=int, default=5120)
    parser.add_argument("--embed-dropout", type=float, default=0.0)
    args = parser.parse_args()

    if args.d_model != args.n_blades * args.d_blade:
        raise ValueError("d_model must equal n_blades * d_blade")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, config = make_model(args)
    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(tag in name for tag in ("norm", "bias")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))

    dataset = StreamingTokenChunks(tokenizer, seq_len=args.seq_len, dataset_name=args.dataset)
    iterator = iter(dataset)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    t0 = time.time()
    tokens_since_log = 0
    running_loss = 0.0
    last_grad_norm = float("nan")

    print("GeoFormer short baseline")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    print(f"  steps={args.steps} batch={args.batch_size} seq={args.seq_len}")
    print(f"  n_layers={args.n_layers} d_model={args.d_model} d_ffn={args.d_ffn}")
    print(f"  tou_attn_layers=[] aux_heads=off")
    print(f"  vocab_size={config.vocab_size}")

    for step in range(1, args.steps + 1):
        lr = get_lr(step - 1, args.steps, args.lr, args.min_lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch_items = [next(iterator) for _ in range(args.batch_size)]
        batch = collate_batch(batch_items)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            outputs = model(input_ids)
            loss = F.cross_entropy(
                outputs["logits"].reshape(-1, config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.detach().cpu())
        best_loss = min(best_loss, loss_value)
        running_loss += loss_value
        tokens_since_log += input_ids.numel()

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            dt = max(time.time() - t0, 1e-6)
            tok_s = tokens_since_log / dt
            avg_loss = running_loss / max(1, args.log_interval if step > 1 else 1)
            print(
                f"step {step:4d}/{args.steps} | "
                f"loss {loss_value:.4f} | avg {avg_loss:.4f} | best {best_loss:.4f} | "
                f"lr {lr:.2e} | gnorm {float(last_grad_norm):.3f} | {tok_s:.1f} tok/s"
            )
            t0 = time.time()
            tokens_since_log = 0
            running_loss = 0.0

    if output_dir:
        ckpt_path = output_dir / "final.pt"
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "config": asdict(config),
                "steps": args.steps,
                "best_loss": best_loss,
            },
            ckpt_path,
        )
        print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
