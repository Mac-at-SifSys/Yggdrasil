"""GeoFormer training loop with curriculum scheduling.

Usage:
    python -m geoformer.training.train --config path/to/config.yaml

Or programmatically:
    from geoformer.training.train import Trainer
    trainer = Trainer(model_config, train_config)
    trainer.train()
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from pathlib import Path

from geoformer.config import GeoFormerConfig
from geoformer.model import GeoFormer
from geoformer.training.config import TrainingConfig
from geoformer.training.losses import GeoFormerLoss
from geoformer.training.data import CurriculumMixer


class Trainer:
    """GeoFormer training loop with curriculum phases."""

    def __init__(
        self,
        model_config: GeoFormerConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        labeled_loader: Optional[DataLoader] = None,
    ):
        self.model_config = model_config
        self.train_config = train_config

        # Device setup
        self.device = torch.device(train_config.device)
        self.dtype = getattr(torch, train_config.dtype)

        # Model
        self.model = GeoFormer(model_config).to(self.device)
        if train_config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Log parameter count
        param_counts = self.model.count_parameters()
        print(f"GeoFormer parameter breakdown:")
        for name, count in param_counts.items():
            print(f"  {name}: {count:,}")

        # Optimizer: AdamW with weight decay exclusions
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embeddings" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2))

        # Loss
        self.loss_fn = GeoFormerLoss(
            lm_weight=train_config.lm_weight,
            blade_weight=train_config.blade_weight,
            cayley_weight=train_config.cayley_weight,
            vocab_size=model_config.vocab_size,
        )

        # Curriculum
        self.curriculum = CurriculumMixer(
            total_steps=train_config.total_steps,
            phase_splits=train_config.phase_splits,
        )

        # Data
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.labeled_loader = labeled_loader
        self.labeled_iter = iter(labeled_loader) if labeled_loader else None

        # State
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        cfg = self.train_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / cfg.warmup_steps
        decay_ratio = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def get_labeled_batch(self) -> Optional[Dict]:
        """Get next batch from labeled data, cycling if needed."""
        if self.labeled_iter is None:
            return None
        try:
            return next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)
            return next(self.labeled_iter)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)

        # Determine if we should use blade labels this step
        phase = self.curriculum.get_phase(self.global_step)
        use_blade_labels = phase >= 2 and "blade_targets" in batch

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            outputs = self.model(
                input_ids,
                return_blade_activations=True,
            )

            # Update loss weights based on curriculum phase
            loss_weights = self.curriculum.get_loss_weights(self.global_step)
            self.loss_fn.lm_weight = loss_weights["lm"]
            self.loss_fn.blade_weight = loss_weights["blade"]
            self.loss_fn.cayley_weight = loss_weights["cayley"]

            # Compute losses
            blade_targets = batch.get("blade_targets", None)
            if blade_targets is not None:
                blade_targets = blade_targets.to(self.device)

            blade_acts = outputs["blade_activations"][-1] if outputs.get("blade_activations") else None

            losses = self.loss_fn(
                logits=outputs["logits"],
                targets=targets,
                blade_logits=outputs.get("blade_logits"),
                blade_targets=blade_targets if use_blade_labels else None,
                blade_activations=blade_acts,
            )

        loss = losses["total"] / self.train_config.gradient_accumulation_steps
        loss.backward()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    outputs["logits"].view(-1, self.model_config.vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:  # Cap eval batches
                break

        return {"eval_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "model_config": self.model_config,
            "train_config": self.train_config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def train(self):
        """Main training loop."""
        cfg = self.train_config
        train_iter = iter(self.train_loader)

        print(f"Starting training: {cfg.total_steps} steps, "
              f"phases at {self.curriculum.phase_boundaries}")

        t0 = time.time()
        running_loss = 0.0

        for step in range(cfg.total_steps):
            self.global_step = step

            # Update learning rate
            lr = self.get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Accumulation loop
            for micro_step in range(cfg.gradient_accumulation_steps):
                # Choose data source based on curriculum
                if self.curriculum.should_use_labeled(step) and self.labeled_loader:
                    batch = self.get_labeled_batch()
                else:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)

                losses = self.train_step(batch)

            # Gradient clipping and optimizer step
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            running_loss += losses["total"]

            # Logging
            if step % cfg.log_interval == 0 and step > 0:
                avg_loss = running_loss / cfg.log_interval
                dt = time.time() - t0
                phase = self.curriculum.get_phase(step)
                tokens_per_sec = cfg.effective_batch_size * cfg.seq_len * cfg.log_interval / dt

                print(
                    f"step {step:6d} | phase {phase} | loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
                    f"lm {losses.get('lm', 0):.4f} | blade {losses.get('blade', 0):.4f}"
                )

                running_loss = 0.0
                t0 = time.time()

            # Evaluation
            if step % cfg.eval_interval == 0 and step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"  eval_loss: {eval_metrics['eval_loss']:.4f}")
                    if eval_metrics["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]
                        self.save_checkpoint(
                            os.path.join(cfg.output_dir, "best.pt")
                        )

            # Checkpoint
            if step % cfg.save_interval == 0 and step > 0:
                self.save_checkpoint(
                    os.path.join(cfg.output_dir, f"step_{step}.pt")
                )

        # Final save
        self.save_checkpoint(os.path.join(cfg.output_dir, "final.pt"))
        print("Training complete.")
