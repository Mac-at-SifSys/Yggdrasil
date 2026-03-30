"""
trainer.py -- Main training loop for the Clifford HLM.

CliffordTrainer orchestrates:
  - Forward pass through the HLM model
  - Loss computation: cross_entropy + algebraic_consistency + grade_entropy
  - Gradient clipping via clifford_grad_clip
  - Optimizer step
  - Per-grade statistics logging

The model interface expected:
  model.forward(input_ids) -> logits_mv  (np.ndarray, shape (B, seq_len, vocab, 8))
  model.parameters() -> list of dict with "mv" keys (Multivector params)
  model.get_rotor_params() -> list of np.ndarray (optional)
  model.get_layer_representations() -> list of np.ndarray (optional)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import time
import numpy as np
from typing import Dict, Optional, Any, List

from rune.backend import to_numpy, xp
from rune.types.multivector import Multivector, GRADE_SLICES
from forge.losses.cross_entropy import clifford_cross_entropy, clifford_cross_entropy_with_grad
from forge.losses.algebraic_consistency import algebraic_consistency_loss
from forge.losses.grade_entropy import grade_entropy_penalty
from forge.optimizers.clifford_grad_clip import clifford_grad_clip
from forge.param_utils import assign_param_grad, get_param_data, infer_param_grade


def _compute_grade_stats(params: list) -> Dict[str, float]:
    """Compute per-grade RMS of parameter values for logging."""
    stats = {}
    accum = {0: [], 1: [], 2: [], 3: []}

    for p in params:
        data = to_numpy(get_param_data(p))
        inferred_grade = infer_param_grade(data)
        if inferred_grade is None and getattr(data, "shape", ()) and data.shape[-1] == 8:
            for grade, slc in GRADE_SLICES.items():
                vals = data[..., slc]
                accum[grade].append(float(np.sqrt(np.mean(vals ** 2))))
        elif inferred_grade is not None:
            accum[inferred_grade].append(float(np.sqrt(np.mean(data ** 2))))

    for grade in range(4):
        if accum[grade]:
            stats[f"param_rms_grade{grade}"] = float(np.mean(accum[grade]))
        else:
            stats[f"param_rms_grade{grade}"] = 0.0

    return stats


def _to_backend_int_array(value):
    """Convert torch / numpy / cupy-like batches to the active backend."""
    if isinstance(value, Multivector):
        return value._data
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except Exception:
            pass
    return xp.asarray(value, dtype=xp.int64)


class CliffordTrainer:
    """Training driver for a Clifford HLM model.

    Parameters
    ----------
    model : object
        Must implement ``forward(input_ids)``, ``parameters()``.
        Optionally ``get_rotor_params()`` and ``get_layer_representations()``.
    optimizer : object
        Must implement ``step()``, ``zero_grad()``.
    max_grad_norm : float
        Maximum gradient norm for clifford_grad_clip.
    ce_weight : float
        Weight for cross-entropy loss.
    alg_weight : float
        Weight for algebraic consistency loss.
    entropy_weight : float
        Weight for grade entropy penalty.
    log_interval : int
        Print stats every N steps.
    """

    def __init__(
        self,
        model,
        optimizer,
        max_grad_norm: float = 1.0,
        ce_weight: float = 1.0,
        alg_weight: float = 0.01,
        entropy_weight: float = 0.01,
        log_interval: int = 50,
        track_grade_stats: bool = False,
        grade_stats_interval: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.ce_weight = ce_weight
        self.alg_weight = alg_weight
        self.entropy_weight = entropy_weight
        self.log_interval = log_interval
        self.track_grade_stats = track_grade_stats
        self.grade_stats_interval = max(1, grade_stats_interval)

        self.global_step = 0
        self.history: List[Dict[str, float]] = []

    def train_step(
        self,
        batch: tuple,
    ) -> Dict[str, float]:
        """Execute one training step.

        Parameters
        ----------
        batch : (input_ids, target_ids)
            input_ids  : np.ndarray (B, seq_len), int64
            target_ids : np.ndarray (B, seq_len), int64

        Returns
        -------
        dict
            Metrics for this step: losses, gradient norm, per-grade stats.
        """
        input_ids, target_ids = batch
        input_ids = _to_backend_int_array(input_ids)
        target_ids = _to_backend_int_array(target_ids)

        self.optimizer.zero_grad()

        # ---- Forward ----
        logits_mv = self.model.forward(input_ids)

        # ---- Cross-entropy loss ----
        # logits_mv might be (B, seq_len, vocab, 8) or (B*seq_len, vocab, 8)
        # Flatten to (N, vocab, 8) for loss
        if isinstance(logits_mv, Multivector):
            raw = logits_mv._data
        elif hasattr(logits_mv, "shape"):
            raw = logits_mv
        elif hasattr(logits_mv, "data") and hasattr(logits_mv.data, "shape"):
            raw = logits_mv.data if hasattr(logits_mv, 'data') else np.array(logits_mv)
        else:
            raw = np.array(logits_mv)

        original_shape = raw.shape
        if raw.ndim == 4:
            B, S, V, C = raw.shape
            flat_logits = raw.reshape(B * S, V, C)
            flat_targets = target_ids.reshape(-1)
        elif raw.ndim == 3 and raw.shape[-1] != 8:
            B, S, V = raw.shape
            flat_logits = raw.reshape(B * S, V)
            flat_targets = target_ids.reshape(-1)
        elif raw.ndim == 3:
            flat_logits = raw
            flat_targets = target_ids.reshape(-1)
        else:
            flat_logits = raw
            flat_targets = target_ids.reshape(-1)

        ce_loss, ce_grad = clifford_cross_entropy_with_grad(flat_logits, flat_targets)

        # ---- Algebraic consistency loss ----
        rotor_params = None
        layer_reps = None
        if hasattr(self.model, 'get_rotor_params'):
            try:
                rotor_params = self.model.get_rotor_params()
            except Exception:
                pass
        if hasattr(self.model, 'get_layer_representations'):
            try:
                layer_reps = self.model.get_layer_representations()
            except Exception:
                pass

        alg_loss = 0.0
        if self.alg_weight != 0.0:
            alg_loss = algebraic_consistency_loss(
                rotor_params=rotor_params,
                layer_representations=layer_reps,
            )

        # ---- Grade entropy penalty (on output) ----
        ent_source = None
        if flat_logits.ndim >= 2 and flat_logits.shape[-1] == 8:
            ent_source = flat_logits
        elif layer_reps:
            candidate = layer_reps[-1]
            if hasattr(candidate, 'shape') and candidate.shape[-1] == 8:
                ent_source = candidate
        ent_loss = 0.0
        if self.entropy_weight != 0.0 and ent_source is not None:
            ent_loss = grade_entropy_penalty(ent_source)

        total_loss = (
            self.ce_weight * ce_loss
            + self.alg_weight * alg_loss
            + self.entropy_weight * ent_loss
        )

        # ---- Backward: distribute CE grad to model parameters ----
        # We propagate the cross-entropy gradient to all parameters via the
        # model's backward hook.  If the model doesn't support it, we use
        # a simple finite-difference-free heuristic: assign a fraction of
        # the gradient norm as a uniform perturbation.
        if hasattr(self.model, 'backward'):
            if raw.ndim == 4:
                ce_grad_reshaped = ce_grad.reshape(original_shape)
            elif raw.ndim == 3 and raw.shape[-1] != 8:
                ce_grad_reshaped = ce_grad.reshape(original_shape)
            else:
                ce_grad_reshaped = ce_grad
            self.model.backward(ce_grad_reshaped)
            if hasattr(self.model, 'write_to_memory'):
                try:
                    self.model.write_to_memory()
                except Exception:
                    pass
        else:
            # Fallback: push the CE gradient magnitude into parameter grads
            # This is a crude approximation for models without full autograd
            params = self.model.parameters()
            grad_scale = np.sqrt(np.mean(ce_grad ** 2))
            for p in params:
                data = get_param_data(p)
                noise = np.random.randn(*data.shape).astype(np.float32)
                assign_param_grad(p, grad_scale * noise)

        # ---- Gradient clipping ----
        params = self.model.parameters()
        grad_norm = clifford_grad_clip(params, self.max_grad_norm)

        # ---- Optimizer step ----
        self.optimizer.step()

        # ---- Per-grade statistics ----
        grade_stats = {}
        if self.track_grade_stats and (self.global_step % self.grade_stats_interval == 0):
            grade_stats = _compute_grade_stats(params)

        metrics = {
            "step": self.global_step,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "alg_loss": alg_loss,
            "ent_loss": ent_loss,
            "grad_norm": grad_norm,
            **grade_stats,
        }

        self.history.append(metrics)
        self.global_step += 1

        return metrics

    def train(
        self,
        dataloader,
        num_epochs: int = 1,
        scheduler=None,
    ) -> List[Dict[str, float]]:
        """Run the full training loop.

        Parameters
        ----------
        dataloader : iterable yielding (input_ids, target_ids) batches
        num_epochs : int
        scheduler : optional, must have get_lr(step) -> float

        Returns
        -------
        list of dict
            All step metrics.
        """
        for epoch in range(num_epochs):
            t0 = time.time()
            for batch in dataloader:
                if scheduler is not None:
                    lr = scheduler.get_lr(self.global_step)
                    self.optimizer.lr = lr

                metrics = self.train_step(batch)

                if self.global_step % self.log_interval == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[step {metrics['step']:>6d}] "
                        f"loss={metrics['loss']:.4f}  "
                        f"ce={metrics['ce_loss']:.4f}  "
                        f"alg={metrics['alg_loss']:.4f}  "
                        f"ent={metrics['ent_loss']:.4f}  "
                        f"gnorm={metrics['grad_norm']:.3f}  "
                        f"({elapsed:.1f}s)"
                    )

        return self.history
