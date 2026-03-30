"""
warmup_cosine.py -- Linear warmup followed by cosine decay.

Standard schedule used in transformer pre-training:
  - Steps [0, warmup_steps):  lr ramps linearly from 0 to peak_lr
  - Steps [warmup_steps, total_steps]:  lr decays via cosine to min_lr
"""

import numpy as np


class WarmupCosineScheduler:
    """Warmup + cosine annealing learning rate schedule.

    Parameters
    ----------
    peak_lr : float
        Maximum learning rate (reached at end of warmup).
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps (including warmup).
    min_lr : float
        Floor learning rate after full decay.
    """

    def __init__(
        self,
        peak_lr: float = 1e-3,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-5,
    ):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        """Return the learning rate for the given step.

        Parameters
        ----------
        step : int
            Current training step (0-indexed).

        Returns
        -------
        float
            Learning rate.
        """
        if step < 0:
            return 0.0

        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine
