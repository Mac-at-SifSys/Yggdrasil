"""
forge -- L4 Training layer for the YGGDRASIL Clifford algebra HLM stack.

Provides Clifford-algebra-aware training infrastructure:
  - Optimizers: CliffordAdam, CliffordSGD, clifford_grad_clip
  - Losses: cross-entropy on scalar projection, algebraic consistency, grade entropy
  - Schedulers: warmup-cosine, grade warmup
  - Data: tokenizer, dataloader
  - Training: trainer loop, checkpointing, distributed stubs

Dependencies:
  L1 rune   -- Multivector types and operations
  L3 holograph -- HLM model and layers
"""

__version__ = "0.1.0"

from forge.optimizers.clifford_adam import CliffordAdam
from forge.optimizers.clifford_sgd import CliffordSGD
from forge.optimizers.clifford_grad_clip import clifford_grad_clip

from forge.losses.cross_entropy import clifford_cross_entropy
from forge.losses.algebraic_consistency import algebraic_consistency_loss
from forge.losses.grade_entropy import grade_entropy_penalty

from forge.schedulers.warmup_cosine import WarmupCosineScheduler
from forge.schedulers.grade_warmup import GradeWarmupScheduler, grade_mask_at_step

from forge.data.tokenizer import BasicTokenizer
from forge.data.dataloader import CliffordDataLoader

from forge.training.trainer import CliffordTrainer
from forge.training.checkpointing import save_checkpoint, load_checkpoint

__all__ = [
    "CliffordAdam", "CliffordSGD", "clifford_grad_clip",
    "clifford_cross_entropy", "algebraic_consistency_loss", "grade_entropy_penalty",
    "WarmupCosineScheduler", "GradeWarmupScheduler", "grade_mask_at_step",
    "BasicTokenizer", "CliffordDataLoader",
    "CliffordTrainer", "save_checkpoint", "load_checkpoint",
]
