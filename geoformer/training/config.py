"""Training configuration for GeoFormer."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # --- Batch ---
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128  # batch_size * grad_accum

    # --- Schedule ---
    total_steps: int = 100_000
    phase_splits: List[float] = field(
        default_factory=lambda: [0.6, 0.25, 0.15]
    )

    # --- Data ---
    seq_len: int = 2048
    general_data_path: Optional[str] = None
    labeled_data_path: Optional[str] = None

    # --- Logging ---
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "checkpoints/geoformer-250m"

    # --- Hardware ---
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True  # torch.compile

    # --- Loss weights (initial, modified by curriculum) ---
    lm_weight: float = 1.0
    blade_weight: float = 0.1
    cayley_weight: float = 0.01
