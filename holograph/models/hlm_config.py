"""
hlm_config.py — Configuration dataclass for the HLM model.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HLMConfig:
    """
    Configuration for the Holographic Language Model.

    The model operates in Cl(3,0) where each "hidden dimension" is
    an 8-component multivector. So d_model=64 means 64 multivectors,
    which is 64*8 = 512 scalar parameters per position.
    """

    # Vocabulary
    vocab_size: int = 32000

    # Model dimensions (in multivector units, not scalars)
    d_model: int = 64          # number of multivector channels
    n_layers: int = 6          # number of HLM transformer blocks
    n_heads: int = 8           # attention heads
    d_ff: int = 256            # feed-forward intermediate dimension

    # ToU parameters
    n_tou_primitives: int = 256   # active primitives (of 1486 total)
    n_blades: int = 9             # number of ToU blades
    tou_layer_interval: int = 2   # insert ToU layer every N blocks

    # Sequence
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.1

    # MoE (optional)
    use_moe: bool = False
    n_experts: int = 4
    moe_top_k: int = 2

    # Grade activation schedule: which activation to use
    # Options: 'gelu', 'relu', 'sigmoid'
    grade_activation: str = 'gelu'

    # Grade activation schedule per layer (optional)
    # If provided, overrides grade_activation per layer
    grade_activation_schedule: Optional[List[str]] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        if self.grade_activation_schedule is not None:
            assert len(self.grade_activation_schedule) == self.n_layers

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def total_scalar_params_per_position(self) -> int:
        """Total scalar parameters per sequence position."""
        return self.d_model * 8

    def small_test_config(self=None) -> 'HLMConfig':
        """Create a small config for testing."""
        return HLMConfig(
            vocab_size=256,
            d_model=8,
            n_layers=2,
            n_heads=2,
            d_ff=16,
            n_tou_primitives=32,
            max_seq_len=64,
            dropout=0.0,
        )
