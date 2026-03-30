"""GeoFormer configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeoFormerConfig:
    """Configuration for GeoFormer-250M."""

    # --- Model dimensions ---
    d_model: int = 640          # Total hidden dimension
    n_blades: int = 8           # Cl(3,0) basis elements
    d_blade: int = 80           # Per-blade dimension (d_model / n_blades)
    n_layers: int = 18          # Transformer blocks
    n_heads: int = 8            # One attention head per blade
    d_ffn: int = 5120           # FFN intermediate dim (8x d_model, shared SwiGLU)

    # --- Vocabulary ---
    vocab_size: int = 50_304    # Qwen2 tokenizer (divisible by 64)
    max_seq_len: int = 2048     # Context length

    # --- ToU Memory ---
    tou_n_primitives: int = 1_486       # Knowledge bank size
    tou_attn_layers: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16]
    )
    tou_bank_path: Optional[str] = None  # Path to bank_full.json

    # --- Blade configuration ---
    blade_names: List[str] = field(default_factory=lambda: [
        "narrative",   # Grade 0: scalar (1)
        "causation",   # Grade 1: e1
        "affect",      # Grade 1: e2
        "wisdom",      # Grade 1: e3
        "relations",   # Grade 2: e12
        "ecology",     # Grade 2: e13
        "epistemics",  # Grade 2: e23
        "temporal",    # Grade 3: e123
    ])
    blade_grades: List[int] = field(
        default_factory=lambda: [0, 1, 1, 1, 2, 2, 2, 3]
    )

    # --- Attention ---
    attn_dropout: float = 0.0
    rope_theta: float = 10_000.0

    # --- Regularization ---
    embed_dropout: float = 0.1
    residual_dropout: float = 0.0
    ffn_dropout: float = 0.0

    # --- Clifford mixing ---
    cayley_mix_init_scale: float = 0.01  # Small init for cross-blade mixing

    # --- Auxiliary heads ---
    use_blade_predictor: bool = True
    use_narrative_parse: bool = True

    # --- Memory optimization ---
    gradient_checkpointing: bool = False
    use_flash_attn: bool = True  # Use F.scaled_dot_product_attention

    # --- Initialization ---
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model == self.n_blades * self.d_blade, (
            f"d_model ({self.d_model}) must equal n_blades * d_blade "
            f"({self.n_blades} * {self.d_blade})"
        )
