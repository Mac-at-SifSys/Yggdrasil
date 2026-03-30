"""MoE-HLM configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoEHLMConfig:
    """Configuration for MoE-HLM 1.3B."""

    # --- Model dimensions ---
    d_model: int = 1536          # Hidden dimension
    n_layers: int = 24           # Transformer blocks
    n_heads: int = 24            # Standard attention heads
    d_head: int = 64             # Per-head dimension (d_model / n_heads)

    # --- MoE ---
    n_experts: int = 8           # Total holographic experts
    top_k: int = 2               # Experts active per token
    router_aux_loss_weight: float = 0.01  # Load balancing loss

    # --- Holographic Expert (HLM-8^3) ---
    n_blades: int = 8            # Cl(3,0) basis elements
    d_blade: int = 128           # Per-blade dimension inside expert
    n_geometric_rounds: int = 3  # Rounds of geometric product (the ^3)
    expert_d_ffn: int = 640      # FFN dim inside expert (between geo rounds)

    # --- Vocabulary ---
    vocab_size: int = 50_304     # Qwen2 tokenizer
    max_seq_len: int = 2048

    # --- ToU Memory (shared across experts) ---
    tou_n_primitives: int = 1_486
    tou_d_prim: int = 128        # Primitive embedding dim (= d_blade)
    tou_every_n_layers: int = 4  # ToU attention at these layers

    # --- Blade configuration ---
    blade_names: List[str] = field(default_factory=lambda: [
        "narrative", "causation", "affect", "wisdom",
        "relations", "ecology", "epistemics", "temporal",
    ])

    # --- Attention ---
    attn_dropout: float = 0.0
    rope_theta: float = 10_000.0

    # --- Regularization ---
    embed_dropout: float = 0.1
    residual_dropout: float = 0.0

    # --- Training ---
    gradient_checkpointing: bool = False
    init_std: float = 0.02

    # --- Derived ---
    @property
    def tou_attn_layers(self) -> List[int]:
        return list(range(self.tou_every_n_layers - 1,
                          self.n_layers,
                          self.tou_every_n_layers))

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) must equal n_heads * d_head "
            f"({self.n_heads} * {self.d_head})"
        )
