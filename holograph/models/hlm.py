"""
hlm.py — Full Holographic Language Model.

Architecture:
    tokens -> CliffordEmbedding -> RotorPositionalEncoding
           -> [HLMBlock] x n_layers (with ToULayer every K blocks)
           -> CliffordLayerNorm
           -> CliffordLinear head -> scalar projection -> logits
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.backend import xp
from holograph.models.hlm_config import HLMConfig
from holograph.models.hlm_block import HLMBlock
from holograph.layers.positional_encoding import RotorPositionalEncoding
from holograph.layers.tou_layer import ToULayer
from holograph.layers.normalization import CliffordLayerNorm
from holograph.layers.clifford_linear import CliffordLinear


class CliffordEmbedding:
    """
    Token embedding into multivector space.

    Each token ID maps to a d_model-dimensional multivector.
    Embedding table: (vocab_size, d_model, 8)
    """

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embeddings with grade-aware Xavier
        scale = 1.0 / xp.sqrt(d_model)
        self.weight = xp.random.randn(vocab_size, d_model, 8).astype(xp.float32) * scale

        # Give scalar components slightly higher magnitude (they carry "meaning")
        self.weight[:, :, 0] *= 2.0

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            token_ids: (batch, seq) — integer token IDs
        Returns:
            (batch, seq, d_model, 8) — multivector embeddings
        """
        return self.weight[token_ids]  # fancy indexing

    def parameters(self):
        return [self.weight]


class HLM:
    """
    Full Holographic Language Model.

    Forward pass:
        tokens (batch, seq) -> logits (batch, seq, vocab_size)
    """

    def __init__(self, config: HLMConfig = None):
        if config is None:
            config = HLMConfig()
        self.config = config

        # Embedding
        self.embedding = CliffordEmbedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.pos_encoding = RotorPositionalEncoding(config.d_model, config.max_seq_len)

        # Transformer blocks
        self.blocks = []
        self.tou_layers = {}  # index -> ToULayer

        for i in range(config.n_layers):
            activation = config.grade_activation
            if config.grade_activation_schedule is not None:
                activation = config.grade_activation_schedule[i]

            block = HLMBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=activation,
                use_moe=config.use_moe,
                n_experts=config.n_experts,
                moe_top_k=config.moe_top_k,
            )
            self.blocks.append(block)

            # Insert ToU layer every tou_layer_interval blocks
            if (i + 1) % config.tou_layer_interval == 0:
                self.tou_layers[i] = ToULayer(
                    config.d_model,
                    n_primitives=config.n_tou_primitives,
                    n_blades=config.n_blades,
                )

        # Final norm
        self.final_norm = CliffordLayerNorm(config.d_model)

        # Language model head: CliffordLinear -> scalar projection
        self.lm_head = CliffordLinear(config.d_model, config.vocab_size, bias=True)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Full forward pass.

        Args:
            tokens: (batch, seq) — integer token IDs
        Returns:
            (batch, seq, vocab_size) — logits
        """
        # Embed tokens: (batch, seq, d_model, 8)
        x = self.embedding.forward(tokens)

        # Apply positional encoding
        x = self.pos_encoding.forward(x)

        # Causal mask: (seq, seq) — True where attention is allowed
        seq_len = tokens.shape[1]
        causal_mask = xp.tril(xp.ones((seq_len, seq_len), dtype=bool))

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block.forward(x, mask=causal_mask)

            # ToU integration after designated blocks
            if i in self.tou_layers:
                x = self.tou_layers[i].forward(x)

        # Final norm
        x = self.final_norm.forward(x)

        # LM head: (batch, seq, d_model, 8) -> (batch, seq, vocab_size, 8)
        logits_mv = self.lm_head.forward(x)

        # Extract scalar component as logits: (batch, seq, vocab_size)
        logits = logits_mv[..., 0]

        return logits

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.pos_encoding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        for tou_layer in self.tou_layers.values():
            params.extend(tou_layer.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def count_parameters(self) -> int:
        """Count total scalar parameters."""
        return sum(p.size for p in self.parameters())

    def __repr__(self):
        return (f"HLM(vocab={self.config.vocab_size}, d_model={self.config.d_model}, "
                f"layers={self.config.n_layers}, heads={self.config.n_heads}, "
                f"params={self.count_parameters():,})")
