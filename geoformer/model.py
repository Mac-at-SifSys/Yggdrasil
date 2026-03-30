"""GeoFormer-250M: The full Geometric Transformer model.

Hybrid Holographic Language Model + LLM that bakes Clifford algebra Cl(3,0)
blade routing and ToU knowledge bank primitives into the transformer architecture.

Architecture flow:
    tokens -> embedding -> blade_projector -> [GeoFormerBlock x N] -> blade_collapse -> logits

Each GeoFormerBlock:
    1. BladeRMSNorm -> GeometricAttention (per-blade + Cayley mixing)
    2. BladeRMSNorm -> GeometricFFN (per-blade SwiGLU + Clifford mixing)
    3. [Optional] ToUMemoryAttention (at designated layers)
    4. Residual connections throughout
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from geoformer.config import GeoFormerConfig
from geoformer.layers.blade_projector import BladeProjector
from geoformer.layers.geo_attention import GeometricAttention
from geoformer.layers.geo_ffn import GeometricFFN
from geoformer.layers.tou_memory import ToUMemoryAttention
from geoformer.layers.blade_collapse import BladeCollapse
from geoformer.layers.norms import BladeRMSNorm
from geoformer.tou.bank import ToUBank


class GeoFormerBlock(nn.Module):
    """A single GeoFormer transformer block.

    Pre-norm architecture:
        mv = mv + attn(norm1(mv))
        mv = mv + ffn(norm2(mv))
        [if tou_layer]: mv = mv + tou_attn(norm3(mv), primitives)
    """

    def __init__(self, config: GeoFormerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_tou = layer_idx in config.tou_attn_layers

        # Pre-attention norm
        self.norm1 = BladeRMSNorm(config.n_blades, config.d_blade)
        self.attn = GeometricAttention(config)

        # Pre-FFN norm
        self.norm2 = BladeRMSNorm(config.n_blades, config.d_blade)
        self.ffn = GeometricFFN(config)

        # Optional ToU memory attention
        if self.has_tou:
            self.norm3 = BladeRMSNorm(config.n_blades, config.d_blade)
            self.tou_attn = ToUMemoryAttention(config)

        self.residual_dropout = nn.Dropout(config.residual_dropout) if config.residual_dropout > 0 else nn.Identity()

    def forward(
        self,
        mv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        primitive_embeddings: Optional[torch.Tensor] = None,
        blade_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mv: (batch, seq, n_blades, d_blade)
            mask: optional causal mask
            primitive_embeddings: (n_primitives, d_blade) for ToU layers
            blade_masks: (n_blades, n_primitives) for ToU layers

        Returns:
            (batch, seq, n_blades, d_blade)
        """
        # Self-attention with residual
        mv = mv + self.residual_dropout(self.attn(self.norm1(mv), mask))

        # FFN with residual
        mv = mv + self.residual_dropout(self.ffn(self.norm2(mv)))

        # ToU memory attention (only at designated layers)
        if self.has_tou and primitive_embeddings is not None:
            mv = mv + self.tou_attn(self.norm3(mv), primitive_embeddings, blade_masks)

        return mv


class BladePredictor(nn.Module):
    """Auxiliary head: predict which blades should be active."""

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.n_blades)

    def forward(self, mv_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mv_flat: (batch, seq, d_model)
        Returns:
            (batch, seq, n_blades) — logits for blade activation
        """
        return self.proj(mv_flat)


class NarrativeParseHead(nn.Module):
    """Auxiliary head: predict narrative parse (tone, urgency, query_type, agency)."""

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.shared = nn.Linear(config.d_model, 128)
        self.tone = nn.Linear(128, 6)       # distressed/negative/mixed/neutral/hopeful/positive
        self.urgency = nn.Linear(128, 3)    # low/moderate/high
        self.query_type = nn.Linear(128, 5) # information/guidance/processing/existential/mixed
        self.agency = nn.Linear(128, 3)     # active/passive/mixed

    def forward(self, mv_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mv_flat: (batch, seq, d_model) — typically use last token
        Returns:
            dict of logits per head
        """
        h = torch.relu(self.shared(mv_flat))
        return {
            "tone": self.tone(h),
            "urgency": self.urgency(h),
            "query_type": self.query_type(h),
            "agency": self.agency(h),
        }


class GeoFormer(nn.Module):
    """GeoFormer-250M: Hybrid Holographic Language Model.

    The core model combining:
    - Standard token embeddings (Qwen2 vocab)
    - Blade projector (flat -> multivector)
    - N geometric transformer blocks with Cayley mixing
    - ToU memory attention at designated layers
    - Blade collapse (multivector -> flat)
    - Weight-tied output head
    """

    def __init__(self, config: GeoFormerConfig):
        super().__init__()
        self.config = config

        # Token embedding (weight-tied with output head)
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.embed_dropout) if config.embed_dropout > 0 else nn.Identity()

        # Input: flat -> multivector
        self.blade_projector = BladeProjector(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GeoFormerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Output: multivector -> flat
        self.blade_collapse = BladeCollapse(config)

        # ToU knowledge bank
        self.tou_bank = ToUBank(
            n_primitives=config.tou_n_primitives,
            d_blade=config.d_blade,
            bank_path=config.tou_bank_path,
        )

        # Output head (weight-tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying

        # Auxiliary heads
        self.blade_predictor = BladePredictor(config) if config.use_blade_predictor else None
        self.narrative_parse = NarrativeParseHead(config) if config.use_narrative_parse else None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_blade_activations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq) token IDs
            targets: (batch, seq) target token IDs for loss computation
            return_blade_activations: if True, return per-layer blade norms

        Returns:
            dict with:
                logits: (batch, seq, vocab_size)
                loss: scalar (if targets provided)
                blade_logits: (batch, seq, n_blades) from aux head
                narrative_parse: dict of parse head logits
                blade_activations: list of (batch, seq, n_blades) per layer
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, T, d_model)
        x = self.embed_dropout(x)

        # Project to multivector space
        mv = self.blade_projector(x)  # (B, T, 8, d_blade)

        # Get ToU primitive embeddings
        prim_embeds, blade_masks = self.tou_bank()

        # Build causal mask once
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

        # Track blade activations for introspection/auxiliary loss
        blade_activations = [] if return_blade_activations else None

        # Forward through transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                mv = torch.utils.checkpoint.checkpoint(
                    block, mv, causal_mask, prim_embeds, blade_masks,
                    use_reentrant=False,
                )
            else:
                mv = block(
                    mv,
                    mask=causal_mask,
                    primitive_embeddings=prim_embeds,
                    blade_masks=blade_masks,
                )

            if return_blade_activations:
                # Blade activation = RMS norm of each blade channel
                blade_norms = mv.pow(2).mean(dim=-1).sqrt()  # (B, T, 8)
                blade_activations.append(blade_norms.detach())

        # Collapse to flat representation
        flat = self.blade_collapse(mv)  # (B, T, d_model)

        # Language modeling head
        logits = self.lm_head(flat)  # (B, T, vocab_size)

        result = {"logits": logits}

        # Compute LM loss if targets provided
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        # Auxiliary heads
        if self.blade_predictor is not None:
            result["blade_logits"] = self.blade_predictor(flat)

        if self.narrative_parse is not None:
            result["narrative_parse"] = self.narrative_parse(flat)

        if blade_activations is not None:
            result["blade_activations"] = blade_activations

        return result

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["token_embed"] = sum(p.numel() for p in self.token_embed.parameters())
        counts["blade_projector"] = sum(p.numel() for p in self.blade_projector.parameters())
        counts["tou_bank"] = sum(p.numel() for p in self.tou_bank.parameters())
        counts["blade_collapse"] = sum(p.numel() for p in self.blade_collapse.parameters())

        attn_params = 0
        ffn_params = 0
        tou_attn_params = 0
        norm_params = 0
        cayley_params = 0

        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attn.qkv.parameters())
            attn_params += sum(p.numel() for p in block.attn.o_proj.parameters())
            cayley_params += sum(p.numel() for p in block.attn.cayley_mixer.parameters())
            ffn_params += sum(p.numel() for p in block.ffn.parameters())
            norm_params += sum(p.numel() for p in block.norm1.parameters())
            norm_params += sum(p.numel() for p in block.norm2.parameters())

            if block.has_tou:
                tou_attn_params += sum(p.numel() for p in block.tou_attn.parameters())
                norm_params += sum(p.numel() for p in block.norm3.parameters())

        counts["attention"] = attn_params
        counts["cayley_mixing"] = cayley_params
        counts["ffn"] = ffn_params
        counts["tou_attention"] = tou_attn_params
        counts["norms"] = norm_params

        if self.blade_predictor:
            counts["blade_predictor"] = sum(p.numel() for p in self.blade_predictor.parameters())
        if self.narrative_parse:
            counts["narrative_parse"] = sum(p.numel() for p in self.narrative_parse.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())
        # Subtract weight-tied lm_head (shared with token_embed)
        counts["total_unique"] = counts["total"] - self.lm_head.weight.numel()

        return counts

    @classmethod
    def from_config(cls, **kwargs) -> "GeoFormer":
        """Create model from keyword arguments."""
        config = GeoFormerConfig(**kwargs)
        return cls(config)
