"""Autoregressive generation for GeoFormer."""

import torch
import torch.nn.functional as F
from typing import Optional, List

from geoformer.model import GeoFormer


@torch.no_grad()
def generate(
    model: GeoFormer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive generation with top-k/top-p sampling.

    Args:
        model: GeoFormer model (eval mode)
        input_ids: (1, seq) prompt token IDs
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: top-k filtering
        top_p: nucleus sampling threshold
        eos_token_id: stop generation at this token

    Returns:
        (1, seq + generated) full sequence including prompt
    """
    model.eval()
    device = input_ids.device

    for _ in range(max_new_tokens):
        # Crop to max_seq_len if needed
        idx_cond = input_ids if input_ids.shape[1] <= model.config.max_seq_len else \
            input_ids[:, -model.config.max_seq_len:]

        # Forward pass
        outputs = model(idx_cond)
        logits = outputs["logits"][:, -1, :]  # (1, vocab_size)

        if temperature == 0:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids


@torch.no_grad()
def generate_with_blades(
    model: GeoFormer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
) -> dict:
    """Generate text and return blade activation traces.

    Returns both the generated text and per-token blade activation
    magnitudes for introspection.
    """
    model.eval()
    device = input_ids.device
    blade_traces = []

    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.shape[1] <= model.config.max_seq_len else \
            input_ids[:, -model.config.max_seq_len:]

        outputs = model(idx_cond, return_blade_activations=True)
        logits = outputs["logits"][:, -1, :]

        # Record blade activations for last token across all layers
        if outputs.get("blade_activations"):
            last_layer_acts = outputs["blade_activations"][-1][:, -1, :]  # (1, 8)
            blade_traces.append(last_layer_acts.cpu())

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return {
        "token_ids": input_ids,
        "blade_traces": torch.cat(blade_traces, dim=0) if blade_traces else None,
    }
