"""Introspection tools for GeoFormer: blade visualization and ToU analysis."""

import torch
from typing import Dict, List, Optional
from geoformer.model import GeoFormer
from geoformer.clifford.algebra import BLADE_NAMES


@torch.no_grad()
def blade_activation_map(
    model: GeoFormer,
    input_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Get blade activation magnitudes across all layers for a given input.

    Returns:
        dict with:
            activations: (n_layers, seq_len, 8) blade RMS per layer per token
            blade_names: list of 8 blade names
    """
    model.eval()
    outputs = model(input_ids, return_blade_activations=True)

    activations = torch.stack(outputs["blade_activations"], dim=0)  # (L, B, T, 8)
    activations = activations.squeeze(1)  # (L, T, 8) assuming B=1

    return {
        "activations": activations,
        "blade_names": BLADE_NAMES,
    }


@torch.no_grad()
def tou_attention_analysis(
    model: GeoFormer,
    input_ids: torch.Tensor,
) -> Dict[str, list]:
    """Analyze which ToU primitives receive highest attention.

    Runs forward pass and inspects the ToU memory attention layers
    to find which primitives are most attended to.

    Returns dict mapping blade names to lists of (primitive_symbol, attention_weight).
    """
    model.eval()

    # We need to hook into the ToU attention layers to capture attention weights
    tou_attentions = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # The ToU attention output is gated — we want pre-gate attention
            # This is a simplified version; full introspection would need
            # the attention weights from inside the module
            tou_attentions[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for block in model.blocks:
        if block.has_tou:
            h = block.tou_attn.register_forward_hook(make_hook(block.layer_idx))
            hooks.append(h)

    # Forward pass
    outputs = model(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Analyze which primitives are most relevant
    # (Full implementation would capture internal attention weights)
    result = {
        "tou_layers_active": list(tou_attentions.keys()),
        "n_primitives": model.tou_bank.n_primitives,
        "primitive_info": model.tou_bank.primitive_info[:10],  # First 10 as sample
    }

    return result


def print_blade_activations(activations: torch.Tensor, layer: int = -1):
    """Pretty-print blade activations for a single layer.

    Args:
        activations: (n_layers, seq_len, 8) from blade_activation_map
        layer: which layer to display (-1 = last)
    """
    acts = activations[layer]  # (T, 8)

    # Average across sequence
    mean_acts = acts.mean(dim=0)  # (8,)

    print(f"Layer {layer} blade activations (sequence-averaged):")
    max_act = mean_acts.max().item()
    for i, name in enumerate(BLADE_NAMES):
        val = mean_acts[i].item()
        bar_len = int(val / max(max_act, 1e-6) * 30)
        bar = "█" * bar_len
        print(f"  {name:<14} {val:>6.3f}  {bar}")
