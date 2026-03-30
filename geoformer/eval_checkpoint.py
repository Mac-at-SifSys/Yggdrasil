#!/usr/bin/env python3
"""GeoFormer checkpoint evaluation.

Run against any saved checkpoint to measure:
1. Perplexity on held-out text
2. Blade specialization analysis
3. Cayley mixing strength
4. ToU primitive attention patterns
5. Sample generation

Usage:
    python -m geoformer.eval_checkpoint --checkpoint ~/geoformer-250m/step_005000.pt

Or on Lambda:
    python -m geoformer.eval_checkpoint --checkpoint ~/geoformer-250m/latest.pt
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoformer.config import GeoFormerConfig
from geoformer.model import GeoFormer
from geoformer.clifford.algebra import BLADE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["model_config"]
    model = GeoFormer(config).to(device)

    # Handle compiled model state dicts
    state_dict = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix if present (from torch.compile)
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    step = ckpt.get("global_step", "unknown")
    log.info(f"  Loaded step {step}, {sum(p.numel() for p in model.parameters()):,} params")
    return model, config, step


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)


# --------------------------------------------------------------------------
# Eval 1: Perplexity
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(model, tokenizer, config, device, n_samples=50):
    """Compute perplexity on held-out FineWeb samples."""
    log.info("=" * 60)
    log.info("EVAL: Perplexity")
    log.info("=" * 60)

    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    total_loss = 0
    total_tokens = 0
    n = 0

    # Skip first 100K examples to get held-out data
    for i, example in enumerate(ds):
        if i < 100_000:
            continue
        if n >= n_samples:
            break

        text = example.get("text", "")
        if not text or len(text) < 200:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 64:
            continue
        tokens = tokens[:config.max_seq_len]

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

        outputs = model(input_ids)
        loss = F.cross_entropy(
            outputs["logits"].view(-1, config.vocab_size),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
        n += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    log.info(f"  Samples: {n}")
    log.info(f"  Tokens:  {total_tokens:,}")
    log.info(f"  Avg loss: {avg_loss:.4f}")
    log.info(f"  Perplexity: {perplexity:.2f}")

    return {"perplexity": perplexity, "avg_loss": avg_loss, "n_samples": n}


# --------------------------------------------------------------------------
# Eval 2: Blade Specialization
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_blade_specialization(model, tokenizer, config, device):
    """Test whether different blades activate for different content types."""
    log.info("=" * 60)
    log.info("EVAL: Blade Specialization")
    log.info("=" * 60)

    test_prompts = {
        "causal": [
            "The earthquake caused widespread destruction because",
            "When interest rates rise, inflation tends to",
            "The chemical reaction produces heat which then",
        ],
        "emotional": [
            "She felt overwhelmed by grief after losing",
            "The joy of seeing his daughter for the first time",
            "He was consumed by rage and could not",
        ],
        "ethical": [
            "The moral dilemma of choosing between saving one life or",
            "Justice requires that we consider the rights of",
            "Is it ethical to sacrifice individual freedom for",
        ],
        "temporal": [
            "Throughout the centuries, empires have risen and fallen",
            "The irreversible march of time means that we cannot",
            "Looking back at the history of civilization, we see",
        ],
        "factual": [
            "The capital of France is Paris, which is located",
            "Water molecules consist of two hydrogen atoms and",
            "The speed of light in a vacuum is approximately",
        ],
        "relational": [
            "The bond between mother and child is fundamentally",
            "Social hierarchies emerge when groups compete for",
            "Trust between allies depends on mutual",
        ],
    }

    results = {}
    for category, prompts in test_prompts.items():
        blade_sums = torch.zeros(8)
        n = 0

        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            outputs = model(input_ids, return_blade_activations=True)

            if outputs.get("blade_activations"):
                # Average across layers and sequence positions
                acts = torch.stack(outputs["blade_activations"])  # (L, 1, T, 8)
                mean_acts = acts.mean(dim=(0, 1, 2)).cpu()  # (8,)
                blade_sums += mean_acts
                n += 1

        if n > 0:
            avg = blade_sums / n
            results[category] = avg.tolist()

            # Find dominant blade
            top_idx = avg.argmax().item()
            log.info(f"  {category:<12} -> dominant: {BLADE_NAMES[top_idx]:<12} "
                     f"({avg[top_idx]:.4f})")

            # Show all blade activations
            for i, name in enumerate(BLADE_NAMES):
                bar = "#" * int(avg[i] / avg.max() * 20)
                log.info(f"    {name:<12} {avg[i]:.4f} {bar}")
            log.info("")

    # Check if blades are actually differentiating
    if results:
        all_vecs = torch.tensor(list(results.values()))
        # Cosine similarity between category activation patterns
        normed = F.normalize(all_vecs, dim=1)
        sim_matrix = normed @ normed.t()
        avg_sim = (sim_matrix.sum() - sim_matrix.trace()) / (sim_matrix.numel() - len(results))
        log.info(f"  Avg cross-category cosine similarity: {avg_sim:.4f}")
        log.info(f"  (Lower = better blade specialization. Random ≈ 0.7, perfect = 0.0)")

    return results


# --------------------------------------------------------------------------
# Eval 3: Cayley Mixing Analysis
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_cayley_mixing(model, config):
    """Analyze learned Cayley mixing strengths."""
    log.info("=" * 60)
    log.info("EVAL: Cayley Mixing Strength")
    log.info("=" * 60)

    for i, block in enumerate(model.blocks):
        # Handle compiled models
        attn = block.attn if not hasattr(block, '_orig_mod') else block._orig_mod.attn
        mixer = attn.cayley_mixer

        alpha = mixer.alpha.data
        signs = mixer.cayley_signs
        mask = mixer.cayley_mask
        effective = (alpha * signs * mask).abs()

        max_val = effective.max().item()
        mean_val = effective[mask > 0].mean().item() if (mask > 0).any() else 0
        log.info(f"  Layer {i:2d}: max={max_val:.6f}, mean={mean_val:.6f}")

    # FFN Clifford mixing
    log.info("\n  FFN Clifford mixing alphas:")
    for i, block in enumerate(model.blocks):
        ffn = block.ffn if not hasattr(block, '_orig_mod') else block._orig_mod.ffn
        alpha = ffn.clifford_mixer.alpha.data
        log.info(f"  Layer {i:2d}: mean={alpha.abs().mean():.6f}, max={alpha.abs().max():.6f}")


# --------------------------------------------------------------------------
# Eval 4: ToU Primitive Analysis
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_tou_primitives(model, tokenizer, config, device):
    """Check if ToU primitives are being attended to."""
    log.info("=" * 60)
    log.info("EVAL: ToU Primitive Attention")
    log.info("=" * 60)

    bank = model.tou_bank
    embeddings = bank.embeddings.weight.data  # (1486, d_blade)

    # Check embedding norms (have they moved from init?)
    norms = embeddings.norm(dim=1)
    log.info(f"  Primitive embedding norms:")
    log.info(f"    mean: {norms.mean():.4f}")
    log.info(f"    std:  {norms.std():.4f}")
    log.info(f"    min:  {norms.min():.4f}")
    log.info(f"    max:  {norms.max():.4f}")

    # Check if embeddings have diverged from initialization
    init_std = config.init_std
    expected_norm = (config.d_blade * init_std ** 2) ** 0.5
    log.info(f"    expected at init: ~{expected_norm:.4f}")
    divergence = (norms.mean() - expected_norm).abs() / expected_norm
    log.info(f"    divergence from init: {divergence:.1%}")

    # Check per-blade clustering
    log.info(f"\n  Per-blade primitive statistics:")
    for b, name in enumerate(BLADE_NAMES):
        mask = bank.blade_masks[b]
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        blade_embs = embeddings[indices]
        blade_norms = blade_embs.norm(dim=1)

        # Intra-blade cosine similarity
        if len(indices) > 1:
            normed = F.normalize(blade_embs, dim=1)
            sim = normed @ normed.t()
            triu = sim[torch.triu_indices(len(indices), len(indices), offset=1).unbind()]
            avg_sim = triu.mean().item()
        else:
            avg_sim = 1.0

        log.info(f"    {name:<12}: {len(indices):>4} prims, "
                 f"norm={blade_norms.mean():.4f}, "
                 f"intra-sim={avg_sim:.4f}")


# --------------------------------------------------------------------------
# Eval 5: Sample Generation
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_generation(model, tokenizer, config, device):
    """Generate samples to qualitatively assess the model."""
    log.info("=" * 60)
    log.info("EVAL: Sample Generation")
    log.info("=" * 60)

    from geoformer.inference.generate import generate

    prompts = [
        "The history of mathematics begins with",
        "Once upon a time, in a kingdom far away,",
        "The relationship between cause and effect is",
        "To understand why empires fall, we must consider",
    ]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        output_ids = generate(
            model, input_ids,
            max_new_tokens=64,
            temperature=0.8,
            top_k=40,
        )

        text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        log.info(f"\n  Prompt: {prompt}")
        log.info(f"  Output: {text[:200]}")
        log.info("")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="GeoFormer Checkpoint Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-perplexity", action="store_true",
                   help="Skip perplexity (needs internet for dataset)")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip text generation")
    p.add_argument("--output", default=None, help="Save results to JSON file")
    args = p.parse_args()

    model, config, step = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer()

    results = {"step": step}

    # Run evals
    if not args.skip_perplexity:
        results["perplexity"] = eval_perplexity(model, tokenizer, config, args.device)

    results["blade_specialization"] = eval_blade_specialization(
        model, tokenizer, config, args.device
    )

    eval_cayley_mixing(model, config)
    eval_tou_primitives(model, tokenizer, config, args.device)

    if not args.skip_generation:
        eval_generation(model, tokenizer, config, args.device)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"\nResults saved to {args.output}")

    log.info("\n" + "=" * 60)
    log.info("  EVALUATION COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
