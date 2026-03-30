"""Smoke test for MoE-HLM."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from geoformer.moe_hlm.config import MoEHLMConfig
from geoformer.moe_hlm.model import MoEHLM
from geoformer.moe_hlm.holographic_expert import HolographicExpert, GeometricRound


def test_geometric_round():
    """Test a single geometric product round."""
    print("=" * 60)
    print("TEST: Geometric Round")
    print("=" * 60)

    config = MoEHLMConfig()
    geo = GeometricRound(config)

    B, T, N, D = 2, 16, 8, config.d_blade
    x = torch.randn(B, T, N, D)

    out = geo(x)
    ok = out.shape == x.shape
    print(f"  [{'PASS' if ok else 'FAIL'}] Shape: {out.shape}")

    # Check gradients flow
    out.sum().backward()
    grad_ok = geo.interaction_weights.grad is not None
    print(f"  [{'PASS' if grad_ok else 'FAIL'}] Gradients flow to interaction_weights")

    return ok and grad_ok


def test_holographic_expert():
    """Test a full HLM-8^3 expert."""
    print("\n" + "=" * 60)
    print("TEST: Holographic Expert (HLM-8^3)")
    print("=" * 60)

    config = MoEHLMConfig()
    expert = HolographicExpert(config)

    B, T, D = 2, 16, config.d_model
    x = torch.randn(B, T, D)

    out = expert(x)
    ok = out.shape == x.shape
    print(f"  [{'PASS' if ok else 'FAIL'}] Shape: {out.shape}")

    # Count expert params
    n_params = sum(p.numel() for p in expert.parameters())
    print(f"  Expert params: {n_params:,}")

    # Check 3 geometric rounds
    n_rounds = len(expert.geo_rounds)
    ok_rounds = n_rounds == 3
    print(f"  [{'PASS' if ok_rounds else 'FAIL'}] Geometric rounds: {n_rounds}")

    return ok and ok_rounds


def test_full_model():
    """Test full MoE-HLM model."""
    print("\n" + "=" * 60)
    print("TEST: Full MoE-HLM Model")
    print("=" * 60)

    config = MoEHLMConfig()  # Uses default 1B config

    model = MoEHLM(config)
    counts = model.count_parameters()

    print(f"\n  Parameter breakdown:")
    for name, count in counts.items():
        print(f"    {name:<25} {count:>12,}")

    # Forward pass
    B, T = 1, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    print(f"\n  Forward pass: batch={B}, seq={T}")
    outputs = model(input_ids, targets=targets)

    tests = []

    logits = outputs["logits"]
    tests.append((f"logits shape: {logits.shape}",
                   logits.shape == (B, T, config.vocab_size)))

    tests.append((f"lm_loss: {outputs['lm_loss'].item():.4f}",
                   outputs["lm_loss"].dim() == 0))

    tests.append((f"aux_loss: {outputs['aux_loss'].item():.4f}",
                   outputs["aux_loss"].dim() == 0))

    tests.append((f"total loss: {outputs['loss'].item():.4f}",
                   outputs["loss"].dim() == 0))

    for desc, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    # Gradient flow
    print(f"\n  Gradient flow check:")
    outputs["loss"].backward()

    grad_checks = [
        ("token_embed", model.token_embed.weight),
        ("attention_qkv", model.blocks[0].attn.qkv.weight),
        ("router_gate", model.blocks[0].moe.router.gate.weight),
        ("expert_proj_in", model.blocks[0].moe.experts[0].proj_in.weight),
        ("expert_geo_weights", model.blocks[0].moe.experts[0].geo_rounds[0].interaction_weights),
        ("tou_bank", model.tou_bank.embeddings.weight),
    ]

    all_grads_ok = True
    for name, param in grad_checks:
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        grad_norm = param.grad.norm().item() if has_grad else 0
        print(f"    [{'PASS' if has_grad else 'FAIL'}] {name}: grad_norm={grad_norm:.6f}")
        all_grads_ok = all_grads_ok and has_grad

    # Parameter range check
    total = counts["total_unique"]
    active = counts["active_per_forward"]
    print(f"\n  Total unique params:      {total:>12,}")
    print(f"  Active per forward pass:  {active:>12,}")
    print(f"  Sparsity ratio:           {active/total:.1%} active")

    return all(ok for _, ok in tests) and all_grads_ok


def test_param_budget():
    """Test that params land in expected range for 1.3B target."""
    print("\n" + "=" * 60)
    print("TEST: Parameter Budget (1.3B target)")
    print("=" * 60)

    config = MoEHLMConfig()
    model = MoEHLM(config)
    counts = model.count_parameters()

    total = counts["total_unique"]
    target_min = 900_000_000   # 900M minimum
    target_max = 1_100_000_000 # 1.1B maximum

    ok = target_min <= total <= target_max
    print(f"  Total unique: {total:,}")
    print(f"  Target range: {target_min:,} - {target_max:,}")
    print(f"  [{'PASS' if ok else 'FAIL'}]")

    return ok


def main():
    print("\n" + "=" * 60)
    print("  MoE-HLM SMOKE TEST")
    print("=" * 60)

    results = []
    results.append(("Geometric Round", test_geometric_round()))
    results.append(("Holographic Expert", test_holographic_expert()))
    results.append(("Param Budget", test_param_budget()))
    results.append(("Full Model", test_full_model()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_pass = all(ok for _, ok in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
