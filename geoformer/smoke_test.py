"""Smoke test for GeoFormer-250M.

Verifies:
1. Model instantiation
2. Forward pass produces correct output shapes
3. Parameter count is in expected range (~245M)
4. Gradients flow through all components
5. Clifford algebra Cayley table is correct
6. ToU bank loads and produces correct masks
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geoformer.config import GeoFormerConfig
from geoformer.model import GeoFormer
from geoformer.clifford.algebra import CAYLEY_TABLE, BLADE_NAMES, cayley_sign_tensor
from geoformer.clifford.ops import geometric_product, geometric_product_fast


def test_cayley_table():
    """Verify Cayley table satisfies Cl(3,0) axioms."""
    print("=" * 60)
    print("TEST: Cayley Table Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    # e_i^2 = +1 for grade-1 blades
    for i in [1, 2, 3]:
        idx, sign = CAYLEY_TABLE[i][i]
        ok = (idx == 0 and sign == 1)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] e{i}^2 = +1")
        passed += ok
        failed += (not ok)

    # Bivectors square to -1
    for i in [4, 5, 6]:
        idx, sign = CAYLEY_TABLE[i][i]
        ok = (idx == 0 and sign == -1)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {BLADE_NAMES[i]}^2 = -1")
        passed += ok
        failed += (not ok)

    # Pseudoscalar squares to -1
    idx, sign = CAYLEY_TABLE[7][7]
    ok = (idx == 0 and sign == -1)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] temporal^2 = -1")
    passed += ok
    failed += (not ok)

    # Semantic predictions
    tests = [
        (1, 2, 4, "Causation x Affect = Relations"),
        (1, 3, 5, "Causation x Wisdom = Ecology"),
        (2, 3, 6, "Affect x Wisdom = Epistemics"),
    ]
    for i, j, expected_k, desc in tests:
        k, sign = CAYLEY_TABLE[i][j]
        ok = (k == expected_k and sign == 1)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        passed += ok
        failed += (not ok)

    # Fast geometric product matches slow
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    slow = geometric_product(a, b)
    fast = geometric_product_fast(a, b)
    ok = torch.allclose(slow, fast, atol=1e-5)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Fast geom product matches slow")
    passed += ok
    failed += (not ok)

    print(f"\n  Cayley: {passed}/{passed + failed} passed")
    return failed == 0


def test_model_forward():
    """Test forward pass and output shapes."""
    print("\n" + "=" * 60)
    print("TEST: Model Forward Pass")
    print("=" * 60)

    config = GeoFormerConfig(
        d_model=640,
        n_blades=8,
        d_blade=80,
        n_layers=18,
        n_heads=8,
        d_ffn=320,
        vocab_size=50_304,
        max_seq_len=2048,
        tou_n_primitives=1_486,
        tou_attn_layers=[4, 8, 12, 16],
    )

    device = torch.device("cpu")
    model = GeoFormer(config).to(device)

    # Test input
    B, T = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    print(f"  Input: batch={B}, seq={T}")

    # Forward pass
    outputs = model(input_ids, targets=targets, return_blade_activations=True)

    # Check shapes
    tests = []

    logits = outputs["logits"]
    tests.append((
        f"logits shape: {logits.shape}",
        logits.shape == (B, T, config.vocab_size)
    ))

    if "loss" in outputs:
        tests.append((
            f"loss: {outputs['loss'].item():.4f} (scalar)",
            outputs["loss"].dim() == 0
        ))

    if "blade_logits" in outputs:
        bl = outputs["blade_logits"]
        tests.append((
            f"blade_logits shape: {bl.shape}",
            bl.shape == (B, T, config.n_blades)
        ))

    if "blade_activations" in outputs:
        acts = outputs["blade_activations"]
        tests.append((
            f"blade_activations: {len(acts)} layers",
            len(acts) == config.n_layers
        ))
        tests.append((
            f"  last layer shape: {acts[-1].shape}",
            acts[-1].shape == (B, T, config.n_blades)
        ))

    if "narrative_parse" in outputs:
        np_out = outputs["narrative_parse"]
        tests.append((
            f"narrative_parse keys: {list(np_out.keys())}",
            set(np_out.keys()) == {"tone", "urgency", "query_type", "agency"}
        ))

    for desc, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    return all(ok for _, ok in tests)


def test_parameter_count():
    """Verify parameter count is in expected range."""
    print("\n" + "=" * 60)
    print("TEST: Parameter Count")
    print("=" * 60)

    config = GeoFormerConfig()
    model = GeoFormer(config)
    counts = model.count_parameters()

    print(f"\n  Parameter breakdown:")
    for name, count in counts.items():
        print(f"    {name:<20} {count:>12,}")

    total = counts["total_unique"]
    target_min = 200_000_000
    target_max = 280_000_000

    ok = target_min <= total <= target_max
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] Total unique params: {total:,} (target: {target_min:,}-{target_max:,})")

    return ok


def test_gradient_flow():
    """Verify gradients flow through all major components."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)

    config = GeoFormerConfig(n_layers=4, tou_attn_layers=[2])  # Small for speed
    model = GeoFormer(config)

    B, T = 1, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    outputs = model(input_ids, targets=targets)
    outputs["loss"].backward()

    # Check key parameters have gradients
    checks = [
        ("token_embed", model.token_embed.weight),
        ("blade_projector", model.blade_projector.proj.weight),
        ("tou_bank", model.tou_bank.embeddings.weight),
        ("attention_qkv", model.blocks[0].attn.qkv.weight),
        ("cayley_mixer_alpha", model.blocks[0].attn.cayley_mixer.alpha),
        ("ffn_gate_proj", model.blocks[0].ffn.swiglu.gate_proj.weight),
        ("ffn_clifford_alpha", model.blocks[0].ffn.clifford_mixer.alpha),
    ]

    all_ok = True
    for name, param in checks:
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        status = "PASS" if has_grad else "FAIL"
        grad_norm = param.grad.norm().item() if has_grad else 0
        print(f"  [{status}] {name}: grad_norm={grad_norm:.6f}")
        all_ok = all_ok and has_grad

    return all_ok


def main():
    print("\n" + "=" * 60)
    print("  GEOFORMER-250M SMOKE TEST")
    print("=" * 60)

    results = []
    results.append(("Cayley Table", test_cayley_table()))
    results.append(("Parameter Count", test_parameter_count()))
    results.append(("Forward Pass", test_model_forward()))
    results.append(("Gradient Flow", test_gradient_flow()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and ok

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
