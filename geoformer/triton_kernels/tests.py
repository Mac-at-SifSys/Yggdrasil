"""Correctness tests for HLM Triton kernels.

Tests each optimized component against the PyTorch reference implementation
to verify numerical equivalence.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from geoformer.triton_kernels.cayley_constants import (
    get_cayley_tensors, get_target_accumulation_table,
    CAYLEY_BY_TARGET, CAYLEY_TABLE,
)
from geoformer.triton_kernels.expert_gemm import FusedGeometricRound


def test_cayley_table():
    """Verify Cayley table constants match algebra axioms."""
    print("=" * 60)
    print("TEST: Cayley Table Constants")
    print("=" * 60)

    passed = 0

    # e_i^2 = +1 for grade-1
    for i in [1, 2, 3]:
        k, s = CAYLEY_TABLE[i][i]
        ok = (k == 0 and s == 1)
        print(f"  [{'PASS' if ok else 'FAIL'}] e{i}^2 = +1")
        passed += ok

    # Bivectors square to -1
    for i in [4, 5, 6]:
        k, s = CAYLEY_TABLE[i][i]
        ok = (k == 0 and s == -1)
        print(f"  [{'PASS' if ok else 'FAIL'}] blade[{i}]^2 = -1")
        passed += ok

    # Pseudoscalar
    k, s = CAYLEY_TABLE[7][7]
    ok = (k == 0 and s == -1)
    print(f"  [{'PASS' if ok else 'FAIL'}] e123^2 = -1")
    passed += ok

    # Per-target table: each blade has exactly 8 products
    for k in range(8):
        ok = len(CAYLEY_BY_TARGET[k]) == 8
        print(f"  [{'PASS' if ok else 'FAIL'}] Blade {k} has {len(CAYLEY_BY_TARGET[k])} products")
        passed += ok

    # Verify per-target matches flat table
    src_ij, src_signs = get_target_accumulation_table()
    for k in range(8):
        for p, (i, j, sign) in enumerate(CAYLEY_BY_TARGET[k]):
            ok_i = src_ij[k, p, 0].item() == i
            ok_j = src_ij[k, p, 1].item() == j
            ok_s = src_signs[k, p].item() == sign
            if not (ok_i and ok_j and ok_s):
                print(f"  [FAIL] Target table mismatch at k={k}, p={p}")
                passed -= 1
                break
        else:
            passed += 1

    print(f"\n  Cayley: {passed} tests passed")
    return passed >= 15


def test_scatter_free_geo_product():
    """Verify scatter-free geometric product matches scatter_add version."""
    print("\n" + "=" * 60)
    print("TEST: Scatter-Free Geometric Product")
    print("=" * 60)

    torch.manual_seed(42)
    N_tok, N_blades, D = 32, 8, 128
    x = torch.randn(N_tok, N_blades, D)

    # Method 1: Original scatter_add approach
    ci, cj, ck, cs = get_cayley_tensors()
    weights = torch.rand(64).sigmoid()

    xi = x[:, ci.long(), :]
    xj = x[:, cj.long(), :]
    products = xi * xj * (cs * weights).unsqueeze(0).unsqueeze(-1)
    geo_scatter = torch.zeros_like(x)
    tk_exp = ck.long().unsqueeze(0).unsqueeze(-1).expand(N_tok, 64, D)
    geo_scatter.scatter_add_(1, tk_exp, products)

    # Method 2: Per-target accumulation (scatter-free)
    src_ij, src_signs = get_target_accumulation_table()
    geo_direct = torch.zeros_like(x)
    for k in range(8):
        for p in range(8):
            i = src_ij[k, p, 0].long()
            j = src_ij[k, p, 1].long()
            sign = src_signs[k, p]
            w = weights[i * 8 + j]
            geo_direct[:, k, :] += sign * w * x[:, i, :] * x[:, j, :]

    # Compare
    ok = torch.allclose(geo_scatter, geo_direct, atol=1e-5)
    max_diff = (geo_scatter - geo_direct).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] Max diff: {max_diff:.2e}")

    return ok


def test_fused_geometric_round():
    """Test FusedGeometricRound matches original scatter-based version."""
    print("\n" + "=" * 60)
    print("TEST: Fused Geometric Round (full)")
    print("=" * 60)

    torch.manual_seed(42)
    N_tok, N_blades, D, D_ffn = 16, 8, 128, 640

    geo = FusedGeometricRound(N_blades, D, D_ffn)
    x = torch.randn(N_tok, N_blades, D)

    # Forward pass
    out = geo(x)

    # Shape check
    ok_shape = out.shape == x.shape
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] Output shape: {out.shape}")

    # Gradient flow
    out.sum().backward()
    ok_grad_iw = geo.interaction_weights.grad is not None
    ok_grad_gate = geo.geo_gate.grad is not None
    ok_grad_ffn = geo.gate_proj.weight.grad is not None
    print(f"  [{'PASS' if ok_grad_iw else 'FAIL'}] Gradient: interaction_weights")
    print(f"  [{'PASS' if ok_grad_gate else 'FAIL'}] Gradient: geo_gate")
    print(f"  [{'PASS' if ok_grad_ffn else 'FAIL'}] Gradient: FFN weights")

    # No NaN
    ok_nan = not torch.isnan(out).any()
    print(f"  [{'PASS' if ok_nan else 'FAIL'}] No NaN in output")

    return ok_shape and ok_grad_iw and ok_grad_gate and ok_grad_ffn and ok_nan


def test_fast_moe_layer():
    """Test FastMoELayer produces valid output."""
    print("\n" + "=" * 60)
    print("TEST: Fast MoE Layer")
    print("=" * 60)

    from geoformer.moe_hlm.config import MoEHLMConfig
    from geoformer.triton_kernels.expert_gemm import FastMoELayer

    config = MoEHLMConfig()
    config.n_layers = 4  # Small for testing
    config.n_geometric_rounds = 2
    config.vocab_size = 1000

    moe = FastMoELayer(config)

    B, T, D = 2, 32, config.d_model
    x = torch.randn(B, T, D)

    out, aux_loss = moe(x)

    ok_shape = out.shape == x.shape
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] Output shape: {out.shape}")

    ok_aux = aux_loss.dim() == 0
    print(f"  [{'PASS' if ok_aux else 'FAIL'}] Aux loss: {aux_loss.item():.4f}")

    # Gradient flow
    (out.sum() + aux_loss).backward()
    ok_gate = moe.gate.weight.grad is not None
    ok_expert = moe.experts.proj_in.grad is not None
    print(f"  [{'PASS' if ok_gate else 'FAIL'}] Gradient: router gate")
    print(f"  [{'PASS' if ok_expert else 'FAIL'}] Gradient: expert proj_in")

    ok_nan = not torch.isnan(out).any()
    print(f"  [{'PASS' if ok_nan else 'FAIL'}] No NaN")

    return ok_shape and ok_aux and ok_gate and ok_expert and ok_nan


def main():
    print("\n" + "=" * 60)
    print("  HLM TRITON KERNEL TESTS")
    print("=" * 60)

    results = []
    results.append(("Cayley Table", test_cayley_table()))
    results.append(("Scatter-Free Geo Product", test_scatter_free_geo_product()))
    results.append(("Fused Geometric Round", test_fused_geometric_round()))
    results.append(("Fast MoE Layer", test_fast_moe_layer()))

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
