"""Benchmarks: compare optimized kernels vs PyTorch baseline.

Measures throughput for:
1. Geometric product (scatter_add vs scatter-free)
2. Geometric round (full: geo + FFN)
3. MoE layer (loop dispatch vs sorted dispatch)
4. Full forward pass

Run on GPU for meaningful results.
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def bench(fn, warmup=5, iters=50, label=""):
    """Benchmark a function. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {label:<45} {median:>8.2f} ms (median of {iters})")
    return median


def bench_geometric_product():
    """Compare scatter_add vs scatter-free geometric product."""
    print("=" * 60)
    print("BENCHMARK: Geometric Product")
    print("=" * 60)

    from geoformer.triton_kernels.cayley_constants import (
        get_cayley_tensors, get_target_accumulation_table
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_tok, N_blades, D = 2048, 8, 128
    x = torch.randn(N_tok, N_blades, D, device=device)
    weights = torch.rand(64, device=device).sigmoid()

    # Setup scatter_add version
    ci, cj, ck, cs = get_cayley_tensors(device)
    tk_exp = ck.long().unsqueeze(0).unsqueeze(-1).expand(N_tok, 64, D)

    def scatter_version():
        xi = x[:, ci.long(), :]
        xj = x[:, cj.long(), :]
        products = xi * xj * (cs * weights).unsqueeze(0).unsqueeze(-1)
        geo = torch.zeros_like(x)
        geo.scatter_add_(1, tk_exp, products)
        return geo

    # Setup scatter-free version
    src_ij, src_signs = get_target_accumulation_table(device)

    def direct_version():
        geo = torch.zeros_like(x)
        for k in range(8):
            for p in range(8):
                i = src_ij[k, p, 0].long()
                j = src_ij[k, p, 1].long()
                sign = src_signs[k, p]
                w = weights[i * 8 + j]
                geo[:, k, :] += sign * w * x[:, i, :] * x[:, j, :]
        return geo

    t_scatter = bench(scatter_version, label="scatter_add (baseline)")
    t_direct = bench(direct_version, label="scatter-free (optimized)")

    print(f"\n  Speedup: {t_scatter/t_direct:.2f}x")
    return t_scatter, t_direct


def bench_geometric_round():
    """Compare original GeometricRound vs FusedGeometricRound."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Geometric Round (geo + FFN)")
    print("=" * 60)

    from geoformer.triton_kernels.expert_gemm import FusedGeometricRound

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_tok, N_blades, D, D_ffn = 2048, 8, 128, 640

    geo_fused = FusedGeometricRound(N_blades, D, D_ffn).to(device)
    x = torch.randn(N_tok, N_blades, D, device=device)

    def fused_version():
        return geo_fused(x)

    t = bench(fused_version, label="FusedGeometricRound")
    print(f"\n  Throughput: {N_tok / (t / 1000):,.0f} tokens/sec per round")
    return t


def bench_moe_layer():
    """Compare original MoELayer vs FastMoELayer."""
    print("\n" + "=" * 60)
    print("BENCHMARK: MoE Layer (full dispatch + experts)")
    print("=" * 60)

    from geoformer.moe_hlm.config import MoEHLMConfig
    from geoformer.triton_kernels.expert_gemm import FastMoELayer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MoEHLMConfig()
    config.n_layers = 1
    config.n_geometric_rounds = 2
    config.vocab_size = 1000

    fast_moe = FastMoELayer(config).to(device)

    B, T = 2, 1024
    x = torch.randn(B, T, config.d_model, device=device)

    def fast_version():
        return fast_moe(x)

    t = bench(fast_version, label="FastMoELayer (sorted dispatch)")
    tokens = B * T
    print(f"\n  Throughput: {tokens / (t / 1000):,.0f} tokens/sec per MoE layer")
    return t


def bench_full_comparison():
    """Compare original vs optimized on a mini model."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Model Forward Pass")
    print("=" * 60)

    from geoformer.moe_hlm.config import MoEHLMConfig
    from geoformer.triton_kernels.moe_dispatch import FastMoEHLMBlock
    # Try to import original for comparison
    try:
        from geoformer.moe_hlm.kaggle_train import MoEHLMBlock as OriginalBlock
        has_original = True
    except ImportError:
        has_original = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MoEHLMConfig()
    config.n_geometric_rounds = 2
    config.vocab_size = 1000

    fast_block = FastMoEHLMBlock(config, layer_idx=3).to(device)

    B, T = 2, 1024
    x = torch.randn(B, T, config.d_model, device=device)

    def fast_forward():
        return fast_block(x)

    t_fast = bench(fast_forward, label="FastMoEHLMBlock (1 layer)")

    tokens = B * T
    estimated_full = t_fast * config.n_layers
    estimated_tps = tokens / (estimated_full / 1000)
    print(f"\n  Single layer: {t_fast:.2f} ms")
    print(f"  Estimated {config.n_layers}-layer forward: {estimated_full:.0f} ms")
    print(f"  Estimated throughput: {estimated_tps:,.0f} tokens/sec")

    return t_fast


def main():
    print("\n" + "=" * 60)
    print("  HLM KERNEL BENCHMARKS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    bench_geometric_product()
    bench_geometric_round()
    bench_moe_layer()
    bench_full_comparison()


if __name__ == "__main__":
    main()
