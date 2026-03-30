"""Test fused expert: correctness + CPU benchmark vs original."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from geoformer.moe_hlm.config import MoEHLMConfig
from geoformer.triton_kernels.fused_expert import FusedMoELayer


class Timer:
    def __init__(self):
        self.times = {}
    @contextmanager
    def track(self, name):
        t0 = time.perf_counter()
        yield
        self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0) * 1000


def test_fused_moe_correctness():
    """Test that FusedMoELayer produces valid output."""
    print("=" * 60)
    print("TEST: FusedMoELayer Correctness")
    print("=" * 60)

    config = MoEHLMConfig()
    config.n_layers = 1
    config.n_geometric_rounds = 2
    config.vocab_size = 1000

    moe = FusedMoELayer(config)
    B, T = 2, 64
    x = torch.randn(B, T, config.d_model)

    # Forward
    out, aux = moe(x)
    tests = []

    tests.append(("shape", out.shape == x.shape))
    tests.append(("no NaN", not torch.isnan(out).any()))
    tests.append(("aux scalar", aux.dim() == 0))
    tests.append(("aux positive", aux.item() > 0))

    # Gradient flow
    (out.sum() + aux).backward()
    tests.append(("grad: gate", moe.gate.weight.grad is not None))
    tests.append(("grad: proj_in", moe.experts.proj_in.grad is not None))
    tests.append(("grad: proj_out", moe.experts.proj_out.grad is not None))
    tests.append(("grad: iw", moe.experts.interaction_weights.grad is not None))
    tests.append(("grad: ffn_gate", moe.experts.ffn_gate.grad is not None))
    tests.append(("grad: geo_gates", moe.experts.geo_gates.grad is not None))

    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    return all(ok for _, ok in tests)


def test_bmm_vs_loop():
    """Benchmark: bmm proj vs sequential loop."""
    print("\n" + "=" * 60)
    print("BENCHMARK: bmm proj_in vs sequential loop (CPU)")
    print("=" * 60)

    n_experts, d_model, expert_dim = 8, 1536, 1024
    max_tokens = 512  # per expert

    W = torch.randn(n_experts, d_model, expert_dim)
    x_batch = torch.randn(n_experts, max_tokens, d_model)

    # Method 1: Sequential loop
    def loop_version():
        results = []
        for e in range(n_experts):
            results.append(x_batch[e] @ W[e])
        return torch.stack(results)

    # Method 2: Single bmm
    def bmm_version():
        return torch.bmm(x_batch, W)

    # Warmup
    for _ in range(3):
        loop_version(); bmm_version()

    # Benchmark
    N_ITER = 20
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        loop_version()
    t_loop = (time.perf_counter() - t0) / N_ITER * 1000

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        bmm_version()
    t_bmm = (time.perf_counter() - t0) / N_ITER * 1000

    # Verify same result
    r_loop = loop_version()
    r_bmm = bmm_version()
    match = torch.allclose(r_loop, r_bmm, atol=1e-5)

    print(f"  Loop (8 matmuls):  {t_loop:.1f} ms")
    print(f"  bmm  (1 matmul):   {t_bmm:.1f} ms")
    print(f"  Speedup:           {t_loop/t_bmm:.2f}x")
    print(f"  Results match:     {match}")

    return match


def test_full_moe_benchmark():
    """Full MoE layer benchmark on CPU."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full FusedMoELayer (CPU)")
    print("=" * 60)

    config = MoEHLMConfig()
    config.n_layers = 1
    config.n_geometric_rounds = 2

    moe = FusedMoELayer(config)
    moe.eval()

    B, T = 2, 256
    x = torch.randn(B, T, config.d_model)

    # Warmup
    for _ in range(2):
        moe(x)

    # Benchmark
    N_ITER = 10
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        with torch.no_grad():
            moe(x)
    t_fused = (time.perf_counter() - t0) / N_ITER * 1000

    tokens = B * T
    print(f"  FusedMoELayer:     {t_fused:.1f} ms")
    print(f"  Throughput:        {tokens/(t_fused/1000):,.0f} tok/s per layer (CPU)")

    return True


def main():
    print("\n" + "=" * 60)
    print("  FUSED EXPERT TESTS")
    print("=" * 60)

    results = []
    results.append(("Correctness", test_fused_moe_correctness()))
    results.append(("bmm vs loop", test_bmm_vs_loop()))
    results.append(("Full MoE bench", test_full_moe_benchmark()))

    print("\n" + "=" * 60)
    all_pass = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES'}")


if __name__ == "__main__":
    main()
