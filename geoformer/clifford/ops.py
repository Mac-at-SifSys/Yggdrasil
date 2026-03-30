"""Differentiable Clifford algebra operations for PyTorch.

These are the core mathematical operations used throughout GeoFormer.
The Cayley table is fixed (math, not learned) but these operations
compose with learned parameters via autograd.
"""

import torch
import torch.nn as nn

from geoformer.clifford.algebra import CAYLEY_TABLE, cayley_sign_tensor


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Cl(3,0) geometric product of two multivectors.

    Args:
        a: (..., 8) multivector components
        b: (..., 8) multivector components

    Returns:
        (..., 8) geometric product a * b
    """
    components_a = a.unbind(dim=-1)
    components_b = b.unbind(dim=-1)
    result = [torch.zeros_like(components_a[0]) for _ in range(8)]

    for i in range(8):
        for j in range(8):
            k, sign = CAYLEY_TABLE[i][j]
            result[k] = result[k] + sign * components_a[i] * components_b[j]

    return torch.stack(result, dim=-1)


def geometric_product_fast(a: torch.Tensor, b: torch.Tensor,
                           _cache: dict = {}) -> torch.Tensor:
    """Optimized geometric product using precomputed sign tensor.

    Same result as geometric_product() but uses einsum for GPU efficiency.

    Args:
        a: (..., 8) multivector components
        b: (..., 8) multivector components

    Returns:
        (..., 8) geometric product a * b
    """
    key = (a.device, a.dtype)
    if key not in _cache:
        _cache[key] = cayley_sign_tensor(device=a.device, dtype=a.dtype)
    signs = _cache[key]

    outer = a.unsqueeze(-1) * b.unsqueeze(-2)  # (..., 8, 8)
    return torch.einsum("...ij,ijk->...k", outer, signs)


class GeometricProductLayer(nn.Module):
    """Differentiable geometric product with learned reference multivectors.

    Computes: output = mix * input + (1-mix) * geom_product(input, ref)
    where ref and mix are learned per-blade.
    """

    def __init__(self, n_refs: int = 2):
        super().__init__()
        self.refs = nn.Parameter(torch.randn(n_refs, 8) * 0.1)
        self.mix = nn.Parameter(torch.ones(8) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (..., 8) multivector. Returns: (..., 8) mixed."""
        mix = self.mix.sigmoid()
        result = mix * x
        for i in range(self.refs.shape[0]):
            ref = self.refs[i].unsqueeze(0).expand_as(x)
            prod = geometric_product_fast(x, ref)
            result = result + (1.0 - mix) / self.refs.shape[0] * prod
        return result
