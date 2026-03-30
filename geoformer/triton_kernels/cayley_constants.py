"""Cayley table constants for Triton kernels.

The Cl(3,0) Cayley table defines all 64 basis element products.
For Triton kernels, we need these as flat arrays that can be loaded
as compile-time constants or passed as kernel arguments.

Format: For each product index p in [0, 64):
  CAYLEY_I[p] = source blade i
  CAYLEY_J[p] = source blade j
  CAYLEY_K[p] = target blade k
  CAYLEY_SIGN[p] = sign (+1 or -1)

Where: basis[i] * basis[j] = CAYLEY_SIGN[p] * basis[k]
"""

import torch
from typing import Tuple


def _derive_cayley_table():
    """Derive Cl(3,0) Cayley table from first principles."""
    basis_gens = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    gens_to_idx = {g: i for i, g in enumerate(basis_gens)}

    def reduce(a, b):
        gens = list(a) + list(b)
        sign = 1
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(gens) - 1:
                if gens[i] == gens[i + 1]:
                    gens.pop(i)
                    gens.pop(i)
                    changed = True
                    if i > 0:
                        i -= 1
                elif gens[i] > gens[i + 1]:
                    gens[i], gens[i + 1] = gens[i + 1], gens[i]
                    sign *= -1
                    changed = True
                    i += 1
                else:
                    i += 1
        return gens_to_idx[tuple(gens)], sign

    table = []
    for i in range(8):
        row = []
        for j in range(8):
            idx, s = reduce(basis_gens[i], basis_gens[j])
            row.append((idx, s))
        table.append(row)
    return table


CAYLEY_TABLE = _derive_cayley_table()

# Flat arrays for kernel use — 64 entries, one per (i, j) pair
CAYLEY_I = []     # Source blade i
CAYLEY_J = []     # Source blade j
CAYLEY_K = []     # Target blade k
CAYLEY_SIGN = []  # Sign of product

for i in range(8):
    for j in range(8):
        k, s = CAYLEY_TABLE[i][j]
        CAYLEY_I.append(i)
        CAYLEY_J.append(j)
        CAYLEY_K.append(k)
        CAYLEY_SIGN.append(s)

# Also build per-target accumulation lists:
# For each target blade k, which (i, j, sign) products contribute to it?
# This is more efficient for the scatter-free accumulation pattern.
CAYLEY_BY_TARGET = {}  # k -> list of (i, j, sign)
for p in range(64):
    k = CAYLEY_K[p]
    if k not in CAYLEY_BY_TARGET:
        CAYLEY_BY_TARGET[k] = []
    CAYLEY_BY_TARGET[k].append((CAYLEY_I[p], CAYLEY_J[p], CAYLEY_SIGN[p]))

# Verify: each target blade should have exactly 8 contributing products
for k in range(8):
    assert len(CAYLEY_BY_TARGET[k]) == 8, f"Blade {k} has {len(CAYLEY_BY_TARGET[k])} products, expected 8"


def get_cayley_tensors(device: torch.device = None) -> Tuple[torch.Tensor, ...]:
    """Get Cayley indices as tensors for kernel arguments.

    Returns:
        cayley_i: (64,) int32 — source blade i
        cayley_j: (64,) int32 — source blade j
        cayley_k: (64,) int32 — target blade k
        cayley_sign: (64,) float32 — product sign
    """
    return (
        torch.tensor(CAYLEY_I, dtype=torch.int32, device=device),
        torch.tensor(CAYLEY_J, dtype=torch.int32, device=device),
        torch.tensor(CAYLEY_K, dtype=torch.int32, device=device),
        torch.tensor(CAYLEY_SIGN, dtype=torch.float32, device=device),
    )


def get_target_accumulation_table(device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get per-target accumulation indices.

    For each target blade k, provides the 8 (i, j, sign) triples.
    Shape: (8, 8, 3) where dim 0 = target blade, dim 1 = contributing product,
    dim 2 = (source_i, source_j, sign).

    This eliminates scatter_add entirely — each output blade directly
    accumulates its 8 contributing products.

    Returns:
        src_indices: (8, 8, 2) int32 — (source_i, source_j) for each target blade
        src_signs: (8, 8) float32 — signs for each product
    """
    src_indices = torch.zeros(8, 8, 2, dtype=torch.int32, device=device)
    src_signs = torch.zeros(8, 8, dtype=torch.float32, device=device)

    for k in range(8):
        for p_idx, (i, j, sign) in enumerate(CAYLEY_BY_TARGET[k]):
            src_indices[k, p_idx, 0] = i
            src_indices[k, p_idx, 1] = j
            src_signs[k, p_idx] = sign

    return src_indices, src_signs


# Print table for verification
def print_cayley_table():
    """Print human-readable Cayley table."""
    labels = ["1", "e1", "e2", "e3", "e12", "e13", "e23", "e123"]
    print(f"{'':>6}", end="")
    for l in labels:
        print(f"{l:>7}", end="")
    print()
    for i in range(8):
        print(f"{labels[i]:>6}", end="")
        for j in range(8):
            k, sign = CAYLEY_TABLE[i][j]
            s = "+" if sign > 0 else "-"
            print(f"  {s}{labels[k]:>4}", end="")
        print()

    print(f"\nPer-target accumulation (scatter-free):")
    for k in range(8):
        products = CAYLEY_BY_TARGET[k]
        terms = []
        for i, j, sign in products:
            s = "+" if sign > 0 else "-"
            terms.append(f"{s}{labels[i]}*{labels[j]}")
        print(f"  {labels[k]:>5} = {' '.join(terms)}")


if __name__ == "__main__":
    print_cayley_table()
