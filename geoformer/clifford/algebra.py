"""Cl(3,0) Clifford algebra — Cayley table and structural constants.

Ported from modal_app/src/training_src/clifford_layer.py (the corrected version).
The Cayley table is derived from first principles, not hardcoded.

Three generators: e1 (World/Causation), e2 (Self/Affect), e3 (Meaning/Wisdom)
Eight basis elements spanning grades 0-3:
  Grade 0: 1        -> narrative (scalar, always present)
  Grade 1: e1       -> causation
           e2       -> affect
           e3       -> wisdom
  Grade 2: e12      -> relations (causation x affect)
           e13      -> ecology  (causation x wisdom)
           e23      -> epistemics (affect x wisdom)
  Grade 3: e123     -> temporal (pseudoscalar, irreversibility)
"""

import torch
from typing import List, Tuple

BLADE_NAMES = [
    "narrative",   # 0: scalar (grade 0)
    "causation",   # 1: e1 (grade 1)
    "affect",      # 2: e2 (grade 1)
    "wisdom",      # 3: e3 (grade 1)
    "relations",   # 4: e12 (grade 2)
    "ecology",     # 5: e13 (grade 2)
    "epistemics",  # 6: e23 (grade 2)
    "temporal",    # 7: e123 (grade 3)
]

BLADE_INDEX = {name: i for i, name in enumerate(BLADE_NAMES)}
BLADE_GRADES = [0, 1, 1, 1, 2, 2, 2, 3]

# Generator tuples for each basis element
_BASIS_GENS = [
    (),        # 0: scalar
    (1,),      # 1: e1
    (2,),      # 2: e2
    (3,),      # 3: e3
    (1, 2),    # 4: e12
    (1, 3),    # 5: e13
    (2, 3),    # 6: e23
    (1, 2, 3), # 7: e123
]

_GENS_TO_IDX = {gens: i for i, gens in enumerate(_BASIS_GENS)}


def _reduce_product(a_gens: tuple, b_gens: tuple) -> Tuple[int, int]:
    """Compute basis element product via bubble sort with anticommutation."""
    gens = list(a_gens) + list(b_gens)
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
    return _GENS_TO_IDX[tuple(gens)], sign


def _derive_cayley_table() -> List[List[Tuple[int, int]]]:
    """Derive full 8x8 Cayley table from Cl(3,0) axioms."""
    table = []
    for i in range(8):
        row = []
        for j in range(8):
            idx, s = _reduce_product(_BASIS_GENS[i], _BASIS_GENS[j])
            row.append((idx, s))
        table.append(row)
    return table


# The definitive Cayley table
CAYLEY_TABLE = _derive_cayley_table()


def cayley_sign_tensor(device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build sparse sign tensor: signs[i, j, k] = sign if table[i][j] -> (k, sign), else 0.

    Shape: [8, 8, 8]. Used for vectorized geometric product.
    """
    signs = torch.zeros(8, 8, 8, device=device, dtype=dtype)
    for i in range(8):
        for j in range(8):
            k, s = CAYLEY_TABLE[i][j]
            signs[i, j, k] = s
    return signs


def cayley_nonzero_entries() -> List[Tuple[int, int, int, int]]:
    """Return list of (i, j, k, sign) for all nonzero Cayley products.

    Useful for sparse Clifford mixing in FFN layers.
    Only includes off-diagonal (i != j) entries where the product
    maps to a different blade (k != i and k != j).
    """
    entries = []
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            k, sign = CAYLEY_TABLE[i][j]
            if k != i and k != j:
                entries.append((i, j, k, sign))
    return entries
