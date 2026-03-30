"""
geometric_fold.py — Tree-reduction via geometric product.

Reduces d_model multivectors to a single summary multivector by
pairwise GP reduction: d_model → d_model/2 → ... → 1.

The grade structure of the result encodes the relational structure
of the entire input sequence.
"""

import numpy as np
from rune.backend import xp
from rune.ops.batched import batched_geom_prod, batched_normalize


def geometric_fold(x: np.ndarray) -> np.ndarray:
    """
    Reduce a sequence of multivectors to one via pairwise GP tree reduction.

    Args:
        x: (d_model, 8) or (N, 8) — sequence of multivectors
    Returns:
        (8,) — single summary multivector (unit norm)
    """
    current = x.copy()

    while current.shape[0] > 1:
        n = current.shape[0]
        if n % 2 == 1:
            # Odd count: keep the last one aside, fold the rest
            last = current[-1:]
            current = current[:-1]
            n -= 1

        # Pairwise GP: elements [0,1], [2,3], [4,5], ...
        a = current[0::2]  # even indices: (n/2, 8)
        b = current[1::2]  # odd indices:  (n/2, 8)
        folded = batched_geom_prod(a, b)  # (n/2, 8)

        if n != current.shape[0]:
            # Was odd — append the leftover
            current = xp.concatenate([folded, last], axis=0)
        else:
            current = folded

    # Normalize to unit multivector
    result = current[0]  # (8,)
    norm = xp.sqrt(xp.sum(result ** 2) + 1e-12)
    return result / norm
