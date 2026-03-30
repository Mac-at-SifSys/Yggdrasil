"""Clifford algebra Cl(3,0) for GeoFormer."""

from geoformer.clifford.algebra import (
    BLADE_NAMES,
    BLADE_GRADES,
    CAYLEY_TABLE,
    cayley_sign_tensor,
    cayley_nonzero_entries,
)
from geoformer.clifford.ops import geometric_product, geometric_product_fast
