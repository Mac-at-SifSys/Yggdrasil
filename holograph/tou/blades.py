"""
blades.py — 9 ToU blades with grade assignments.

Each blade is a processing pathway that focuses on specific grades
of the Clifford algebra. Blades act as projection + transformation
pipelines that specialize in different aspects of multivector content.

The 9 blades correspond to the meaningful grade combinations in Cl(3,0):
1. Scalar blade     — grade 0 only (magnitude/intensity)
2. Vector blade     — grade 1 only (direction/position)
3. Bivector blade   — grade 2 only (rotation/area)
4. Trivector blade  — grade 3 only (volume/orientation)
5. Even blade       — grades 0+2 (rotors/spinors)
6. Odd blade        — grades 1+3 (vectors+pseudovectors)
7. Full blade       — all grades (general multivector)
8. Motor blade      — grades 0+1+2 (motors for rigid motion)
9. Dual blade       — grades 1+2+3 (everything except scalar)
"""

import numpy as np
from rune.backend import xp
from rune.types.multivector import GRADE_SLICES


# Blade definitions: name -> set of active grades
BLADE_DEFINITIONS = {
    'scalar':    {0},
    'vector':    {1},
    'bivector':  {2},
    'trivector': {3},
    'even':      {0, 2},
    'odd':       {1, 3},
    'full':      {0, 1, 2, 3},
    'motor':     {0, 1, 2},
    'dual':      {1, 2, 3},
}

N_BLADES = 9


class Blade:
    """
    A single blade: a grade-projected processing pathway.

    Attributes:
        name: blade identifier
        grades: set of active grades
        mask: (8,) binary mask for active components
        projection_matrix: (8, 8) diagonal projection
    """

    def __init__(self, name: str, grades: set):
        self.name = name
        self.grades = grades

        # Build component mask
        self.mask = xp.zeros(8, dtype=xp.float32)
        for g in grades:
            slc = GRADE_SLICES[g]
            self.mask[slc] = 1.0

        # Active component count
        self.n_active = int(self.mask.sum())

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project multivector data to this blade's active grades.
        x: (..., 8) -> (..., 8) with inactive grades zeroed.
        """
        return x * self.mask

    def __repr__(self):
        return f"Blade('{self.name}', grades={self.grades}, n_active={self.n_active})"


class ToUBlades:
    """
    Collection of 9 ToU blades.

    Provides methods to route multivectors through blades
    and recombine the results.
    """

    def __init__(self):
        self.blades = []
        self.blade_names = []
        self.blade_map = {}

        for name, grades in BLADE_DEFINITIONS.items():
            blade = Blade(name, grades)
            self.blades.append(blade)
            self.blade_names.append(name)
            self.blade_map[name] = blade

    def __getitem__(self, key) -> Blade:
        if isinstance(key, int):
            return self.blades[key]
        return self.blade_map[key]

    def __len__(self):
        return N_BLADES

    def project_all(self, x: np.ndarray) -> list:
        """
        Project input through all 9 blades.
        Returns list of 9 projected arrays, each (..., 8).
        """
        return [blade.project(x) for blade in self.blades]

    def recombine(self, blade_outputs: list, weights: np.ndarray) -> np.ndarray:
        """
        Weighted recombination of blade outputs.

        Args:
            blade_outputs: list of 9 arrays, each (..., 8)
            weights: (..., 9) — per-blade weights
        Returns:
            (..., 8) — weighted sum of blade outputs
        """
        result = xp.zeros_like(blade_outputs[0])
        for i, output in enumerate(blade_outputs):
            w = weights[..., i:i + 1]  # (..., 1)
            result = result + output * w
        return result

    def __repr__(self):
        return f"ToUBlades(n_blades={N_BLADES})"
