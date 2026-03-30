"""
primitives.py — 1,486 ToU primitives as Cl(3,0) multivectors.

The Tensor of Understanding defines 1,486 cognitive primitives organized
into semantic categories. Each primitive is a multivector whose grade
structure encodes its semantic type:

- Scalar-dominant: quantity/magnitude primitives
- Vector-dominant: directional/relational primitives
- Bivector-dominant: transformational/rotational primitives
- Mixed: complex compositional primitives

The primitives are initialized with structure reflecting their category,
then made learnable during training.
"""

import numpy as np
from rune.backend import xp


# Primitive category definitions with counts
PRIMITIVE_CATEGORIES = {
    'logical': 64,        # AND, OR, NOT, XOR, IMPLIES, etc.
    'arithmetic': 48,     # ADD, SUB, MUL, DIV, MOD, etc.
    'comparison': 32,     # EQ, NEQ, LT, GT, etc.
    'spatial': 96,        # UP, DOWN, LEFT, RIGHT, NEAR, FAR, etc.
    'temporal': 80,       # BEFORE, AFTER, DURING, SINCE, etc.
    'causal': 64,         # CAUSE, EFFECT, ENABLE, PREVENT, etc.
    'modal': 48,          # POSSIBLE, NECESSARY, PROBABLE, etc.
    'quantifier': 32,     # ALL, SOME, NONE, MOST, FEW, etc.
    'entity': 128,        # PERSON, PLACE, THING, CONCEPT, etc.
    'action': 192,        # MOVE, THINK, SAY, MAKE, GIVE, etc.
    'property': 160,      # BIG, SMALL, RED, FAST, OLD, etc.
    'relation': 144,      # PART_OF, MEMBER_OF, INSTANCE_OF, etc.
    'emotion': 80,        # HAPPY, SAD, ANGRY, FEARFUL, etc.
    'cognition': 96,      # KNOW, BELIEVE, REMEMBER, IMAGINE, etc.
    'communication': 64,  # TELL, ASK, PROMISE, THREATEN, etc.
    'structure': 96,      # SEQUENCE, SET, MAP, TREE, GRAPH, etc.
    'meta': 62,           # ABSTRACT, COMPOSE, TRANSFORM, etc.
}
# Total: 1,486

# Grade profiles for each category
# (scalar_weight, vector_weight, bivector_weight, trivector_weight)
CATEGORY_GRADE_PROFILES = {
    'logical':        (0.8, 0.1, 0.05, 0.05),
    'arithmetic':     (0.7, 0.2, 0.05, 0.05),
    'comparison':     (0.6, 0.3, 0.05, 0.05),
    'spatial':        (0.1, 0.7, 0.15, 0.05),
    'temporal':       (0.2, 0.5, 0.2, 0.1),
    'causal':         (0.2, 0.3, 0.4, 0.1),
    'modal':          (0.5, 0.1, 0.3, 0.1),
    'quantifier':     (0.7, 0.1, 0.1, 0.1),
    'entity':         (0.3, 0.4, 0.2, 0.1),
    'action':         (0.1, 0.4, 0.4, 0.1),
    'property':       (0.4, 0.4, 0.1, 0.1),
    'relation':       (0.1, 0.3, 0.5, 0.1),
    'emotion':        (0.3, 0.3, 0.2, 0.2),
    'cognition':      (0.2, 0.2, 0.4, 0.2),
    'communication':  (0.2, 0.4, 0.3, 0.1),
    'structure':      (0.3, 0.2, 0.4, 0.1),
    'meta':           (0.1, 0.1, 0.3, 0.5),
}


class ToUPrimitives:
    """
    1,486 cognitive primitives as Cl(3,0) multivectors.

    Each primitive is an 8-component multivector initialized with
    a grade profile matching its semantic category.
    """

    def __init__(self, learnable: bool = True):
        self.n_primitives = sum(PRIMITIVE_CATEGORIES.values())
        assert self.n_primitives == 1486

        # All primitives: (1486, 8)
        self.data = xp.zeros((self.n_primitives, 8), dtype=xp.float32)
        self.learnable = learnable

        # Category index mapping
        self.category_ranges = {}
        self._category_labels = []

        self._initialize()

    def _initialize(self):
        """Initialize primitives with category-aware grade profiles."""
        idx = 0
        for cat_name, count in PRIMITIVE_CATEGORIES.items():
            start = idx
            profile = CATEGORY_GRADE_PROFILES[cat_name]

            for p in range(count):
                # Initialize with grade-weighted random values
                # Grade 0: scalar
                self.data[idx, 0] = xp.random.randn() * profile[0]
                # Grade 1: vector
                self.data[idx, 1:4] = xp.random.randn(3) * profile[1] / xp.sqrt(3)
                # Grade 2: bivector
                self.data[idx, 4:7] = xp.random.randn(3) * profile[2] / xp.sqrt(3)
                # Grade 3: trivector
                self.data[idx, 7] = xp.random.randn() * profile[3]

                self._category_labels.append(cat_name)
                idx += 1

            self.category_ranges[cat_name] = (start, idx)

        # Normalize all primitives to unit norm for stable routing
        norms = xp.sqrt(xp.sum(self.data ** 2, axis=-1, keepdims=True) + 1e-12)
        self.data = self.data / norms

    def get_category(self, name: str) -> np.ndarray:
        """Get all primitives for a category. Returns (count, 8)."""
        start, end = self.category_ranges[name]
        return self.data[start:end]

    def get_primitive(self, index: int) -> np.ndarray:
        """Get a single primitive by global index. Returns (8,)."""
        return self.data[index]

    def category_of(self, index: int) -> str:
        """Return the category name for a primitive index."""
        return self._category_labels[index]

    def parameters(self):
        if self.learnable:
            return [self.data]
        return []

    def __repr__(self):
        return f"ToUPrimitives(n={self.n_primitives}, categories={len(PRIMITIVE_CATEGORIES)})"
