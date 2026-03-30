"""ToU Knowledge Bank: loads primitives from bank_full.json into learnable embeddings.

The knowledge bank contains 1,486 primitives organized into 244 families
across 8 Clifford blades. Each primitive has a rich semantic definition.

At model init, we:
1. Load the bank structure (blade -> family -> primitive)
2. Create a learnable embedding for each primitive (d_blade dims)
3. Build blade-to-primitive assignment masks
4. Optionally initialize embeddings from text using a frozen encoder
"""

import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from geoformer.clifford.algebra import BLADE_NAMES, BLADE_INDEX


class ToUBank(nn.Module):
    """Learnable embedding bank for ToU primitives.

    Attributes:
        embeddings: nn.Embedding(n_primitives, d_blade) — learnable
        blade_masks: (n_blades, n_primitives) binary mask
        primitive_info: list of dicts with metadata per primitive
    """

    def __init__(self, n_primitives: int, d_blade: int, bank_path: Optional[str] = None):
        super().__init__()
        self.n_primitives = n_primitives
        self.d_blade = d_blade

        # Learnable primitive embeddings
        self.embeddings = nn.Embedding(n_primitives, d_blade)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        # Will be filled by load_bank or set_masks
        self.register_buffer(
            "blade_masks",
            torch.zeros(len(BLADE_NAMES), n_primitives)
        )

        self.primitive_info: List[dict] = []
        self.blade_to_indices: Dict[str, List[int]] = {name: [] for name in BLADE_NAMES}

        if bank_path is not None:
            self.load_bank(bank_path)
        else:
            # Default: distribute primitives evenly across blades
            self._distribute_evenly()

    def _distribute_evenly(self):
        """Distribute primitives evenly across blades when no bank file is provided."""
        n_blades = len(BLADE_NAMES)
        per_blade = self.n_primitives // n_blades
        remainder = self.n_primitives % n_blades

        masks = torch.zeros(n_blades, self.n_primitives)
        idx = 0
        for b in range(n_blades):
            count = per_blade + (1 if b < remainder else 0)
            masks[b, idx:idx + count] = 1.0
            self.blade_to_indices[BLADE_NAMES[b]] = list(range(idx, idx + count))
            idx += count

        self.blade_masks = masks

    def load_bank(self, bank_path: str):
        """Load primitive structure from bank_full.json.

        Assigns each primitive a global index and builds blade masks.
        """
        path = Path(bank_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge bank not found: {bank_path}")

        with open(path, "r", encoding="utf-8") as f:
            bank = json.load(f)

        idx = 0
        primitive_info = []
        blade_masks = torch.zeros(len(BLADE_NAMES), self.n_primitives)

        for blade_name in BLADE_NAMES:
            if blade_name not in bank.get("blades", {}):
                continue

            blade_data = bank["blades"][blade_name]
            blade_idx = BLADE_INDEX[blade_name]

            for family_code, family_data in blade_data.get("families", {}).items():
                for prim in family_data.get("primitives", []):
                    if idx >= self.n_primitives:
                        break

                    primitive_info.append({
                        "index": idx,
                        "blade": blade_name,
                        "blade_idx": blade_idx,
                        "family": family_code,
                        "symbol": prim.get("symbol", f"P_{idx}"),
                        "name": prim.get("name", ""),
                        "definition": prim.get("definition", ""),
                    })

                    blade_masks[blade_idx, idx] = 1.0
                    self.blade_to_indices[blade_name].append(idx)
                    idx += 1

        self.primitive_info = primitive_info
        self.blade_masks = blade_masks

        # Resize if bank has fewer primitives than configured
        actual_count = idx
        if actual_count < self.n_primitives:
            # Mask out unused slots
            self.blade_masks[:, actual_count:] = 0

    def get_blade_primitives(self, blade_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get primitive embeddings and indices for a specific blade.

        Returns:
            embeddings: (n_blade_prims, d_blade)
            indices: (n_blade_prims,) global indices
        """
        mask = self.blade_masks[blade_idx]
        indices = mask.nonzero(as_tuple=True)[0]
        return self.embeddings(indices), indices

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all embeddings and blade masks.

        Returns:
            embeddings: (n_primitives, d_blade)
            blade_masks: (n_blades, n_primitives)
        """
        return self.embeddings.weight, self.blade_masks
