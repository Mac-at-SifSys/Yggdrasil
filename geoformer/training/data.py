"""Dataset and DataLoader for GeoFormer training.

Supports:
- Standard text for causal LM pretraining
- WZ1-labeled data with blade activation targets
- Curriculum-based phase switching
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Optional, Callable
from pathlib import Path


class TextDataset(Dataset):
    """Simple token-based dataset for causal LM training.

    Each sample is a fixed-length chunk of tokenized text.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 2048,
    ):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.n_samples = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.token_ids[start:end]
        targets = self.token_ids[start + 1:end + 1]

        return {"input_ids": input_ids, "targets": targets}


class WZ1LabeledDataset(Dataset):
    """Dataset with blade activation labels for auxiliary training.

    JSONL format:
    {
        "messages": [{"role": "...", "content": "..."}],
        "blade_activations": [0.0, 0.8, 0.6, ...],  // 8 floats
        "primitives": ["K_CAS_SYS_1", "K_WIS_POL_2"],  // optional
        "narrative_parse": {"tone": "...", ...}  // optional
    }
    """

    def __init__(
        self,
        path: str,
        tokenizer: Callable,
        max_seq_len: int = 2048,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Build text from chat messages
        messages = sample.get("messages", [])
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"

        # Tokenize
        token_ids = self.tokenizer(text)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)

        result = {"input_ids": input_ids, "targets": targets}

        # Blade activation targets
        if "blade_activations" in sample:
            result["blade_targets"] = torch.tensor(
                sample["blade_activations"], dtype=torch.float32
            )

        # Narrative parse targets
        if "narrative_parse" in sample:
            result["narrative_parse"] = sample["narrative_parse"]

        return result


class CurriculumMixer:
    """Manages curriculum phase transitions during training.

    Phase 1 (Language Foundation): 100% general text
    Phase 2 (Blade Alignment): 70% general + 30% WZ1 labeled
    Phase 3 (Knowledge Grounding): 50% general + 50% WZ1 labeled
    """

    PHASES = {
        1: {"name": "language_foundation", "general_ratio": 1.0},
        2: {"name": "blade_alignment", "general_ratio": 0.7},
        3: {"name": "knowledge_grounding", "general_ratio": 0.5},
    }

    def __init__(
        self,
        total_steps: int,
        phase_splits: List[float] = None,
    ):
        if phase_splits is None:
            phase_splits = [0.6, 0.25, 0.15]  # Default: 60/25/15
        assert len(phase_splits) == 3 and abs(sum(phase_splits) - 1.0) < 1e-6

        self.total_steps = total_steps
        self.phase_boundaries = [
            int(phase_splits[0] * total_steps),
            int((phase_splits[0] + phase_splits[1]) * total_steps),
            total_steps,
        ]

    def get_phase(self, step: int) -> int:
        """Return current phase (1, 2, or 3) based on training step."""
        for phase_idx, boundary in enumerate(self.phase_boundaries):
            if step < boundary:
                return phase_idx + 1
        return 3

    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Return loss component weights for the current phase."""
        phase = self.get_phase(step)

        if phase == 1:
            return {"lm": 1.0, "blade": 0.1, "cayley": 0.01}
        elif phase == 2:
            return {"lm": 1.0, "blade": 0.3, "cayley": 0.01}
        else:  # phase 3
            return {"lm": 1.0, "blade": 0.3, "cayley": 0.05}

    def should_use_labeled(self, step: int) -> bool:
        """Whether to sample from labeled data at this step."""
        phase = self.get_phase(step)
        general_ratio = self.PHASES[phase]["general_ratio"]
        return torch.rand(1).item() > general_ratio


def collate_variable_length(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function that pads variable-length sequences."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    targets = []
    blade_targets = []
    has_blades = "blade_targets" in batch[0]

    for item in batch:
        T = item["input_ids"].shape[0]
        pad_len = max_len - T

        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=0))
        targets.append(F.pad(item["targets"], (0, pad_len), value=-100))

        if has_blades and "blade_targets" in item:
            blade_targets.append(item["blade_targets"])

    result = {
        "input_ids": torch.stack(input_ids),
        "targets": torch.stack(targets),
    }

    if blade_targets:
        result["blade_targets"] = torch.stack(blade_targets)

    return result


# Need F for collate function
import torch.nn.functional as F
