"""
holograph.model -- Loadable top-level HLM wrapper for serving and tooling.

This module bridges the experimental HLM implementation into the serving and
agent layers by providing a stable ``HolographLM`` interface with load/save,
generation, embedding, and analysis helpers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Iterable, List, Optional

import numpy as np

from rune.backend import set_backend, to_numpy

if TYPE_CHECKING:
    from hlm_experiment.models.hlm_125m import HLM125M, HLM125MConfig


def _load_hlm_classes():
    from hlm_experiment.models.hlm_125m import HLM125M, HLM125MConfig

    return HLM125M, HLM125MConfig


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "model.npz")
        if os.path.exists(candidate):
            return candidate
        npz_files = sorted(
            os.path.join(path, name) for name in os.listdir(path)
            if name.endswith(".npz")
        )
        if npz_files:
            return npz_files[0]
        raise FileNotFoundError(f"No .npz checkpoint found in {path}")
    return path


def _configure_backend(device: str) -> None:
    if device == "cuda":
        try:
            set_backend("cuda")
            return
        except Exception:
            set_backend("cpu")
            return
    set_backend("cpu")


def _load_state_arrays_into_model(model, data, checkpoint_path: str) -> None:
    if any(key.startswith("param__") and "__grade_" in key for key in data.files):
        from forge.training.checkpointing import load_checkpoint

        model_state, _, _ = load_checkpoint(checkpoint_path)
        if all(name.startswith("param_") for name in model_state):
            ordered = [model_state[f"param_{i}"] for i in range(len(model.parameters()))]
            for param, value in zip(model.parameters(), ordered):
                param[...] = value
        else:
            model.load_state_dict(model_state, strict=False)
        return

    if any(key.startswith("param__") for key in data.files):
        state = {
            key.split("param__", 1)[1]: data[key]
            for key in data.files
            if key.startswith("param__")
        }
        model.load_state_dict(state, strict=False)
        return

    if any(key.startswith("param_") for key in data.files):
        ordered = sorted(
            (key for key in data.files if key.startswith("param_")),
            key=lambda key: int(key.split("_")[1]),
        )
        for param, key in zip(model.parameters(), ordered):
            param[...] = data[key]


def load_native_hlm(path: str, device: str = "cpu"):
    """Load an ``HLM125M`` model from a serving export or Forge checkpoint."""
    _configure_backend(device)

    checkpoint_path = _resolve_checkpoint_path(path)
    with np.load(checkpoint_path, allow_pickle=False) as data:
        HLM125M, HLM125MConfig = _load_hlm_classes()
        config = HLM125MConfig()
        if "config_json" in data.files:
            config = HLM125MConfig(**json.loads(str(data["config_json"])))

        model = HLM125M(config)
        _load_state_arrays_into_model(model, data, checkpoint_path)
        return model


def _compute_grade_norms(components: np.ndarray) -> dict:
    return {
        0: float(np.linalg.norm(components[0:1])),
        1: float(np.linalg.norm(components[1:4])),
        2: float(np.linalg.norm(components[4:7])),
        3: float(np.linalg.norm(components[7:8])),
    }


class _TokenizerAdapter:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self._enc = None

    def encode(self, text: str) -> List[int]:
        if not text:
            return [0]
        if self._enc is not None:
            return [int(tok) % self.vocab_size for tok in self._enc.encode(text)]
        return [byte % self.vocab_size for byte in text.encode("utf-8")]

    def decode(self, token_ids: Iterable[int]) -> str:
        token_ids = [int(tok) % self.vocab_size for tok in token_ids]
        if not token_ids:
            return ""
        if self._enc is not None:
            return self._enc.decode(token_ids)
        return bytes(tok % 256 for tok in token_ids).decode("utf-8", errors="ignore")


@dataclass
class HolographEncoding:
    components: List[float]
    grade_0_norm: float
    grade_1_norm: float
    grade_2_norm: float
    grade_3_norm: float
    algebraic_consistency: float = 1.0


@dataclass
class HolographGeneration:
    text: str
    confidence: float
    token_count: int
    grade_0_norm: float
    grade_1_norm: float
    grade_2_norm: float
    grade_3_norm: float
    algebraic_consistency: float = 1.0
    multivector_components: Optional[List[float]] = None


class HolographLM:
    def __init__(
        self,
        model: Optional["HLM125M"] = None,
        config: Optional["HLM125MConfig"] = None,
    ):
        HLM125M, HLM125MConfig = _load_hlm_classes()
        self._model = model or HLM125M(config or HLM125MConfig())
        self.config = self._model.config
        self._tokenizer = _TokenizerAdapter(self.config.vocab_size)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HolographLM":
        return cls(model=load_native_hlm(path, device=device))

    def save(self, path: str) -> str:
        if path.endswith(".npz"):
            save_path = path
        else:
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, "model.npz")

        payload = {"config_json": np.array(json.dumps(self.config.to_dict()))}
        for name, value in self._model.state_dict().items():
            payload[f"param__{name}"] = value
        np.savez_compressed(save_path, **payload)
        return save_path

    @property
    def model(self):
        return self._model

    def _build_encoding(self, prompt: str) -> HolographEncoding:
        token_ids = self._tokenizer.encode(prompt)
        token_ids = token_ids[-self.config.max_seq_len :]
        tokens = np.array([token_ids], dtype=np.int64)
        self._model.forward(tokens)

        hidden = self._model.get_last_hidden_state()
        if hidden is None:
            components = np.zeros(8, dtype=np.float32)
        else:
            from holograph.memory.geometric_fold import geometric_fold

            pooled = to_numpy(hidden[0]).mean(axis=0)
            components = to_numpy(geometric_fold(pooled))

        from forge.losses.algebraic_consistency import algebraic_consistency_loss

        grade_norms = _compute_grade_norms(components)
        return HolographEncoding(
            components=components.tolist(),
            grade_0_norm=grade_norms[0],
            grade_1_norm=grade_norms[1],
            grade_2_norm=grade_norms[2],
            grade_3_norm=grade_norms[3],
            algebraic_consistency=float(
                algebraic_consistency_loss(rotor_params=self._model.get_rotor_params())
            ),
        )

    def encode(self, text: str) -> HolographEncoding:
        return self._build_encoding(text)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.encode(text).components, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(text) for text in texts], axis=0)

    def grade_energy(self, text: str) -> dict:
        components = self.embed(text)
        return {
            0: float(np.sum(components[0:1] ** 2)),
            1: float(np.sum(components[1:4] ** 2)),
            2: float(np.sum(components[4:7] ** 2)),
            3: float(np.sum(components[7:8] ** 2)),
        }

    def attention_map(self, text: str) -> List[List[float]]:
        token_ids = self._tokenizer.encode(text)
        token_ids = token_ids[-self.config.max_seq_len :]
        tokens = np.array([token_ids], dtype=np.int64)
        self._model.forward(tokens)

        try:
            head_maps = []
            for head in self._model.blocks[0].attn._cache["heads"]:
                head_maps.append(to_numpy(head["attn_weights"][0]))
            return np.mean(head_maps, axis=0).tolist()
        except Exception:
            n = max(len(token_ids), 1)
            return np.eye(n, dtype=np.float32).tolist()

    def _sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[int, float]:
        if temperature <= 0:
            idx = int(np.argmax(logits))
            return idx, 1.0

        scaled = logits / max(temperature, 1e-6)
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum() + 1e-12

        if 0 < top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            keep = cumulative <= top_p
            if not np.any(keep):
                keep[0] = True
            filtered = probs[order] * keep
            filtered /= filtered.sum() + 1e-12
            choice = int(np.random.choice(len(order), p=filtered))
            idx = int(order[choice])
            return idx, float(probs[idx])

        idx = int(np.random.choice(len(probs), p=probs))
        return idx, float(probs[idx])

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        grade_mask: Optional[List[int]] = None,
    ) -> HolographGeneration:
        context = self._tokenizer.encode(prompt)
        if not context:
            context = [0]

        generated: List[int] = []
        confidences: List[float] = []

        for _ in range(max_tokens):
            window = context[-self.config.max_seq_len :]
            tokens = np.array([window], dtype=np.int64)
            logits = to_numpy(self._model.forward(tokens))
            next_token, confidence = self._sample_next_token(
                logits[0, -1], temperature=temperature, top_p=top_p
            )
            context.append(next_token)
            generated.append(next_token)
            confidences.append(confidence)

        text = self._tokenizer.decode(generated)
        encoding = self._build_encoding(prompt + text)
        return HolographGeneration(
            text=text,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            token_count=len(generated),
            grade_0_norm=encoding.grade_0_norm,
            grade_1_norm=encoding.grade_1_norm,
            grade_2_norm=encoding.grade_2_norm,
            grade_3_norm=encoding.grade_3_norm,
            algebraic_consistency=encoding.algebraic_consistency,
            multivector_components=encoding.components,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Generator[str, None, None]:
        result = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for token in result.text.split():
            yield token + " "
