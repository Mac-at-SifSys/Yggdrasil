"""
Native Clifford API — full multivector output for downstream
Clifford-aware consumers (KL-speaking agents).

Endpoints:
    POST /v1/clifford/embed          full 8-component MV embeddings
    POST /v1/clifford/grade_analysis per-grade energy breakdown
    POST /v1/clifford/attention_map  Clifford attention scores
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


# Grade labels for human-readable output
GRADE_LABELS: Dict[int, str] = {
    0: "scalar",
    1: "vector (e1,e2,e3)",
    2: "bivector (e12,e13,e23)",
    3: "trivector (e123)",
}

GRADE_SLICES: Dict[int, slice] = {
    0: slice(0, 1),
    1: slice(1, 4),
    2: slice(4, 7),
    3: slice(7, 8),
}


def _embed_texts(model: Any, texts: List[str]) -> np.ndarray:
    """Prefer the model batch path so Clifford embeds fan out inside one call."""
    if hasattr(model, "embed_batch"):
        embeddings = model.embed_batch(texts)
    else:
        embeddings = [model.embed(text) for text in texts]

    embeddings = np.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


# ---------------------------------------------------------------------------
# /v1/clifford/embed
# ---------------------------------------------------------------------------
def handle_clifford_embed(handler: Any, body: Dict, model: Any) -> None:
    """Return full 8-component multivector embeddings.

    Request body:
        {"text": "hello"}            single text
        {"text": ["hello", "world"]} batch

    Response:
        {
          "embeddings": [
            {
              "text": "hello",
              "multivector": [s, e1, e2, e3, e12, e13, e23, e123],
              "grades": {
                "0": {"label": "scalar", "components": [s]},
                "1": {"label": "vector ...", "components": [e1, e2, e3]},
                ...
              }
            },
            ...
          ]
        }
    """
    texts = body.get("text", "")
    if isinstance(texts, str):
        texts = [texts]

    embeddings = _embed_texts(model, texts)

    results = []
    for t, mv in zip(texts, embeddings):
        mv_list = mv.tolist() if hasattr(mv, "tolist") else list(mv)
        grades_info = {}
        for g, slc in GRADE_SLICES.items():
            components = mv_list[slc]
            grades_info[str(g)] = {
                "label": GRADE_LABELS[g],
                "components": components,
            }
        results.append(
            {
                "text": t,
                "multivector": mv_list,
                "grades": grades_info,
            }
        )

    handler._send_json({"embeddings": results})


# ---------------------------------------------------------------------------
# /v1/clifford/grade_analysis
# ---------------------------------------------------------------------------
def handle_grade_analysis(handler: Any, body: Dict, model: Any) -> None:
    """Return per-grade energy breakdown.

    Request body:
        {"text": "hello"}

    Response:
        {
          "text": "hello",
          "grade_energy": {
            "0": {"label": "scalar", "energy": ..., "fraction": ...},
            ...
          },
          "total_energy": ...,
          "dominant_grade": 1
        }
    """
    text = body.get("text", "")
    energy_map = model.grade_energy(text)

    total = sum(energy_map.values())
    total = max(total, 1e-12)

    grade_energy_resp: Dict[str, Any] = {}
    dominant_grade = 0
    max_energy = -1.0
    for g in range(4):
        e = energy_map.get(g, 0.0)
        frac = e / total
        grade_energy_resp[str(g)] = {
            "label": GRADE_LABELS[g],
            "energy": e,
            "fraction": frac,
        }
        if e > max_energy:
            max_energy = e
            dominant_grade = g

    handler._send_json(
        {
            "text": text,
            "grade_energy": grade_energy_resp,
            "total_energy": total,
            "dominant_grade": dominant_grade,
        }
    )


# ---------------------------------------------------------------------------
# /v1/clifford/attention_map
# ---------------------------------------------------------------------------
def handle_attention_map(handler: Any, body: Dict, model: Any) -> None:
    """Return Clifford attention scores.

    Request body:
        {"text": "the cat sat"}

    Response:
        {
          "text": "the cat sat",
          "tokens": ["the", "cat", "sat"],
          "attention_scores": [[...], [...], [...]],
          "shape": [3, 3]
        }
    """
    text = body.get("text", "")
    tokens = text.split() if text else []
    attn = model.attention_map(text)

    # normalise to plain list
    if hasattr(attn, "tolist"):
        attn = attn.tolist()

    handler._send_json(
        {
            "text": text,
            "tokens": tokens,
            "attention_scores": attn,
            "shape": [len(tokens), len(tokens)] if tokens else [0, 0],
        }
    )
