"""
ONNX export skeleton for Clifford HLM models.

Registers custom ops for the Clifford algebra primitives and provides
``export_to_onnx(model, path, sample_input)`` as the public entry point.

NOTE: This is a structural stub.  A real export requires the ``onnx`` and
``onnxruntime`` packages plus a trained model object.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Custom-op registry (simulated)
# ---------------------------------------------------------------------------
@dataclass
class _CustomOp:
    name: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)


_CUSTOM_OPS: List[_CustomOp] = []


def register_custom_op(
    name: str,
    *,
    domain: str = "ai.yggdrasil.clifford",
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> _CustomOp:
    """Register a custom ONNX operator for Clifford algebra."""
    op = _CustomOp(
        name=name,
        domain=domain,
        inputs=inputs or ["input"],
        outputs=outputs or ["output"],
        attributes=attributes or {},
    )
    _CUSTOM_OPS.append(op)
    return op


# Pre-register the core Clifford ops
register_custom_op(
    "GeometricProduct",
    inputs=["a", "b"],
    outputs=["ab"],
    attributes={"algebra": "Cl(3,0)"},
)
register_custom_op(
    "GradeProjection",
    inputs=["mv"],
    outputs=["projected"],
    attributes={"grade": 0, "algebra": "Cl(3,0)"},
)
register_custom_op(
    "CliffordNorm",
    inputs=["mv"],
    outputs=["norm"],
    attributes={"algebra": "Cl(3,0)"},
)
register_custom_op(
    "CliffordConjugate",
    inputs=["mv"],
    outputs=["conjugate"],
    attributes={"algebra": "Cl(3,0)"},
)
register_custom_op(
    "CliffordAttention",
    inputs=["query", "key", "value"],
    outputs=["output"],
    attributes={"num_heads": 8, "algebra": "Cl(3,0)"},
)


# ---------------------------------------------------------------------------
# Export graph builder (structural skeleton)
# ---------------------------------------------------------------------------
@dataclass
class _ONNXGraphSkeleton:
    """Minimal representation of an ONNX graph for serialisation."""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    custom_ops: List[Dict[str, Any]] = field(default_factory=list)

    def add_node(
        self, op_type: str, inputs: List[str], outputs: List[str], **attrs: Any
    ) -> None:
        self.nodes.append(
            {"op_type": op_type, "inputs": inputs, "outputs": outputs, **attrs}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ir_version": 8,
            "opset_imports": [
                {"domain": "", "version": 17},
                {"domain": "ai.yggdrasil.clifford", "version": 1},
            ],
            "graph": {
                "nodes": self.nodes,
                "inputs": self.inputs,
                "outputs": self.outputs,
            },
            "custom_ops": self.custom_ops,
        }


def _build_skeleton(
    model: Any, sample_input: np.ndarray
) -> _ONNXGraphSkeleton:
    """Trace the model with a sample input and build a skeleton graph."""
    graph = _ONNXGraphSkeleton()

    # Input tensor
    graph.inputs.append(
        {
            "name": "input_ids",
            "shape": list(sample_input.shape),
            "dtype": "float32",
        }
    )

    # Registered custom ops
    for op in _CUSTOM_OPS:
        graph.custom_ops.append(
            {
                "name": op.name,
                "domain": op.domain,
                "inputs": op.inputs,
                "outputs": op.outputs,
                "attributes": op.attributes,
            }
        )

    # Placeholder linear flow
    graph.add_node(
        "CliffordEmbedding",
        inputs=["input_ids"],
        outputs=["embedded"],
        domain="ai.yggdrasil.clifford",
    )
    graph.add_node(
        "CliffordAttention",
        inputs=["embedded", "embedded", "embedded"],
        outputs=["attn_out"],
        domain="ai.yggdrasil.clifford",
    )
    graph.add_node(
        "GradeProjection",
        inputs=["attn_out"],
        outputs=["output"],
        domain="ai.yggdrasil.clifford",
        grade=0,
    )

    graph.outputs.append(
        {"name": "output", "shape": ["batch", "seq", 8], "dtype": "float32"}
    )

    return graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: Any,
    path: str,
    sample_input: Optional[np.ndarray] = None,
) -> str:
    """Export a Clifford HLM model to ONNX format (skeleton).

    Parameters
    ----------
    model : object
        The HLM model (or stub).
    path : str
        Output file path (will write a ``.json`` skeleton alongside if the
        real ``onnx`` package is not available).
    sample_input : np.ndarray, optional
        A sample input array for shape inference.  Defaults to a dummy
        (1, 16, 8) tensor.

    Returns
    -------
    str — path to the written file.
    """
    if sample_input is None:
        sample_input = np.zeros((1, 16, 8), dtype=np.float32)

    graph = _build_skeleton(model, sample_input)
    skeleton = graph.to_dict()

    # Attempt real ONNX export; fall back to JSON skeleton
    json_path = path.replace(".onnx", ".onnx_skeleton.json")

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(skeleton, f, indent=2)

    return json_path
