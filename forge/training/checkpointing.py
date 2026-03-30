"""
checkpointing.py -- Grade-stratified model checkpointing.

Saves and loads model/optimizer state in a format that separates each
multivector parameter into its grade components.  This allows:
  - Inspecting per-grade weight magnitudes without loading the whole model
  - Selective loading (e.g., load only scalar weights for a transfer baseline)
  - Clean diffs between checkpoints at each grade
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from typing import Dict, Any, Optional, Tuple

from rune.backend import to_numpy
from rune.types.multivector import GRADE_SLICES
from forge.param_utils import get_param_data, get_param_name


def _stratify_params(params: list) -> dict:
    """Split each parameter's multivector data into grade-keyed arrays.

    Returns
    -------
    dict
        ``{param_name: {grade_k: np.ndarray for k in 0..3}}``
    """
    stratified = {}
    for i, p in enumerate(params):
        name = get_param_name(p, i)
        data = to_numpy(get_param_data(p))
        grades = {}
        if data.shape and data.shape[-1] == 8:
            for grade, slc in GRADE_SLICES.items():
                grades[f"grade_{grade}"] = data[..., slc].copy()
        else:
            grades["raw"] = data.copy()
        stratified[name] = grades
    return stratified


def _reconstruct_params(stratified: dict) -> dict:
    """Reconstruct flat (8,) arrays from grade-stratified data.

    Returns
    -------
    dict
        ``{param_name: np.ndarray shape (..., 8)}``
    """
    reconstructed = {}
    for name, grades in stratified.items():
        if "raw" in grades:
            reconstructed[name] = grades["raw"].copy()
            continue
        # Determine batch shape from grade_0
        g0 = grades["grade_0"]
        batch_shape = g0.shape[:-1]
        data = np.zeros((*batch_shape, 8), dtype=np.float32)
        for grade, slc in GRADE_SLICES.items():
            key = f"grade_{grade}"
            if key in grades:
                data[..., slc] = grades[key]
        reconstructed[name] = data
    return reconstructed


def save_checkpoint(
    model,
    optimizer,
    step: int,
    path: str,
    extra: Optional[dict] = None,
):
    """Save a grade-stratified checkpoint to disk.

    Parameters
    ----------
    model : object
        Must implement ``parameters() -> list of dict``.
    optimizer : object
        Must implement ``state_dict() -> dict``.
    step : int
        Current training step.
    path : str
        Output file path (.npz).
    extra : dict, optional
        Additional metadata to save.
    """
    params = model.parameters()
    stratified = _stratify_params(params)

    save_dict = {
        "step": np.array(step, dtype=np.int64),
    }

    # Flatten stratified params for np.savez
    for param_name, grades in stratified.items():
        for grade_key, arr in grades.items():
            save_key = f"param__{param_name}__{grade_key}"
            save_dict[save_key] = arr

    # Optimizer state
    opt_state = optimizer.state_dict()
    # Serialize optimizer scalars
    save_dict["opt__t"] = np.array(opt_state.get("t", 0), dtype=np.int64)
    save_dict["opt__lr"] = np.array(opt_state.get("lr", 0.0), dtype=np.float64)

    # Optimizer per-parameter state (m, v buffers)
    if "states" in opt_state:
        for i, s in enumerate(opt_state["states"]):
            for key, arr in s.items():
                save_dict[f"opt__state_{i}__{key}"] = arr

    # Extra metadata
    if extra:
        for k, v in extra.items():
            if isinstance(v, np.ndarray):
                save_dict[f"extra__{k}"] = v
            else:
                save_dict[f"extra__{k}"] = np.array(v)

    np.savez_compressed(path, **save_dict)


def load_checkpoint(
    path: str,
) -> Tuple[dict, dict, int]:
    """Load a grade-stratified checkpoint.

    Parameters
    ----------
    path : str
        Path to the .npz checkpoint file.

    Returns
    -------
    (model_state, optimizer_state, step)
        model_state : dict mapping param_name -> np.ndarray shape (..., 8)
        optimizer_state : dict with 't', 'lr', and per-param 'm'/'v' buffers
        step : int
    """
    data = np.load(path, allow_pickle=False)

    step = int(data["step"])

    # Reconstruct model parameters
    stratified = {}  # param_name -> {grade_key: array}
    for key in data.files:
        if key.startswith("param__"):
            parts = key.split("__")
            # param__<name>__grade_<k>
            param_name = parts[1]
            grade_key = parts[2]
            if param_name not in stratified:
                stratified[param_name] = {}
            stratified[param_name][grade_key] = data[key]

    model_state = _reconstruct_params(stratified)

    # Reconstruct optimizer state
    opt_state = {}
    opt_state["t"] = int(data.get("opt__t", 0))
    opt_state["lr"] = float(data.get("opt__lr", 0.0))

    # Per-parameter optimizer buffers
    param_states = {}
    for key in data.files:
        if key.startswith("opt__state_"):
            # opt__state_<idx>__<buf_name>
            parts = key.split("__")
            idx = int(parts[1].replace("state_", ""))
            buf_name = parts[2]
            if idx not in param_states:
                param_states[idx] = {}
            param_states[idx][buf_name] = data[key]

    if param_states:
        max_idx = max(param_states.keys())
        opt_state["states"] = [
            param_states.get(i, {}) for i in range(max_idx + 1)
        ]

    return model_state, opt_state, step
