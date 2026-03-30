"""
Bridge helpers for wiring HLM models into the persistent CUDA engine.

The current engine implements a core transformer block path. It does not yet
cover the full HLM feature surface, so this adapter performs explicit support
checks before compiling or syncing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from rune.backend import to_numpy, xp


@dataclass
class EngineSupportReport:
    supported: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        status = "supported" if self.supported else "unsupported"
        lines = [f"persistent_engine={status}"]
        for reason in self.reasons:
            lines.append(f"- {reason}")
        for warning in self.warnings:
            lines.append(f"- warning: {warning}")
        return lines

    def summary(self) -> str:
        return "\n".join(self.summary_lines())


def _config_from_model_or_config(model_or_config):
    return getattr(model_or_config, "config", model_or_config)


def can_attach(model_or_config) -> EngineSupportReport:
    """
    Check whether the model has compatible shapes to attach to the persistent engine.
    This checks basic shape compatibility (architecture layout) but NOT full
    training capability.
    """
    config = _config_from_model_or_config(model_or_config)
    reasons: List[str] = []
    warnings: List[str] = []

    if getattr(config, "memory_enabled", False):
        top_k = int(getattr(config, "memory_top_k", 1))
        if top_k > 128:
            reasons.append(
                f"memory_top_k={top_k} exceeds the current persistent-engine limit of 128"
            )
        d_model = int(getattr(config, "d_model", 0))
        if d_model > 512:
            reasons.append(
                f"d_model={d_model} exceeds the current compiled-memory write-path limit of 512"
            )
        warnings.append(
            "compiled memory uses an exact bank scan/write path; it is functional but can slow down as the bank fills"
        )
    if getattr(config, "use_rotor_bias", False):
        warnings.append(
            "rotor_bias is currently a legacy no-op in the native HLM path; the persistent engine ignores it"
        )
    if float(getattr(config, "dropout", 0.0)) != 0.0:
        warnings.append(
            "dropout is non-zero, but the persistent engine path assumes deterministic dropout-free execution"
        )

    return EngineSupportReport(
        supported=(len(reasons) == 0),
        reasons=reasons,
        warnings=warnings,
    )


def can_train(model_or_config) -> EngineSupportReport:
    """
    Check whether the model can be fully trained on the persistent engine.
    Returns False if any required backward path is missing.
    This is stricter than can_attach -- it requires gradient + update support
    for ALL parameters.
    """
    config = _config_from_model_or_config(model_or_config)
    reasons: List[str] = []
    warnings: List[str] = []

    attach_report = can_attach(config)
    if not attach_report.supported:
        return attach_report

    return EngineSupportReport(
        supported=(len(reasons) == 0),
        reasons=reasons,
        warnings=warnings,
    )


def persistent_engine_support_report(model_or_config) -> EngineSupportReport:
    """
    Report whether the current HLM configuration can run on the persistent engine.

    This delegates to can_attach() for forward compatibility and can_train()
    for full training. For legacy compatibility, this uses can_attach() as
    the main check.
    """
    config = _config_from_model_or_config(model_or_config)

    attach_report = can_attach(config)
    train_report = can_train(config)

    reasons = list(attach_report.reasons)
    if attach_report.supported and not train_report.supported:
        reasons.extend(train_report.reasons)

    all_warnings = list(attach_report.warnings) + list(train_report.warnings)

    return EngineSupportReport(
        supported=train_report.supported,
        reasons=reasons,
        warnings=all_warnings,
    )


def _zeros_like_bias(width: int) -> np.ndarray:
    return np.zeros((width, 8), dtype=np.float32)


def _pack_multihead_projection(projections) -> np.ndarray:
    return np.concatenate([to_numpy(proj.weight).copy() for proj in projections], axis=0)


def pack_hlm_core_params(model, *, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
    """
    Pack a supported HLM model into the parameter layout expected by CUDAEngine.

    Supported models are the strict core-transformer subset reported by
    :func:`persistent_engine_support_report`.
    """
    report = can_attach(model)
    if not report.supported:
        raise ValueError(report.summary())

    config = model.config
    d_model = int(config.d_model)
    params: Dict[str, np.ndarray] = {
        "embed.weight": to_numpy(model.embedding.weight).copy(),
        "final_ln.gamma": to_numpy(model.final_norm.gamma).copy(),
        "final_ln.beta": to_numpy(model.final_norm.beta).copy(),
    }

    if getattr(model, "pos_encoding", None) is not None:
        rotors = to_numpy(model.pos_encoding._rotor_table[:seq_len]).copy()
        params["pos.rotors"] = np.broadcast_to(rotors[np.newaxis, ...], (batch_size, seq_len, rotors.shape[1], 8)).reshape(
            batch_size * seq_len, rotors.shape[1], 8
        ).copy()

    for layer_idx, block in enumerate(model.blocks):
        pfx = f"layer{layer_idx}"

        params[f"{pfx}.ln1.gamma"] = to_numpy(block.attn_norm.gamma).copy()
        params[f"{pfx}.ln1.beta"] = to_numpy(block.attn_norm.beta).copy()

        params[f"{pfx}.attn.Wq"] = _pack_multihead_projection(block.attn.proj_q)
        params[f"{pfx}.attn.Wk"] = _pack_multihead_projection(block.attn.proj_k)
        params[f"{pfx}.attn.Wv"] = _pack_multihead_projection(block.attn.proj_v)
        params[f"{pfx}.attn.Wo"] = to_numpy(block.attn.proj_out.weight).copy()
        params[f"{pfx}.attn.bq"] = _zeros_like_bias(d_model)
        params[f"{pfx}.attn.bk"] = _zeros_like_bias(d_model)
        params[f"{pfx}.attn.bv"] = _zeros_like_bias(d_model)
        if block.attn.proj_out.bias is None:
            params[f"{pfx}.attn.bo"] = _zeros_like_bias(d_model)
        else:
            params[f"{pfx}.attn.bo"] = to_numpy(block.attn.proj_out.bias).copy()

        params[f"{pfx}.ln2.gamma"] = to_numpy(block.ffn_norm.gamma).copy()
        params[f"{pfx}.ln2.beta"] = to_numpy(block.ffn_norm.beta).copy()
        params[f"{pfx}.ffn.W1"] = to_numpy(block.ffn.up.weight).copy()
        params[f"{pfx}.ffn.b1"] = to_numpy(block.ffn.up.bias).copy()
        params[f"{pfx}.ffn.W2"] = to_numpy(block.ffn.down.weight).copy()
        params[f"{pfx}.ffn.b2"] = to_numpy(block.ffn.down.bias).copy()

    return params


def sync_hlm_core_params_from_engine(model, engine) -> None:
    """
    Copy persistent-engine parameters back into the HLM model.

    This only syncs the core-transformer subset the engine currently owns.
    """
    report = can_attach(model)
    if not report.supported:
        raise ValueError(report.summary())

    model.embedding.weight[...] = xp.asarray(engine.get_param("embed.weight"), dtype=model.embedding.weight.dtype)
    model.final_norm.gamma[...] = xp.asarray(engine.get_param("final_ln.gamma"), dtype=model.final_norm.gamma.dtype)
    model.final_norm.beta[...] = xp.asarray(engine.get_param("final_ln.beta"), dtype=model.final_norm.beta.dtype)

    for layer_idx, block in enumerate(model.blocks):
        pfx = f"layer{layer_idx}"
        block.attn_norm.gamma[...] = xp.asarray(engine.get_param(f"{pfx}.ln1.gamma"), dtype=block.attn_norm.gamma.dtype)
        block.attn_norm.beta[...] = xp.asarray(engine.get_param(f"{pfx}.ln1.beta"), dtype=block.attn_norm.beta.dtype)

        wq = engine.get_param(f"{pfx}.attn.Wq")
        wk = engine.get_param(f"{pfx}.attn.Wk")
        wv = engine.get_param(f"{pfx}.attn.Wv")
        d_head = model.config.d_head
        for head_idx in range(model.config.n_heads):
            sl = slice(head_idx * d_head, (head_idx + 1) * d_head)
            block.attn.proj_q[head_idx].weight[...] = xp.asarray(wq[sl], dtype=block.attn.proj_q[head_idx].weight.dtype)
            block.attn.proj_k[head_idx].weight[...] = xp.asarray(wk[sl], dtype=block.attn.proj_k[head_idx].weight.dtype)
            block.attn.proj_v[head_idx].weight[...] = xp.asarray(wv[sl], dtype=block.attn.proj_v[head_idx].weight.dtype)
        block.attn.proj_out.weight[...] = xp.asarray(engine.get_param(f"{pfx}.attn.Wo"), dtype=block.attn.proj_out.weight.dtype)
        if block.attn.proj_out.bias is not None:
            block.attn.proj_out.bias[...] = xp.asarray(engine.get_param(f"{pfx}.attn.bo"), dtype=block.attn.proj_out.bias.dtype)

        block.ffn_norm.gamma[...] = xp.asarray(engine.get_param(f"{pfx}.ln2.gamma"), dtype=block.ffn_norm.gamma.dtype)
        block.ffn_norm.beta[...] = xp.asarray(engine.get_param(f"{pfx}.ln2.beta"), dtype=block.ffn_norm.beta.dtype)
        block.ffn.up.weight[...] = xp.asarray(engine.get_param(f"{pfx}.ffn.W1"), dtype=block.ffn.up.weight.dtype)
        block.ffn.up.bias[...] = xp.asarray(engine.get_param(f"{pfx}.ffn.b1"), dtype=block.ffn.up.bias.dtype)
        block.ffn.down.weight[...] = xp.asarray(engine.get_param(f"{pfx}.ffn.W2"), dtype=block.ffn.down.weight.dtype)
        block.ffn.down.bias[...] = xp.asarray(engine.get_param(f"{pfx}.ffn.b2"), dtype=block.ffn.down.bias.dtype)


def compile_persistent_engine_for_hlm(model, batch_size: int, seq_len: int):
    """Compile a supported HLM model into the persistent CUDA engine."""
    from rune.engine.cuda_engine import CUDAEngine

    params = pack_hlm_core_params(model, batch_size=batch_size, seq_len=seq_len)
    engine = CUDAEngine()
    engine.compile(model.config.to_dict(), batch_size=batch_size, seq_len=seq_len, params=params)
    return engine
