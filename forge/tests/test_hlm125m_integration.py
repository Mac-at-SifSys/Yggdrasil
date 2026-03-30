"""
Integration tests for wiring HLM125M into the Forge training stack.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from forge.optimizers.clifford_adam import CliffordAdam
from forge.training.trainer import CliffordTrainer
from hlm_experiment.models.hlm_125m import HLM125M, HLM125MConfig


def _tiny_config():
    return HLM125MConfig(
        vocab_size=32,
        d_model=4,
        n_layers=1,
        n_heads=2,
        d_ff=8,
        max_seq_len=8,
        use_rotor_bias=False,
        memory_enabled=False,
    )


def test_hlm125m_train_step_runs_through_forge():
    np.random.seed(42)

    model = HLM125M(_tiny_config())
    optimizer = CliffordAdam(model.parameters(), lr=1e-3)
    trainer = CliffordTrainer(
        model=model,
        optimizer=optimizer,
        alg_weight=0.0,
        entropy_weight=0.01,
        log_interval=999,
    )

    before = model.embedding.weight.copy()
    input_ids = np.random.randint(0, model.config.vocab_size, (1, 4), dtype=np.int64)
    target_ids = np.random.randint(0, model.config.vocab_size, (1, 4), dtype=np.int64)

    metrics = trainer.train_step((input_ids, target_ids))

    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["ce_loss"])
    assert np.isfinite(metrics["grad_norm"])
    assert not np.allclose(before, model.embedding.weight)


def test_hlm125m_train_step_accepts_torch_batches():
    torch.manual_seed(11)

    model = HLM125M(_tiny_config())
    optimizer = CliffordAdam(model.parameters(), lr=1e-3)
    trainer = CliffordTrainer(
        model=model,
        optimizer=optimizer,
        alg_weight=0.0,
        entropy_weight=0.0,
        log_interval=999,
    )

    input_ids = torch.randint(0, model.config.vocab_size, (1, 4), dtype=torch.long)
    target_ids = torch.randint(0, model.config.vocab_size, (1, 4), dtype=torch.long)

    metrics = trainer.train_step((input_ids, target_ids))

    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["ce_loss"])
